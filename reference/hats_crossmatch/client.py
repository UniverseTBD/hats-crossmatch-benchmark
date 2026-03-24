"""Streaming crossmatch client that returns an HF IterableDataset.

Wraps lsdb's CatalogStream (which yields pandas DataFrames) into a
datasets.IterableDataset (which yields dicts row-by-row), giving
downstream code a unified interface whether data comes from a live
lsdb crossmatch or a pre-computed HuggingFace dataset.

When ``sharded=True`` (the default), each LSDB HEALPix partition becomes
a separate HF shard, so ``DataLoader(ds, num_workers=N)`` can parallelise
across partitions without Dask distributed.
"""

from __future__ import annotations

from typing import Any

import lsdb
import nested_pandas as npd
import pandas as pd
from datasets import Features, IterableDataset, Value
from lsdb.dask import merge_catalog_functions
from lsdb.streams import CatalogStream

# Module-level cache so each DataLoader worker process only opens catalogs once.
_catalog_cache: dict[tuple, Any] = {}


def _get_or_open_crossmatch(
    url_a: str,
    url_b: str,
    kwargs_a: tuple[tuple[str, Any], ...],
    kwargs_b: tuple[tuple[str, Any], ...],
    xmatch_kwargs: tuple[tuple[str, Any], ...],
):
    """Return a cached lazy crossmatch catalog, opening it if needed."""
    key = (url_a, url_b, kwargs_a, kwargs_b, xmatch_kwargs)
    if key not in _catalog_cache:
        cat_a = lsdb.open_catalog(url_a, **dict(kwargs_a))
        cat_b = lsdb.open_catalog(url_b, **dict(kwargs_b))
        _catalog_cache[key] = cat_a.crossmatch(cat_b, **dict(xmatch_kwargs))
    return _catalog_cache[key]


def _iter_rows_sharded(
    partition_indices,
    url_a,
    url_b,
    kwargs_a,
    kwargs_b,
    xmatch_kwargs,
):
    """Yield one dict per row, computing only the given partition indices.

    ``partition_indices`` is a ``list`` so that HF's ``from_generator``
    treats it as the shard axis — each worker gets a disjoint subset.
    All other parameters are strings or tuples (broadcast, not sharded).
    """
    xmatch = _get_or_open_crossmatch(url_a, url_b, kwargs_a, kwargs_b, xmatch_kwargs)
    for idx in partition_indices:
        chunk = xmatch._ddf.get_partition(idx).compute()
        for record in chunk.to_dict("records"):
            yield record


def _concat_partition_and_margin(
    partition: npd.NestedFrame, margin: npd.NestedFrame | None
) -> npd.NestedFrame:
    """Concatenate partition and margin, handling empty frames without warning."""
    if margin is None or len(margin) == 0:
        return partition
    if len(partition) == 0:
        return npd.NestedFrame(margin)
    return npd.NestedFrame(pd.concat([partition, margin]))


merge_catalog_functions.concat_partition_and_margin = _concat_partition_and_margin


def _iter_rows(xmatch, stats=None, n_workers=None, partitions_per_chunk=1):
    """Yield one dict per row from the crossmatch CatalogStream.

    If *stats* is a dict, it is updated in-place with partition progress:
    ``npartitions``, ``partitions_done``.
    """
    client = None
    if n_workers is not None:
        from distributed import Client

        client = Client(n_workers=n_workers, threads_per_worker=1)

    stream = CatalogStream(
        catalog=xmatch,
        client=client,
        partitions_per_chunk=partitions_per_chunk,
        shuffle=False,
    )
    npartitions = xmatch.npartitions
    if stats is not None:
        stats["npartitions"] = npartitions
        stats["partitions_done"] = 0
    for chunk in stream:
        for record in chunk.to_dict("records"):
            yield record
        if stats is not None:
            stats["partitions_done"] += partitions_per_chunk


def stream_crossmatch(
    url_a: str,
    url_b: str,
    *,
    radius_arcsec: float = 1.0,
    n_neighbors: int = 1,
    suffixes: tuple[str, str] = ("_a", "_b"),
    search_filter: Any | None = None,
    storage_options_a: dict | None = None,
    storage_options_b: dict | None = None,
    partitions_per_chunk: int = 1,
    n_workers: int | None = None,
    sharded: bool = True,
) -> IterableDataset:
    """Stream crossmatch results as an HF IterableDataset.

    Parameters
    ----------
    url_a : str
        HATS catalog URL (e.g. "hf://datasets/UniverseTBD/mmu_sdss_sdss").
    url_b : str
        HATS catalog URL.
    radius_arcsec : float
        Crossmatch radius in arcseconds.
    n_neighbors : int
        Maximum number of neighbors to return per source.
    suffixes : tuple[str, str]
        Column suffixes to disambiguate the two catalogs.
    search_filter : optional
        lsdb search filter (e.g. ``lsdb.ConeSearch(...)``).
    storage_options_a : dict or None
        Storage options for catalog A (e.g. ``{"anon": True}`` for S3).
    storage_options_b : dict or None
        Storage options for catalog B.
    partitions_per_chunk : int
        Number of Dask partitions to compute per chunk. Higher values
        increase memory usage but may improve throughput. Only used when
        ``sharded=False``.
    n_workers : int or None
        If given, spin up a ``dask.distributed.Client`` with this many
        single-threaded workers and pass it to ``CatalogStream``. Only
        used when ``sharded=False``.
    sharded : bool
        If True (default), expose each LSDB partition as a separate HF
        shard. This lets ``DataLoader(ds, num_workers=N)`` parallelise
        across partitions without Dask distributed. When True,
        ``n_workers`` and ``partitions_per_chunk`` are ignored — use
        ``DataLoader`` workers instead.

    Returns
    -------
    datasets.IterableDataset
        An iterable dataset where each element is a dict representing
        one crossmatch row.
    """
    kwargs_a: dict[str, Any] = {}
    kwargs_b: dict[str, Any] = {}

    if search_filter is not None:
        kwargs_a["search_filter"] = search_filter
        kwargs_b["search_filter"] = search_filter

    if storage_options_a is not None:
        kwargs_a["storage_options"] = storage_options_a
    elif url_a.startswith("s3://"):
        kwargs_a["storage_options"] = {"anon": True}

    if storage_options_b is not None:
        kwargs_b["storage_options"] = storage_options_b
    elif url_b.startswith("s3://"):
        kwargs_b["storage_options"] = {"anon": True}

    cat_a = lsdb.open_catalog(url_a, **kwargs_a)
    cat_b = lsdb.open_catalog(url_b, **kwargs_b)

    xmatch = cat_a.crossmatch(
        cat_b,
        n_neighbors=n_neighbors,
        radius_arcsec=radius_arcsec,
        suffixes=suffixes,
        suffix_method="all_columns",
    )

    stats: dict[str, int] = {}
    stats["npartitions"] = xmatch.npartitions

    if sharded:
        # Package dicts as tuples of items so HF doesn't treat dict
        # values as shard axes (only lists trigger sharding).
        xmatch_kwargs = tuple(
            {
                "n_neighbors": n_neighbors,
                "radius_arcsec": radius_arcsec,
                "suffixes": suffixes,
                "suffix_method": "all_columns",
            }.items()
        )
        ds = IterableDataset.from_generator(
            _iter_rows_sharded,
            gen_kwargs={
                "partition_indices": list(range(xmatch.npartitions)),
                "url_a": url_a,
                "url_b": url_b,
                "kwargs_a": tuple(kwargs_a.items()),
                "kwargs_b": tuple(kwargs_b.items()),
                "xmatch_kwargs": xmatch_kwargs,
            },
        )
    else:
        ds = IterableDataset.from_generator(
            _iter_rows,
            gen_kwargs={
                "xmatch": xmatch,
                "stats": stats,
                "n_workers": n_workers,
                "partitions_per_chunk": partitions_per_chunk,
            },
        )

    ds.crossmatch_stats = stats
    ds.total_rows_a = cat_a.hc_structure.catalog_info.total_rows
    ds.total_rows_b = cat_b.hc_structure.catalog_info.total_rows
    return ds
