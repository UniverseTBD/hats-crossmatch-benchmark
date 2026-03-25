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
from datasets import IterableDataset
from lsdb.dask import merge_catalog_functions
from lsdb.streams import CatalogStream


def _make_async_read_patch():
    """Create a patched ``read_parquet_file_to_pandas`` that uses async HTTP.

    Returns the patched function.  The patch resolves ``hf://`` URLs to
    ``https://`` and reads via ``fsspec.implementations.http.HTTPFileSystem``
    (which inherits from ``AsyncFileSystem`` and uses aiohttp in a background
    thread, bypassing the GIL).  Non-HF paths fall through to the original.
    """
    import re

    import fsspec.implementations.http
    import hats.io.file_io.file_io as _file_io
    from huggingface_hub import hf_hub_url

    _orig = _file_io.read_parquet_file_to_pandas
    _http_fs = fsspec.implementations.http.HTTPFileSystem()

    _HF_RE = re.compile(r"^hf://datasets/([^/]+/[^/]+)/(.+)$")

    def _patched_read(path, *args, **kwargs):
        path_str = str(path)
        m = _HF_RE.match(path_str)
        if m:
            repo_id, file_path = m.group(1), m.group(2)
            url = hf_hub_url(repo_id=repo_id, filename=file_path, repo_type="dataset")
            return npd.read_parquet(url, filesystem=_http_fs, **kwargs)
        return _orig(path, *args, **kwargs)

    return _patched_read


def _iter_rows_sharded(partition_indices, dask_partitions, prefetch=16):
    """Yield one dict per row, computing only the given partition indices.

    Each element of ``partition_indices`` is a single int (one HF shard per
    LSDB partition).  HF's ``from_generator`` distributes these across
    DataLoader workers so each worker gets a disjoint subset.

    ``dask_partitions`` is a ``tuple`` (broadcast, not sharded) of
    lightweight Dask graph references.  With ``fork`` start method,
    workers inherit these via shared memory.

    Within each worker, indices are grouped into ``prefetch``-sized batches
    and computed via a double-buffered pipeline to overlap I/O with yielding.
    """
    import os

    import dask
    import hats.io.file_io.file_io as _file_io

    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

    _orig_read = _file_io.read_parquet_file_to_pandas
    _file_io.read_parquet_file_to_pandas = _make_async_read_patch()

    try:
        # HF wraps each shard element in a list — unwrap to plain ints.
        indices = [
            x if isinstance(x, int) else x[0]
            for x in partition_indices
        ]

        if prefetch <= 1:
            for idx in indices:
                chunk = dask_partitions[idx].compute(scheduler="synchronous")
                yield from chunk.to_dict("records")
            return

        from concurrent.futures import ThreadPoolExecutor

        def compute_batch(batch_indices):
            parts = [dask_partitions[i] for i in batch_indices]
            return dask.compute(*parts, scheduler="threads", num_workers=prefetch)

        batches = [
            indices[i : i + prefetch]
            for i in range(0, len(indices), prefetch)
        ]

        with ThreadPoolExecutor(max_workers=1) as bg:
            future = bg.submit(compute_batch, batches[0])
            for i, batch in enumerate(batches):
                results = future.result()
                if i + 1 < len(batches):
                    future = bg.submit(compute_batch, batches[i + 1])
                for chunk in results:
                    yield from chunk.to_dict("records")
    finally:
        _file_io.read_parquet_file_to_pandas = _orig_read


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
    columns_a: list[str] | None = None,
    columns_b: list[str] | None = None,
    partitions_per_chunk: int = 1,
    n_workers: int | None = None,
    sharded: bool = True,
    prefetch: int = 16,
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
    columns_a : list[str] or None
        Columns to read from catalog A. ``None`` reads all columns.
        Selecting only needed columns avoids fetching large columns
        (e.g. ``spectrum``) and can dramatically improve throughput.
    columns_b : list[str] or None
        Columns to read from catalog B. ``None`` reads all columns.
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
    prefetch : int
        Number of partitions to compute concurrently using a thread pool.
        Overlaps network I/O with CPU crossmatch work (both release the
        GIL). Set to 1 to disable prefetching. Only used when
        ``sharded=True``.

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

    if columns_a is not None:
        kwargs_a["columns"] = columns_a
    if columns_b is not None:
        kwargs_b["columns"] = columns_b

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
        # Pre-extract Dask partition references — lightweight graph objects,
        # not computed data. As a tuple, HF broadcasts them to all shards
        # (only lists trigger sharding). With fork start method, DataLoader
        # workers inherit these via shared memory — no catalog re-opening.
        dask_partitions = tuple(
            xmatch._ddf.get_partition(i) for i in range(xmatch._ddf.npartitions)
        )
        # Each partition index is its own HF shard (1:1 mapping), so
        # DataLoader(num_workers=N) can distribute individual partitions
        # across N workers.  Prefetch batching happens inside the generator.
        n = len(dask_partitions)
        partition_indices = list(range(n))
        ds = IterableDataset.from_generator(
            _iter_rows_sharded,
            gen_kwargs={
                "partition_indices": partition_indices,
                "dask_partitions": dask_partitions,
                "prefetch": prefetch,
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
