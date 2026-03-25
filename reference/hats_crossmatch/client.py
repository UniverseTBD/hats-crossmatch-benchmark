"""Streaming crossmatch client that returns an HF IterableDataset.

Each LSDB HEALPix partition becomes a separate HF shard, so
``DataLoader(ds, num_workers=N)`` can parallelise across partitions
without Dask distributed.
"""

from __future__ import annotations

from typing import Any

import lsdb
import nested_pandas as npd
import pandas as pd
from datasets import IterableDataset
from lsdb.dask import merge_catalog_functions

# Module-level global used by forked worker processes.
# With fork context, children inherit this via copy-on-write shared memory.
_WORKER_PARTITIONS = None


def _reset_fsspec_after_fork():
    """Make fsspec's async filesystems usable after fork.

    After ``fork()``, inherited ``AsyncFileSystem`` instances hold a dead
    event loop and aiohttp session.  This resets the global loop state
    and patches the ``loop`` property so that inherited instances
    re-initialise their loop *and* session instead of raising
    ``RuntimeError("not fork-safe")``.
    """
    import os

    import fsspec.asyn

    fsspec.asyn.reset_after_fork()
    fsspec.asyn.reset_lock()

    _pid = os.getpid()

    def _loop_auto_reset(self):
        if self._pid != _pid:
            self._pid = _pid
            self._loop = fsspec.asyn.get_loop()
            # Also clear the inherited aiohttp session so a fresh
            # one is created on the new event loop.
            if hasattr(self, '_session'):
                self._session = None
        return self._loop

    fsspec.asyn.AsyncFileSystem.loop = property(_loop_auto_reset)


def _pool_worker(idx):
    """Compute a single partition in a forked worker process."""
    import os

    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

    _reset_fsspec_after_fork()

    chunk = _WORKER_PARTITIONS[idx].compute(scheduler="synchronous")
    return chunk


def _iter_rows_multiproc(dask_partitions, num_proc):
    """Yield one dict per row, using a multiprocessing pool for parallelism."""
    global _WORKER_PARTITIONS
    _WORKER_PARTITIONS = dask_partitions

    import multiprocessing

    ctx = multiprocessing.get_context("fork")
    with ctx.Pool(num_proc) as pool:
        for chunk in pool.imap_unordered(_pool_worker, range(len(dask_partitions))):
            yield from chunk.to_dict("records")


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

    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

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
    prefetch: int = 16,
    num_proc: int = 0,
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
    prefetch : int
        Number of partitions to compute concurrently using a thread pool.
        Overlaps network I/O with CPU crossmatch work (both release the
        GIL). Set to 1 to disable prefetching. Only used when
        ``num_proc=0``.
    num_proc : int
        Number of worker processes for multiprocess parallelism. Uses
        ``multiprocessing.Pool`` with ``fork`` context so children inherit
        the Dask graph via copy-on-write shared memory. Set to 0 (default)
        to use the single-process path.

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
    if num_proc > 0:
        ds = IterableDataset.from_generator(
            _iter_rows_multiproc,
            gen_kwargs={
                "dask_partitions": dask_partitions,
                "num_proc": num_proc,
            },
        )
    else:
        partition_indices = list(range(n))
        ds = IterableDataset.from_generator(
            _iter_rows_sharded,
            gen_kwargs={
                "partition_indices": partition_indices,
                "dask_partitions": dask_partitions,
                "prefetch": prefetch,
            },
        )

    ds.crossmatch_stats = stats
    ds.total_rows_a = cat_a.hc_structure.catalog_info.total_rows
    ds.total_rows_b = cat_b.hc_structure.catalog_info.total_rows
    return ds
