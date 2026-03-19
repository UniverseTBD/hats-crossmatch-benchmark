import time
import warnings
from datetime import datetime, timezone

import click
import lsdb
import numpy as np
from upath import UPath

warnings.filterwarnings("ignore", message="The behavior of array concatenation with empty entries")

from benchmarks.config import BenchmarkConfig, TEST_CONE_DEC, TEST_CONE_RA, resolve_catalog
from benchmarks.metrics import BenchmarkResult, PeakMemoryTracker


def _suppress_concat_warning():
    """Suppress the FutureWarning about array concatenation with empty entries."""
    import warnings
    warnings.filterwarnings("ignore", message="The behavior of array concatenation with empty entries")


def run_stream_benchmark(config: BenchmarkConfig) -> BenchmarkResult:
    """Execute a streaming crossmatch benchmark using lsdb CatalogStream.

    Loads catalogs, plans the crossmatch lazily, then streams partitions
    via CatalogStream with background pre-fetching.
    """
    from lsdb.streams import CatalogStream

    url_a = resolve_catalog(config.catalog_a)
    url_b = resolve_catalog(config.catalog_b)

    result = BenchmarkResult(
        config=config,
        timestamp=datetime.now(timezone.utc).isoformat(),
        mode="stream",
    )

    region = None
    if config.test:
        click.echo(f"  [test mode] Cone search: ({TEST_CONE_RA}, {TEST_CONE_DEC}), r={config.test_radius_deg}°")
        region = lsdb.ConeSearch(TEST_CONE_RA, TEST_CONE_DEC, config.test_radius_deg * 3600)

    path_a = UPath(url_a, anon=True) if url_a.startswith("s3://") else url_a
    path_b = UPath(url_b, anon=True) if url_b.startswith("s3://") else url_b

    # Load catalogs and plan crossmatch (lazy)
    try:
        cat_a = lsdb.open_catalog(path_a, search_filter=region)
        cat_b = lsdb.open_catalog(path_b, search_filter=region)
    except ValueError as e:
        if "no coverage" in str(e).lower():
            click.echo("  Sky region has no coverage — skipping.")
            return result
        raise

    try:
        xmatch = cat_a.crossmatch(
            cat_b,
            n_neighbors=config.n_neighbors,
            radius_arcsec=config.radius_arcsec,
            suffixes=config.suffixes,
            suffix_method="all_columns",
        )
    except RuntimeError as e:
        if "do not overlap" in str(e).lower():
            click.echo("  Catalogs do not overlap — no matches possible.")
            return result
        raise

    # Set up Dask client — synchronous in test mode, distributed otherwise
    from dask.distributed import Client

    client = None
    if config.test:
        import dask
        dask.config.set(scheduler="synchronous")
    else:
        client_kwargs = {"threads_per_worker": 1}
        if config.n_workers is not None:
            client_kwargs["n_workers"] = config.n_workers
        import dask
        dask.config.set({"distributed.admin.large-graph-warning-threshold": "100 MiB"})
        client = Client(**client_kwargs)
        client.run(_suppress_concat_warning)

    n_partitions = xmatch.npartitions
    stream = CatalogStream(catalog=xmatch, client=client, shuffle=False)

    tracker = PeakMemoryTracker()
    tracker.start()
    total_start = time.perf_counter()

    num_matches = 0
    num_batches = 0
    dist_arrays = []
    time_to_first_batch = None

    from tqdm import tqdm

    pbar = tqdm(stream, total=n_partitions, desc="Streaming partitions", unit="part")
    for chunk in pbar:
        if time_to_first_batch is None:
            time_to_first_batch = time.perf_counter() - total_start
        num_batches += 1
        batch_len = len(chunk)
        num_matches += batch_len
        if batch_len > 0 and "_dist_arcsec" in chunk.columns:
            dist_arrays.append(chunk["_dist_arcsec"].to_numpy())
        pbar.set_postfix(matches=f"{num_matches:,}")
    pbar.close()

    time_compute = time.perf_counter() - total_start
    result.memory_peak = tracker.stop()

    if client is not None:
        client.close()

    result.time_compute = time_compute
    result.time_total = time_compute
    result.time_to_first_batch = time_to_first_batch
    result.num_batches = num_batches
    result.num_matches = num_matches
    result.throughput_rows_per_sec = num_matches / time_compute if time_compute > 0 else 0.0

    if dist_arrays:
        dists = np.concatenate(dist_arrays)
        result.dist_mean = float(np.mean(dists))
        result.dist_median = float(np.median(dists))
        result.dist_std = float(np.std(dists))
        result.dist_min = float(np.min(dists))
        result.dist_max = float(np.max(dists))

    return result


def run_stream_hf_benchmark(config: BenchmarkConfig) -> BenchmarkResult:
    """Execute a streaming benchmark reading pre-computed crossmatch results from HF Datasets."""
    import datasets

    result = BenchmarkResult(
        config=config,
        timestamp=datetime.now(timezone.utc).isoformat(),
        mode="stream-hf",
    )

    tracker = PeakMemoryTracker()
    tracker.start()
    total_start = time.perf_counter()

    ds = datasets.load_dataset(config.hf_repo_id, streaming=True, split="train")

    num_rows = 0
    time_to_first_batch = None

    for row in ds:
        if time_to_first_batch is None:
            time_to_first_batch = time.perf_counter() - total_start
        num_rows += 1

    time_compute = time.perf_counter() - total_start
    result.memory_peak = tracker.stop()

    result.time_compute = time_compute
    result.time_total = time_compute
    result.time_to_first_batch = time_to_first_batch
    result.num_matches = num_rows
    result.throughput_rows_per_sec = num_rows / time_compute if time_compute > 0 else 0.0

    return result


def run_stream_repeated(config: BenchmarkConfig) -> list[BenchmarkResult]:
    """Run a streaming benchmark multiple times and return all results."""
    func = run_stream_hf_benchmark if config.mode == "stream-hf" else run_stream_benchmark
    results = []
    for i in range(config.repeat):
        result = func(config)
        results.append(result)
    return results
