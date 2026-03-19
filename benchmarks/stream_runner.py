import time
from datetime import datetime, timezone

import click
import lsdb
import numpy as np

from benchmarks.config import BenchmarkConfig, TEST_CONE_DEC, TEST_CONE_RA, resolve_catalog
from benchmarks.metrics import BenchmarkResult, PeakMemoryTracker


def run_stream_benchmark(config: BenchmarkConfig) -> BenchmarkResult:
    """Execute a streaming crossmatch benchmark using lsdb.stream_crossmatch()."""
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

    kwargs_a = {}
    kwargs_b = {}
    if url_a.startswith("s3://"):
        kwargs_a["storage_options"] = {"anon": True}
    if url_b.startswith("s3://"):
        kwargs_b["storage_options"] = {"anon": True}

    tracker = PeakMemoryTracker()
    tracker.start()
    total_start = time.perf_counter()

    num_matches = 0
    num_batches = 0
    dist_arrays = []
    time_to_first_batch = None

    stream = lsdb.stream_crossmatch(
        url_a,
        url_b,
        radius_arcsec=config.radius_arcsec,
        n_neighbors=config.n_neighbors,
        suffixes=config.suffixes,
        search_filter=region,
        **{**kwargs_a, **kwargs_b},
    )

    for batch in stream:
        if time_to_first_batch is None:
            time_to_first_batch = time.perf_counter() - total_start
        num_batches += 1
        batch_len = len(batch)
        num_matches += batch_len
        if batch_len > 0 and "_dist_arcsec" in batch.columns:
            dist_arrays.append(batch["_dist_arcsec"].to_numpy())
        click.echo(f"  Batch {num_batches}: {batch_len:,} rows (total: {num_matches:,})")

    time_compute = time.perf_counter() - total_start
    result.memory_peak = tracker.stop()

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
