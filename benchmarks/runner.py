import time
from datetime import datetime, timezone

import lsdb
import numpy as np
from dask.distributed import Client

from benchmarks.config import BenchmarkConfig, resolve_catalog
from benchmarks.metrics import BenchmarkResult, PeakMemoryTracker


def run_benchmark(config: BenchmarkConfig) -> BenchmarkResult:
    """Execute a single crossmatch benchmark run."""
    url_a = resolve_catalog(config.catalog_a)
    url_b = resolve_catalog(config.catalog_b)

    result = BenchmarkResult(
        config=config,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )

    total_start = time.perf_counter()

    # Set up Dask client
    client_kwargs = {}
    if config.n_workers is not None:
        client_kwargs["n_workers"] = config.n_workers
    client = Client(**client_kwargs)

    try:
        # Phase 1: Load catalogs
        t0 = time.perf_counter()
        cat_a = lsdb.open_catalog(url_a)
        cat_b = lsdb.open_catalog(url_b)
        result.time_load = time.perf_counter() - t0

        result.num_partitions_a = cat_a.get_healpix_pixels().shape[0]
        result.num_partitions_b = cat_b.get_healpix_pixels().shape[0]

        # Phase 2: Plan crossmatch (lazy)
        t0 = time.perf_counter()
        xmatch = cat_a.crossmatch(
            cat_b,
            n_neighbors=config.n_neighbors,
            radius_arcsec=config.radius_arcsec,
            suffixes=config.suffixes,
        )
        result.time_plan = time.perf_counter() - t0

        # Phase 3: Compute (trigger execution with peak memory tracking)
        tracker = PeakMemoryTracker()
        tracker.start()
        t0 = time.perf_counter()
        computed = xmatch.compute()
        result.time_compute = time.perf_counter() - t0
        result.memory_peak = tracker.stop()

        # Phase 4: Analyze results
        result.num_rows_a = len(cat_a)
        result.num_rows_b = len(cat_b)
        result.num_matches = len(computed)

        min_rows = min(result.num_rows_a, result.num_rows_b)
        result.match_rate = result.num_matches / min_rows if min_rows > 0 else 0.0

        if result.num_matches > 0 and "_dist_arcsec" in computed.columns:
            dists = computed["_dist_arcsec"].to_numpy()
            result.dist_mean = float(np.mean(dists))
            result.dist_median = float(np.median(dists))
            result.dist_std = float(np.std(dists))
            result.dist_min = float(np.min(dists))
            result.dist_max = float(np.max(dists))

    finally:
        client.close()

    result.time_total = time.perf_counter() - total_start
    return result


def run_repeated(config: BenchmarkConfig) -> list[BenchmarkResult]:
    """Run a benchmark multiple times (config.repeat) and return all results."""
    results = []
    for i in range(config.repeat):
        result = run_benchmark(config)
        results.append(result)
    return results
