import time
from datetime import datetime, timezone

import click
import lsdb
import numpy as np
from dask.distributed import Client, progress

from benchmarks.config import BenchmarkConfig, TEST_CONE_DEC, TEST_CONE_RA, resolve_catalog
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

    # Set up Dask — use synchronous scheduler in test mode to avoid overhead
    client = None
    if config.test:
        import dask
        dask.config.set(scheduler="synchronous")
    else:
        client_kwargs = {"threads_per_worker": 1}
        if config.n_workers is not None:
            client_kwargs["n_workers"] = config.n_workers
        client = Client(**client_kwargs)

    region = None
    if config.test:
        click.echo(f"  [test mode] Cone search: ({TEST_CONE_RA}, {TEST_CONE_DEC}), r={config.test_radius_deg}°")
        region = lsdb.ConeSearch(TEST_CONE_RA, TEST_CONE_DEC, config.test_radius_deg * 3600)
    
    try:
        # Phase 1: Load catalogs
        t0 = time.perf_counter()
        kwargs_a = {"search_filter": region}
        kwargs_b = {"search_filter": region}
        if url_a.startswith("s3://"):
            kwargs_a["storage_options"] = {"anon": True}
        if url_b.startswith("s3://"):
            kwargs_b["storage_options"] = {"anon": True}
        cat_a = lsdb.open_catalog(url_a, **kwargs_a)
        cat_b = lsdb.open_catalog(url_b, **kwargs_b)

        result.time_load = time.perf_counter() - t0

        result.num_partitions_a = len(cat_a.get_healpix_pixels())
        result.num_partitions_b = len(cat_b.get_healpix_pixels())

        # Phase 2: Plan crossmatch (lazy)
        t0 = time.perf_counter()
        try:
            xmatch = cat_a.crossmatch(
                cat_b,
                n_neighbors=config.n_neighbors,
                radius_arcsec=config.radius_arcsec,
                suffixes=config.suffixes,
                suffix_method="overlapping_columns",
            )
        except RuntimeError as e:
            if "do not overlap" in str(e).lower():
                click.echo("  Catalogs do not overlap — no matches possible.")
                result.time_plan = time.perf_counter() - t0
                result.num_rows_a = len(cat_a)
                result.num_rows_b = len(cat_b)
                result.time_total = time.perf_counter() - total_start
                return result
            raise
        result.time_plan = time.perf_counter() - t0

        # Phase 3: Compute (trigger execution with peak memory tracking)
        tracker = PeakMemoryTracker()
        tracker.start()
        t0 = time.perf_counter()
        if client is not None:
            future = client.compute(xmatch._ddf)
            progress(future)
            computed = future.result()
        else:
            from dask.diagnostics import ProgressBar
            with ProgressBar():
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
        if client is not None:
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
