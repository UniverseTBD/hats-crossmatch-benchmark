import csv
import json
import os
from pathlib import Path

from benchmarks.metrics import BenchmarkResult

RESULTS_DIR = Path(__file__).parent.parent / "results"


def _format_time(seconds: float) -> str:
    if seconds < 1:
        return f"{seconds * 1000:.0f}ms"
    if seconds < 60:
        return f"{seconds:.2f}s"
    return f"{seconds / 60:.1f}m"


def _format_bytes(nbytes: int) -> str:
    if nbytes < 1024**2:
        return f"{nbytes / 1024:.1f} KB"
    if nbytes < 1024**3:
        return f"{nbytes / 1024**2:.1f} MB"
    return f"{nbytes / 1024**3:.2f} GB"


def console_report(results: list[BenchmarkResult]) -> None:
    """Print a human-readable summary to stdout."""
    for i, r in enumerate(results):
        if len(results) > 1:
            print(f"\n--- Run {i + 1}/{len(results)} ---")
        print()
        print(f"  Catalogs:    {r.config.catalog_a} x {r.config.catalog_b}")
        print(f"  Radius:      {r.config.radius_arcsec} arcsec")
        print(f"  N neighbors: {r.config.n_neighbors}")
        print()
        print(f"  Timing:")
        print(f"    Load:      {_format_time(r.time_load)}")
        print(f"    Plan:      {_format_time(r.time_plan)}")
        print(f"    Compute:   {_format_time(r.time_compute)}")
        print(f"    Total:     {_format_time(r.time_total)}")
        print()
        print(f"  Memory peak: {_format_bytes(r.memory_peak)}")
        print()
        print(f"  Rows A:      {r.num_rows_a:,}")
        print(f"  Rows B:      {r.num_rows_b:,}")
        print(f"  Partitions:  {r.num_partitions_a} x {r.num_partitions_b}")
        print(f"  Matches:     {r.num_matches:,}")
        print(f"  Match rate:  {r.match_rate:.2%}")
        if r.num_matches > 0:
            print()
            print(f"  Distance (arcsec):")
            print(f"    Mean:      {r.dist_mean:.4f}")
            print(f"    Median:    {r.dist_median:.4f}")
            print(f"    Std:       {r.dist_std:.4f}")
            print(f"    Min:       {r.dist_min:.4f}")
            print(f"    Max:       {r.dist_max:.4f}")

    if len(results) > 1:
        times = [r.time_total for r in results]
        import numpy as np

        print(f"\n--- Summary ({len(results)} runs) ---")
        print(f"  Total time:  {np.mean(times):.2f}s +/- {np.std(times):.2f}s")


def json_report(results: list[BenchmarkResult]) -> Path:
    """Write full results to a JSON file. Returns the output path."""
    RESULTS_DIR.mkdir(exist_ok=True)
    r = results[0]
    slug_a = r.config.catalog_a.replace("/", "_")
    slug_b = r.config.catalog_b.replace("/", "_")
    ts = r.timestamp.replace(":", "-").replace("+", "p")
    filename = f"{ts}_{slug_a}_x_{slug_b}.json"
    path = RESULTS_DIR / filename

    data = [r.to_dict() for r in results]
    path.write_text(json.dumps(data, indent=2))
    print(f"\n  JSON written to {path}")
    return path


def csv_report(results: list[BenchmarkResult]) -> Path:
    """Append results to history.csv. Returns the output path."""
    RESULTS_DIR.mkdir(exist_ok=True)
    path = RESULTS_DIR / "history.csv"
    file_exists = path.exists()

    fieldnames = list(results[0].to_dict().keys())
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        for r in results:
            writer.writerow(r.to_dict())

    print(f"  CSV appended to {path}")
    return path
