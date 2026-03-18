import threading
import time
from dataclasses import dataclass, field

import psutil

from benchmarks.config import BenchmarkConfig


@dataclass
class BenchmarkResult:
    config: BenchmarkConfig
    timestamp: str

    # Timing (seconds)
    time_load: float = 0.0
    time_plan: float = 0.0
    time_compute: float = 0.0
    time_total: float = 0.0

    # Memory (bytes)
    memory_peak: int = 0

    # Match statistics
    num_rows_a: int = 0
    num_rows_b: int = 0
    num_matches: int = 0
    match_rate: float = 0.0

    # Distance stats (arcsec)
    dist_mean: float = 0.0
    dist_median: float = 0.0
    dist_std: float = 0.0
    dist_min: float = 0.0
    dist_max: float = 0.0

    # Catalog metadata
    num_partitions_a: int = 0
    num_partitions_b: int = 0

    def to_dict(self) -> dict:
        return {
            "catalog_a": self.config.catalog_a,
            "catalog_b": self.config.catalog_b,
            "radius_arcsec": self.config.radius_arcsec,
            "n_neighbors": self.config.n_neighbors,
            "repeat": self.config.repeat,
            "timestamp": self.timestamp,
            "time_load": self.time_load,
            "time_plan": self.time_plan,
            "time_compute": self.time_compute,
            "time_total": self.time_total,
            "memory_peak": self.memory_peak,
            "num_rows_a": self.num_rows_a,
            "num_rows_b": self.num_rows_b,
            "num_matches": self.num_matches,
            "match_rate": self.match_rate,
            "dist_mean": self.dist_mean,
            "dist_median": self.dist_median,
            "dist_std": self.dist_std,
            "dist_min": self.dist_min,
            "dist_max": self.dist_max,
            "num_partitions_a": self.num_partitions_a,
            "num_partitions_b": self.num_partitions_b,
        }


class PeakMemoryTracker:
    """Background thread that samples RSS to find peak memory usage."""

    def __init__(self, interval: float = 0.1):
        self._interval = interval
        self._process = psutil.Process()
        self._peak = self._process.memory_info().rss
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self):
        self._peak = self._process.memory_info().rss
        self._stop.clear()
        self._thread = threading.Thread(target=self._sample, daemon=True)
        self._thread.start()

    def _sample(self):
        while not self._stop.is_set():
            rss = self._process.memory_info().rss
            if rss > self._peak:
                self._peak = rss
            self._stop.wait(self._interval)

    def stop(self) -> int:
        self._stop.set()
        if self._thread:
            self._thread.join()
        # One final sample
        rss = self._process.memory_info().rss
        if rss > self._peak:
            self._peak = rss
        return self._peak
