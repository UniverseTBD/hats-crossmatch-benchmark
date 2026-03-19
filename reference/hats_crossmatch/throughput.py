"""Throughput counter addon for streaming crossmatch."""

from __future__ import annotations

import time
from typing import Iterable, Iterator


class ThroughputCounter:
    """Wraps a crossmatch IterableDataset and tracks two throughput counters.

    * **matched rows** — crossmatch output rows yielded so far.
    * **source rows scanned** — estimated from partition progress and
      the known total rows in both source catalogs.

    Parameters
    ----------
    dataset : IterableDataset
        A dataset returned by :func:`stream_crossmatch`.  Must carry
        ``crossmatch_stats``, ``total_rows_a``, and ``total_rows_b``
        attributes (set automatically by ``stream_crossmatch``).

    Examples
    --------
    >>> ds = stream_crossmatch(...)
    >>> counter = ThroughputCounter(ds)
    >>> for row in counter:
    ...     pass
    >>> print(counter.summary())
    """

    def __init__(self, dataset) -> None:
        self._dataset = dataset
        self._stats: dict = getattr(dataset, "crossmatch_stats", {})
        self.total_source_rows: int = (
            getattr(dataset, "total_rows_a", 0)
            + getattr(dataset, "total_rows_b", 0)
        )
        self.matched_rows: int = 0
        self.source_rows_scanned: int = 0
        self.elapsed: float = 0.0
        self._t_start: float | None = None
        self._t_first: float | None = None

    @property
    def matched_per_sec(self) -> float:
        if self.elapsed <= 0:
            return 0.0
        return self.matched_rows / self.elapsed

    @property
    def source_per_sec(self) -> float:
        if self.elapsed <= 0:
            return 0.0
        return self.source_rows_scanned / self.elapsed

    @property
    def time_to_first(self) -> float | None:
        return self._t_first

    def _update_source_estimate(self):
        npartitions = self._stats.get("npartitions", 0)
        done = self._stats.get("partitions_done", 0)
        if npartitions > 0 and self.total_source_rows > 0:
            self.source_rows_scanned = int(
                done / npartitions * self.total_source_rows
            )

    def __iter__(self) -> Iterator:
        self.matched_rows = 0
        self.source_rows_scanned = 0
        self._t_start = time.perf_counter()
        self._t_first = None
        for item in self._dataset:
            if self._t_first is None:
                self._t_first = time.perf_counter() - self._t_start
            self.matched_rows += 1
            self.elapsed = time.perf_counter() - self._t_start
            self._update_source_estimate()
            yield item
        self.elapsed = time.perf_counter() - self._t_start
        self._update_source_estimate()

    def tqdm_postfix(self) -> dict:
        """Return a dict suitable for ``tqdm.set_postfix()``."""
        d = {"matched": f"{self.matched_rows:,}"}
        if self.total_source_rows > 0:
            d["scanned"] = f"~{self.source_rows_scanned:,}/{self.total_source_rows:,}"
        if self.elapsed > 0:
            d["rate"] = f"{self.matched_per_sec:,.0f} rows/s"
        return d

    def summary(self) -> str:
        lines = [
            f"  Matched rows:    {self.matched_rows:,}",
        ]
        if self.total_source_rows > 0:
            lines.append(f"  Source scanned:  ~{self.source_rows_scanned:,} / {self.total_source_rows:,}")
            if self.total_source_rows > 0:
                lines.append(f"  Match rate:      {self.matched_rows / self.total_source_rows:.2%}")
        if self._t_first is not None:
            lines.append(f"  Time to first:   {self._t_first:.3f}s")
        lines.append(f"  Total time:      {self.elapsed:.2f}s")
        if self.elapsed > 0:
            lines.append(f"  Throughput:      {self.matched_per_sec:,.0f} matched/s, ~{self.source_per_sec:,.0f} scanned/s")
        return "\n".join(lines)
