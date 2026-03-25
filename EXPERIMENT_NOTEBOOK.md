# Experiment Notebook

## Experiment 1: Parameter Sweep Baseline (2026-03-25)

**Setup:** `hf://` URLs with monkey-patched `read_parquet_file_to_pandas` to intercept
`hf://` paths, convert to `https://` via `huggingface_hub.hf_hub_url()`, and read through
`fsspec.implementations.http.HTTPFileSystem`. The monkey-patch was applied/restored around
each partition compute in both the multiprocess and sharded paths.

**Catalogs:**
- A: `hf://datasets/UniverseTBD/mmu_sdss_sdss`
- B: `hf://datasets/UniverseTBD/mmu_gz10`

**Parameters:** 5000 rows, radius=1.0 arcsec, n_neighbors=1

### Group A: Prefetch sweep (num_proc=0, dataloader_workers=0)

| prefetch | Setup (s) | TTFR (s) | Total (s) | rows/s |
|----------|-----------|----------|-----------|--------|
| 1        | 8.2       | 1.3      | 297.2     | 16.8   |
| 2        | 8.2       | 1.2      | 274.1     | 18.2   |
| 4        | 7.9       | 1.2      | 278.1     | 18.0   |
| 8        | 8.5       | 1.2      | 276.6     | 18.1   |
| 12       | 8.2       | 1.2      | 276.0     | 18.1   |
| 16       | 8.0       | 1.1      | 274.8     | 18.2   |

**Observation:** Prefetch has minimal impact on throughput beyond prefetch=2. Single-process
throughput plateaus around 18 rows/s, suggesting the bottleneck is per-partition compute
(crossmatch), not I/O overlap.

### Group B: Multiprocess pool (dataloader_workers=0, prefetch=16)

| num_proc | Setup (s) | TTFR (s) | Total (s) | rows/s |
|----------|-----------|----------|-----------|--------|
| 1        | 8.1       | 1.4      | 293.7     | 17.0   |
| 2        | 10.3      | 1.7      | 197.8     | 25.3   |
| 4        | 8.1       | 1.8      | 196.5     | 25.4   |
| 8        | 8.5       | 2.1      | 198.2     | 25.2   |

**Observation:** Multiprocessing gives a ~1.5x speedup at num_proc=2 but plateaus there.
No further gains at 4 or 8 processes. This suggests the monkey-patching approach (creating
a new HTTPFileSystem per worker, re-resolving hf:// URLs) has overhead that limits scaling.

### Group C: DataLoader workers (num_proc=0, prefetch=16)

| dataloader_workers | Setup (s) | TTFR (s) | Total (s) | rows/s |
|--------------------|-----------|----------|-----------|--------|
| 1                  | 8.0       | 1.2      | 277.3     | 18.0   |
| 2                  | 8.5       | 1.5      | 267.3     | 18.7   |
| 4                  | 7.9       | 2.1      | 256.7     | 19.5   |
| 8                  | 8.0       | 2.4      | 253.0     | 19.8   |

**Observation:** DataLoader workers provide marginal improvement (~10% at dw=8). The
sharded generator approach with prefetch already overlaps I/O, so adding DataLoader
parallelism on top adds little.

---

## Experiment 2: Replace hf:// with native HTTPS (2026-03-25)

**Motivation:** Kostya Malanchev suggested that instead of monkey-patching
`hats.io.file_io.read_parquet_file_to_pandas` to intercept `hf://` paths and redirect
through `HTTPFileSystem`, we should pass HTTPS URLs directly to `lsdb.open_catalog()`.
This lets fsspec natively use `HTTPFileSystem` for all I/O — no patching of upstream code.

**Changes:**
1. Replaced `hf://datasets/UniverseTBD/...` URLs with
   `https://huggingface.co/datasets/UniverseTBD/.../resolve/main/.../` pointing directly
   to the HATS catalog subdirectory
2. Removed the `_make_async_read_patch()` function entirely
3. Removed monkey-patching from `_pool_worker()` and `_iter_rows_sharded()`
4. Added `_reset_fsspec_after_fork()` to handle fsspec's async event loop in forked
   workers (fsspec's `AsyncFileSystem` is not fork-safe — inherited instances hold a dead
   event loop and aiohttp session after `fork()`)

**Catalogs:**
- A: `https://huggingface.co/datasets/UniverseTBD/mmu_sdss_sdss/resolve/main/mmu_sdss_sdss/`
- B: `https://huggingface.co/datasets/UniverseTBD/mmu_gz10/resolve/main/mmu_gz10/`

**Parameters:** 5000 rows, radius=1.0 arcsec, n_neighbors=1

### Group B: Multiprocess pool (dataloader_workers=0, prefetch=16)

| num_proc | Setup (s) | TTFR (s) | Total (s) | rows/s |
|----------|-----------|----------|-----------|--------|
| 1        | 7.9       | 1.2      | 281.1     | 17.8   |
| 2        | 7.8       | 1.4      | 186.0     | 26.9   |
| 4        | 8.9       | 2.3      | 146.5     | 34.1   |
| 8        | 9.0       | 2.6      | 140.9     | 35.5   |

### Comparison: hf:// monkey-patch vs native HTTPS

| num_proc | hf:// + patch (rows/s) | https:// native (rows/s) | Change |
|----------|------------------------|--------------------------|--------|
| 1        | 17.0                   | 17.8                     | +4.7%  |
| 2        | 25.3                   | 26.9                     | +6.3%  |
| 4        | 25.4                   | 34.1                     | +34.3% |
| 8        | 25.2                   | 35.5                     | +40.9% |

**Key finding:** The native HTTPS approach scales significantly better with more processes.
The old monkey-patching approach plateaued at ~25 rows/s regardless of process count (2, 4,
or 8). The HTTPS approach continues scaling up to 35.5 rows/s at 8 processes — a 40%
improvement over the patched version.

**Why:** With `hf://` URLs, the monkey-patch had to create a new `HTTPFileSystem` and
resolve each `hf://` path via `hf_hub_url()` on every parquet read. With native HTTPS,
fsspec uses `HTTPFileSystem` directly from the Dask graph — the paths are already resolved,
and after the fork-safety reset, each worker gets a clean async event loop without the
per-read URL resolution overhead.

**Note on environment:** fsspec's `HTTPFileSystem` uses aiohttp, which requires a valid
SSL CA bundle. On systems where the default CA bundle is not found, set
`SSL_CERT_FILE=$(python -c "import certifi; print(certifi.where())")` before running.
