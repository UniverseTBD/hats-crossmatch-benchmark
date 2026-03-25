# Streaming Crossmatch Client — Reference Implementation

A lightweight wrapper that exposes **lsdb crossmatch results as a HuggingFace `IterableDataset`**, giving downstream code (ML pipelines, PyTorch DataLoaders) a unified streaming interface whether data comes from a live crossmatch or a pre-computed HF dataset.

## Quick start

```python
from hats_crossmatch import stream_crossmatch

ds = stream_crossmatch(
    url_a="hf://datasets/UniverseTBD/mmu_sdss_sdss",
    url_b="hf://datasets/UniverseTBD/mmu_plasticc",
    radius_arcsec=1.0,
    n_neighbors=1,
)

for row in ds:
    print(row["ra_a"], row["_dist_arcsec"])
```

The returned `ds` is a real `datasets.IterableDataset` — it supports `.take()`, `.map()`, `.filter()`, and can be passed directly to a PyTorch `DataLoader`.

## Performance

### The bottleneck: KD-tree crossmatch is CPU-bound

Throughput is limited by scipy's KD-tree crossmatch, which is CPU-bound and holds the GIL. With `scheduler="threads"`, threads only overlap I/O — they cannot parallelize CPU work. Single-threaded throughput is structurally capped at ~10 rows/sec for typical catalog pairs.

Network I/O is *not* the bottleneck. Loading catalog metadata and partition data from HuggingFace takes ~8–11s regardless of catalog size, because HATS spatial indexing streams only overlapping HEALPix partitions.

### Multiprocess parallelism (`num_proc`)

The `num_proc` parameter uses `multiprocessing.Pool` with `fork` context. Each child process gets its own GIL, so N processes compute N partitions simultaneously. The Dask graph is inherited via copy-on-write shared memory — only integer partition indices are sent through the pipe.

```bash
# Baseline: single-process (~10 rows/sec)
uv run python examples/speedtest.py --prefetch 16

# 4 workers (~40 rows/sec)
uv run python examples/speedtest.py --num-proc 4

# 8 workers (~80 rows/sec)
uv run python examples/speedtest.py --num-proc 8
```

Scaling is roughly linear until CPU or network saturation.

### PyTorch DataLoader parallelism

For PyTorch users, the fine-grained sharding (1 LSDB partition = 1 HF shard) enables DataLoader multi-worker parallelism via the same `fork` mechanism:

```bash
uv run python examples/speedtest.py --prefetch 16 --dataloader-workers 4
```

### Choosing a parallelism strategy

| Strategy | Flag | Use case |
|---|---|---|
| Single-process + thread prefetch | `--prefetch 16` (default) | Simple scripts, low core count |
| Multiprocess pool | `--num-proc N` | Batch jobs, servers, no PyTorch dependency |
| DataLoader workers | `--dataloader-workers N` | PyTorch training pipelines |

## API

### `stream_crossmatch(url_a, url_b, **kwargs) -> IterableDataset`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `url_a` | `str` | required | HATS catalog URL |
| `url_b` | `str` | required | HATS catalog URL |
| `radius_arcsec` | `float` | `1.0` | Crossmatch radius in arcseconds |
| `n_neighbors` | `int` | `1` | Max neighbors per source |
| `suffixes` | `tuple[str, str]` | `("_a", "_b")` | Column name suffixes |
| `search_filter` | | `None` | lsdb filter (e.g. `lsdb.ConeSearch(...)`) |
| `storage_options_a` | `dict \| None` | `None` | fsspec options for catalog A |
| `storage_options_b` | `dict \| None` | `None` | fsspec options for catalog B |
| `columns_a` | `list[str] \| None` | `None` | Columns to read from catalog A |
| `columns_b` | `list[str] \| None` | `None` | Columns to read from catalog B |
| `prefetch` | `int` | `16` | Thread-pool prefetch depth (single-process path) |
| `num_proc` | `int` | `0` | Multiprocess workers (0 = single-process) |

## Examples

```bash
# From the reference/ directory:
python examples/basic_stream.py       # Stream and print first N rows
python examples/hf_comparison.py      # Side-by-side: lsdb vs HF dataset
python examples/speedtest.py          # Throughput benchmark
```

## Motivation

HuggingFace Datasets has become the standard interface for streaming tabular data into ML pipelines. By wrapping lsdb crossmatch results into an `IterableDataset`, the same downstream code works identically regardless of whether the crossmatch is computed live or read from a pre-built dataset on the Hub.
