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

## API

### `stream_crossmatch(url_a, url_b, **kwargs) -> IterableDataset`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `url_a` | `str` | required | HATS catalog URL |
| `url_b` | `str` | required | HATS catalog URL |
| `radius_arcsec` | `float` | `1.0` | Crossmatch radius in arcseconds |
| `n_neighbors` | `int` | `1` | Max neighbors per source |
| `suffixes` | `tuple[str, str]` | `("_a", "__b")` | Column name suffixes |
| `search_filter` | | `None` | lsdb filter (e.g. `lsdb.ConeSearch(...)`) |
| `storage_options_a` | `dict \| None` | `None` | fsspec options for catalog A |
| `storage_options_b` | `dict \| None` | `None` | fsspec options for catalog B |

## Examples

```bash
# From the reference/ directory:
python examples/basic_stream.py       # Stream and print first N rows
python examples/hf_comparison.py      # Side-by-side: lsdb vs HF dataset
```

## Motivation

HuggingFace Datasets has become the standard interface for streaming tabular data into ML pipelines. By wrapping lsdb's `CatalogStream` into an `IterableDataset`, the same downstream code works identically regardless of whether the crossmatch is computed live or read from a pre-built dataset on the Hub.
