# HATS Crossmatch Benchmark

Benchmarking tool for measuring [LSDB](https://lsdb.readthedocs.io/) crossmatch performance on HATS-formatted catalogs hosted on Hugging Face.

## Setup

```bash
uv venv .venv && uv pip install '.'
```

## Usage

### Single crossmatch benchmark

```bash
python -m benchmarks.cli run \
  --catalog-a sdss --catalog-b plasticc \
  --radius 1.0 --n-neighbors 1 \
  --output json console
```

### Parameter sweep (vary radius)

```bash
python -m benchmarks.cli sweep \
  --catalog-a sdss --catalog-b plasticc \
  --radii 0.1,0.5,1.0,2.0,5.0
```

### Run standard benchmark suite

```bash
python -m benchmarks.cli suite
```

### List available catalogs

```bash
python -m benchmarks.cli list
```

### Direct HF URLs

```bash
python -m benchmarks.cli run \
  --catalog-a-path hf://datasets/UniverseTBD/mmu_sdss_sdss \
  --catalog-b-path hf://datasets/UniverseTBD/mmu_plasticc \
  --output console json
```

## Output formats

- **console** — Human-readable summary to stdout
- **json** — Full results to `results/<timestamp>_<a>_x_<b>.json`
- **csv** — Append row to `results/history.csv`
