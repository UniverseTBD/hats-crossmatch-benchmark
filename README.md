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

### Streaming crossmatch benchmark

Iterates `lsdb.stream_crossmatch()` partition-by-partition, measuring time-to-first-batch and throughput:

```bash
python -m benchmarks.cli stream \
  --catalog-a sdss --catalog-b plasticc \
  --radius 1.0 --n-neighbors 1 \
  --output console json
```

### Stream from HuggingFace Datasets

Streams pre-computed crossmatch results row-by-row from a HF dataset:

```bash
python -m benchmarks.cli stream-hf \
  --repo-id UniverseTBD/mmu_sdss_sdss \
  --output console
```

### Run standard benchmark suite

```bash
python -m benchmarks.cli suite
```

Use `--mode stream` to run the suite in streaming mode:

```bash
python -m benchmarks.cli suite --mode stream
```

### Run S3 benchmark suite

```bash
python -m benchmarks.cli s3-suite
```

### Run stream-hf benchmark suite

```bash
python -m benchmarks.cli stream-hf-suite
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
