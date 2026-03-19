"""Side-by-side: stream the same data from lsdb crossmatch vs HF dataset.

Both paths use the same `for row in ds` interface.
"""

import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))

import datasets

from hats_crossmatch import stream_crossmatch

N = 3

# --- Path 1: Live crossmatch via lsdb, returned as IterableDataset ---
print("=== lsdb crossmatch (streaming) ===")
ds_lsdb = stream_crossmatch(
    url_a="hf://datasets/UniverseTBD/mmu_sdss_sdss",
    url_b="hf://datasets/UniverseTBD/mmu_plasticc",
    radius_arcsec=1.0,
    n_neighbors=1,
)
print(f"Type: {type(ds_lsdb).__name__}")

for i, row in enumerate(ds_lsdb):
    if i >= N:
        break
    cols = list(row.keys())[:5]
    print(f"  Row {i}: {{{', '.join(f'{k}={row[k]}' for k in cols)}}} ...")

# --- Path 2: Pre-computed results from HuggingFace Hub ---
print("\n=== HuggingFace dataset (streaming) ===")
ds_hf = datasets.load_dataset(
    "UniverseTBD/mmu_sdss_sdss", streaming=True, split="train"
)
print(f"Type: {type(ds_hf).__name__}")

for i, row in enumerate(ds_hf):
    if i >= N:
        break
    cols = list(row.keys())[:5]
    print(f"  Row {i}: {{{', '.join(f'{k}={row[k]}' for k in cols)}}} ...")

# --- Both support .take(), .map(), etc. ---
print("\n=== .take() works on both ===")
print(f"  lsdb take(2): {list(ds_lsdb.take(2)).__len__()} rows")
print(f"  HF    take(2): {list(ds_hf.take(2)).__len__()} rows")

print("\nBoth backends produce the same IterableDataset interface.")
