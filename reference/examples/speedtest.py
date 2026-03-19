"""Speed test: measure throughput of streaming crossmatch via IterableDataset."""

import sys
import time

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))

from hats_crossmatch import stream_crossmatch

URL_A = "hf://datasets/UniverseTBD/mmu_sdss_sdss"
URL_B = "hf://datasets/UniverseTBD/mmu_gz10"

print(f"Catalogs: {URL_A.split('/')[-1]} x {URL_B.split('/')[-1]}")
print("Setting up crossmatch...")

t_setup = time.perf_counter()
ds = stream_crossmatch(url_a=URL_A, url_b=URL_B, radius_arcsec=1.0, n_neighbors=1)
t_setup = time.perf_counter() - t_setup
print(f"Setup: {t_setup:.2f}s\n")

from tqdm import tqdm

print("Streaming rows...")
t_start = time.perf_counter()
t_first = None
num_rows = 0

pbar = tqdm(ds, desc="Streaming", unit=" rows", unit_scale=True)
for row in pbar:
    if t_first is None:
        t_first = time.perf_counter() - t_start
    num_rows += 1
    if num_rows % 1_000 == 0:
        elapsed = time.perf_counter() - t_start
        pbar.set_postfix(rate=f"{num_rows / elapsed:,.0f} rows/s")
pbar.close()

t_total = time.perf_counter() - t_start

print(f"\n{'— Results —':^40}")
print(f"  Total rows:      {num_rows:,}")
print(f"  Time to first:   {t_first:.3f}s" if t_first else "  Time to first:   N/A (no rows)")
print(f"  Total time:      {t_total:.2f}s")
if t_total > 0:
    print(f"  Throughput:      {num_rows / t_total:,.0f} rows/s")
