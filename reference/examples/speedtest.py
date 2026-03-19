"""Speed test: measure throughput of streaming crossmatch via IterableDataset."""

import sys
import time

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))

from tqdm import tqdm

from hats_crossmatch import ThroughputCounter, stream_crossmatch

URL_A = "hf://datasets/UniverseTBD/mmu_sdss_sdss"
URL_B = "hf://datasets/UniverseTBD/mmu_gz10"

print(f"Catalogs: {URL_A.split('/')[-1]} x {URL_B.split('/')[-1]}")
print("Setting up crossmatch...")

t_setup = time.perf_counter()
ds = stream_crossmatch(url_a=URL_A, url_b=URL_B, radius_arcsec=1.0, n_neighbors=1)
t_setup = time.perf_counter() - t_setup
print(f"Setup: {t_setup:.2f}s")
print(f"Source rows: {ds.total_rows_a:,} + {ds.total_rows_b:,} = {ds.total_rows_a + ds.total_rows_b:,}\n")

counter = ThroughputCounter(ds)

print("Streaming rows...")
pbar = tqdm(counter, desc="Streaming", unit=" rows", unit_scale=True)
for row in pbar:
    if counter.matched_rows % 1_000 == 0:
        pbar.set_postfix(counter.tqdm_postfix())
pbar.close()

print(f"\n{'— Results —':^40}")
print(counter.summary())
