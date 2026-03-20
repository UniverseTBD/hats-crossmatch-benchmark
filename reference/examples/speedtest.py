"""Speed test: measure throughput of streaming crossmatch via IterableDataset."""

import argparse
import sys
import time

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))

from tqdm import tqdm

from hats_crossmatch import ThroughputCounter, stream_crossmatch

URL_A = "hf://datasets/UniverseTBD/mmu_sdss_sdss"
URL_B = "hf://datasets/UniverseTBD/mmu_gz10"


def main():
    parser = argparse.ArgumentParser(description="Speed test for streaming crossmatch")
    parser.add_argument("--radius", type=float, default=1.0, help="Crossmatch radius in arcseconds (default: 1.0)")
    parser.add_argument("--partitions-per-chunk", type=int, default=1, help="Partitions per chunk (default: 1)")
    parser.add_argument("--n-workers", type=int, default=None, help="Number of Dask workers (default: None)")
    args = parser.parse_args()

    print(f"Catalogs: {URL_A.split('/')[-1]} x {URL_B.split('/')[-1]}")
    print("Setting up crossmatch...")

    t_setup = time.perf_counter()
    ds = stream_crossmatch(
        url_a=URL_A,
        url_b=URL_B,
        radius_arcsec=args.radius,
        n_neighbors=1,
        partitions_per_chunk=args.partitions_per_chunk,
        n_workers=args.n_workers,
    )
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


if __name__ == "__main__":
    main()
