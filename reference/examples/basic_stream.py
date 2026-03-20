"""Minimal example: stream crossmatch results and print the first N rows."""

import argparse
import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))

from hats_crossmatch import stream_crossmatch

def main():
    parser = argparse.ArgumentParser(description="Stream crossmatch and print first N rows")
    parser.add_argument("--partitions-per-chunk", type=int, default=1, help="Partitions per chunk (default: 1)")
    parser.add_argument("--n-workers", type=int, default=None, help="Number of Dask workers (default: None)")
    args = parser.parse_args()

    N = 5

    ds = stream_crossmatch(
        url_a="hf://datasets/UniverseTBD/mmu_sdss_sdss",
        url_b="hf://datasets/UniverseTBD/mmu_plasticc",
        radius_arcsec=1.0,
        n_neighbors=1,
        partitions_per_chunk=args.partitions_per_chunk,
        n_workers=args.n_workers,
    )

    print(f"Type: {type(ds).__name__}")
    print(f"First {N} rows:\n")

    for i, row in enumerate(ds):
        if i >= N:
            break
        print(f"Row {i}: ra_a={row.get('ra_a')}, dec_a={row.get('dec_a')}, "
              f"dist={row.get('_dist_arcsec')}")

    print("\nDone.")


if __name__ == "__main__":
    main()
