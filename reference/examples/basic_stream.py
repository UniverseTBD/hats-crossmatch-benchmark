"""Minimal example: stream crossmatch results and print the first N rows."""

import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))

from hats_crossmatch import stream_crossmatch

def main():
    N = 5

    ds = stream_crossmatch(
        url_a="hf://datasets/UniverseTBD/mmu_sdss_sdss",
        url_b="hf://datasets/UniverseTBD/mmu_plasticc",
        radius_arcsec=1.0,
        n_neighbors=1,
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
