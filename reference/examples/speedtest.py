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
    parser.add_argument("--n-workers", type=int, default=None, help="Number of Dask workers (default: None, only used with --no-sharded)")
    parser.add_argument("--no-sharded", action="store_true", help="Disable sharded mode (use legacy CatalogStream path)")
    parser.add_argument("--dataloader-workers", type=int, default=0, help="Number of DataLoader workers (default: 0 = main process)")
    args = parser.parse_args()

    sharded = not args.no_sharded

    print(f"Catalogs: {URL_A.split('/')[-1]} x {URL_B.split('/')[-1]}")
    print(f"Mode: {'sharded' if sharded else 'legacy (CatalogStream)'}")
    if sharded and args.dataloader_workers > 0:
        print(f"DataLoader workers: {args.dataloader_workers}")
    print("Setting up crossmatch...")

    t_setup = time.perf_counter()
    ds = stream_crossmatch(
        url_a=URL_A,
        url_b=URL_B,
        radius_arcsec=args.radius,
        n_neighbors=1,
        partitions_per_chunk=args.partitions_per_chunk,
        n_workers=args.n_workers,
        sharded=sharded,
    )
    t_setup = time.perf_counter() - t_setup
    print(f"Setup: {t_setup:.2f}s")
    print(f"Source rows: {ds.total_rows_a:,} + {ds.total_rows_b:,} = {ds.total_rows_a + ds.total_rows_b:,}")
    print(f"Shards (partitions): {ds.n_shards}\n")

    if sharded and args.dataloader_workers > 0:
        _run_with_dataloader(ds, args.dataloader_workers)
    else:
        _run_simple(ds)


def _run_simple(ds):
    """Iterate the dataset directly (single-process)."""
    counter = ThroughputCounter(ds)

    print("Streaming rows...")
    pbar = tqdm(counter, desc="Streaming", unit=" rows", unit_scale=True)
    for row in pbar:
        if counter.matched_rows % 1_000 == 0:
            pbar.set_postfix(counter.tqdm_postfix())
    pbar.close()

    print(f"\n{'— Results —':^40}")
    print(counter.summary())


def _run_with_dataloader(ds, num_workers):
    """Iterate using a PyTorch DataLoader for multi-worker parallelism."""
    import torch.utils.data

    dl = torch.utils.data.DataLoader(ds, batch_size=None, num_workers=num_workers)

    total = 0
    t0 = time.perf_counter()
    print("Streaming rows via DataLoader...")
    pbar = tqdm(dl, desc="Streaming", unit=" rows", unit_scale=True)
    for row in pbar:
        total += 1
        if total % 1_000 == 0:
            elapsed = time.perf_counter() - t0
            pbar.set_postfix({"rows/s": f"{total / elapsed:,.0f}"})
    pbar.close()

    elapsed = time.perf_counter() - t0
    print(f"\n{'— Results —':^40}")
    print(f"Total rows:  {total:,}")
    print(f"Elapsed:     {elapsed:.1f}s")
    print(f"Throughput:  {total / elapsed:,.0f} rows/s")


if __name__ == "__main__":
    main()
