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
    parser.add_argument("--prefetch", type=int, default=16, help="Thread-pool prefetch depth (default: 16, set 1 to disable)")
    parser.add_argument("--num-proc", type=int, default=0, help="Number of multiprocessing workers (default: 0 = single-process)")
    parser.add_argument("--dataloader-workers", type=int, default=0, help="Number of DataLoader workers (default: 0 = main process)")
    parser.add_argument("--columns-a", type=str, default=None, help="Comma-separated columns for catalog A (default: all)")
    parser.add_argument("--columns-b", type=str, default=None, help="Comma-separated columns for catalog B (default: all)")
    args = parser.parse_args()

    columns_a = [c.strip() for c in args.columns_a.split(",")] if args.columns_a else None
    columns_b = [c.strip() for c in args.columns_b.split(",")] if args.columns_b else None

    print(f"Catalogs: {URL_A.split('/')[-1]} x {URL_B.split('/')[-1]}")
    if args.num_proc > 0:
        print(f"Multiprocess workers: {args.num_proc}")
    else:
        print(f"Prefetch threads: {args.prefetch}")
    if args.dataloader_workers > 0:
        print(f"DataLoader workers: {args.dataloader_workers}")
    print(f"Columns A: {columns_a or 'all'}")
    print(f"Columns B: {columns_b or 'all'}")
    print("Setting up crossmatch...")

    t_setup = time.perf_counter()
    ds = stream_crossmatch(
        url_a=URL_A,
        url_b=URL_B,
        radius_arcsec=args.radius,
        n_neighbors=1,
        prefetch=args.prefetch,
        num_proc=args.num_proc,
        columns_a=columns_a,
        columns_b=columns_b,
    )
    t_setup = time.perf_counter() - t_setup
    print(f"Setup: {t_setup:.2f}s")
    print(f"Source rows: {ds.total_rows_a:,} + {ds.total_rows_b:,} = {ds.total_rows_a + ds.total_rows_b:,}")
    print(f"Shards (partitions): {ds.n_shards}\n")

    if args.dataloader_workers > 0:
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

    # Use "fork" context so workers inherit the pre-built Dask graph from the
    # parent process without pickling.  Python 3.14+ defaults to forkserver/spawn.
    dl = torch.utils.data.DataLoader(
        ds, batch_size=None, num_workers=num_workers, multiprocessing_context="fork",
    )

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
