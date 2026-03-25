"""Single-config benchmark runner. Prints one JSON line to stdout."""

import argparse
import json
import sys
import time

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent / "reference"))

from hats_crossmatch import stream_crossmatch

URL_A = "hf://datasets/UniverseTBD/mmu_sdss_sdss/"
URL_B = "hf://datasets/UniverseTBD/mmu_gz10/"
#URL_A = "https://huggingface.co/datasets/UniverseTBD/mmu_sdss_sdss/resolve/main/mmu_sdss_sdss/"
#URL_B = "https://huggingface.co/datasets/UniverseTBD/mmu_gz10/resolve/main/mmu_gz10/"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-proc", type=int, default=0)
    parser.add_argument("--prefetch", type=int, default=16)
    parser.add_argument("--dataloader-workers", type=int, default=0)
    parser.add_argument("--max-rows", type=int, default=5000)
    args = parser.parse_args()

    # --- setup ---
    t_setup_start = time.perf_counter()
    ds = stream_crossmatch(
        url_a=URL_A,
        url_b=URL_B,
        radius_arcsec=1.0,
        n_neighbors=1,
        prefetch=args.prefetch,
        num_proc=args.num_proc,
    )
    setup_time = time.perf_counter() - t_setup_start

    # --- iterate ---
    if args.dataloader_workers > 0:
        total, time_to_first, total_time = _run_dataloader(
            ds, args.dataloader_workers, args.max_rows
        )
    else:
        total, time_to_first, total_time = _run_simple(ds, args.max_rows)

    throughput = total / total_time if total_time > 0 else 0.0

    result = {
        "num_proc": args.num_proc,
        "dataloader_workers": args.dataloader_workers,
        "prefetch": args.prefetch,
        "max_rows": args.max_rows,
        "setup_time_s": round(setup_time, 3),
        "time_to_first_row_s": round(time_to_first, 3),
        "total_time_s": round(total_time, 3),
        "total_rows": total,
        "throughput_rows_per_sec": round(throughput, 1),
    }
    print(json.dumps(result), flush=True)


def _run_simple(ds, max_rows):
    total = 0
    time_to_first = None
    t0 = time.perf_counter()
    for _ in ds:
        if time_to_first is None:
            time_to_first = time.perf_counter() - t0
        total += 1
        if total >= max_rows:
            break
    total_time = time.perf_counter() - t0
    return total, time_to_first or 0.0, total_time


def _run_dataloader(ds, num_workers, max_rows):
    import torch.utils.data

    dl = torch.utils.data.DataLoader(
        ds, batch_size=None, num_workers=num_workers, multiprocessing_context="fork",
    )
    total = 0
    time_to_first = None
    t0 = time.perf_counter()
    for _ in dl:
        if time_to_first is None:
            time_to_first = time.perf_counter() - t0
        total += 1
        if total >= max_rows:
            break
    total_time = time.perf_counter() - t0
    return total, time_to_first or 0.0, total_time


if __name__ == "__main__":
    main()
