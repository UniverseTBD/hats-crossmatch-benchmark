#!/usr/bin/env bash
set -euo pipefail

OUTDIR="results/param_sweep"
JSONL="$OUTDIR/sweep_results.jsonl"
JSON="$OUTDIR/sweep_results.json"

mkdir -p "$OUTDIR"
> "$JSONL"  # truncate

run() {
    local np="$1" dw="$2" pf="$3"
    echo ""
    echo ">>> num_proc=$np  dataloader_workers=$dw  prefetch=$pf"
    uv run python sweep_bench.py \
        --num-proc "$np" \
        --dataloader-workers "$dw" \
        --prefetch "$pf" \
        --max-rows 5000 \
        | tee -a "$JSONL"
}

echo "=== Group A: Prefetch only (num_proc=0, dataloader_workers=0) ==="
for pf in 1 2 4 8 12 16; do
    run 0 0 "$pf"
done

echo ""
echo "=== Group B: Multiprocess pool (dataloader_workers=0, prefetch=16) ==="
for np in 1 2 4 8; do
    run "$np" 0 16
done

echo ""
echo "=== Group C: DataLoader workers only, prefetch 16 ==="
for dw in 1 2 4 8; do
    run 0 "$dw" 16
done

# Convert JSONL to JSON array
echo "Converting to JSON array..."
python3 -c "
import json, sys
lines = open('$JSONL').read().strip().split('\n')
results = [json.loads(l) for l in lines if l.strip()]
with open('$JSON', 'w') as f:
    json.dump(results, f, indent=2)
print(f'Wrote {len(results)} results to $JSON')
"

# Summary table
echo ""
echo "=== Summary ==="
python3 -c "
import json
results = json.load(open('$JSON'))
print(f'{'Config':<45} {'Setup':>7} {'TTFR':>7} {'Total':>7} {'Rows':>6} {'rows/s':>8}')
print('-' * 85)
for r in results:
    cfg = f\"np={r['num_proc']} dw={r['dataloader_workers']} pf={r['prefetch']}\"
    print(f'{cfg:<45} {r[\"setup_time_s\"]:>7.1f} {r[\"time_to_first_row_s\"]:>7.1f} {r[\"total_time_s\"]:>7.1f} {r[\"total_rows\"]:>6} {r[\"throughput_rows_per_sec\"]:>8.1f}')
"
