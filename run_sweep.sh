#!/usr/bin/env bash
# Sweep over worker/partition configurations and log results.

set -euo pipefail

OUTDIR="results/sweep"
mkdir -p "$OUTDIR"

configs=(
  "1 1"
  "1 8"
  "4 4"
  "8 8"
  "8 1"
)

for cfg in "${configs[@]}"; do
  read -r workers partitions <<< "$cfg"
  label="w${workers}_p${partitions}"
  outfile="$OUTDIR/${label}.txt"

  echo "=== Running: workers=$workers partitions_per_chunk=$partitions ==="
  uv run reference/examples/speedtest.py \
    --n-workers="$workers" \
    --partitions-per-chunk="$partitions" \
    2>&1 | tee "$outfile"
  echo ""
done

echo "All runs complete. Results in $OUTDIR/"
