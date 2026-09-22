#!/bin/bash
# PCA-rotation A/B for the D=12 / N=271663 / K=1000 embedding.
# Same binary (build-ab-mod), same flags; only the input CSV differs.
#   A = raw embedding (grid on raw columns 0-3)
#   B = PCA-rotated embedding (grid on top-4 variance axes)
set -euo pipefail

ROOT=/global/u1/d/dratnam/libFRNN
BIN=$ROOT/build-ab-mod/frnn_benchmark
RAW=/global/cfs/cdirs/m3443/www/xju/frnn_data/embedding_data.csv
ROT=$ROOT/benchmarks/results/embedding_data_pca.csv
OUT=$ROOT/benchmarks/results/ab
mkdir -p "$OUT"

echo "=== GPU ==="; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

echo "=== A: RAW axes ==="
"$BIN" --embedding "$RAW" --warmup 5 --iterations 30 \
  --output "$OUT/pca_A_raw.csv"

echo "=== B: PCA-rotated axes ==="
"$BIN" --embedding "$ROT" --warmup 5 --iterations 30 \
  --output "$OUT/pca_B_rotated.csv"

echo "=== stage_neighbor_search comparison ==="
for tag in A_raw B_rotated; do
  f="$OUT/pca_${tag}.csv"
  # header then the neighbor-search stage row
  line=$(grep "stage_neighbor_search" "$f" | head -1)
  echo "$tag : $line"
done
echo "done."
