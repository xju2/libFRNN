#!/bin/bash
# PCA-rotation A/B for the D=12 / N=271663 / K=1000 embedding.
# Same binary, input, and flags; only FRNN_PCA_ROTATE differs.
# Usage: run_pca_ab.sh [embedding.csv] [output-directory]
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)
BIN=${FRNN_BENCHMARK_BIN:-$ROOT/build/frnn_benchmark}
INPUT=${1:-$ROOT/data/embedding_data.csv}
OUT=${2:-$ROOT/benchmarks/results/ab}
mkdir -p "$OUT"

echo "=== GPU ==="; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

echo "=== A: PCA disabled ==="
FRNN_PCA_ROTATE=0 "$BIN" --embedding "$INPUT" --warmup 5 --iterations 30 \
  --output "$OUT/pca_A_off.csv"

echo "=== B: PCA enabled ==="
FRNN_PCA_ROTATE=1 "$BIN" --embedding "$INPUT" --warmup 5 --iterations 30 \
  --output "$OUT/pca_B_on.csv"

echo "=== stage_neighbor_search comparison ==="
for tag in A_off B_on; do
  f="$OUT/pca_${tag}.csv"
  # header then the neighbor-search stage row
  line=$(grep "stage_neighbor_search" "$f" | head -1)
  echo "$tag : $line"
done
echo "done."
