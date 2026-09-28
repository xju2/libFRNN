#!/usr/bin/env python3
"""PCA-rotation prototype front-end for the libFRNN D=12 A/B experiment.

Two jobs:
  1. Diagnose how much variance lives in the RAW first-4 columns (what the
     grid prunes on today) vs. the TOP-4 principal directions (what it would
     prune on after rotation). This is the go/no-go check.
  2. If warranted, write a rotated copy of the embedding CSV whose columns are
     ordered by descending variance, so the existing binary grids on the
     highest-variance axes with no kernel changes.

The rotation is an orthonormal transform, so all pairwise Euclidean distances
are preserved exactly (up to float32 roundoff) -> identical neighbors.
"""
import sys
import numpy as np

SRC = "/global/cfs/cdirs/m3443/www/xju/frnn_data/embedding_data.csv"
DST = "/global/u1/d/dratnam/libFRNN/benchmarks/results/embedding_data_pca.csv"
DIM = 12
GRID_AXES = 4  # libFRNN grids on the first min(D,4) columns for D>4


def main():
    print(f"loading {SRC} ...", flush=True)
    X = np.loadtxt(SRC, delimiter=",", dtype=np.float64)
    assert X.shape[1] == DIM, X.shape
    n = X.shape[0]
    print(f"loaded {n} x {DIM}", flush=True)

    mu = X.mean(axis=0)
    Xc = X - mu

    # Per-axis variance in the raw frame.
    raw_var = Xc.var(axis=0)
    total_var = raw_var.sum()

    # PCA: eigendecomposition of the covariance (12x12 -> trivial).
    cov = (Xc.T @ Xc) / n
    eigvals, eigvecs = np.linalg.eigh(cov)          # ascending
    order = np.argsort(eigvals)[::-1]               # descending
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]                      # columns = components

    raw_top4 = np.sort(raw_var)[::-1][:GRID_AXES].sum()
    pca_top4 = eigvals[:GRID_AXES].sum()

    print("\n=== VARIANCE DIAGNOSTIC ===")
    print(f"total variance:                {total_var:.6g}")
    print(f"raw columns 0-3 (as-is):       {raw_var[:GRID_AXES].sum():.6g} "
          f"({100*raw_var[:GRID_AXES].sum()/total_var:.1f}% of total)")
    print(f"best 4 raw columns:            {raw_top4:.6g} "
          f"({100*raw_top4/total_var:.1f}% of total)")
    print(f"top-4 principal directions:    {pca_top4:.6g} "
          f"({100*pca_top4/total_var:.1f}% of total)")
    print(f"\nper-axis raw variance:  "
          + ", ".join(f"{v:.4g}" for v in raw_var))
    print(f"eigenvalues (desc):     "
          + ", ".join(f"{v:.4g}" for v in eigvals))

    gain = pca_top4 / max(raw_var[:GRID_AXES].sum(), 1e-30)
    print(f"\nPCA top-4 captures {gain:.2f}x the variance of raw cols 0-3 "
          f"on the pruning axes.")
    if gain < 1.05:
        print(">>> Raw columns already near-optimal; PCA unlikely to help.")
    else:
        print(">>> PCA concentrates more variance on the grid axes; "
              "pruning should tighten. Worth an A/B.")

    if len(sys.argv) > 1 and sys.argv[1] == "--write":
        Xr = (Xc @ eigvecs).astype(np.float32)      # rotate, cols by var desc
        print(f"\nwriting rotated CSV -> {DST} ...", flush=True)
        np.savetxt(DST, Xr, fmt="%.7g", delimiter=",")
        # Correctness self-check: distances preserved on a random sample.
        rng = np.random.default_rng(0)
        idx = rng.choice(n, size=200, replace=False)
        d_raw = np.linalg.norm(Xc[idx][:, None] - Xc[idx][None], axis=2)
        d_rot = np.linalg.norm(Xr[idx][:, None] - Xr[idx][None], axis=2)
        max_abs = np.abs(d_raw - d_rot).max()
        print(f"distance preservation check (200 pts): "
              f"max |d_raw - d_rot| = {max_abs:.3e}")
        print("done.", flush=True)


if __name__ == "__main__":
    main()
