"""Generate synthetic FRNN benchmark clouds from the real embedding distribution.

Model: correlated multivariate Gaussians fitted from the *raw* (un-rotated) real
D=12 embedding, preserving its full mean and 12x12 covariance -- including the
off-diagonal cross-axis structure that the PCA-rotation front-end (v1.1.0) exists
to exploit. Sampling an already-diagonalized ("_pca") spectrum would hand the
library data that is already in its own principal basis, hiding the rotation's
benefit; fitting the raw covariance avoids that.

Dimension handling:
  D <= 12: top-left DxD block of the real covariance -- the marginal of the first
           D raw coordinates, which retains cross-axis correlation.
  D  > 12: real 12x12 covariance plus extra independent axes whose variances
           continue the eigenvalue spectrum's geometric tail.

Each (N, D) uses a deterministic seed so all FRNN versions run on identical inputs.

Usage:
  python benchmarks/make_synthetic.py generate [--outdir DIR]
  python benchmarks/make_synthetic.py calibrate [--outdir DIR] [--neighbors K]
"""
import argparse
import hashlib
from pathlib import Path

import numpy as np

REFERENCE = Path('data/embedding_data.csv')  # raw, un-rotated embedding
REFERENCE_SHAPE = (271663, 12)
N_VALUES = [1_000, 5_000, 10_000, 50_000, 100_000, 500_000, 1_000_000]
D_VALUES = [3, 4, 12, 24]
TARGET_NEIGHBORS = 32  # expected neighbors per query (excluding self) for radius calibration


def fit_distribution() -> tuple[np.ndarray, np.ndarray]:
    """Mean (D,) and full covariance (D, D) of the raw real embedding (float64)."""
    real = np.loadtxt(REFERENCE, delimiter=',', dtype=np.float32)
    if real.shape != REFERENCE_SHAPE:
        raise ValueError(f'Expected reference {REFERENCE_SHAPE}, got {real.shape}')
    real = real.astype(np.float64)
    return real.mean(axis=0), np.cov(real, rowvar=False)


def dist_for_dim(mean: np.ndarray, cov: np.ndarray,
                 dimension: int) -> tuple[np.ndarray, np.ndarray]:
    """Mean/covariance for a target D: top-left DxD block, or spectral extension."""
    full = cov.shape[0]
    if dimension <= full:
        return mean[:dimension].copy(), cov[:dimension, :dimension].copy()
    # Continue the eigenvalue tail's geometric decay as extra independent axes.
    eig = np.linalg.eigvalsh(cov)[::-1]          # variances, descending
    ratio = float(eig[-1] / eig[-2])             # ~0.77 per step at the tail
    extra = dimension - full
    tail = eig[-1] * ratio ** np.arange(1, extra + 1)
    big = np.zeros((dimension, dimension), dtype=cov.dtype)
    big[:full, :full] = cov
    idx = np.arange(full, dimension)
    big[idx, idx] = tail
    return np.concatenate([mean, np.zeros(extra, dtype=mean.dtype)]), big


def seed_for(count: int, dimension: int) -> int:
    """Deterministic 32-bit seed so all FRNN versions share identical inputs."""
    digest = hashlib.sha256(f'{count}x{dimension}'.encode()).hexdigest()
    return int(digest[:8], 16)


def make_cloud(mean: np.ndarray, cov: np.ndarray,
               count: int, dimension: int) -> np.ndarray:
    m, c = dist_for_dim(mean, cov, dimension)
    rng = np.random.default_rng(seed_for(count, dimension))
    return rng.multivariate_normal(m, c, size=count).astype(np.float32)


def path_for(outdir: Path, count: int, dimension: int) -> Path:
    return outdir / f'synthetic_N{count}_D{dimension}.csv'


def generate(outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    mean, cov = fit_distribution()
    std = np.sqrt(np.diag(cov))
    offdiag = cov[~np.eye(cov.shape[0], dtype=bool)]
    print(f'Fitted from {REFERENCE} (raw, correlated)')
    print(f'  per-axis std: {np.array2string(std, precision=4)}')
    print(f'  max |corr off-diagonal|: {np.abs(offdiag).max() / np.diag(cov).mean():.4f}')
    for dimension in D_VALUES:
        for count in N_VALUES:
            cloud = make_cloud(mean, cov, count, dimension)
            out = path_for(outdir, count, dimension)
            np.savetxt(out, cloud, delimiter=',', fmt='%.7g')
            print(f'  wrote {out.name:32} shape={cloud.shape} seed={seed_for(count, dimension)}')


def empirical_radius(tree, points: np.ndarray, target: int, sample: int = 512) -> float:
    """Radius that yields ~target neighbors/query, estimated on a random sample."""
    rng = np.random.default_rng(0)
    idx = rng.choice(len(points), size=min(sample, len(points)), replace=False)
    # (target+1)th nearest distance includes self at distance 0, so k=target+1.
    dist, _ = tree.query(points[idx], k=target + 1)
    return float(np.median(dist[:, target]))


def calibrate(outdir: Path, target: int, emit: Path | None = None) -> None:
    from scipy.spatial import cKDTree

    mean, cov = fit_distribution()
    print(f'Calibration: radius targeting ~{target} neighbors/query (median kth distance)\n')
    header = f'{"file":34} {"radius":>10} {"mean_nbrs":>10} {"p95_nbrs":>9}'
    print(header)
    print('-' * len(header))
    rows = []
    for dimension in D_VALUES:
        for count in N_VALUES:
            path = path_for(outdir, count, dimension)
            if path.exists():
                points = np.loadtxt(path, delimiter=',', dtype=np.float32)
            else:
                points = make_cloud(mean, cov, count, dimension)
            tree = cKDTree(points)
            radius = empirical_radius(tree, points, target)
            # Verify achieved neighbor counts on a fresh sample (exclude self).
            rng = np.random.default_rng(1)
            qidx = rng.choice(len(points), size=min(256, len(points)), replace=False)
            counts = np.array([len(n) - 1 for n in
                               tree.query_ball_point(points[qidx], radius)])
            p95 = float(np.percentile(counts, 95))
            print(f'{path.name:34} {radius:10.5f} {counts.mean():10.2f} {p95:9.1f}')
            rows.append((count, dimension, radius, float(counts.mean()), p95))

    if emit is not None:
        emit.parent.mkdir(parents=True, exist_ok=True)
        with emit.open('w') as f:
            f.write('N,D,radius,mean_nbrs,p95_nbrs\n')
            for count, dimension, radius, mean_nbrs, p95 in rows:
                f.write(f'{count},{dimension},{radius:.6f},{mean_nbrs:.2f},{p95:.1f}\n')
        print(f'\nWrote radius table: {emit}')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    g = sub.add_parser('generate', help='write the N x D synthetic CSV grid')
    g.add_argument('--outdir', type=Path, default=Path('benchmarks/data/synthetic'))
    c = sub.add_parser('calibrate', help='report per-(N,D) radius and neighbor counts')
    c.add_argument('--outdir', type=Path, default=Path('benchmarks/data/synthetic'))
    c.add_argument('--neighbors', type=int, default=TARGET_NEIGHBORS)
    c.add_argument('--emit', type=Path, default=None,
                   help='write the radius table to this CSV path')
    args = parser.parse_args()

    if args.command == 'generate':
        generate(args.outdir)
    elif args.command == 'calibrate':
        calibrate(args.outdir, args.neighbors, args.emit)


if __name__ == '__main__':
    main()
