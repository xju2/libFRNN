# Exact FRNN performance report

## Scope and revisions

- Baseline implementation: `a35c71322750`
- Measured optimized implementation: `2a5da2907916`
- Branch: `improve`

The optimized implementation preserves the standalone C++17/CUDA API and has
no production dependency on Torch, ATen, c10, Python, or TBB. Python remains
an optional NumPy binding over the same C++ library.

## Measurement environment

- GPU: NVIDIA GeForce RTX 2070 SUPER, compute capability 7.5, 8 GiB
- Driver: 580.173.02
- CUDA Toolkit/runtime: 12.0.140
- CMake: 3.28.3
- C++ compiler: GNU 13.3.0
- CPU: AMD Ryzen 5 3600 6-Core Processor
- Build: Release
- CMake CUDA architecture: 52 (the environment's detected/default value)
- Synthetic warmup/measured iterations: 10/50
- Real warmup/measured iterations: 10/30

CUDA-event timings record on the call's stream and synchronize the stop event.
Host timings use `std::chrono::steady_clock` and synchronize before stopping.
The raw machine-readable results are:

- `benchmarks/results/baseline_a35c713.csv`
- `benchmarks/results/final_2a5da29.csv`
- `benchmarks/results/final_real_2a5da29.csv`

## Final architecture

Automatic dispatch uses exact CUDA brute force when the overflow-safe
coordinate/selection work estimate is at most 2,000,000. All larger workloads
use the uniform grid. Both paths evaluate every dimension, use an inclusive
radius, and retain candidates by `(squared_distance, original_index)`.

The grid pipeline:

1. computes database bounds;
2. chooses radius-sized cells for up to three grid dimensions and radius/2
   cells for four grid dimensions;
3. caps dense metadata at 131,072 cells for dimensions 1–4 and 2,097,152 cells
   for dimensions 5–32;
4. assigns cells, scans counts, and counting-sorts points;
5. processes identical-set queries in spatial order while restoring original
   output rows;
6. evaluates specialized dimensions 1/2/3/4/8/12/16 or the generic path;
7. exits distance accumulation as soon as its partial sum exceeds the radius;
8. uses sorted insertion for short rows and converts to a deterministic
   max-heap after 24 candidates when K is at least 64;
9. counts, scans, and writes final edges.

Internal neighbor rows use 32-bit indices and one lazy `-1` sentinel. The
public edge type remains signed 64-bit. The device API stays fully
asynchronous and supports both full-output and count-only modes. The host API
uses count-only search, allocates exact edge storage, then writes without
rerunning search.

## Correctness

The independent CPU oracle in `tests/test_frnn.cpp` computes all-pairs
`float32` squared distance, inclusive radius membership, deterministic
distance/index sorting, K truncation, and edge post-processing.

The core test runs 2,500 randomized cases through forced grid, forced CUDA
brute force, and automatic dispatch: 7,500 path comparisons per run. Coverage
includes dimensions 1/2/3/4/8/16/32; K 0/1/4/8/16/32/64; empty and dense
neighborhoods; identical and separate sets; duplicate, clustered, correlated,
sorted, anisotropic, and degenerate coordinates; exact and adjacent
`nextafter` radius boundaries; ties; reused workspaces; alternating sizes;
count-only calls; and default/non-default streams.

The real reference validation is exact:

| Item | Result |
|---|---:|
| Embedding | 271,663 points × 12 dimensions |
| Radius / K | 0.12 / 1000 |
| Reference directed edges | 9,279,672 |
| Output directed edges | 9,279,672 |
| Missing / extra | 0 / 0 |
| Precision / recall | 1.0 / 1.0 |
| Repeated host/Python time | 0.675 s |

The CUDA core, Python binding suite, and installed CMake consumer all form
separate validation layers. Compute Sanitizer could not run because this
machine's installation lacks `libsanitizer-collection.so`; Nsight limitations
are detailed in `docs/optimization_log.md`.

## Performance

Warm device latency:

| Workload | Baseline p50 / p95 | Final p50 / p95 | p50 speedup |
|---|---:|---:|---:|
| 128, D=3, K=4, sparse | 0.168 / 0.170 ms | 0.058 / 0.059 ms | 2.88x |
| 512, D=3, K=32, dense | 1.031 / 1.037 ms | 0.776 / 0.792 ms | 1.33x |
| 4,096, D=3, K=16, sparse | 0.454 / 0.459 ms | 0.235 / 0.240 ms | 1.93x |
| 10,000, D=8, K=16, sparse | 8.240 / 8.252 ms | 3.228 / 3.406 ms | 2.55x |
| 10,000 separate, D=4, K=32 | 2.733 / 2.754 ms | 2.751 / 2.769 ms | 0.99x |

The p50 geometric-mean speedup over these five immutable-baseline cases is
1.80x. The only primary-case regression is 0.6%, below the 10% acceptance
limit.

Host convenience API latency:

| Workload | Baseline p50 | Final p50 | Speedup |
|---|---:|---:|---:|
| 128, D=3, K=4 | 0.706 ms | 0.119 ms | 5.94x |
| 512, D=3, K=32 | 1.604 ms | 1.027 ms | 1.56x |
| 4,096, D=3, K=16 | 1.372 ms | 0.552 ms | 2.49x |
| 10,000, D=8, K=16 | 9.520 ms | 4.246 ms | 2.24x |
| 10,000 separate, D=4, K=32 | 5.109 ms | 4.170 ms | 1.23x |

Real device breakdown:

| Phase | p50 | p95 |
|---|---:|---:|
| One-time workspace reserve | 45.245 ms | 46.958 ms |
| Cold device call | 401.182 ms | 401.985 ms |
| Warm device GPU | 356.277 ms | 358.379 ms |
| Warm synchronized host | 356.552 ms | 358.439 ms |
| 9.28M-edge device-to-host copy | 7.169 ms | 7.253 ms |

The brute-force crossover measurements support the dispatch rule:

- 128/D=16/K=1: brute force 0.089 ms, grid 0.185 ms;
- 128/D=3/K=4: brute force 0.055 ms, grid 0.167 ms;
- 512/D=3/K=32 dense: grid 0.835 ms, brute force 1.121 ms;
- 1,000/D=4/K=8 sparse: grid 0.217 ms, brute force 0.277 ms at the time of
  the dispatch experiment.

The later 3D-grid/cell-cap optimization reduces the automatic 1,000/D=4/K=8
case further to 0.125 ms.

## Memory, compilation, and regressions

For the real host call, major device allocations are approximately:

| Storage | Size |
|---|---:|
| Neighbor distance/index workspace | 2,173,304,000 bytes |
| Exact edge output | 148,474,752 bytes |
| Input and sorted coordinates | 26,079,648 bytes |
| Grid counts/offsets | 16,777,216 bytes |
| Point cell metadata | 2,173,304 bytes |
| Edge count/offset metadata | 4,346,624 bytes |
| Approximate total before scan scratch | 2,261 MiB |

Count-first host allocation avoids the 4,146 MiB conservative edge output
that would otherwise be required for N×K capacity.

The Release shared library grows from 168 KiB to 399 KiB (+231 KiB, 2.38x).
A clean parallel build on this machine grows from 4.01 seconds to 8.40 seconds
(2.09x), mainly from dimension-specialized kernels and the exact heap path.

No immutable primary benchmark regresses by 10%. Relative to the first broad
post-optimization matrix, the D=8 duplicate distribution rises from 2.097 ms
to 2.364 ms (+12.7%) while the rest of that matrix generally improves. This
edge-case regression is retained because the final structural changes produce
large gains on empty, boundary, clustered, dense-K, large D=8, and 500k-point
cases, and the duplicate case remains exact.

## Retained and rejected experiments

Retained:

- exact CUDA brute-force fallback and documented dispatch;
- early distance exit and common-dimension specializations;
- spatially ordered identical-set queries;
- lazy neighbor sentinel and 32-bit internal indices;
- dimension-sensitive 3D/4D grids and metadata ceilings;
- exact count-first host output allocation;
- density-triggered hybrid top-K heap.

Rejected or narrowed:

- unconditional brute force for 512/D=3/K=32 and 1,000/D=4/K=8;
- a universal 131,072-cell grid cap, which made the real D=12 case 2.25x
  slower;
- radius-sized cells for four grid dimensions, which regressed large D=8;
- an unconditional K=64 heap, which doubled a sparse 50k-point case;
- a fourth grid dimension for D=4, which regressed the small D=4 case;
- thread-local K arrays, avoided after resource inspection showed existing
  D=8/D=16 stack and register pressure.

## Remaining bottlenecks

- High-dimensional search still indexes only four coordinates and can visit
  many false candidates for correlated or anisotropic data.
- Dense grid scans operate on a dimensionality-selected ceiling rather than a
  compact occupied-cell structure.
- Per-query selection is serial even after switching to a heap; a
  warp-cooperative design may help dense cells.
- Edge count, scan, write, and count copy remain separate stages.
- The synchronous host API still allocates a fresh workspace per call.
- CUDA Graph capture and a sorted-key/occupied-cell redesign remain credible
  larger projects, but neither had enough measured evidence to retain here.

## Reproduction

```bash
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DFRNN_BUILD_TESTS=ON \
  -DFRNN_BUILD_BENCHMARKS=ON
cmake --build build --parallel
ctest --test-dir build --output-on-failure

./build/frnn_benchmark \
  --full --warmup 10 --iterations 50 \
  --output benchmarks/results/reproduction.csv

./build/frnn_benchmark \
  --embedding data/embedding_data.csv \
  --warmup 10 --iterations 30 \
  --output benchmarks/results/reproduction-real.csv

python tests/compare_reference_edges.py --minimum-agreement 1.0
```

For the optional Python validation:

```bash
python -m pip install .
python tests/test_python.py
```

For installed-library consumer validation:

```bash
cmake --install build --prefix "$PWD/install"
cmake -S tests/consumer -B build-consumer \
  -DCMAKE_PREFIX_PATH="$PWD/install"
cmake --build build-consumer --parallel
```

## Final validation results

The required Release configuration, build, and CTest command completed
successfully. The shared library installed, configured in
`tests/consumer`, built, and ran with output `1 0`. A separate
`BUILD_SHARED_LIBS=OFF` Release build passed CTest, installed, and its static
consumer also built and ran with output `1 0`.

`ldd build/libfrnn.so` reports CUDA runtime and standard system libraries
only. Source/dependency inspection confirms no production Torch, ATen, c10,
Python, or TBB linkage. The only Python discovery in CMake is under the
disabled-by-default `FRNN_BUILD_PYTHON` option.

## Suggested pull request

Title:

```text
Optimize exact standalone FRNN search and edge construction
```

Description:

```text
Adds an exact CUDA brute-force fallback, dimension-sensitive 3D/4D grids,
specialized early-exit distance kernels, spatially ordered same-set queries,
lazy compact neighbor rows, a density-triggered exact top-K heap, and
count-first host output allocation. Expands differential testing to all
dispatch paths and adds reproducible synthetic/real benchmarks. The supplied
271,663×12 dataset matches all 9,279,672 reference edges exactly. Representative
warm GPU p50 improves by 1.80x geometric mean, including 2.88x on small sparse
and 2.55x on medium D=8 workloads.
```
