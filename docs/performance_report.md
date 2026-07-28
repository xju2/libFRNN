# Exact FRNN performance report

## Scope and revisions

- Baseline implementation: `a35c71322750`
- Measured optimized implementation: `fc4e73cb48da`
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
- Acceptance comparison CUDA architecture: 52, matching the immutable baseline
- Native GPU measurement CUDA architecture: 75
- Synthetic warmup/measured iterations: 10/50
- Real warmup/measured iterations: 10/30

CUDA-event timings record on the call's stream and synchronize the stop event.
Host timings use `std::chrono::steady_clock` and synchronize before stopping.
The raw machine-readable results are:

- `benchmarks/results/baseline_a35c713.csv`
- `benchmarks/results/final_fc4e73c_sm52.csv`
- `benchmarks/results/final_real_fc4e73c_sm52.csv`
- `benchmarks/results/final_fc4e73c_sm75.csv`
- `benchmarks/results/final_real_fc4e73c_sm75.csv`

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
4. assigns cells, scans counts, and counting-sorts points into AoS for
   dimensions 1–4 or SoA for dimensions 5–32;
5. processes identical-set queries in spatial order while restoring original
   output rows, while separate-query kernels read the original AoS database;
6. rejects non-intersecting corner cells in four-dimensional grids;
7. evaluates specialized dimensions 1/2/3/4/8/12/16 or the generic path,
   using 256-thread low-dimensional and 128-thread high-dimensional kernels;
8. screens high-dimensional candidates with grouped SoA loads in a
   conservative reordered pass, then recomputes survivors with canonical
   axis-order float32 fused multiply-adds;
9. uses sorted insertion for short rows and converts to a deterministic
   max-heap after 24 candidates when K is at least 64;
10. counts, scans, and writes final edges.

Internal neighbor rows use 32-bit indices and one lazy `-1` sentinel. The
public edge type remains signed 64-bit. The device API stays fully
asynchronous and supports both full-output and count-only modes. The host API
uses count-only search, allocates exact edge storage, then writes without
rerunning search.

## Correctness

The independent CPU oracle in `tests/test_frnn.cpp` computes all-pairs
`float32` squared distance with explicit axis-order `std::fma`, inclusive
radius membership, deterministic distance/index sorting, K truncation, and
edge post-processing.

The core test runs 2,500 randomized cases through forced grid, forced CUDA
brute force, and automatic dispatch: 7,500 path comparisons per run. Coverage
includes dimensions 1/2/3/4/8/16/32; K 0/1/4/8/16/32/64; empty and dense
neighborhoods; identical and separate sets; duplicate, clustered, correlated,
sorted, anisotropic, and degenerate coordinates; exact and adjacent
`nextafter` radius boundaries; ties; reused workspaces; alternating sizes;
count-only calls; default/non-default streams; and concurrent independent
streams. An additional 4,096-candidate D=8/12/16/32 suite targets rounding
sensitivity exactly at the radius. Dedicated cases cover grid-cell
boundaries, tiny and large finite coordinates/cells, highly occupied cells,
and empty cells between occupied cells.

The real reference validation is exact:

| Item | Result |
|---|---:|
| Embedding | 271,663 points × 12 dimensions |
| Radius / K | 0.12 / 1000 |
| Reference directed edges | 9,279,672 |
| Output directed edges | 9,279,672 |
| Missing / extra | 0 / 0 |
| Precision / recall | 1.0 / 1.0 |
| Repeated host/Python time | 0.576 s |

The CUDA core, Python binding suite, and installed CMake consumer all form
separate validation layers. Compute Sanitizer could not run because this
machine's installation lacks `libsanitizer-collection.so`; Nsight limitations
are detailed in `docs/optimization_log.md`.

## Performance

Warm device latency:

| Workload | Baseline p50 / p95 | Final p50 / p95 | p50 speedup |
|---|---:|---:|---:|
| 128, D=3, K=4, sparse | 0.168 / 0.170 ms | 0.059 / 0.059 ms | 2.87x |
| 512, D=3, K=32, dense | 1.031 / 1.037 ms | 0.793 / 0.811 ms | 1.30x |
| 4,096, D=3, K=16, sparse | 0.454 / 0.459 ms | 0.239 / 0.244 ms | 1.89x |
| 10,000, D=8, K=16, sparse | 8.240 / 8.252 ms | 2.213 / 2.235 ms | 3.72x |
| 10,000 separate, D=4, K=32 | 2.733 / 2.754 ms | 2.881 / 2.900 ms | 0.95x |

The p50 geometric-mean speedup over these five immutable-baseline cases is
1.90x. The only primary-case regression is 5.4%, below the 10% acceptance
limit. Architecture-75 results are effectively identical on these cases and
are retained separately rather than mixed into the baseline comparison.

Host convenience API latency:

| Workload | Baseline p50 | Final p50 | Speedup |
|---|---:|---:|---:|
| 128, D=3, K=4 | 0.706 ms | 0.118 ms | 5.99x |
| 512, D=3, K=32 | 1.604 ms | 1.038 ms | 1.55x |
| 4,096, D=3, K=16 | 1.372 ms | 0.554 ms | 2.48x |
| 10,000, D=8, K=16 | 9.520 ms | 3.272 ms | 2.91x |
| 10,000 separate, D=4, K=32 | 5.109 ms | 4.328 ms | 1.18x |

Real device breakdown:

| Phase | p50 | p95 |
|---|---:|---:|
| One-time workspace reserve | 45.260 ms | 47.051 ms |
| Cold device call | 308.930 ms | 311.633 ms |
| Warm device GPU | 263.300 ms | 265.539 ms |
| Grid bounds | 0.129 ms | 0.156 ms |
| Cell assignment | 0.069 ms | 0.070 ms |
| Dense-grid scan | 0.047 ms | 0.049 ms |
| SoA point reorder | 0.526 ms | 0.535 ms |
| Neighbor search | 259.343 ms | 261.822 ms |
| Edge count / scan / write | 3.234 ms | 3.302 ms |
| Warm synchronized host | 263.708 ms | 265.177 ms |
| 9.28M-edge device-to-host copy | 6.987 ms | 7.323 ms |

The real warm device path is 1.35x faster than the previous 356.277 ms
result. Neighbor search is now 98.5% of measured device latency; grid
construction and edge post-processing are no longer credible primary
bottlenecks.

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

The architecture-52 Release shared library grows from 168 KiB to 573 KiB
(3.41x); the architecture-75 build is 645 KiB. A clean parallel build on this
machine grows from 4.01 seconds to 9.47 seconds (2.36x). The additional code
comes mainly from common-dimension kernels specialized for identical versus
separate inputs and from the exact heap path. Resource inspection reports 64
registers and no stack for the retained identical D=8/D=12 kernels; identical
D=16 uses 64 registers and 128 bytes of stack.

No immutable primary benchmark regresses by 10%. Relative to the previous
broad optimized matrix, no case regresses by 10% either: the largest increases
are 6.7% for 500,000/D=3 and 4.7% for separate D=4. D=8 and D=12 cases improve
by 18–32%, including the formerly regressed duplicate distribution, which
drops from 2.364 ms to 1.690 ms.

## Retained and rejected experiments

Retained:

- exact CUDA brute-force fallback and documented dispatch;
- early distance exit and common-dimension specializations;
- spatially ordered identical-set queries;
- lazy neighbor sentinel and 32-bit internal indices;
- dimension-sensitive 3D/4D grids and metadata ceilings;
- dimension-sensitive AoS/SoA reorder and 256/128-thread search kernels;
- four-dimensional cell bounding-box pruning;
- conservative reordered distance screening with canonical recomputation;
- exact count-first host output allocation;
- density-triggered hybrid top-K heap.

Rejected or narrowed:

- unconditional brute force for 512/D=3/K=32 and 1,000/D=4/K=8;
- a universal 131,072-cell grid cap, which made the real D=12 case 2.25x
  slower;
- radius-sized cells for four grid dimensions, which regressed large D=8;
- an unconditional K=64 heap, which doubled a sparse 50k-point case;
- a fourth grid dimension for D=4, which regressed the small D=4 case;
- unconditional SoA and 128-thread search, which regressed low-dimensional
  and separate-query cases;
- direct reordered/grouped distance accumulation, which failed adversarial
  float32 radius-boundary exactness;
- a compiled but undispatched warp-cooperative kernel, removed because it had
  no validated performance result;
- thread-local K arrays, avoided after resource inspection showed existing
  D=8/D=16 stack and register pressure.

## Remaining bottlenecks

- Neighbor search accounts for 98.5% of the real device path. Further material
  gain requires a validated cooperative candidate/selection redesign rather
  than more grid-setup tuning.
- High-dimensional search still indexes only four coordinates and can visit
  false candidates for correlated or anisotropic data, although reordered
  screening now rejects them earlier.
- Per-query selection remains serial after switching to a heap. The attempted
  warp-cooperative path was incomplete and removed; a correct bounded
  cooperative top-K is a separate substantial design.
- Dense grid scans and edge post-processing together cost under 4 ms on the
  263 ms real path, so compact occupied-cell metadata, kernel fusion, and CUDA
  Graph capture have little credible end-to-end benefit for this workload.
- The synchronous host API still allocates a fresh workspace per call; callers
  needing minimum repeated latency should use the asynchronous workspace API.

## Reproduction

```bash
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=52 \
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

Both architecture-52 and architecture-75 Release configurations built and
passed the expanded CTest suite. The architecture-52 shared library installed,
configured in `tests/consumer`, built, and ran with output `1 0`. A separate
`BUILD_SHARED_LIBS=OFF` Release build passed CTest, installed, and its static
consumer also built and ran with output `1 0`.

The optional Python package passed all 12 binding tests. The final package
then reproduced all 9,279,672 supplied real reference edges with zero missing,
zero extra, and no duplicates.

`ldd build-final-sm52/libfrnn.so` reports CUDA runtime and standard system
libraries only. Source/dependency inspection confirms no production Torch,
ATen, c10, Python, or TBB linkage. The only Python discovery in CMake is under
the disabled-by-default `FRNN_BUILD_PYTHON` option. Allocation and
synchronization inspection confirms they remain outside the reusable
asynchronous hot path.

## Suggested pull request

Title:

```text
Optimize exact standalone FRNN search and edge construction
```

Description:

```text
Adds an exact CUDA brute-force fallback, dimension-sensitive 3D/4D grids,
AoS/SoA search layouts, conservative high-dimensional screening,
spatially ordered same-set queries, lazy compact neighbor rows, a
density-triggered exact top-K heap, and count-first host output allocation.
Expands differential testing to all dispatch paths and adds reproducible
synthetic/real benchmarks. The supplied 271,663×12 dataset matches all
9,279,672 reference edges exactly. Representative warm GPU p50 improves by
1.90x geometric mean, including 2.87x on small sparse and 3.72x on medium D=8
workloads; the real device path improves from 356.3 ms to 263.3 ms.
```
