# Exact FRNN optimization log

## Baseline

The immutable implementation baseline is commit `a35c71322750`
(`a35c713`). The baseline benchmark is
`benchmarks/results/baseline_a35c713.csv`.

The measurements in this log were collected on:

- NVIDIA GeForce RTX 2070 SUPER, compute capability 7.5, 8 GiB;
- NVIDIA driver 580.173.02;
- CUDA Toolkit 12.0.140;
- CMake 3.28.3;
- GNU C++ 13.3.0;
- AMD Ryzen 5 3600 6-Core Processor;
- Release build, with CMake's detected/default CUDA architecture 52.

The benchmark uses a non-blocking caller stream unless the case name says
`default_stream`. GPU latency is measured with CUDA events on that stream.
Host latency uses `std::chrono::steady_clock` and synchronizes before stopping
the timer. Each result reports minimum, p50, p95, mean, and standard
deviation. The benchmark separately reports workspace reservation, cold
device, warm device GPU, warm device host, device-to-host output copy, and
host convenience API latency.

The baseline source always used the dense grid. Representative warm GPU p50
latencies were:

| Case | Baseline p50 |
|---|---:|
| 128 points, D=3, K=4, sparse | 0.168 ms |
| 512 points, D=3, K=32, dense | 1.031 ms |
| 4,096 points, D=3, K=16, sparse | 0.454 ms |
| 10,000 points, D=8, K=16, sparse | 8.240 ms |
| 10,000 separate queries, D=4, K=32 | 2.733 ms |

## Initial profiling observations

Source inspection and CUDA-event measurements identified these structural
costs:

1. Every grid call clears and prefix-scans 2,097,152 dense cells, even when
   only a small grid is occupied.
2. Small clouds pay for bounds, grid construction, counting sort, two scans,
   neighbor search, and edge compaction even when direct exact search is less
   work.
3. Distance calculation always evaluated every dimension after the partial
   squared distance had already exceeded the radius.
4. Identical query/reference inputs were searched in original query order,
   even though a spatial ordering had already been built for the database.
5. Every query initialized all K distance/index slots. The real workload uses
   K=1000 but has far fewer valid neighbors, making this an avoidable O(NK)
   write.
6. Internal neighbor indices used 64 bits even though input sizes are already
   limited to `INT_MAX`.
7. The synchronous host API allocates and frees all device storage per call
   and synchronizes once for the count and once for edge output.
8. The one-thread-per-query top-K insertion path remains sensitive to dense
   cells, large K, and high dimensions.

Nsight Systems 2025.5.2 is installed, but its QDSTRM importer is missing, so
reports cannot be generated. Nsight Compute can attach, but hardware
performance-counter access is denied (`ERR_NVGPUCTRPERM`). Resource usage was
therefore inspected with `cuobjdump --dump-resource-usage`. The final
specialized D=8 grid kernel uses 54 registers and 64 bytes of stack per thread;
D=16 uses 62 registers and 96 bytes of stack. This is why a large thread-local
K array was not added without a separate cooperative-selection design.
Compute Sanitizer is also installed, but cannot start because its injection
library `libsanitizer-collection.so` is absent from the installation.

## Experiments

### Exact CUDA brute-force fallback — retained

Hypothesis: direct exact search removes fixed grid overhead for sufficiently
small pair counts.

An exact CUDA brute-force kernel now uses the same inclusive radius check,
full-dimensional squared distance, `(distance, original index)` ordering, and
edge post-processing as the grid path. `SearchAlgorithm` permits forced grid
and brute-force runs for tests and benchmarks. Automatic dispatch selects
brute force when:

```text
query_count * database_count * dimension * ceil(K / 8) <= 2,000,000
```

The multiplication is evaluated with overflow-safe divisions.

Retained result: the 128/D=3/K=4 case improved from 0.167 ms to 0.055 ms
(3.0x). A 128/D=16/K=1 case measures 0.089 ms brute force versus 0.185 ms
grid.

Negative crossover results:

- 512/D=3/K=32 dense: brute force 1.121 ms versus grid 0.835 ms;
- 1,000/D=4/K=8 sparse: brute force 0.277 ms versus grid 0.217 ms.

The K-weighted threshold deliberately routes both negative cases to the grid.

### Early-exit and dimension-specialized distance — retained

Hypothesis: for D greater than 3, most grid candidates can be rejected after
only a few non-negative squared terms.

The distance loop now exits as soon as its partial sum exceeds the squared
radius. This cannot turn a rejected candidate into an accepted one because
the remaining terms are non-negative. Dimensions 1, 2, 3, 4, 8, and 16 have
specialized kernels; dimensions through 32 retain the generic exact path.
Query coordinates are cached once per specialized thread.

The 10,000/D=8/K=16 case improved from 8.240 ms to 4.94 ms before query
reordering. The separate 10,000/D=4/K=32 case improved from 2.733 ms to
2.43 ms.

### Spatially ordered identical-set queries — retained

Hypothesis: adjacent GPU threads should traverse similar cells and reuse
candidate cache lines if identical-set queries use the already sorted
database ordering.

For `inputs_are_same`, the grid path now searches the sorted database as the
query sequence and writes each neighbor row to its original query index.
Self-loop comparison and deterministic tie-breaking both use original
indices. No second sort or coordinate copy is introduced.

The 10,000/D=8/K=16 case improved further to 3.48 ms. The 4,096/D=3/K=16
case improved to 0.363 ms. Differential tests verify that final edge order is
unchanged.

### Lazy neighbor sentinel and 32-bit internal indices — retained

Hypothesis: initializing all K slots is unnecessary because insertion only
reads the first `found` entries and edge counting stops at a sentinel.

The kernels now write one `-1` sentinel after the last valid neighbor rather
than initializing every slot. Internal indices are 32-bit; public edges
remain signed 64-bit. Workspace storage falls from 12 to 8 bytes per reserved
neighbor.

Representative final experiment p50 values are:

| Case | Baseline | Current | Speedup |
|---|---:|---:|---:|
| 128/D=3/K=4 sparse | 0.168 ms | 0.058 ms | 2.91x |
| 512/D=3/K=32 dense | 1.031 ms | 0.831 ms | 1.24x |
| 4,096/D=3/K=16 sparse | 0.454 ms | 0.342 ms | 1.33x |
| 10,000/D=8/K=16 sparse | 8.240 ms | 3.424 ms | 2.41x |
| 10,000 separate/D=4/K=32 | 2.733 ms | 2.165 ms | 1.26x |

No case in this baseline subset regressed.

### D=12 specialization and capped 4D grid — retained

The real embedding has 12 dimensions. Adding D=12 to the specialized,
query-cached distance kernels reduced its exact host call from 1.164 seconds
to 0.749 seconds. The D=12 grid kernel uses 51 registers and 80 bytes of stack
per thread.

For dimensions 5 and higher, the dense grid now indexes four coordinates.
Grid finalization increases cell size until the dense metadata fits within
2,097,152 cells, so candidate coverage remains exact. Dimensions through 4
use at most three grid coordinates because a 1,000/D=4/K=8 case showed that
the fourth nested cell loop cost more than it saved. The final routing measures
0.126 ms for that case versus 0.215 ms before the grid-cap experiments.

Grid metadata is workload-sensitive:

- 1D–4D inputs use a 131,072-cell allocation/scan ceiling;
- dimensions 5–32 use a 2,097,152-cell ceiling for the 4D grid;
- 1D–3D use radius-sized cells;
- higher dimensions retain radius/2 cells.

A rejected universal 131,072-cell cap was exact and faster on the synthetic
matrix, but slowed the real D=12 case from 0.700 seconds to 1.574 seconds by
making its 4D cells too coarse. The dimension-aware cap retains the small-grid
speedups without that regression.

### Count-first host output allocation — retained

The device API now supports count-only execution when `edges == nullptr`.
The synchronous host path uses that mode, synchronizes the exact count,
allocates exactly the required edge storage, and writes from the already
computed neighbor/offset workspace. It does not repeat neighbor search.

For the real directed workload, temporary device edge storage falls from the
4.35 GB conservative capacity to 148 MB for 9,279,672 exact edges. The warm
device-to-host edge copy is about 7.1 ms.

### Hybrid top-K heap — retained

Insertion shifting is efficient for short neighbor lists but scales poorly
for dense, large-K queries. For K at least 64, each query now starts with the
sorted insertion path. Only after it accepts a 25th candidate does it convert
the row in place to a deterministic max-heap. Heap replacement retains the
best K by `(distance, original index)`, and an exact heap sort restores public
neighbor order at the end.

The dense 10,000/D=3/K=64 case improves from about 7.9 ms to 3.7 ms. A rejected
always-heap K=64 experiment made a 50,000-point sparse case take 3.82 ms; the
hybrid takes about 1.94 ms because sparse queries never convert. On the real
K=1000 workload, warm device p50 improves from 370.8 ms with insertion to
357.7 ms with the 24-candidate hybrid. Thresholds of 32 and an always-heap
variant are retained in machine-readable experiment results for comparison.

### Structure-of-arrays database reorder — retained

Hypothesis: candidate threads read the same coordinate axis at adjacent sorted
indices, so a structure-of-arrays layout should coalesce distance-filter loads
better than the original array-of-structures layout.

For dimensions above four, the counting-sort output now stores every
coordinate axis contiguously while retaining original point indices
separately. This reduced real D=12 neighbor-search time from about 394 ms to
336 ms in the initial experiment and made later independent-load scheduling
possible.

The first version applied SoA and 128-thread blocks unconditionally. The full
matrix then exposed regressions of 20.7% for separate D=4 queries and 14.9%
for the 500,000-point D=3 case. The retained layout is workload-sensitive:
dimensions through four keep sorted AoS records and use 256-thread blocks;
higher dimensions use SoA and 128 threads. Separate query/reference kernels
read original AoS database records because unrelated query cells do not
coalesce sorted SoA loads. Identical and separate modes are compile-time
kernel specializations to avoid carrying both paths in registers. The
affected warm p50 values recovered to 3.53 ms for 500,000/D=3 and 2.88 ms for
separate D=4, within 7% and 5% of the prior optimized matrix respectively.

### Four-dimensional cell AABB filter and 128-thread search blocks — retained

Hypothesis: the radius-aligned nested cell range contains corner cells whose
minimum Euclidean distance already exceeds the radius, and the high-register
search kernel does not benefit from 256-thread blocks.

Four-dimensional grid traversal now rejects such cells with an
axis-aligned-box distance test before loading their points, and the
high-dimensional search kernel uses 128 threads with an occupancy launch
bound. Applying that arithmetic to three-dimensional grids was not beneficial
and is disabled. The real workload improved from about 336 ms to 311 ms.
Differential cases spanning boundary, empty-cell, and randomized layouts
retained exact output.

### Reordered grouped distance screening — narrowed for exactness

The first experiment accumulated dimensions 4..D-1 before the four indexed
grid dimensions and grouped four squared terms into each addition. It reduced
the real warm device time to roughly 256 ms, but changed float32 evaluation
order. A new adversarial suite that normalizes 4,096 D=8/12/16/32 candidates
onto the radius boundary exposed an edge-set mismatch, so that direct
reordered result was rejected.

The retained design uses reordered four-load groups only as a conservative
screen. Its rejection threshold expands `radius_squared` by 2^-16 relative
and 2^-142 absolute, exceeding the roundoff difference bound for at most 32
non-negative float32 terms. Candidates not safely outside that threshold are
recomputed with fused multiply-adds in canonical dimension order. This keeps
neighbor membership and distance ordering identical to the explicit FMA
oracle while preserving most of the filter benefit. Preliminary warm p50 is
263.2 ms on the real D=12 workload and 2.21 ms on the 10,000-point D=8 sparse
case, versus 356.3 ms and 3.23 ms before the SoA/screening work.

An experimental warp-cooperative large-K kernel was compiled but never
dispatched and had no validated performance result. It was removed rather
than retained as an abandoned alternate path.

## Correctness evidence

The independent CPU oracle computes every dimension with an explicit
float32 fused multiply-add in axis order, includes
`distance <= radius * radius`, sorts by `(distance, original index)`,
truncates to K, and then applies edge post-processing.

The C++ differential suite currently covers 2,500 randomized cases and all
three dispatch choices, producing 7,500 forced/automatic comparisons per
run. It varies dimensions 1, 2, 3, 4, 8, 16, and 32; K; radii; identical and
separate inputs; directed and undirected output; empty neighborhoods; dense
neighborhoods; duplicates; correlated, sorted, clustered, and degenerate
coordinates. Dedicated cases cover exact radius boundaries, `nextafter`
inside/outside values, ties, repeated calls, and a caller-provided stream.

The supplied real workload (`271,663 x 12`, radius 0.12, K=1000) exactly
matches all 9,279,672 directed edges in `data/edge_list.csv`: zero missing,
zero extra, precision and recall both 1.0.

## Current bottlenecks and next experiments

- The grid still scans a dense metadata ceiling selected by dimensionality.
- High-dimensional grids index only the first four dimensions; early exit
  preserves exactness but candidate traffic remains high.
- Dense and clustered cases still use a serial per-query selection structure.
- Separate query/reference mode cannot reuse the database ordering for query
  locality.
- Edge counting, scan, writing, and final count copy remain separate stages.
- CUDA Graph capture and a compact occupied-cell/sorted-key grid have not yet
  shown enough evidence to justify their complexity.

The broad final matrix is in
`benchmarks/results/final_2a5da29.csv`. It covers point counts through
500,000, dimensions 3/4/8/16, K 1 through 64, densities from empty through
256 neighbors, identical and separate query sets, default and non-default
streams, and uniform, clustered, separated-cluster, boundary, duplicate,
correlated, anisotropic, degenerate, and sorted distributions.
