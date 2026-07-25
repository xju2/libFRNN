## Prerequisite

This goal starts from the completed PyTorch-removal work.

The repository must already:

* build without PyTorch, ATen, c10, Python, or TBB;
* expose a standalone C++/CUDA API;
* support an explicit CUDA stream;
* include a brute-force correctness reference;
* include a reusable workspace or equivalent allocation mechanism.

If those prerequisites are not present, stop and clearly report that Goal 1 is incomplete. Do not reintroduce PyTorch as a shortcut.

## Primary outcome

Minimize end-to-end and steady-state latency of exact FRNN search and edge construction without changing the defined results.

This is an algorithm and GPU-systems optimization task, not a launch-parameter-only exercise.

Do not stop after changing block size, compiler flags, grid resolution, or a few constants. Those experiments are allowed, but the work must also investigate structural bottlenecks and substantial algorithmic improvements.

Continue profiling, forming hypotheses, implementing experiments, testing correctness, and benchmarking until:

* the correctness contract is fully satisfied;
* meaningful latency improvement is demonstrated;
* the major remaining bottlenecks are understood;
* further changes have no credible expected benefit or require an explicitly out-of-scope redesign.

## Canonical correctness contract

Implement and document one canonical brute-force oracle.

Unless the existing standalone API deliberately documents otherwise, use these semantics:

1. Coordinates are contiguous `float32` values.

2. Squared Euclidean distance is calculated across every input dimension:

```text
squared_distance(i, j) =
    sum_d (points1[i, d] - points2[j, d])^2
```

3. A candidate is included when:

```text
squared_distance <= radius * radius
```

The boundary is inclusive.

4. For every query point:

   * collect all candidates within the radius;
   * order them by `(squared_distance, original_point_index)`;
   * return the first K candidates.

The original point index is the deterministic tie-breaker.

5. Raw neighbor search may include the query itself when the query and reference sets are the same. Self-loop removal belongs to the edge-building post-processing step.

6. Missing neighbors use the public API's documented sentinel values. Preserve the established values unless changing them is absolutely necessary and explicitly approved by the tests and documentation.

7. Neighbor indices must exactly match the brute-force oracle.

8. Squared distances must:

   * be bitwise identical where the implementation performs equivalent arithmetic; or
   * match within a documented tight absolute/relative or ULP tolerance where compiler-generated floating-point behavior differs.

9. Edge output must exactly match the result obtained by applying the documented edge post-processing rules to the brute-force neighbor matrix.

10. No approximate search, probabilistic pruning, reduced-dimensional distance, FP16 distance, relaxed radius, or recall-based acceptance is allowed.

When duplicated coordinates or equal distances occur, deterministic ordering is mandatory.

## Phase 1: Establish an immutable baseline

Before optimizing:

1. Run the complete correctness suite.

2. Record:

   * baseline commit;
   * GPU model;
   * CUDA Toolkit version;
   * compiler version;
   * build type and flags;
   * CUDA architecture;
   * CPU model;
   * driver version;
   * benchmark dataset parameters.

3. Add a benchmark executable under a clear location such as:

```text
benchmarks/frnn_benchmark.cu
```

4. Measure at least:

   * host API end-to-end latency;
   * device API end-to-end latency;
   * steady-state latency with workspace reuse;
   * one-time initialization/allocation cost;
   * grid construction;
   * key or cell assignment;
   * prefix sum or sorting;
   * point reordering;
   * neighbor search;
   * edge post-processing;
   * device-to-host transfer for the host convenience API.

5. Use CUDA events for GPU timing and an appropriate host clock for full end-to-end timing.

6. Include warmup iterations and enough measured iterations to report:

   * minimum;
   * median or p50;
   * p95;
   * mean;
   * standard deviation when useful.

7. Do not time asynchronous launches without synchronizing the timing event correctly.

Commit or save baseline benchmark results before changing the implementation.

## Benchmark matrix

Use a matrix broad enough to prevent overfitting.

Cover available practical subsets of:

### Point counts

```text
128
512
1,000
4,096
10,000
50,000
100,000
500,000
```

Eventually, use the real life example in `data/embedding_data.csv`. Use the `data/edge_list.csv` file as the canonical reference for correctness.
A testing script is provided in `tests/compare_reference_edges.py` to compare the output of the optimized implementation against the reference edge list.

Use larger cases when hardware memory permits.

The following parameter matrix is for your reference. It should be designed to be close to the real-world use cases of the library. 

### Dimensions

Include the dimensions most relevant to the library's intended use and multiple supported values, for example:

```text
3
4
8
16
```

Do not silently optimize only 3D if the public API supports higher dimensions.

### K values

```text
1
4
8
16
32
64
```

Use only values supported by the public contract.

### Neighborhood density

Generate radii or datasets giving approximately:

```text
0–1
4
16
64
256+
```

valid neighbors per query.

Measure both sparse and dense neighborhoods.

### Point distributions

Include:

* uniform random;
* highly clustered;
* several separated clusters;
* points concentrated near cell boundaries;
* duplicated points;
* sorted or correlated coordinates;
* anisotropic distributions;
* degenerate distributions where one or more coordinates are constant;
* realistic sample input if one exists in the repository's intended application.

### Query modes

Benchmark:

* query and reference sets being identical;
* separate query and reference sets;
* cold workspace;
* reused workspace;
* default stream;
* non-default stream;
* one call;
* repeated calls.

Store machine-readable benchmark results, preferably CSV or JSON, so before/after results can be compared.

## Phase 2: Profile before redesigning

Use the best profiling tools available in the environment, such as:

* Nsight Systems;
* Nsight Compute;
* CUDA event instrumentation;
* compiler resource reports;
* SASS or PTX inspection when useful.

Identify and document:

* kernel launch overhead;
* host/device synchronization;
* device allocation and deallocation;
* host/device copies;
* prefix-sum cost;
* sorting or counting-sort cost;
* global-memory bandwidth;
* uncoalesced accesses;
* temporary-memory traffic;
* local-memory spills;
* register pressure from K-sized arrays;
* shared-memory utilization;
* occupancy;
* branch divergence;
* load imbalance from dense cells;
* redundant cell visits;
* duplicated distance computation;
* poor behavior for small point clouds;
* poor behavior for dense neighborhoods;
* fixed launch configurations that underutilize the GPU;
* costs introduced by edge post-processing.

Write the initial findings in:

```text
docs/optimization_log.md
```

## Phase 3: Perform hypothesis-driven experiments

For each substantial experiment:

1. State the bottleneck and hypothesis.
2. Record the baseline affected by the change.
3. Implement the smallest useful experiment.
4. Run the complete correctness suite.
5. Run the relevant benchmark subset.
6. Keep the change only if it is correct and beneficial.
7. Revert failed or regressive experiments.
8. Record the result, including negative results, in `docs/optimization_log.md`.

Do not accumulate unvalidated changes.

## Structural optimization areas to investigate

Investigate all relevant areas rather than assuming one preferred solution.

### A. Eliminate per-call allocation and synchronization

* Reuse all temporary device storage through the workspace.
* Avoid `cudaMalloc` and `cudaFree` in the hot path.
* Consider `cudaMallocAsync` only when it improves the design and compatibility is documented.
* Avoid host reads of device counters during the pipeline.
* Avoid implicit synchronization between stages.
* Allow the full device operation to remain asynchronous.
* Consider CUDA Graph capture for stable repeated workloads, but keep it optional and validate its benefit.

### B. Replace or redesign the grid-construction pipeline

Evaluate alternatives such as:

* direct cell-key generation;
* CUB device radix sort by cell key;
* run-length encoding of sorted cell keys;
* CUB scans;
* fused cell assignment and key generation;
* avoiding a separate counting-sort pass;
* retaining original indices without redundant full-coordinate copies;
* building compact occupied-cell metadata rather than arrays for every empty cell;
* using a dense grid only when occupancy justifies it;
* using hash or sorted-key lookup when dense grid metadata is too large.

Use CUDA Toolkit primitives when beneficial. Do not introduce PyTorch.

### C. Optimize neighbor-cell traversal

* Precompute or specialize the stencil of neighboring cells.
* Avoid repeatedly calculating identical grid bounds.
* Clamp grid coordinates efficiently.
* Reduce integer divisions and expensive floor operations.
* Eliminate visits to cells that cannot intersect the radius.
* Improve handling of empty cells.
* Investigate compact occupied-cell ranges.
* Load grid parameters efficiently.
* Keep correctness for points exactly on cell and radius boundaries.

### D. Optimize distance calculation

* Ensure coalesced coordinate loads.
* Evaluate array-of-structures versus structure-of-arrays or tiled layouts.
* Use vectorized loads where alignment and dimension permit.
* Specialize common dimensions.
* Keep a correct generic path for other supported dimensions.
* Reuse query coordinates from registers or shared memory.
* Avoid rereading candidate coordinates unnecessarily.
* Investigate warp-cooperative processing for dense cells.
* Do not reduce the number of dimensions used in the exact distance.

### E. Redesign top-K selection

The current family of approaches may place K-element arrays in thread-local storage, causing register pressure or spills.

Investigate:

* specialized register selection for very small K;
* warp-level top-K;
* block-cooperative selection for larger K;
* bounded heaps;
* bitonic or merge-based methods;
* staged selection;
* different algorithms for sparse and dense candidate sets;
* direct unsorted collection followed by an efficient selection when appropriate.

Preserve deterministic ordering by `(distance, original index)`.

Measure register count, occupancy, spills, and latency for each K regime.

### F. Fuse kernels where beneficial

Evaluate fusion of stages such as:

* grid assignment and key generation;
* neighbor filtering and edge generation;
* self-loop removal and edge compaction;
* output counting and writing.

Do not fuse kernels merely to reduce launch count when it increases memory traffic, register pressure, or complexity without measured benefit.

### G. Add an exact brute-force fallback

A grid is not necessarily fastest for small point clouds, very dense radii, or unfavorable dimensionality.

Implement and benchmark an exact CUDA brute-force path.

Develop a documented dispatch heuristic based on measurable inputs such as:

* number of query points;
* number of reference points;
* dimension;
* K;
* radius;
* expected or observed grid occupancy;
* expected neighborhood density;
* workspace state.

The fallback must use the same canonical semantics and output ordering.

Prefer a simple, robust heuristic supported by benchmark evidence over an opaque collection of magic thresholds.

### H. Exploit the identical-query/reference case carefully

When query and reference point clouds are identical, investigate:

* reuse of grid metadata;
* avoiding duplicate sorting;
* avoiding redundant copies;
* symmetric distance reuse where it genuinely reduces latency;
* specialized edge generation.

Do not assume the top-K relation is symmetric. Any symmetry optimization must still match the brute-force result exactly.

### I. Tune launch configuration based on workload

Replace unconditional fixed launch choices where they are suboptimal.

Consider:

* occupancy-informed block sizes;
* separate configurations by dimension and K;
* different kernels for small and large clouds;
* grid-stride versus one-query-per-thread mappings;
* persistent kernels only if profiling justifies them.

Treat launch tuning as one component, not the entire optimization.

## Correctness test requirements

Expand the brute-force differential suite to include:

* thousands of randomized small cases;
* randomized supported dimensions;
* randomized K;
* randomized radii;
* empty neighborhoods;
* all points within the radius;
* duplicate points;
* equal-distance ties;
* NaN and infinity rejection or documented handling;
* coordinates near floating-point limits relevant to the API;
* points exactly at the radius;
* `nextafter` values just inside and outside the radius;
* points on grid-cell boundaries;
* very small and very large cells;
* highly occupied single cells;
* empty cells between occupied cells;
* repeated invocation with reused workspace;
* multiple streams;
* alternating input sizes;
* separate and identical query/reference sets;
* all runtime dispatch paths;
* brute-force fallback thresholds.

Run the full differential suite after every retained structural change.

Use sanitizers or CUDA correctness tools when available.

## Performance acceptance rules

Correctness has absolute priority.

A change must not be accepted because it is faster on one hand-selected input.

At minimum:

1. All exactness tests pass.

2. The benchmark report separates:

   * cold latency;
   * warm latency with workspace reuse;
   * device-only latency;
   * host convenience API latency.

3. The final implementation demonstrates a meaningful reduction in p50 steady-state latency on representative workloads.

4. Aim for at least a 2x speedup on one or more important medium/large sparse workloads, but do not sacrifice correctness to reach that number.

5. Small point-cloud latency should improve through an exact fallback or lower-overhead path when practical.

6. No primary benchmark regime should regress by more than 10% without a documented and compelling overall tradeoff.

7. Large regressions in edge cases must be documented even if the geometric mean improves.

8. Report speedup both per benchmark and as an appropriately summarized aggregate. Do not report only the best case.

9. Include comparison against the exact CUDA brute-force implementation so users can see the crossover point.

10. The no-PyTorch dependency requirement remains enforced.

## API and maintainability constraints

* Preserve the public API whenever practical.
* Preserve explicit CUDA-stream support.
* Preserve asynchronous behavior in the device API.
* Preserve workspace reuse.
* Keep a generic correct implementation in addition to specialized fast paths.
* Document dispatch decisions.
* Do not expose internal tuning constants as unexplained public API.
* Avoid architecture-specific assumptions unless guarded and documented.
* Keep compilation time and binary-size growth visible, especially when templating across many D and K combinations.
* Preserve license notices and upstream attribution.
* Do not add unconditional logging or profiling overhead to release builds.

## Required deliverables

Produce:

1. An optimized standalone implementation.
2. An independent brute-force CPU or CUDA oracle.
3. An exact CUDA brute-force fallback suitable for production dispatch.
4. A comprehensive differential test suite.
5. A reproducible benchmark executable.
6. Machine-readable baseline and final benchmark results.
7. `docs/optimization_log.md` containing:

   * profiling observations;
   * hypotheses;
   * experiments;
   * retained changes;
   * reverted changes;
   * unresolved bottlenecks.
8. Updated README documentation covering:

   * algorithm overview;
   * exactness guarantee;
   * supported dimensions and K;
   * device and host APIs;
   * workspace reuse;
   * stream semantics;
   * benchmark reproduction;
   * dispatch behavior;
   * known limitations.
9. A final performance report with hardware and software metadata.

## Required final validation

Run and report:

```bash
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DFRNN_BUILD_TESTS=ON \
  -DFRNN_BUILD_BENCHMARKS=ON

cmake --build build --parallel

ctest --test-dir build --output-on-failure
```

Run the complete benchmark matrix or the largest subset supported by the available GPU.

Also verify:

* no production Torch, ATen, c10, Python, or TBB dependency;
* no unexpected synchronization in the device API;
* no per-call allocation in the warm workspace path;
* no CUDA errors after any kernel stage;
* deterministic output over repeated runs;
* exact neighbor-index agreement with brute force;
* documented distance tolerance;
* successful static or shared-library consumer build.

## Git discipline

Work on a dedicated branch such as:

```text
goal/optimize-exact-frnn
```

Use logically separated commits, for example:

1. benchmark and profiling baseline;
2. brute-force oracle and expanded exactness suite;
3. workspace and allocation improvements;
4. grid pipeline redesign;
5. neighbor-search kernel changes;
6. top-K specialization;
7. brute-force fallback and dispatch;
8. edge post-processing improvements;
9. final benchmarks and documentation.

Do not leave abandoned kernels, dead experimental code, unexplained constants, or commented-out implementations in production paths.

## Final report

At completion, report:

* the baseline commit and final commit;
* hardware and CUDA environment;
* the final algorithm and dispatch architecture;
* exactness-test coverage and results;
* before/after benchmark table;
* p50 and p95 latency;
* speedups by workload;
* brute-force crossover points;
* memory usage;
* compilation and binary-size impact;
* retained and rejected experiments;
* remaining bottlenecks;
* any workloads that regressed;
* exact commands for reproducing tests and benchmarks;
* suggested pull-request title and description.

Do not declare success based only on compilation or a single faster kernel. Success requires exact brute-force equivalence, reproducible measurements, and demonstrated end-to-end latency improvement.
