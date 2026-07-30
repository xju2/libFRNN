# libFRNN

libFRNN is a standalone C++17/CUDA fixed-radius nearest-neighbor library. Its
core has no dependency on Python, PyTorch, ATen, c10, TBB, NumPy, or a Python
binding framework. The optional `frnn` Python package is a thin NumPy binding
over the same installed C++ API.

The implementation supports dimensions 1 through 32. It automatically
dispatches small workloads to exact CUDA brute force and larger workloads to
a dense uniform grid. The grid indexes up to three coordinates for dimensions
1–4 and four coordinates for dimensions 5–32; candidate distances always use
every input dimension.

## Algorithm and exactness

Both production search paths implement the same canonical result:

1. accumulate squared Euclidean distance in `float32` dimension order with
   one fused multiply-add per coordinate;
2. include candidates whose squared distance is less than or equal to
   `radius * radius`;
3. order each query row by `(squared_distance, original_database_index)`;
4. retain the first `max_neighbors` entries;
5. apply self-loop and directed/undirected edge filtering.

Neighbor indices and final edges are exact, not recall-based. Duplicate
coordinates retain distinct original indices, radius boundaries are
inclusive, and equal-distance ties prefer the lower original index. Internal
squared distances are not part of the public output. The independent CPU
oracle uses `std::fma` to make the same accumulation contract explicit and
includes exact-boundary, adjacent `nextafter`, and rounding-sensitive
high-dimensional cases.

The grid path builds bounds, assigns database points to dense cells, performs
a prefix sum and counting sort, and searches only cells intersecting each
query radius. Sorted coordinates use AoS through dimension 4 and SoA above
dimension 4; separate query/reference calls read the original AoS database.
Identical query/reference calls process queries in the existing spatial
ordering and write results back by original index. Four-dimensional grids
reject corner cells whose bounding boxes cannot intersect the radius. Query
cell ranges, cell boxes, and rejection thresholds are conservatively expanded
for float32 roundoff.
Specialized distance kernels cover dimensions 1, 2, 3, 4, 8, 12, and 16,
with a generic exact path through dimension 32.

High-dimensional kernels first use grouped SoA loads as a conservative
reordered screen. Candidates close enough to matter are recomputed in
canonical dimension order, so the screen cannot change membership or
ordering. Low-dimensional search uses 256-thread blocks; high-dimensional
search uses 128 threads.

For large K, selection begins as a sorted insertion list and converts to a
deterministic max-heap only after a query accepts 24 candidates. This avoids
heap overhead for sparse queries and avoids quadratic insertion shifting for
dense queries. An internal `-1` index marks the end of a short neighbor row;
sentinels are not exposed in edge output.

Automatic brute-force dispatch uses this overflow-safe work rule:

```text
query_count * database_count * dimension * ceil(max_neighbors / 8)
    <= 2,000,000
```

Callers can force either exact path for measurement or diagnosis:

```cpp
frnn::BuildOptions options;
options.algorithm = frnn::SearchAlgorithm::grid;
// or frnn::SearchAlgorithm::brute_force
```

The forced choices change only the algorithm, not result semantics.

## C++ installation and use

Required tools are CMake 3.20 or newer, a C++17 compiler, and a compatible CUDA
Toolkit. CUDA architectures are configured through CMake's standard
`CMAKE_CUDA_ARCHITECTURES` variable.

```bash
cmake -S . -B build-cpp \
  -DCMAKE_BUILD_TYPE=Release \
  -DFRNN_BUILD_TESTS=ON \
  -DFRNN_BUILD_PYTHON=OFF
cmake --build build-cpp --parallel
ctest --test-dir build-cpp --output-on-failure
cmake --install build-cpp --prefix "$PWD/install"
```

`BUILD_SHARED_LIBS=ON` is the default. Set it to `OFF` for a static core
library. A downstream CMake project consumes either form with:

```cmake
find_package(frnn CONFIG REQUIRED)
target_link_libraries(my_target PRIVATE frnn::frnn)
```

The synchronous host interface accepts row-major coordinates:

```cpp
#include <frnn/frnn.hpp>

std::vector<float> points = /* [num_points, dimension] */;
auto edges = frnn::buildEdges(
    frnn::PointView{points.data(), num_points, dimension},
    0.5F,
    32);
```

The returned `std::vector<frnn::Edge>` owns row-major `[num_edges, 2]` signed
64-bit pairs. The one-input convenience overload removes self-loops and emits
an undirected pair once, with `source > target`. Duplicate coordinates remain
distinct points; duplicate edge pairs are not emitted. Radius comparison is
inclusive. Candidates are ordered by squared distance, then by target index,
before `max_neighbors` truncation.

The historical vector interface remains available:

```cpp
frnn::buildEdges(query, database, edge_list, num_points, dimension,
                 radius, max_neighbors);
```

Its output is flattened `[2, num_edges]`: all sources followed by all targets.
It preserves the historical same-set, no-self-loop, one-undirected-pair
behavior.

### Asynchronous device API

`frnn::buildEdgesAsync` accepts `DevicePointView` values whose pointers already
reside on the active CUDA device, a caller-owned `DeviceEdgeBuffer`, a reusable
`Workspace`, and a caller-provided `cudaStream_t`. The output edge storage is
row-major `[capacity, 2]`; `edge_count` is a device-resident signed 64-bit
scalar. Use `requiredEdgeCapacity(query_count, max_neighbors)` to size it.
Passing `edges=nullptr` requests count-only execution; in that mode `capacity`
is ignored and the exact device-resident count is still populated.

Every memory operation, prefix sum, and kernel is enqueued on the supplied
stream. The device API does not copy input coordinates to the host. Call
`Workspace::reserve` before an asynchronous region to avoid allocation-related
synchronization when a workspace grows. A reserved workspace has no per-call
allocation on the warm path. Use a separate workspace for concurrently
executing calls.

The device API does not synchronize the supplied stream. It reports launch and
CUDA API errors immediately; execution errors are observed when the caller
synchronizes. Device callers must provide finite coordinates because checking
device values on the host would break asynchronous execution. The synchronous
host API validates all coordinates, copies inputs to the device, runs
count-first, allocates exact edge storage, and copies the output to the host.

## Python installation and use

The optional package requires Python 3.9+, NumPy, pybind11, and
scikit-build-core at build time. It does not require or import PyTorch.

The build machine must provide an NVIDIA CUDA Toolkit with `nvcc` and CUDA
headers/libraries compatible with its compiler and installed NVIDIA driver.
Failure to locate CUDA is reported by CMake's required CUDA language/toolkit
checks.

Build and install with a standard command:

```bash
python -m pip install .
```

or build a wheel:

```bash
python -m pip install build
python -m build
python -m pip install dist/frnn-*.whl
```

The Python API is:

```python
frnn.build_edges(
    query,
    database=None,
    *,
    radius: float,
    max_neighbors: int,
    exclude_self: bool = True,
    directed: bool | None = None,
) -> numpy.ndarray
```

For example:

```python
import numpy as np
import frnn

points = np.asarray(points, dtype=np.float32, order="C")
edges = frnn.build_edges(points, radius=0.5, max_neighbors=32)
```

Inputs must be C-contiguous NumPy `float32` arrays shaped
`[num_points, dimension]`. The result is a NumPy `int64` array shaped
`[2, num_edges]`. Non-array inputs, non-`float32` dtypes, non-contiguous or
non-2D arrays, dimensions outside `[1, 32]`, non-finite coordinates, invalid
radii, and negative neighbor limits are rejected.

When `database` is omitted, the input is treated as one point set:
`directed=None` produces unique undirected pairs, and `exclude_self=True`
removes self-loops. Setting `directed=True` emits both directions where each is
selected. When `database` is supplied, directed query-to-database output is
required and self-loop removal has no effect because the arrays represent
different sets.

The radius boundary is inclusive. `max_neighbors` is applied per query after
self-loop exclusion. Equal-distance candidates are resolved by the lower
target index, so tie handling is deterministic. Duplicate coordinates are
allowed and retain their separate indices.

The Python call is synchronous. It copies both NumPy inputs from host memory to
CUDA memory, calls the public standalone C++ API, synchronizes, and copies
edges back to NumPy-owned host memory. Python users never manage internal grid
or workspace buffers. Device-resident Python array interoperability is not yet
provided.

To test an installed package rather than the source tree:

```bash
python tests/test_python.py
```

## Implementation layers

- `frnn` is the independently buildable/installable C++/CUDA core. It owns
  grid construction, prefix sums, counting sort, nearest-neighbor selection,
  workspace allocation, edge filtering, and CUDA error reporting.
- `<frnn/frnn.hpp>` provides host and device C++ interfaces over that core.
- `_frnn` is built only with `FRNN_BUILD_PYTHON=ON`; it validates NumPy arrays
  and calls the public host API without reimplementing the algorithm.

The original kernel sources were derived from
<https://github.com/murnanedaniel/FRNN> and
<https://github.com/lxxue/prefix_sum>.

## Benchmarking

Build and run the reproducible benchmark with:

```bash
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DFRNN_BUILD_TESTS=ON \
  -DFRNN_BUILD_BENCHMARKS=ON
cmake --build build --parallel
./build/frnn_benchmark \
  --full \
  --warmup 10 \
  --iterations 50 \
  --output benchmarks/results/local.csv
```

The CSV contains build/compiler/GPU metadata and separate rows for workspace
allocation, cold device latency, warm CUDA-event latency, synchronized device
host latency, device-to-host output transfer, and host API end-to-end latency.
It reports minimum, p50, p95, mean, and standard deviation.

The supplied real embedding can be measured without including CSV parsing in
the timed region:

```bash
./build/frnn_benchmark \
  --embedding data/embedding_data.csv \
  --warmup 10 \
  --iterations 30 \
  --output benchmarks/results/local-real.csv

python tests/compare_reference_edges.py --minimum-agreement 1.0
```

Use `--algorithm grid` or `--algorithm brute_force` to measure crossover
points. The immutable baseline, retained experiments, and final machine-
readable data live in `benchmarks/results/`. Profiling observations, rejected
experiments, and remaining bottlenecks are in
`docs/optimization_log.md`; the final comparison is in
`docs/performance_report.md`.

## Known limitations

- Dimensions above four use the first four coordinates for grid indexing,
  although distance and acceptance use every coordinate.
- Dense or highly correlated high-dimensional clouds can still generate many
  candidates and dominate latency.
- The host convenience API is synchronous and creates a fresh device workspace
  per call; repeated low-latency applications should use the device API and a
  reserved workspace.
- Device output capacity remains caller-managed. Count-only mode is available
  when allocating the conservative capacity is undesirable.
- The current library supports any non-negative K that fits device memory; it
  does not expose neighbor distances as a public output.
