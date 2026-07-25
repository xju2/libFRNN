# libFRNN

libFRNN is a standalone C++17/CUDA fixed-radius nearest-neighbor library. Its
core has no dependency on Python, PyTorch, ATen, c10, TBB, NumPy, or a Python
binding framework. The optional `frnn` Python package is a thin NumPy binding
over the same installed C++ API.

The current implementation supports dimensions 1 through 32 and uses a
three-dimensional uniform grid (or all available dimensions for 1D/2D) to
select candidates. Distances are evaluated over every input dimension.

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

Every memory operation, prefix sum, and kernel is enqueued on the supplied
stream. The device API does not copy input coordinates to the host. Call
`Workspace::reserve` before an asynchronous region to avoid allocation-related
synchronization when a workspace grows. The host convenience API performs
host-to-device input copies, synchronizes, and copies its output to the host.

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
