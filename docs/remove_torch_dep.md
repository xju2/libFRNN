> This document records the standalone-core migration requirements. The core
> remains PyTorch-free; a later, separately built `FRNN_BUILD_TORCH` adapter is
> documented in the README and does not change those core guarantees.

Convert libFRNN into a standalone C++/CUDA library that can be integrated into either:

1. a native C++ project through CMake; or
2. a Python project through an optional Python extension package.

The core implementation must have no dependency on PyTorch, ATen, c10, Python, or TBB.

Python support must be implemented as a thin optional binding over the standalone C++/CUDA API. Building or using the C++ library must not require Python or the Python binding dependencies.

At the end of this goal:

* a C++ user must be able to install libFRNN and consume it with `find_package`;
* a Python user must be able to build or install a Python package and call FRNN from Python without installing PyTorch;
* both interfaces must use the same core implementation and produce the same results.

The required core dependencies are:

* a compatible C++ compiler;
* CMake;
* the CUDA Toolkit;
* the C++ standard library.

Optional Python bindings may additionally depend on:

* Python;
* NumPy;
* pybind11 or nanobind;
* a Python build backend such as scikit-build-core.

Do not introduce PyTorch as an optional or testing dependency.

## Integration architecture

Organize the repository into three conceptual layers.

### Layer 1: Core C++/CUDA library

The core library must contain:

* all CUDA kernels;
* workspace and memory management;
* grid construction;
* prefix sum or sorting operations;
* nearest-neighbor search;
* edge construction and filtering;
* error handling;
* public C++ APIs.

This layer must not include or link against:

* Python headers;
* pybind11 or nanobind;
* NumPy;
* PyTorch;
* ATen;
* c10;
* TBB.

The core library must be independently buildable, installable, and testable.

### Layer 2: C++ interface

Provide a documented C++ interface supporting:

* host-resident input through ordinary C++ containers or pointer-and-size views;
* device-resident input through CUDA pointers or explicit device views;
* caller-provided `cudaStream_t`;
* reusable workspace allocation;
* synchronous host convenience calls;
* asynchronous device calls;
* explicit output ownership and memory layout.

Where practical, preserve the existing interface:

```cpp
frnn::buildEdges(
    std::vector<float>& query,
    std::vector<float>& database,
    std::vector<int64_t>& edge_list,
    int64_t num_spacepoints,
    int embedding_dim,
    float r_max,
    int k_max);
```

A cleaner modern overload may be added, but the compatibility interface should remain available unless there is a documented technical reason to remove it.

### Layer 3: Optional Python interface

Add a Python extension that calls the public C++ interface rather than duplicating the algorithm.

The Python extension must be optional and controlled by a CMake option such as:

```text
FRNN_BUILD_PYTHON
```

The default C++-only build must not search for Python or binding libraries unless this option is enabled.

Provide a Python API with a simple interface similar to:

```python
import frnn_cuda

edges = frnn_cuda.build_edges(
    query,
    database,
    r_max: float,
    k_max: int
)
```

The exact API may be refined, but it must clearly document:

* accepted input types;
* expected shape;
* supported data type;
* output shape and data type;
* radius semantics;
* maximum-neighbor behavior;
* self-loop behavior;
* duplicate-edge behavior;
* whether the operation is synchronous;
* whether a host/device copy occurs.

At minimum, support contiguous NumPy `float32` arrays with shape:

```text
[num_points, dimension]
```

Return edges in a clearly documented representation, preferably either:

```text
[2, num_edges]
```

or

```text
[num_edges, 2]
```

using signed 64-bit integer indices.

Reject unsupported dtypes, dimensions, non-finite configuration values, and malformed arrays with useful Python exceptions.

The first implementation may copy NumPy input from host to device and copy results back to the host. This behavior must be documented.

Optionally support device-resident Python arrays through one of these interoperability standards:

* DLPack;
* `__cuda_array_interface__`.

Such support must not require PyTorch. CuPy-specific integration may be tested when CuPy is available, but CuPy must not become a mandatory core dependency.

Do not expose raw internal grid buffers or require Python users to manage internal CUDA allocations.

## Python packaging

Provide a Python package that can be built with a standard command such as:

```bash
python -m pip install .
```

or:

```bash
python -m build
python -m pip install dist/*.whl
```

Prefer a modern CMake-based Python packaging arrangement, such as:

* `pyproject.toml`;
* scikit-build-core;
* pybind11 or nanobind.

Python packaging requirements:

* the package imports as `frnn_cuda` to avoid conflicting with the original PyTorch package;
* importing the package must not import PyTorch;
* the built extension must link to the standalone FRNN core;
* the algorithm must not be separately reimplemented in the binding;
* Python package metadata must state the required CUDA compatibility assumptions;
* editable installation should work when practical;
* wheel creation should work on the supported build environment;
* the source distribution should contain all files required to build the extension;
* failure to locate CUDA must produce a useful diagnostic.

Keep Python packaging isolated enough that downstream C++ users do not need Python packaging tools.

## CMake and packaging requirements

Modernize the build to support both C++ and optional Python integration.

Requirements:

* declare the required C++ and CUDA project languages;
* use `find_package(CUDAToolkit REQUIRED)` or equivalent modern CUDA CMake support;
* remove `find_package(Torch)`;
* remove unconditional Python discovery;
* remove TBB discovery;
* remove `TORCH_CXX_FLAGS`;
* link the core library only to required CUDA Toolkit targets and standard system libraries;
* define an explicit core target;
* export the core target using a namespace such as `frnn::frnn`;
* keep Python binding targets separate from the core target;
* search for Python and pybind11 or nanobind only when `FRNN_BUILD_PYTHON=ON`;
* avoid propagating Python include directories or libraries through the C++ target;
* provide independent options such as:

```cmake
option(FRNN_BUILD_TESTS "Build FRNN tests" ON)
option(FRNN_BUILD_BENCHMARKS "Build FRNN benchmarks" OFF)
option(FRNN_BUILD_PYTHON "Build Python bindings" OFF)
```

* install the C++ library, public headers, exported targets, package configuration, and version file;
* make both shared and static core-library builds possible when practical;
* make CUDA architectures configurable;
* ensure that the Python extension can either link to the installed core library or build it as part of the Python package;
* avoid embedding absolute build paths in installed files or wheels.

A C++ consumer must be able to use:

```cmake
find_package(frnn CONFIG REQUIRED)
target_link_libraries(my_target PRIVATE frnn::frnn)
```

A Python consumer must be able to use:

```python
import numpy as np
import frnn_cuda

points = np.asarray(points, dtype=np.float32)
edges = frnn_cuda.build_edges(points, radius=0.5, max_neighbors=32)
```

## test requirements

Maintain a CTest-based core test suite that requires neither Python nor PyTorch.

Also add Python binding tests when `FRNN_BUILD_PYTHON=ON`. Use the exsting test if possible.

Python tests must cover:

1. successful package import without PyTorch installed;
2. NumPy input validation;
3. output dtype and shape;
4. ordinary random point clouds;
5. empty input;
6. a single point;
7. duplicated points;
8. points exactly on the radius boundary;
9. K truncation;
10. deterministic tie handling;
11. self-loop and duplicate-edge rules;
12. comparison with the independent brute-force reference;
13. repeated Python calls;
14. useful exceptions for invalid input;
15. agreement between Python and C++ results for identical test data.

Do not use PyTorch to construct expected results.

The Python tests may use NumPy and a straightforward NumPy or pure-Python brute-force reference for small inputs.

## required validation

Before declaring completion, validate the C++-only build:

```bash
cmake -S . -B build-cpp \
  -DCMAKE_BUILD_TYPE=Release \
  -DFRNN_BUILD_TESTS=ON \
  -DFRNN_BUILD_PYTHON=OFF

cmake --build build-cpp --parallel

ctest --test-dir build-cpp --output-on-failure

cmake --install build-cpp --prefix "$PWD/install"
```

Create and compile a small external C++ consumer using the installed package.

Then validate the Python build in a clean virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install .

python -c "import frnn_cuda; print(frnn_cuda)"
```

Run the Python test suite using the package as installed, not by relying only on imports from the source directory.

Where supported, also validate wheel creation:

```bash
python -m pip install build
python -m build
```

Inspect the resulting C++ library and Python extension with tools such as `ldd` or `readelf`.

Confirm that neither links to:

* Torch;
* ATen;
* c10;
* TBB;
* Python libraries in the core C++ target.

The Python extension may naturally link to Python runtime components, but the standalone core library must not.

## acceptance criteria

This goal is complete only when all of the following are true:

1. A clean C++ build does not require PyTorch, Python, or TBB.
2. The core production code contains no Torch, ATen, or c10 APIs.
3. The core CMake configuration does not search for Torch, Python, or TBB.
4. Python discovery occurs only when Python bindings are explicitly enabled.
5. The standalone core target does not link to Python or Python-binding libraries.
6. A C++ project can consume the installed library through `find_package(frnn CONFIG REQUIRED)`.
7. A Python project can install and import the package without PyTorch.
8. The Python binding calls the same C++/CUDA implementation used by C++ consumers.
9. NumPy inputs and outputs are supported and documented.
10. C++ and Python interfaces produce identical results for identical inputs.
11. Correctness tests pass against an independent brute-force reference.
12. CUDA kernels consistently use the caller-provided stream in the device API.
13. The device C++ API does not perform hidden whole-input host transfers.
14. The Python API documents any host/device copies and synchronization.
15. The host C++ and Python convenience APIs work without exposing internal grid buffers.
16. Installation, package export, and external C++ consumption work.
17. Python package installation and wheel creation work on the supported environment.
18. README documentation contains separate C++ and Python usage sections.
19. There are no unconditional debug prints during normal library or Python-module use.
20. No major FRNN algorithm optimization has been mixed into the dependency-removal work without a documented correctness reason.

## final report

At completion, provide:

* a concise architecture summary covering the core, C++ API, and Python binding;
* a list of public C++ API changes;
* the Python API signature and examples;
* a list of removed dependencies;
* the optional Python build dependencies;
* C++ test and installation results;
* Python installation and test results;
* wheel-build results where available;
* dynamic-link dependency inspection results;
* any host/device copies made by each interface;
* any known behavioral differences;
* any unresolved correctness or packaging issues;
* a suggested commit or pull-request title;
* a clear starting point for the subsequent optimization goal.
