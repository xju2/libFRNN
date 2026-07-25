# libFRNN

A C++ library for fixed-radius Nearest Neighbour algorithm running on CUDA.

These source files are copied from https://github.com/murnanedaniel/FRNN
and https://github.com/lxxue/prefix_sum


## Dependencies

libFRNN requires PyTorch and a CUDA toolkit compatible with the CUDA version
used by PyTorch. If PyTorch was installed with `uv`, install the matching CUDA
compiler into the same virtual environment:

```bash
source .venv/bin/activate
uv pip install 'cuda-toolkit[nvcc]'
```

The build detects `nvcc` from the virtual environment automatically. An
explicit `CUDACXX` or `CUDA_TOOLKIT_ROOT_DIR` takes precedence.

## To build

```bash
source .venv/bin/activate
cmake -DCMAKE_PREFIX_PATH=$(python -c 'import torch;print(torch.utils.cmake_prefix_path)') -S . -B build
cmake --build build --config Release --parallel
```

## Reference edge-list test

After installing the Python module that exposes `frnn.build_edges`, compare its
output with the supplied reference data:

```bash
python tests/compare_reference_edges.py
```

The script calls:

```python
build_edges(embedding, embedding, r_max=0.12, k_max=1000)
```

It compares directed edges without depending on their order and requires at
least 99% precision and 99% recall by default. Use
`--minimum-agreement=1.0` for exact set equality. If the binding accepts Torch
tensors rather than NumPy arrays, run with `--input-backend=torch`. Pass
`--output=data/libfrnn_edge_list.csv` to save the produced edge list.
