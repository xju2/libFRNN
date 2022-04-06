# libFRNN

A C++ library for fixed-radius Nearest Neighbour algorithm running on CUDA.

These source files are copied from https://github.com/murnanedaniel/FRNN
and https://github.com/lxxue/prefix_sum


## Dependencies
It depends on `PyTorch` and `CUDA`, and `TBB` (needed for Pytorch).
## To build

```bash
cmake .. -DCMAKE_PREFIX_PATH=$(python -c 'import torch;print(torch.utils.cmake_prefix_path)')
```

