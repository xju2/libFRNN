#!/bin/bash
# Install the CUDA compiler, runtime headers and CCCL (CUB) into a manylinux_2_28
# (AlmaLinux 8) container for cibuildwheel. CUDA 12.8 is the first release that
# supports the image's GCC 14.
set -euo pipefail
CUDA_VERSION="${CUDA_VERSION:-12-8}"
dnf install -y dnf-plugins-core
dnf config-manager --add-repo \
  https://developer.download.nvidia.com/compute/cuda/repos/rhel8/x86_64/cuda-rhel8.repo
dnf install -y "cuda-nvcc-${CUDA_VERSION}" "cuda-cudart-devel-${CUDA_VERSION}" \
  "cuda-cccl-${CUDA_VERSION}"
dnf clean all
ln -sfn "/usr/local/cuda-${CUDA_VERSION/-/.}" /usr/local/cuda
/usr/local/cuda/bin/nvcc --version
