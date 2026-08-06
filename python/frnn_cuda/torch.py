"""Optional PyTorch CUDA interface to libFRNN."""

import math
from importlib import import_module

import torch

try:
    _frnn_torch = import_module(f"{__package__}._frnn_torch")
except ModuleNotFoundError as error:
    if error.name != f"{__package__}._frnn_torch":
        raise
    raise ImportError(
        "frnn_cuda was built without PyTorch support; reinstall with "
        "FRNN_BUILD_TORCH=ON"
    ) from error


def _validate_tensor(value, name):
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if not value.is_cuda:
        raise ValueError(f"{name} must be a CUDA tensor")
    if value.dtype is not torch.float32:
        raise TypeError(f"{name} must have dtype torch.float32")
    if value.ndim != 2:
        raise ValueError(f"{name} must have shape [num_points, dimension]")
    if not value.is_contiguous():
        raise ValueError(f"{name} must be contiguous")
    if not 1 <= value.shape[1] <= 32:
        raise ValueError(f"{name} dimension must be in the range [1, 32]")


def build_edges(
    query,
    database=None,
    *,
    radius,
    max_neighbors,
    exclude_self=True,
    directed=None,
):
    """Build fixed-radius edges directly from CUDA PyTorch tensors.

    Inputs are contiguous ``torch.float32`` tensors shaped ``[N, D]`` on the
    same CUDA device. The returned ``torch.int64`` tensor has shape ``[2, E]``
    and remains on that device. The call uses and synchronizes PyTorch's current
    stream to obtain the data-dependent edge count; point and edge data are
    never copied to the host.
    """
    _validate_tensor(query, "query")
    if database is not None:
        _validate_tensor(database, "database")
        if query.device != database.device:
            raise ValueError("query and database must be on the same CUDA device")
        if query.shape[1] != database.shape[1]:
            raise ValueError("query and database must have the same dimension")
        if directed is False:
            raise ValueError(
                "directed=False is only supported when database is omitted"
            )
    if not math.isfinite(radius) or radius <= 0:
        raise ValueError("radius must be finite and greater than zero")
    if not isinstance(max_neighbors, int) or isinstance(max_neighbors, bool):
        raise TypeError("max_neighbors must be an integer")
    if max_neighbors < 0:
        raise ValueError("max_neighbors must be non-negative")
    if directed is not None and not isinstance(directed, bool):
        raise TypeError("directed must be bool or None")
    if not isinstance(exclude_self, bool):
        raise TypeError("exclude_self must be bool")

    return torch.ops.frnn_cuda.build_edges(
        query,
        database,
        float(radius),
        max_neighbors,
        exclude_self,
        directed,
    )


__all__ = ["build_edges"]
