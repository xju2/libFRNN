"""NumPy interface to the standalone libFRNN C++/CUDA implementation."""

from ._frnn import __version__, build_edges

__all__ = ["__version__", "build_edges"]
