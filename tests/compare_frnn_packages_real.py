#!/usr/bin/env python3
"""Benchmark both FRNN packages on the supplied real-world data."""

from pathlib import Path
from statistics import median
from time import perf_counter

import frnn
import frnn_cuda
import numpy as np
import torch
from compare_reference_edges import encode_edges, load_embedding, load_reference

RADIUS = 0.12
MAX_NEIGHBORS = 1000
ITERATIONS = 5


def original(points):
    # Original FRNN includes the query point itself, while libFRNN excludes it.
    return frnn.frnn_grid_points(
        points, points, K=MAX_NEIGHBORS + 1, r=RADIUS
    )[1]


def original_host(points):
    device_points = torch.from_numpy(points).cuda().unsqueeze(0)
    return original(device_points).cpu()


def milliseconds(call):
    call()
    torch.cuda.synchronize()
    samples = []
    for _ in range(ITERATIONS):
        start = perf_counter()
        call()
        torch.cuda.synchronize()
        samples.append((perf_counter() - start) * 1_000)
    return median(samples)


def original_edge_codes(indices, num_nodes):
    neighbors = indices[0]
    sources = torch.arange(num_nodes, device=neighbors.device).unsqueeze(1)
    valid = (neighbors >= 0) & (neighbors != sources)
    codes = sources.expand_as(neighbors)[valid] * num_nodes + neighbors[valid]
    return np.unique(codes.cpu().numpy().astype(np.uint64, copy=False))


def require_exact(label, actual, reference):
    if np.array_equal(actual, reference):
        return
    common = np.intersect1d(actual, reference, assume_unique=True).size
    raise AssertionError(
        f"{label} edge mismatch: {actual.size - common:,} extra, "
        f"{reference.size - common:,} missing"
    )


def main():
    root = Path(__file__).resolve().parents[1]
    points = load_embedding(root / "data" / "embedding_data.csv")
    reference, reference_duplicates = load_reference(
        root / "data" / "edge_list.csv", points.shape[0]
    )
    if reference_duplicates:
        raise AssertionError(
            f"reference contains {reference_duplicates:,} duplicate edges"
        )

    device_points = torch.from_numpy(points).cuda().unsqueeze(0)
    cuda_edges = frnn_cuda.build_edges(
        points,
        radius=RADIUS,
        max_neighbors=MAX_NEIGHBORS,
        directed=True,
    )
    cuda_codes = np.unique(encode_edges(cuda_edges, points.shape[0], "libFRNN"))
    torch_indices = original(device_points)
    torch_codes = original_edge_codes(torch_indices, points.shape[0])
    require_exact("libFRNN", cuda_codes, reference)
    require_exact("original FRNN", torch_codes, reference)

    cuda_ms = milliseconds(
        lambda: frnn_cuda.build_edges(
            points,
            radius=RADIUS,
            max_neighbors=MAX_NEIGHBORS,
            directed=True,
        )
    )
    torch_ms = milliseconds(lambda: original(device_points))
    torch_host_ms = milliseconds(lambda: original_host(points))

    print()
    print(
        "points  dimensions  directed_edges  agreement  frnn_cuda_host_ms  "
        "frnn_torch_gpu_ms  frnn_torch_host_ms"
    )
    print(
        f"{points.shape[0]:>6}  {points.shape[1]:>10}  {reference.size:>14}  "
        f"{'100%':>9}  {cuda_ms:>17.3f}  {torch_ms:>17.3f}  "
        f"{torch_host_ms:>18.3f}"
    )


if __name__ == "__main__":
    main()
