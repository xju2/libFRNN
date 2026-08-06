"""Compare this package with the original PyTorch FRNN package."""

from statistics import median
from time import perf_counter

import numpy as np
import torch

import frnn
import frnn_cuda


MAX_NEIGHBORS = 32
TARGET_NEIGHBORS = 8


def radius_for(point_count):
    return (TARGET_NEIGHBORS / (point_count * 4.0 / 3.0 * np.pi)) ** (1.0 / 3.0)


def original(points, radius):
    return frnn.frnn_grid_points(
        points, points, K=MAX_NEIGHBORS + 1, r=radius
    )[1]


def original_host(points, radius):
    device_points = torch.from_numpy(points).cuda().unsqueeze(0)
    return original(device_points, radius).cpu()


def milliseconds(call, iterations=5):
    call()
    torch.cuda.synchronize()
    samples = []
    for _ in range(iterations):
        start = perf_counter()
        call()
        torch.cuda.synchronize()
        samples.append((perf_counter() - start) * 1_000)
    return median(samples)


def edge_set(indices):
    return {
        (source, int(target))
        for source, row in enumerate(indices[0].cpu().numpy())
        for target in row
        if target >= 0 and target != source
    }


def main():
    print(
        "points  edges  agreement  frnn_cuda_host_ms  "
        "frnn_torch_gpu_ms  frnn_torch_host_ms"
    )
    for point_count in (10_000, 100_000):
        points = np.random.default_rng(0).random(
            (point_count, 3), dtype=np.float32
        )
        device_points = torch.from_numpy(points).cuda().unsqueeze(0)
        radius = radius_for(point_count)
        cuda_edges = frnn_cuda.build_edges(
            points,
            radius=radius,
            max_neighbors=MAX_NEIGHBORS,
            directed=True,
        )
        torch_indices = original(device_points, radius)
        if point_count == 10_000:
            expected = edge_set(torch_indices)
            actual = set(map(tuple, cuda_edges.T.tolist()))
            assert actual == expected, (
                f"edge mismatch: {len(actual - expected)} extra, "
                f"{len(expected - actual)} missing"
            )
            agreement = "100%"
        else:
            agreement = "checked@10k"
        cuda_ms = milliseconds(
            lambda: frnn_cuda.build_edges(
                points,
                radius=radius,
                max_neighbors=MAX_NEIGHBORS,
                directed=True,
            )
        )
        torch_ms = milliseconds(lambda: original(device_points, radius))
        torch_host_ms = milliseconds(lambda: original_host(points, radius))
        print(
            f"{point_count:>6}  {cuda_edges.shape[1]:>6}  {agreement:>10}  "
            f"{cuda_ms:>17.3f}  {torch_ms:>17.3f}  {torch_host_ms:>18.3f}"
        )


if __name__ == "__main__":
    main()
