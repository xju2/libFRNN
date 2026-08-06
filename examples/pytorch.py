import torch
from frnn_cuda.torch import build_edges

points = torch.rand((100_000, 3), dtype=torch.float32, device="cuda")
edges = build_edges(points, radius=0.03, max_neighbors=32)

print(edges.shape, edges.dtype, edges.device)
