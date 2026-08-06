import math
import unittest

import numpy as np

try:
    import torch
except ImportError as error:
    raise unittest.SkipTest(f"PyTorch is unavailable: {error}")

try:
    from frnn_cuda.torch import build_edges
except ImportError as error:
    cause = error.__cause__
    if isinstance(cause, ModuleNotFoundError) and cause.name.endswith(
        "._frnn_torch"
    ):
        raise unittest.SkipTest(f"optional binding is unavailable: {error}")
    raise


def reference_edges(
    query,
    database=None,
    *,
    radius,
    max_neighbors,
    exclude_self=True,
    directed=None,
):
    same_input = database is None
    database = query if same_input else database
    directed = (not same_input) if directed is None else directed
    edges = []
    for source, point in enumerate(query):
        candidates = []
        for target, other in enumerate(database):
            if same_input and exclude_self and source == target:
                continue
            distance = float(np.sum((point - other) ** 2))
            if distance <= radius * radius:
                candidates.append((distance, target))
        candidates.sort(key=lambda item: (item[0], item[1]))
        for _, target in candidates[:max_neighbors]:
            if directed or source > target:
                edges.append((source, target))
    if not edges:
        return np.empty((2, 0), dtype=np.int64)
    return np.asarray(edges, dtype=np.int64).T


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
class TorchBindingTests(unittest.TestCase):
    def assert_edges_equal(self, actual, expected):
        self.assertTrue(actual.is_cuda)
        self.assertEqual(actual.dtype, torch.int64)
        self.assertEqual(actual.shape[0], 2)
        np.testing.assert_array_equal(actual.cpu().numpy(), expected)

    def test_random_cloud_matches_reference(self):
        points = np.random.default_rng(7).random((32, 3), dtype=np.float32)
        actual = build_edges(
            torch.from_numpy(points).cuda(), radius=0.35, max_neighbors=7
        )
        self.assert_edges_equal(
            actual, reference_edges(points, radius=0.35, max_neighbors=7)
        )

    def test_directed_query_database(self):
        query = np.asarray([[0, 0], [1, 0]], dtype=np.float32)
        database = np.asarray([[0, 0], [0.5, 0]], dtype=np.float32)
        actual = build_edges(
            torch.from_numpy(query).cuda(),
            torch.from_numpy(database).cuda(),
            radius=0.5,
            max_neighbors=2,
        )
        self.assert_edges_equal(
            actual,
            reference_edges(query, database, radius=0.5, max_neighbors=2),
        )

    def test_empty_duplicates_boundary_and_ties(self):
        empty = torch.empty((0, 3), dtype=torch.float32, device="cuda")
        self.assert_edges_equal(
            build_edges(empty, radius=1.0, max_neighbors=4),
            np.empty((2, 0), dtype=np.int64),
        )

        points = np.asarray(
            [[0, 0], [0, 0], [1, 0], [-1, 0], [0, 1]], dtype=np.float32
        )
        actual = build_edges(
            torch.from_numpy(points).cuda(),
            radius=1.0,
            max_neighbors=2,
            directed=True,
        )
        expected = reference_edges(
            points, radius=1.0, max_neighbors=2, directed=True
        )
        self.assert_edges_equal(actual, expected)

    def test_uses_current_stream(self):
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            points = torch.empty((3, 2), dtype=torch.float32, device="cuda")
            points.copy_(
                torch.tensor([[0, 0], [0.5, 0], [2, 0]], dtype=torch.float32)
            )
            edges = build_edges(points, radius=0.5, max_neighbors=2)
        self.assertTrue(stream.query())
        self.assert_edges_equal(edges, np.asarray([[1], [0]], dtype=np.int64))

    def test_validation(self):
        valid = torch.zeros((2, 3), dtype=torch.float32, device="cuda")
        invalid = [
            (torch.zeros((2, 3), dtype=torch.float32), ValueError),
            (valid.double(), TypeError),
            (torch.zeros(6, dtype=torch.float32, device="cuda"), ValueError),
            (valid[:, ::2], ValueError),
        ]
        for value, error in invalid:
            with self.subTest(shape=value.shape, dtype=value.dtype):
                with self.assertRaises(error):
                    build_edges(value, radius=1.0, max_neighbors=2)
        for radius in (0.0, -1.0, math.nan, math.inf):
            with self.assertRaises(ValueError):
                build_edges(valid, radius=radius, max_neighbors=2)
        with self.assertRaises(ValueError):
            build_edges(valid, radius=1.0, max_neighbors=-1)


if __name__ == "__main__":
    unittest.main()
