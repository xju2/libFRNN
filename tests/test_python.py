import importlib.util
import math
import subprocess
import sys
import unittest

import numpy as np

import frnn_cuda


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
    if directed is None:
        directed = not same_input
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


class PythonBindingTests(unittest.TestCase):
    def assert_edges_equal(self, actual, expected):
        self.assertEqual(actual.dtype, np.dtype(np.int64))
        self.assertEqual(actual.shape[0], 2)
        np.testing.assert_array_equal(actual, expected)

    def test_import_does_not_import_torch(self):
        completed = subprocess.run(
            [
                sys.executable,
                "-I",
                "-c",
                "import sys, frnn_cuda; assert 'torch' not in sys.modules",
            ],
            check=False,
            text=True,
            capture_output=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_random_cloud_matches_reference(self):
        points = np.random.default_rng(7).random((32, 3), dtype=np.float32)
        actual = frnn_cuda.build_edges(points, radius=0.35, max_neighbors=7)
        expected = reference_edges(
            points, radius=0.35, max_neighbors=7
        )
        self.assert_edges_equal(actual, expected)

    def test_directed_query_database(self):
        query = np.asarray([[0, 0], [1, 0]], dtype=np.float32)
        database = np.asarray([[0, 0], [0.5, 0]], dtype=np.float32)
        actual = frnn_cuda.build_edges(
            query, database, radius=0.5, max_neighbors=2
        )
        expected = reference_edges(
            query, database, radius=0.5, max_neighbors=2
        )
        self.assert_edges_equal(actual, expected)

    def test_empty_input(self):
        points = np.empty((0, 3), dtype=np.float32)
        self.assert_edges_equal(
            frnn_cuda.build_edges(points, radius=1.0, max_neighbors=4),
            np.empty((2, 0), dtype=np.int64),
        )

    def test_single_point(self):
        points = np.zeros((1, 3), dtype=np.float32)
        self.assertEqual(
            frnn_cuda.build_edges(points, radius=1.0, max_neighbors=4).shape,
            (2, 0),
        )

    def test_duplicated_points_and_self_loop_rule(self):
        points = np.zeros((3, 3), dtype=np.float32)
        actual = frnn_cuda.build_edges(points, radius=0.1, max_neighbors=3)
        expected = np.asarray([[1, 2, 2], [0, 0, 1]], dtype=np.int64)
        self.assert_edges_equal(actual, expected)

    def test_radius_boundary_is_inclusive(self):
        points = np.asarray([[0, 0, 0], [1, 0, 0]], dtype=np.float32)
        self.assert_edges_equal(
            frnn_cuda.build_edges(points, radius=1.0, max_neighbors=1),
            np.asarray([[1], [0]], dtype=np.int64),
        )

    def test_k_truncation_and_tie_order(self):
        points = np.asarray(
            [[0, 0], [1, 0], [-1, 0], [0, 1], [0, -1]],
            dtype=np.float32,
        )
        actual = frnn_cuda.build_edges(
            points, radius=1.0, max_neighbors=1, directed=True
        )
        expected = reference_edges(
            points,
            radius=1.0,
            max_neighbors=1,
            directed=True,
        )
        self.assert_edges_equal(actual, expected)
        for _ in range(3):
            np.testing.assert_array_equal(
                frnn_cuda.build_edges(
                    points, radius=1.0, max_neighbors=1, directed=True
                ),
                actual,
            )

    def test_no_duplicate_edges(self):
        points = np.asarray([[0, 0], [0.1, 0], [0.2, 0]], dtype=np.float32)
        edges = frnn_cuda.build_edges(points, radius=1.0, max_neighbors=3)
        pairs = list(map(tuple, edges.T.tolist()))
        self.assertEqual(len(pairs), len(set(pairs)))
        self.assertTrue(all(source > target for source, target in pairs))

    def test_repeated_calls(self):
        points = np.zeros((4, 4), dtype=np.float32)
        first = frnn_cuda.build_edges(points, radius=1.0, max_neighbors=4)
        for _ in range(10):
            np.testing.assert_array_equal(
                frnn_cuda.build_edges(points, radius=1.0, max_neighbors=4),
                first,
            )

    def test_validation(self):
        valid = np.zeros((2, 3), dtype=np.float32)
        invalid_cases = [
            (np.zeros((2, 3), dtype=np.float64), TypeError),
            (np.zeros((6,), dtype=np.float32), ValueError),
            (np.zeros((2, 0), dtype=np.float32), ValueError),
            (valid[:, ::2], ValueError),
            ([[0.0, 0.0]], TypeError),
        ]
        for value, error in invalid_cases:
            with self.subTest(value=type(value).__name__):
                with self.assertRaises(error):
                    frnn_cuda.build_edges(
                        value, radius=1.0, max_neighbors=2
                    )
        for radius in (0.0, -1.0, math.nan, math.inf):
            with self.assertRaises(ValueError):
                frnn_cuda.build_edges(
                    valid, radius=radius, max_neighbors=2
                )
        with self.assertRaises(ValueError):
            frnn_cuda.build_edges(valid, radius=1.0, max_neighbors=-1)
        nonfinite = valid.copy()
        nonfinite[0, 0] = np.nan
        with self.assertRaises(ValueError):
            frnn_cuda.build_edges(nonfinite, radius=1.0, max_neighbors=2)

    def test_cpp_python_shared_fixture(self):
        # This fixture is also asserted by tests/test_frnn.cpp.
        points = np.asarray(
            [
                [0.0, 0.0, 0.0],
                [0.2, 0.0, 0.0],
                [0.0, 0.3, 0.0],
                [0.0, 0.0, 0.4],
                [0.9, 0.9, 0.9],
                [0.2, 0.2, 0.0],
            ],
            dtype=np.float32,
        )
        self.assert_edges_equal(
            frnn_cuda.build_edges(points, radius=0.42, max_neighbors=4),
            reference_edges(points, radius=0.42, max_neighbors=4),
        )


if __name__ == "__main__":
    unittest.main()
