#!/usr/bin/env python3
"""Compare libFRNN output against the provided reference edge list.

The comparison treats edges as a set of directed (source, destination) pairs.
Edge ordering therefore does not affect the result, but reversed edges do.
"""

from __future__ import annotations

import argparse
import importlib
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

EXPECTED_EMBEDDING_SHAPE = (271_663, 12)
EXPECTED_REFERENCE_SHAPE = (2, 9_279_672)


def parse_args() -> argparse.Namespace:
    repository_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Run libFRNN and compare its edges with data/edge_list.csv."
    )
    parser.add_argument(
        "--embedding",
        type=Path,
        default=repository_root / "data" / "embedding_data.csv",
        help="Input embedding CSV (default: data/embedding_data.csv).",
    )
    parser.add_argument(
        "--reference",
        type=Path,
        default=repository_root / "data" / "edge_list.csv",
        help="Reference edge-list CSV (default: data/edge_list.csv).",
    )
    parser.add_argument(
        "--module",
        default="frnn",
        help="Python module containing build_edges (default: frnn).",
    )
    parser.add_argument(
        "--function",
        default="build_edges",
        help="Edge-building callable in --module (default: build_edges).",
    )
    parser.add_argument("--r-max", type=float, default=0.12)
    parser.add_argument("--k-max", type=int, default=1000)
    parser.add_argument(
        "--minimum-agreement",
        type=float,
        default=0.99,
        help=(
            "Minimum required precision and recall, in [0, 1] "
            "(default: 0.99). Use 1.0 to require identical edge sets."
        ),
    )
    parser.add_argument(
        "--examples",
        type=int,
        default=10,
        help="Maximum missing and extra edges to print (default: 10).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optionally write the libFRNN edge list to this CSV path.",
    )
    args = parser.parse_args()

    if not 0.0 <= args.minimum_agreement <= 1.0:
        parser.error("--minimum-agreement must be between 0 and 1")
    if args.r_max <= 0.0:
        parser.error("--r-max must be positive")
    if args.k_max <= 0:
        parser.error("--k-max must be positive")
    if args.examples < 0:
        parser.error("--examples cannot be negative")

    return args


def load_embedding(path: Path) -> np.ndarray:
    print(f"Loading embedding from {path} ...", flush=True)
    embedding = np.loadtxt(path, delimiter=",", dtype=np.float32)
    if embedding.shape != EXPECTED_EMBEDDING_SHAPE:
        raise ValueError(
            f"embedding shape is {embedding.shape}; expected {EXPECTED_EMBEDDING_SHAPE}"
        )
    if not np.isfinite(embedding).all():
        raise ValueError("embedding contains NaN or infinity")
    return np.ascontiguousarray(embedding)


def load_reference(path: Path, num_nodes: int) -> tuple[np.ndarray, int]:
    print(f"Loading reference edges from {path} ...", flush=True)

    # The CSV stores integer node IDs with a decimal suffix. float32 represents
    # every ID in this data set exactly and halves the initial load footprint.
    raw_edges = np.loadtxt(path, delimiter=",", dtype=np.float32)
    if raw_edges.shape != EXPECTED_REFERENCE_SHAPE:
        raise ValueError(
            f"reference edge shape is {raw_edges.shape}; "
            f"expected {EXPECTED_REFERENCE_SHAPE}"
        )

    codes = encode_edges(raw_edges, num_nodes, label="reference")
    unique_codes = np.unique(codes)
    return unique_codes, codes.size - unique_codes.size


def load_callable(module_name: str, function_name: str) -> Any:
    try:
        module = importlib.import_module(module_name)
    except ImportError as error:
        raise RuntimeError(
            f"could not import {module_name!r}; install/build the libFRNN "
            "Python extension or select it with --module"
        ) from error

    try:
        function = getattr(module, function_name)
    except AttributeError as error:
        raise RuntimeError(
            f"module {module_name!r} has no callable {function_name!r}"
        ) from error
    if not callable(function):
        raise TypeError(f"{module_name}.{function_name} is not callable")
    return function


def to_numpy_edges(value: Any) -> np.ndarray:
    edges = np.asarray(value)
    if edges.ndim != 2:
        raise ValueError(f"build_edges returned shape {edges.shape}; expected 2-D")
    if edges.shape[0] == 2:
        pass
    elif edges.shape[1] == 2:
        edges = edges.T
    else:
        raise ValueError(
            f"build_edges returned shape {edges.shape}; "
            "expected [2, num_edges] or [num_edges, 2]"
        )
    return edges


def encode_edges(edges: np.ndarray, num_nodes: int, label: str) -> np.ndarray:
    if edges.size == 0:
        return np.empty(0, dtype=np.uint64)
    if not np.isfinite(edges).all():
        raise ValueError(f"{label} edges contain NaN or infinity")
    if np.any(edges != np.floor(edges)):
        raise ValueError(f"{label} edges contain non-integer node IDs")

    minimum = float(edges.min())
    maximum = float(edges.max())
    if minimum < 0 or maximum >= num_nodes:
        raise ValueError(
            f"{label} node IDs span [{minimum:g}, {maximum:g}], "
            f"but valid IDs span [0, {num_nodes - 1}]"
        )

    sources = edges[0].astype(np.uint64, copy=True)
    sources *= np.uint64(num_nodes)
    sources += edges[1].astype(np.uint64, copy=False)
    return sources


def count_common(sorted_left: np.ndarray, sorted_right: np.ndarray) -> int:
    if sorted_left.size > sorted_right.size:
        sorted_left, sorted_right = sorted_right, sorted_left
    positions = np.searchsorted(sorted_right, sorted_left)
    in_bounds = positions < sorted_right.size
    return int(
        np.count_nonzero(
            in_bounds
            & (
                sorted_right[np.minimum(positions, max(sorted_right.size - 1, 0))]
                == sorted_left
            )
        )
    )


def difference_examples(
    sorted_left: np.ndarray, sorted_right: np.ndarray, limit: int
) -> np.ndarray:
    """Return up to limit values in left which are absent from right."""
    if limit == 0 or sorted_left.size == 0:
        return np.empty(0, dtype=np.uint64)
    if sorted_right.size == 0:
        return sorted_left[:limit]

    examples: list[np.ndarray] = []
    remaining = limit
    chunk_size = 1_000_000
    for start in range(0, sorted_left.size, chunk_size):
        chunk = sorted_left[start : start + chunk_size]
        positions = np.searchsorted(sorted_right, chunk)
        present = positions < sorted_right.size
        present[present] = sorted_right[positions[present]] == chunk[present]
        missing = chunk[~present]
        if missing.size:
            examples.append(missing[:remaining])
            remaining -= min(remaining, missing.size)
            if remaining == 0:
                break
    return np.concatenate(examples) if examples else np.empty(0, dtype=np.uint64)


def format_codes(codes: np.ndarray, num_nodes: int) -> str:
    return ", ".join(
        f"({int(code // num_nodes)}, {int(code % num_nodes)})" for code in codes
    )


def main() -> int:
    args = parse_args()
    build_edges = load_callable(args.module, args.function)

    embedding_array = load_embedding(args.embedding)
    num_nodes = embedding_array.shape[0]
    reference, reference_duplicates = load_reference(args.reference, num_nodes)

    print(
        f"Calling {args.module}.{args.function}"
        f"(embedding, radius={args.r_max}, max_neighbors={args.k_max}) ...",
        flush=True,
    )
    start = time.perf_counter()
    actual_value = build_edges(
        embedding_array, radius=args.r_max, max_neighbors=args.k_max
    )

    actual_edges = to_numpy_edges(actual_value)
    elapsed = time.perf_counter() - start
    actual_codes = encode_edges(actual_edges, num_nodes, label="libFRNN")

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        np.savetxt(
            args.output,
            actual_edges.astype(np.int64, copy=False),
            delimiter=",",
            fmt="%d",
        )
        print(f"Wrote libFRNN edges to {args.output}", flush=True)

    actual = np.unique(actual_codes)
    actual_duplicates = actual_codes.size - actual.size

    common = count_common(actual, reference)
    missing = reference.size - common
    extra = actual.size - common
    precision = common / actual.size if actual.size else float(reference.size == 0)
    recall = common / reference.size if reference.size else float(actual.size == 0)
    union = actual.size + reference.size - common
    jaccard = common / union if union else 1.0

    print()
    print(f"build_edges time:       {elapsed:.3f} s")
    print(f"reference edges:        {reference.size:,}")
    print(f"libFRNN edges:          {actual.size:,}")
    print(f"common edges:           {common:,}")
    print(f"missing edges:          {missing:,}")
    print(f"extra edges:            {extra:,}")
    print(f"reference duplicates:  {reference_duplicates:,}")
    print(f"libFRNN duplicates:     {actual_duplicates:,}")
    print(f"precision:              {precision:.8f}")
    print(f"recall:                 {recall:.8f}")
    print(f"Jaccard similarity:     {jaccard:.8f}")

    if missing and args.examples:
        examples = difference_examples(reference, actual, args.examples)
        print(f"missing examples:       {format_codes(examples, num_nodes)}")
    if extra and args.examples:
        examples = difference_examples(actual, reference, args.examples)
        print(f"extra examples:         {format_codes(examples, num_nodes)}")

    passed = precision >= args.minimum_agreement and recall >= args.minimum_agreement
    threshold = args.minimum_agreement
    if passed:
        print(f"\nPASS: precision and recall are both at least {threshold:.2%}.")
        return 0

    print(
        f"\nFAIL: precision and recall must both be at least {threshold:.2%}.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(2) from error
