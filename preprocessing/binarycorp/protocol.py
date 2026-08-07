"""Model-independent BinaryCorp-3M indexing and retrieval protocol."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

OPTIMIZATION_PAIRS = (
    ("O0", "O3"),
    ("O1", "O3"),
    ("O2", "O3"),
    ("O0", "Os"),
    ("O1", "Os"),
    ("O2", "Os"),
)
OPTIMIZATION_LEVELS = frozenset(level for pair in OPTIMIZATION_PAIRS for level in pair)
DEFAULT_POOL_SIZE = 10_000


@dataclass(frozen=True)
class IndexStats:
    files: int
    functions: int
    groups: int


@dataclass(frozen=True)
class SplitArtifact:
    """One reduced artifact loaded from a BinaryCorp split."""

    project: str
    optimization: str
    functions: list[dict[str, Any]]


@dataclass(frozen=True)
class EmbeddingTable:
    """Contiguous embeddings and their deterministic dataset positions."""

    keys: list[tuple[str, str]]
    ordinals: np.ndarray
    vectors: np.ndarray

    def __post_init__(self) -> None:
        if len(self.keys) != len(self.ordinals) or len(self.keys) != len(self.vectors):
            raise ValueError("Embedding keys, ordinals, and vectors must have equal lengths")
        if self.ordinals.ndim != 1 or self.vectors.ndim != 2:
            raise ValueError("Embedding ordinals and vectors must be one- and two-dimensional")


def iter_split_artifacts(
    data_dir: str | Path,
    split: str = "test",
) -> Iterator[SplitArtifact]:
    """Load one artifact at a time instead of materializing a whole split."""
    root = Path(data_dir) / split
    if not root.is_dir():
        raise FileNotFoundError(f"BinaryCorp split directory not found: {root}")

    for project_dir in sorted(root.iterdir()):
        if not project_dir.is_dir():
            continue
        for artifact in sorted(project_dir.glob("*.pt")):
            payload = torch.load(artifact, map_location="cpu", weights_only=True)
            if not isinstance(payload, dict):
                raise ValueError(f"Expected a mapping in {artifact}")
            project = payload.get("project")
            if project != project_dir.name:
                raise ValueError(
                    f"Project in {artifact} must match its directory: "
                    f"expected {project_dir.name!r}, got {project!r}"
                )
            optimization = payload.get("opt")
            if optimization not in OPTIMIZATION_LEVELS:
                raise ValueError(
                    f"Invalid or missing optimization level in {artifact}: {optimization!r}"
                )
            records = payload.get("functions")
            if not isinstance(records, list):
                raise ValueError(f"Expected a functions list in {artifact}")
            for index, function in enumerate(records):
                if not isinstance(function, dict):
                    raise ValueError(f"Expected a function mapping in {artifact} at index {index}")
                function_name = function.get("function_name")
                if not isinstance(function_name, str) or not function_name:
                    raise ValueError(f"Missing function_name in {artifact} at index {index}")
            yield SplitArtifact(project_dir.name, optimization, records)


def matchable_counts(
    data_dir: str | Path,
    split: str = "test",
) -> tuple[dict[str, int], IndexStats]:
    """Count official pairs with a compact bit mask per function key."""
    optimization_bits = {
        optimization: 1 << index for index, optimization in enumerate(sorted(OPTIMIZATION_LEVELS))
    }
    masks: dict[tuple[str, str], int] = {}
    files = functions = 0
    for artifact in iter_split_artifacts(data_dir, split):
        files += 1
        bit = optimization_bits[artifact.optimization]
        for function in artifact.functions:
            key = (artifact.project, function["function_name"])
            mask = masks.get(key, 0)
            if mask & bit:
                raise ValueError(
                    "Duplicate BinaryCorp identity "
                    f"({key[0]!r}, {key[1]!r}, {artifact.optimization!r})"
                )
            masks[key] = mask | bit
            functions += 1

    counts = {
        f"{source}-{target}": sum(
            bool(mask & optimization_bits[source]) and bool(mask & optimization_bits[target])
            for mask in masks.values()
        )
        for source, target in OPTIMIZATION_PAIRS
    }
    return counts, IndexStats(files, functions, len(masks))


def _unit_rows(matrix: np.ndarray, name: str) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[0] == 0:
        raise ValueError(f"{name} must be a non-empty two-dimensional array")
    if not np.isfinite(matrix).all():
        raise ValueError(f"{name} contains non-finite values")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError(f"{name} contains a zero embedding")
    return matrix / norms


def ranks_for_pool(
    query: np.ndarray,
    candidates: np.ndarray,
    *,
    query_block_size: int = 512,
) -> np.ndarray:
    """Rank aligned positives with the protocol's pessimistic tie policy."""
    if query.shape != candidates.shape:
        raise ValueError("Aligned query and candidate arrays must have the same shape")
    if query_block_size < 1:
        raise ValueError("query_block_size must be positive")
    query = _unit_rows(query, "query")
    candidates = _unit_rows(candidates, "candidates")
    ranks = np.empty(query.shape[0], dtype=np.int32)
    for start in range(0, query.shape[0], query_block_size):
        stop = min(start + query_block_size, query.shape[0])
        similarities = query[start:stop] @ candidates.T
        positives = similarities[
            np.arange(stop - start),
            np.arange(start, stop),
        ]
        ranks[start:stop] = (similarities >= positives[:, None]).sum(axis=1)
    return ranks


def _indexed_pool_retrieval(
    query: np.ndarray,
    candidates: np.ndarray,
    query_indices: np.ndarray,
    candidate_indices: np.ndarray,
    pool_size: int,
    rng: np.random.Generator,
    *,
    query_block_size: int,
) -> dict[str, Any]:
    """Evaluate aligned row indices without copying an entire optimization pair."""
    if pool_size < 1:
        raise ValueError("pool_size must be positive")
    if query.ndim != 2 or candidates.ndim != 2 or query.shape[1] != candidates.shape[1]:
        raise ValueError("Query and candidate embedding widths must match")
    if query_indices.shape != candidate_indices.shape or query_indices.ndim != 1:
        raise ValueError("Query and candidate indices must be aligned one-dimensional arrays")
    pair_count = len(query_indices)
    pool_count = pair_count // pool_size
    permutation = rng.permutation(pair_count)
    pool_metrics: list[dict[str, float]] = []
    for pool_index in range(pool_count):
        indices = permutation[pool_index * pool_size : (pool_index + 1) * pool_size]
        ranks = ranks_for_pool(
            query[query_indices[indices]],
            candidates[candidate_indices[indices]],
            query_block_size=query_block_size,
        )
        pool_metrics.append(
            {
                "mrr": float((1.0 / ranks).mean()),
                "recall_at_1": float((ranks == 1).mean()),
            }
        )
    return {
        "n_pairs": pair_count,
        "n_pools": pool_count,
        "n_dropped": pair_count - pool_count * pool_size,
        "mrr": (float(np.mean([pool["mrr"] for pool in pool_metrics])) if pool_metrics else None),
        "recall_at_1": (
            float(np.mean([pool["recall_at_1"] for pool in pool_metrics])) if pool_metrics else None
        ),
        "pools": pool_metrics,
    }


def evaluate_optimization_pairs(
    embeddings: dict[str, EmbeddingTable],
    *,
    pool_size: int = DEFAULT_POOL_SIZE,
    seed: int = 0,
    query_block_size: int = 512,
    require_all_pairs: bool = True,
) -> dict[str, Any]:
    """Evaluate and equally average the six official optimization pairs."""
    rng = np.random.default_rng(seed)
    per_pair: dict[str, dict[str, Any]] = {}
    for source, target in OPTIMIZATION_PAIRS:
        name = f"{source}-{target}"
        source_table = embeddings.get(source)
        target_table = embeddings.get(target)
        if source_table is None or target_table is None:
            source_indices = np.empty(0, dtype=np.int64)
            target_indices = np.empty(0, dtype=np.int64)
        else:
            target_by_key = {key: index for index, key in enumerate(target_table.keys)}
            source_rows: list[int] = []
            target_rows: list[int] = []
            for source_index, key in enumerate(source_table.keys):
                target_index = target_by_key.get(key)
                if target_index is not None:
                    source_rows.append(source_index)
                    target_rows.append(target_index)
            source_indices = np.asarray(source_rows, dtype=np.int64)
            target_indices = np.asarray(target_rows, dtype=np.int64)
            if source_indices.size:
                order = np.argsort(source_table.ordinals[source_indices], kind="stable")
                source_indices = source_indices[order]
                target_indices = target_indices[order]

        if source_indices.size == 0:
            result = {
                "n_pairs": 0,
                "n_pools": 0,
                "n_dropped": 0,
                "mrr": None,
                "recall_at_1": None,
                "pools": [],
            }
        else:
            result = _indexed_pool_retrieval(
                source_table.vectors,
                target_table.vectors,
                source_indices,
                target_indices,
                pool_size,
                rng,
                query_block_size=query_block_size,
            )
        per_pair[name] = result

    missing = [name for name, result in per_pair.items() if result["n_pools"] == 0]
    if missing and require_all_pairs:
        raise ValueError(
            f"No complete pool of size {pool_size} for optimization pairs: {', '.join(missing)}"
        )
    complete = [result for result in per_pair.values() if result["n_pools"] > 0]
    return {
        "protocol": {
            "optimization_pairs": [f"{source}-{target}" for source, target in OPTIMIZATION_PAIRS],
            "pool_size": pool_size,
            "seed": seed,
            "pooling": "seeded shuffle, disjoint full pools, remainder dropped",
            "tie_policy": "pessimistic",
            "pair_aggregation": "equal mean over optimization pairs",
        },
        "aggregate": {
            "mrr": float(np.mean([result["mrr"] for result in complete])) if complete else None,
            "recall_at_1": (
                float(np.mean([result["recall_at_1"] for result in complete])) if complete else None
            ),
            "pairs_with_pool": len(complete),
        },
        "per_pair": per_pair,
    }
