"""Helpers for reading structural edges from paired model batches."""

from collections.abc import Mapping
from typing import Any


def structural_edges(batch: Mapping[str, Any], side: str) -> tuple[Any, Any]:
    """Select augmented typed edges when present, otherwise base CFG edges."""
    if side not in {"a", "b"}:
        raise ValueError(f"side must be 'a' or 'b', got {side!r}")
    edge_index = batch.get(f"gnn_edge_index_{side}")
    if edge_index is None:
        edge_index = batch.get(f"edge_index_{side}")
    return edge_index, batch.get(f"gnn_edge_types_{side}")
