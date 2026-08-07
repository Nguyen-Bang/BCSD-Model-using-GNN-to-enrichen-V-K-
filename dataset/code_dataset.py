"""Tensor-backed Siamese dataset for reduced BCSD artifacts."""

import logging
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

logger = logging.getLogger("bcsd.dataset")

VALID_SPLITS = frozenset({"train", "validation", "test"})
MINHASH_DIM = 128
FUNC_FEATURE_DIM = 4 + MINHASH_DIM

NODE_NUMERIC_FEATURES = [
    "size",
    "instruction_count",
    "successor_count",
    "predecessor_count",
    "incoming_xref_count",
    "outgoing_xref_count",
    "string_ref_count",
    "call_count",
    "data_ref_count",
    "unique_register_count",
    "arithmetic_density",
    "multiply_divide_density",
    "logic_density",
    "comparison_density",
    "call_density",
    "mov_density",
    "lea_density",
    "stack_op_density",
    "memory_access_density",
    "constant_density",
]
NODE_FEATURE_DIM = len(NODE_NUMERIC_FEATURES)
PALMTREE_EMB_DIM = 128


def _torch_load(path: Path) -> dict[str, Any]:
    """Load the tensor-only reduced-data schema onto CPU."""
    data = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a mapping in {path}, got {type(data).__name__}")
    return data


def _token_tensor(payload: Any, context: str) -> torch.Tensor:
    try:
        tensor = torch.as_tensor(payload)
    except (TypeError, ValueError, RuntimeError) as error:
        raise ValueError(f"Invalid token sequence in {context}: {error}") from error
    if tensor.ndim != 1 or tensor.numel() == 0:
        raise ValueError(f"Expected a non-empty 1D token sequence in {context}")
    if tensor.dtype == torch.bool or tensor.is_floating_point() or tensor.is_complex():
        raise ValueError(f"Token IDs must be integers in {context}")
    if torch.any(tensor < 0):
        raise ValueError(f"Token IDs must be nonnegative in {context}")
    if torch.any(tensor > torch.iinfo(torch.int32).max):
        raise ValueError(f"Token IDs exceed int32 storage in {context}")
    return tensor.to(dtype=torch.int32, device="cpu")


def _extract_node_features(nodes: list[dict[str, Any]]) -> list[list[float]]:
    """Extract the 20 numeric features used by the 3L GNN."""
    if not nodes:
        return [[0.0] * NODE_FEATURE_DIM]
    return [[float(node.get(feature, 0.0)) for feature in NODE_NUMERIC_FEATURES] for node in nodes]


def _expected_node_count(func: dict[str, Any]) -> int:
    nodes = func.get("nodes", []) or []
    declared = int(func.get("node_count", 0) or 0)
    return max(declared, len(nodes), 1)


def _extract_palmtree_tensor(
    func: dict[str, Any],
    expected_rows: int,
    context: str,
) -> torch.Tensor:
    payload = func.get("node_features")
    if payload is None:
        payload = func.get("palmtree_node_embeddings")
    try:
        tensor = torch.as_tensor(payload)
    except (TypeError, ValueError, RuntimeError) as error:
        raise ValueError(f"Invalid inline PalmTree node features in {context}: {error}") from error

    if tensor.ndim != 2 or tensor.shape[1] != PALMTREE_EMB_DIM:
        shape = tuple(tensor.shape)
        raise ValueError(
            f"PalmTree feature shape mismatch in {context}: "
            f"expected [nodes, {PALMTREE_EMB_DIM}], got {shape}"
        )
    if tensor.shape[0] != expected_rows:
        raise ValueError(
            f"PalmTree node count mismatch in {context}: "
            f"expected {expected_rows}, got {tensor.shape[0]}"
        )
    return tensor.to(dtype=torch.float16, device="cpu")


def _functions(data: dict[str, Any], path: Path) -> list[dict[str, Any]]:
    functions = data.get("functions")
    if not isinstance(functions, list):
        raise ValueError(f"Expected a functions list in {path}")
    if not all(isinstance(func, dict) for func in functions):
        raise ValueError(f"Expected function mappings in {path}")
    return functions


class ReducedDataset(Dataset):
    """Preload reduced artifacts and sample same-function compilation pairs."""

    def __init__(
        self,
        data_dir: str = "reduced_data",
        split: str = "train",
        preload: bool = True,
        gnn_sidecar_dir: str | None = None,
    ):
        if split not in VALID_SPLITS:
            raise ValueError(f"Unknown split {split!r}; expected one of {sorted(VALID_SPLITS)}")
        if preload is not True:
            raise ValueError("ReducedDataset supports compact preloading only")
        if gnn_sidecar_dir is not None:
            raise ValueError("External GNN sidecars are unsupported; use inline node_features")

        self.data_dir = Path(data_dir)
        self.split = split
        self.preload = True

        split_dir = self.data_dir / split
        if not split_dir.is_dir():
            raise FileNotFoundError(f"Split directory not found: {split_dir}")

        self._index, self._groups = self._build_index(split_dir)
        self._group_keys = [key for key, entries in self._groups.items() if len(entries) >= 2]

        total_functions = sum(len(entries) for entries in self._groups.values())
        pairable_functions = sum(
            len(entries) for entries in self._groups.values() if len(entries) >= 2
        )
        logger.info(
            "ReducedDataset [%s]: %d functions, %d pairable groups "
            "(%d pairable functions), %d projects",
            split,
            total_functions,
            len(self._group_keys),
            pairable_functions,
            len({key[0] for key in self._groups}),
        )

    def _build_index(
        self, split_dir: Path
    ) -> tuple[list[dict[str, Any]], dict[tuple[str, str], list[int]]]:
        index: list[dict[str, Any]] = []
        groups: dict[tuple[str, str], list[int]] = defaultdict(list)

        for project_dir in sorted(split_dir.iterdir()):
            if not project_dir.is_dir():
                continue
            project = project_dir.name

            for pt_path in sorted(project_dir.glob("*.pt")):
                data = _torch_load(pt_path)
                functions = _functions(data, pt_path)
                binary_name = data.get("binary_name", pt_path.stem)

                for func_index, func in enumerate(functions):
                    function_name = func.get("function_name", "unknown")
                    context = f"{pt_path} function {func_index} ({function_name})"

                    entry_index = len(index)
                    index.append(
                        {
                            "project": project,
                            "binary_name": binary_name,
                            "function_name": function_name,
                            "data": self._to_plain(
                                func,
                                context=context,
                            ),
                        }
                    )
                    groups[(project, function_name)].append(entry_index)

        return index, dict(groups)

    @staticmethod
    def _to_plain(
        func: dict[str, Any],
        *,
        context: str = "function",
    ) -> dict[str, Any]:
        token_ids = _token_tensor(func.get("token_ids"), context)

        attention_mask = func.get("attention_mask")
        attention_mask = (
            torch.ones(token_ids.numel(), dtype=torch.int8)
            if attention_mask is None
            else torch.as_tensor(attention_mask, dtype=torch.int8)
        )
        token_type_ids = func.get("token_type_ids")
        token_type_ids = (
            torch.zeros(token_ids.numel(), dtype=torch.int32)
            if token_type_ids is None
            else torch.as_tensor(token_type_ids, dtype=torch.int32)
        )
        if attention_mask.shape != token_ids.shape or token_type_ids.shape != token_ids.shape:
            raise ValueError(
                f"Token, attention-mask, and token-type shapes must match in {context}"
            )

        function_features = func.get("function_features")
        function_features = (
            torch.zeros(FUNC_FEATURE_DIM, dtype=torch.float32)
            if function_features is None
            else torch.as_tensor(function_features, dtype=torch.float32)
        )
        if function_features.shape != (FUNC_FEATURE_DIM,):
            raise ValueError(
                f"Expected {FUNC_FEATURE_DIM} function features in {context}, "
                f"got {tuple(function_features.shape)}"
            )

        edge_type_map = {
            "conditional_jump": 0,
            "unconditional_jump": 1,
            "fallthrough": 2,
            "switch": 3,
            "indirect_jump": 4,
        }
        edge_pairs = []
        edge_types = []
        for edge in func.get("edges", []) or []:
            if isinstance(edge, (list, tuple)) and len(edge) >= 2:
                edge_pairs.append([edge[0], edge[1]])
                edge_types.append(
                    edge_type_map.get(edge[2] if len(edge) >= 3 else "fallthrough", 2)
                )
            elif isinstance(edge, dict):
                edge_pairs.append([edge.get("source", 0), edge.get("target", 0)])
                edge_types.append(edge_type_map.get(edge.get("type", "fallthrough"), 2))

        if "node_features" in func:
            if func["node_features"] is None:
                raise ValueError(f"Missing inline PalmTree node_features in {context}")
            node_features = _extract_palmtree_tensor(
                func,
                _expected_node_count(func),
                context,
            )
        elif "palmtree_node_embeddings" in func:
            if func["palmtree_node_embeddings"] is None:
                raise ValueError(f"Missing inline PalmTree node features in {context}")
            node_features = _extract_palmtree_tensor(
                func,
                _expected_node_count(func),
                context,
            )
        else:
            node_features = torch.as_tensor(
                _extract_node_features(func.get("nodes", []) or []),
                dtype=torch.float32,
            )

        return {
            "token_ids": token_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
            "function_features": function_features,
            "node_features": node_features,
            "edges": edge_pairs,
            "edge_types": edge_types,
            "node_count": func.get("node_count", 0),
            "edge_count": func.get("edge_count", 0),
        }

    def __len__(self) -> int:
        return len(self._group_keys)

    def __getitem__(self, index: int) -> dict[str, Any]:
        project, function_name = self._group_keys[index]
        left_index, right_index = random.sample(self._groups[(project, function_name)], 2)
        left = self._index[left_index]
        right = self._index[right_index]
        return {
            "binary_1": left["data"],
            "binary_2": right["data"],
            "label": 1,
            "metadata": {
                "project": project,
                "function_name": function_name,
                "binary_1_name": left["binary_name"],
                "binary_2_name": right["binary_name"],
            },
        }
