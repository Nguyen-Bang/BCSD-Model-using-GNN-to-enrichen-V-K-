"""Convert tokenized functions to the compact on-disk dataset schema."""

import math

import torch

MINHASH_DIM = 128
FUNCTION_SCALAR_FEATURES = (
    "node_count",
    "edge_count",
    "function_size",
    "function_cyclomatic_complexity",
)
NODE_KEEP_FIELDS = (
    "id",
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
    "call_targets",
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
)


def build_function_features(function: dict) -> list[float]:
    """Build the four scalar and 128 MinHash function features."""
    scalars = [
        math.log1p(max(0.0, float(function.get(key, 0)))) for key in FUNCTION_SCALAR_FEATURES
    ]
    minhash = function.get("opcode_minhash", [])
    if minhash and len(minhash) >= MINHASH_DIM:
        minhash_features = [float(value) / (2**32) for value in minhash[:MINHASH_DIM]]
    else:
        minhash_features = [0.0] * MINHASH_DIM
    return scalars + minhash_features


def reduce_node(node: dict) -> dict:
    """Keep only node fields consumed by the dataset loader."""
    return {key: node[key] for key in NODE_KEEP_FIELDS if key in node}


def reduce_function(function: dict) -> dict | None:
    """Convert one tokenized function to the compact dataset schema."""
    token_ids = function.get("token_ids")
    if not token_ids:
        return None

    attention_mask = function.get("attention_mask", [1] * len(token_ids))
    token_type_ids = function.get("token_type_ids", [0] * len(token_ids))

    return {
        "function_name": function.get("function_name", "unknown"),
        "token_ids": torch.tensor(token_ids, dtype=torch.int32),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.int8),
        "token_type_ids": torch.tensor(token_type_ids, dtype=torch.int32),
        "function_features": torch.tensor(build_function_features(function), dtype=torch.float32),
        "node_count": function.get("node_count", 0),
        "edge_count": function.get("edge_count", 0),
        "function_size": function.get("function_size", 0),
        "function_cyclomatic_complexity": function.get("function_cyclomatic_complexity", 0),
        "nodes": [reduce_node(node) for node in function.get("nodes", [])],
        "edges": function.get("edges", []),
    }
