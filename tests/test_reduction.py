import math

import torch

from dataset.reduction import build_function_features, reduce_function, reduce_node


def test_build_function_features_scales_scalars_and_minhash():
    function = {
        "node_count": 3,
        "edge_count": -2,
        "function_size": 7,
        "function_cyclomatic_complexity": 1,
        "opcode_minhash": list(range(130)),
    }

    features = build_function_features(function)

    assert len(features) == 132
    assert features[:4] == [math.log1p(3), 0.0, math.log1p(7), math.log1p(1)]
    assert features[4] == 0.0
    assert features[-1] == 127 / (2**32)


def test_build_function_features_uses_zeroes_for_incomplete_minhash():
    features = build_function_features({"opcode_minhash": [1, 2]})

    assert features == [0.0] * 132


def test_reduce_node_keeps_only_dataset_fields():
    assert reduce_node({"id": 4, "size": 12, "instructions": ["mov"], "address": 8}) == {
        "id": 4,
        "size": 12,
    }


def test_reduce_function_builds_compact_tensor_schema():
    reduced = reduce_function(
        {
            "function_name": "entry",
            "token_ids": [3, 4],
            "node_count": 1,
            "nodes": [{"id": 0, "size": 2, "instructions": ["ret"]}],
            "edges": [[0, 0, "fallthrough"]],
        }
    )

    assert reduced is not None
    assert reduced["token_ids"].dtype == torch.int32
    assert torch.equal(reduced["attention_mask"], torch.tensor([1, 1], dtype=torch.int8))
    assert torch.equal(reduced["token_type_ids"], torch.tensor([0, 0], dtype=torch.int32))
    assert reduced["function_features"].shape == (132,)
    assert reduced["nodes"] == [{"id": 0, "size": 2}]
    assert reduced["edges"] == [[0, 0, "fallthrough"]]


def test_reduce_function_skips_missing_or_empty_tokens():
    assert reduce_function({}) is None
    assert reduce_function({"token_ids": []}) is None
