import torch

from dataset.collate import collate_functions


def _function() -> dict:
    return {
        "token_ids": torch.tensor([3, 4]),
        "attention_mask": torch.tensor([1, 1]),
        "token_type_ids": torch.tensor([0, 0]),
        "function_features": torch.zeros(132),
        "node_features": torch.zeros(2, 20),
        "edges": [[0, 1]],
        "edge_types": [2],
    }


def test_single_function_collator_omits_pair_and_unused_feature_tensors():
    batch = collate_functions(
        [_function()],
        include_function_features=False,
        include_graph=False,
    )

    assert set(batch) == {"input_ids", "attention_mask", "token_type_ids"}
    assert all(not key.endswith(("_a", "_b")) for key in batch)


def test_single_function_collator_builds_typed_graph_once():
    batch = collate_functions(
        [_function()],
        include_graph=True,
        use_edge_features=True,
        use_reverse_edges=True,
    )

    assert batch["node_features"].shape == (2, 20)
    assert batch["gnn_edge_index"].shape == (2, 4)
    assert batch["gnn_edge_types"].tolist() == [2, 7, 10, 10]
    assert all(not key.endswith(("_a", "_b")) for key in batch)
