"""Regression tests for tensor-backed ReducedDataset preloading."""

import torch

from dataset.code_dataset import ReducedDataset
from dataset.collate import PAD_TOKEN_ID, _collate_binaries


def _legacy_collate(binary_list):
    sequences = [b["token_ids"] for b in binary_list]
    type_sequences = [b["token_type_ids"] for b in binary_list]
    mask_sequences = [b["attention_mask"] for b in binary_list]
    max_len = max(len(sequence) for sequence in sequences)

    padded_ids = []
    padded_types = []
    padded_masks = []
    for ids, types, mask in zip(sequences, type_sequences, mask_sequences, strict=False):
        pad_len = max_len - len(ids)
        padded_ids.append(ids + [PAD_TOKEN_ID] * pad_len)
        padded_types.append(types + [PAD_TOKEN_ID] * pad_len)
        padded_masks.append(mask + [0] * pad_len)

    return {
        "input_ids": torch.tensor(padded_ids, dtype=torch.long),
        "attention_mask": torch.tensor(padded_masks, dtype=torch.long),
        "token_type_ids": torch.tensor(padded_types, dtype=torch.long),
        "function_features": torch.tensor(
            [b["function_features"] for b in binary_list],
            dtype=torch.float32,
        ),
    }


def test_tensor_backed_preload_matches_legacy_collate():
    legacy = [
        {
            "token_ids": [7, 8, 9],
            "attention_mask": [1, 1, 1],
            "token_type_ids": [0, 0, 0],
            "function_features": [0.25] * 132,
        },
        {
            "token_ids": [10],
            "attention_mask": [1],
            "token_type_ids": [0],
            "function_features": [0.5] * 132,
        },
    ]
    compact = [{key: torch.as_tensor(value) for key, value in binary.items()} for binary in legacy]

    expected = _legacy_collate(legacy)
    actual = _collate_binaries(compact)
    for key in expected:
        assert torch.equal(actual[key], expected[key]), key


def test_to_plain_keeps_compact_tensor_dtypes():
    func = {
        "token_ids": torch.tensor([7, 8], dtype=torch.int32),
        "attention_mask": torch.tensor([1, 1], dtype=torch.int8),
        "token_type_ids": torch.tensor([0, 0], dtype=torch.int32),
        "function_features": torch.arange(132, dtype=torch.float32),
        "nodes": [
            {"size": 3.0, "instruction_count": 2.0},
            {"size": 1.0, "instruction_count": 1.0},
        ],
        "edges": [[0, 1, "fallthrough"]],
    }

    compact = ReducedDataset._to_plain(func)

    assert compact["token_ids"].dtype == torch.int32
    assert compact["attention_mask"].dtype == torch.int8
    assert compact["token_type_ids"].dtype == torch.int32
    assert compact["function_features"].dtype == torch.float32
    assert compact["node_features"].dtype == torch.float32
    assert compact["node_features"].shape == (2, 20)
