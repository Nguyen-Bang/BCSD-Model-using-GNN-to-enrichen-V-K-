"""Hermetic tests for the reduced-data runtime."""

from pathlib import Path

import pytest
import torch

import dataset.code_dataset as code_dataset
from dataset.code_dataset import ReducedDataset
from dataset.collate import (
    MASK_TOKEN_ID,
    SELF_LOOP_TYPE,
    collate_baseline,
    collate_baseline_inference,
    collate_gnn,
)
from dataset.sampler import SqrtBalancedSampler
from scripts.baseline.reduce_tokenized import validate_output_layout


def _function(
    name: str = "target",
    token_ids=(7, 8),
    *,
    node_count: int = 2,
):
    stored_tokens = (
        torch.tensor(token_ids, dtype=torch.int32) if isinstance(token_ids, tuple) else token_ids
    )
    return {
        "function_name": name,
        "token_ids": stored_tokens,
        "attention_mask": torch.ones(len(token_ids), dtype=torch.int8),
        "token_type_ids": torch.zeros(len(token_ids), dtype=torch.int32),
        "function_features": torch.arange(132, dtype=torch.float32),
        "node_count": node_count,
        "edge_count": 1,
        "nodes": [
            {"size": 2.0, "instruction_count": 1.0},
            {"size": 3.0, "instruction_count": 2.0},
        ][:node_count],
        "edges": [[0, 1, "fallthrough"]] if node_count > 1 else [],
    }


def _write_base(
    root: Path,
    *,
    split: str = "train",
    token_ids=((7, 8), (9,)),
) -> list[Path]:
    project_dir = root / split / "project"
    project_dir.mkdir(parents=True)
    paths = []
    for index, tokens in enumerate(token_ids):
        path = project_dir / f"binary_{index}.pt"
        torch.save(
            {
                "binary_name": f"binary_{index}",
                "functions": [_function(token_ids=tokens)],
            },
            path,
        )
        paths.append(path)
    return paths


def _add_inline_node_features(paths: list[Path]) -> None:
    for index, path in enumerate(paths):
        data = torch.load(path, weights_only=True)
        data["functions"][0]["node_features"] = torch.full(
            (2, 128), float(index + 1), dtype=torch.float16
        )
        torch.save(data, path)


def test_reducer_rejects_stale_project_in_another_split(tmp_path):
    (tmp_path / "train" / "project").mkdir(parents=True)
    split_map = {"train": [], "validation": [], "test": ["project"]}
    with pytest.raises(ValueError, match="clean output directory"):
        validate_output_layout(tmp_path, split_map)


def test_compact_preload_and_public_collate(tmp_path):
    _write_base(tmp_path)
    dataset = ReducedDataset(tmp_path, split="train", preload=True)

    assert len(dataset) == 1
    stored = dataset._index[0]["data"]
    assert stored["token_ids"].dtype == torch.int32
    assert stored["attention_mask"].dtype == torch.int8
    assert stored["token_type_ids"].dtype == torch.int32
    assert stored["function_features"].dtype == torch.float32
    assert stored["node_features"].shape == (2, 20)
    assert stored["node_features"].dtype == torch.float32

    sample = dataset[0]
    batch = collate_baseline_inference([sample])
    assert batch["input_ids_a"].dtype == torch.long
    assert batch["input_ids_a"].shape[0] == 1
    assert batch["function_features_a"].shape == (1, 132)

    training_batch = collate_baseline([sample])
    for side in ("a", "b"):
        masked = training_batch[f"mlm_labels_{side}"] != -100
        assert masked.any()
        assert torch.all(training_batch[f"masked_input_ids_{side}"][masked] == MASK_TOKEN_ID)

    graph_batch = collate_gnn([sample], use_edge_features=True, use_reverse_edges=True)
    assert graph_batch["node_features_a"].shape == (2, 20)
    assert graph_batch["node_features_a"].dtype == torch.float32
    assert SELF_LOOP_TYPE in graph_batch["gnn_edge_types_a"].tolist()


def test_split_is_exact_and_preload_is_required(tmp_path):
    (tmp_path / "validation").mkdir()

    with pytest.raises(FileNotFoundError, match="test"):
        ReducedDataset(tmp_path, split="test")
    with pytest.raises(ValueError, match="Unknown split"):
        ReducedDataset(tmp_path, split="validation_backup")
    with pytest.raises(ValueError, match="preloading only"):
        ReducedDataset(tmp_path, split="validation", preload=False)


@pytest.mark.parametrize("tokens", [[], torch.tensor([], dtype=torch.int32)])
def test_empty_tokens_are_rejected_with_context(tmp_path, tokens):
    project_dir = tmp_path / "train" / "project"
    project_dir.mkdir(parents=True)
    path = project_dir / "empty.pt"
    torch.save(
        {"binary_name": "empty", "functions": [_function(token_ids=tokens)]},
        path,
    )

    with pytest.raises(ValueError, match=r"empty\.pt function 0 \(target\)"):
        ReducedDataset(tmp_path, split="train")


@pytest.mark.parametrize(
    "tokens",
    [
        [1.5, 2.0],
        [True, False],
        [[1, 2]],
        "bad",
        [2**31],
    ],
)
def test_malformed_tokens_are_rejected_with_context(tmp_path, tokens):
    _write_base(tmp_path, token_ids=(tokens, (9,)))
    with pytest.raises(ValueError, match=r"binary_0\.pt function 0 \(target\)"):
        ReducedDataset(tmp_path, split="train")


def test_inline_palmtree_features_are_compact_and_collate(tmp_path):
    base_paths = _write_base(tmp_path)
    _add_inline_node_features(base_paths)
    dataset = ReducedDataset(tmp_path, split="train")
    for entry in dataset._index:
        features = entry["data"]["node_features"]
        assert features.shape == (2, 128)
        assert features.dtype == torch.float16

    batch = collate_gnn([dataset[0]])
    assert batch["node_features_a"].shape == (2, 128)
    assert batch["node_features_a"].dtype == torch.float32


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("width", "feature shape mismatch"),
        ("node_count", "node count mismatch"),
        ("missing_payload", "Missing inline PalmTree node_features"),
    ],
)
def test_invalid_inline_palmtree_features_fail_fast(tmp_path, case, message):
    base_paths = _write_base(tmp_path)
    path = base_paths[0]
    data = torch.load(path, weights_only=True)
    if case == "width":
        data["functions"][0]["node_features"] = torch.zeros((2, 64))
    elif case == "node_count":
        data["functions"][0]["node_features"] = torch.zeros((1, 128))
    else:
        data["functions"][0]["node_features"] = None
    torch.save(data, path)

    with pytest.raises(ValueError, match=message):
        ReducedDataset(tmp_path, split="train")


def test_external_sidecars_are_rejected(tmp_path):
    _write_base(tmp_path)
    with pytest.raises(ValueError, match="External GNN sidecars are unsupported"):
        ReducedDataset(tmp_path, split="train", gnn_sidecar_dir="legacy")


def test_load_is_restricted_and_cpu_safe(tmp_path, monkeypatch):
    _write_base(tmp_path)
    calls = []
    original_load = torch.load

    def recording_load(*args, **kwargs):
        calls.append(kwargs.copy())
        return original_load(*args, **kwargs)

    monkeypatch.setattr(code_dataset.torch, "load", recording_load)
    ReducedDataset(tmp_path, split="train")

    assert calls
    assert all(call["weights_only"] is True for call in calls)
    assert all(call["map_location"] == "cpu" for call in calls)


class _DatasetStub:
    def __init__(self, group_keys):
        self._group_keys = group_keys


def test_sampler_uses_no_replacement_and_compresses_projects():
    full = _DatasetStub([("project", str(index)) for index in range(5)])
    full_sample = list(SqrtBalancedSampler(full, scale_factor=50, seed=4))
    assert sorted(full_sample) == list(range(5))

    compressed = _DatasetStub([("project", str(index)) for index in range(9)])
    sampler = SqrtBalancedSampler(compressed, scale_factor=1, seed=4)
    first = list(sampler)
    assert len(first) == 3
    assert len(set(first)) == 3
    assert first == list(SqrtBalancedSampler(compressed, scale_factor=1, seed=4))
    sampler.set_epoch(1)
    assert list(sampler) != first


def test_sampler_padding_matches_reported_rank_length():
    dataset = _DatasetStub([("project", "target")])
    for rank in range(8):
        sampler = SqrtBalancedSampler(dataset, rank=rank, world_size=8)
        assert len(list(sampler)) == len(sampler) == 1


@pytest.mark.parametrize(
    "kwargs",
    [
        {"scale_factor": 0},
        {"scale_factor": float("nan")},
        {"scale_factor": True},
        {"world_size": 0},
        {"world_size": 2, "rank": 2},
        {"world_size": 2, "rank": -1},
    ],
)
def test_sampler_rejects_invalid_configuration(kwargs):
    with pytest.raises((TypeError, ValueError)):
        SqrtBalancedSampler(_DatasetStub([("project", "target")]), **kwargs)


def test_collate_rejects_empty_tokens():
    binary = {
        "token_ids": [],
        "function_features": torch.zeros(132),
        "node_features": torch.zeros((1, 20)),
        "edges": [],
        "edge_types": [],
    }
    sample = {
        "binary_1": binary,
        "binary_2": binary,
        "label": 1,
        "metadata": {},
    }
    with pytest.raises(ValueError, match="non-empty"):
        collate_baseline_inference([sample])
