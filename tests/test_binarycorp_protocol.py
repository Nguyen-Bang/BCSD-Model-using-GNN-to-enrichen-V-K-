import numpy as np
import pytest
import torch

from dataset.code_dataset import ReducedDataset
from evaluation.binarycorp import runner
from evaluation.binarycorp.runner import encode_split
from preprocessing.binarycorp.protocol import (
    OPTIMIZATION_LEVELS,
    EmbeddingTable,
    SplitArtifact,
    evaluate_optimization_pairs,
    matchable_counts,
    ranks_for_pool,
)


def _complete_embeddings(count: int, width: int = 8):
    rng = np.random.default_rng(4)
    keys = [("project", f"function-{index}") for index in range(count)]
    vectors = rng.normal(size=(count, width)).astype(np.float32)
    ordinals = np.arange(count, dtype=np.int64)
    return {
        optimization: EmbeddingTable(keys.copy(), ordinals.copy(), vectors.copy())
        for optimization in OPTIMIZATION_LEVELS
    }


def test_exact_six_pairs_use_disjoint_full_pools_and_drop_remainder():
    result = evaluate_optimization_pairs(_complete_embeddings(7), pool_size=3, seed=0)

    assert result["aggregate"] == {
        "mrr": 1.0,
        "recall_at_1": 1.0,
        "pairs_with_pool": 6,
    }
    assert len(result["per_pair"]) == 6
    assert all(pair["n_pools"] == 2 for pair in result["per_pair"].values())
    assert all(pair["n_dropped"] == 1 for pair in result["per_pair"].values())


def test_pessimistic_tie_policy_uses_worst_tied_rank():
    query = np.array([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32)
    candidates = query.copy()
    assert ranks_for_pool(query, candidates).tolist() == [2, 2]


def test_exact_protocol_rejects_a_missing_optimization_pair():
    keys = [("project", "function")]
    table = EmbeddingTable(keys, np.array([0]), np.array([[1.0, 0.0]]))
    embeddings = {"O0": table, "O3": table}
    with pytest.raises(ValueError, match="No complete pool"):
        evaluate_optimization_pairs(embeddings, pool_size=1)


def test_index_split_rejects_duplicate_identity(tmp_path):
    root = tmp_path / "test" / "project"
    root.mkdir(parents=True)
    function = {"function_name": "static", "token_ids": torch.tensor([1])}
    torch.save(
        {"project": "project", "opt": "O0", "functions": [function, function]},
        root / "a.pt",
    )

    with pytest.raises(ValueError, match="Duplicate BinaryCorp identity"):
        matchable_counts(tmp_path, "test")


def test_index_split_rejects_project_directory_mismatch(tmp_path):
    root = tmp_path / "test" / "directory-project"
    root.mkdir(parents=True)
    torch.save(
        {"project": "payload-project", "opt": "O0", "functions": []},
        root / "a.pt",
    )

    with pytest.raises(ValueError, match="must match its directory"):
        matchable_counts(tmp_path, "test")


def _function(name: str, token: int) -> dict:
    return {
        "function_name": name,
        "token_ids": torch.tensor([token], dtype=torch.int32),
        "attention_mask": torch.tensor([1], dtype=torch.int8),
        "token_type_ids": torch.tensor([0], dtype=torch.int32),
        "function_features": torch.zeros(132),
        "nodes": [{}],
        "edges": [],
    }


def test_encode_split_streams_into_preallocated_optimization_tables(tmp_path, monkeypatch):
    root = tmp_path / "test" / "p"
    root.mkdir(parents=True)
    torch.save({"project": "p", "opt": "O0", "functions": [_function("f", 3)]}, root / "a.pt")
    torch.save({"project": "p", "opt": "O3", "functions": [_function("f", 4)]}, root / "b.pt")

    seen_batches = []

    class TokenAdapter:
        include_function_features = False
        include_graph = False

        def encode(self, batch):
            seen_batches.append(set(batch))
            token = batch["input_ids"][:, 0].float()
            return torch.stack([token, torch.ones_like(token)], dim=1)

    def fail_concatenate(*_args, **_kwargs):
        raise AssertionError("encode_split must not accumulate and concatenate chunks")

    def fail_full_conversion(*_args, **_kwargs):
        raise AssertionError("token-only adapters must not materialize graph tensors")

    monkeypatch.setattr(np, "concatenate", fail_concatenate)
    monkeypatch.setattr(ReducedDataset, "_to_plain", fail_full_conversion)

    encoded, stats = encode_split(
        tmp_path,
        "test",
        TokenAdapter(),
        batch_size=1,
        use_edge_features=False,
        use_reverse_edges=False,
    )

    assert set(encoded) == {"O0", "O3"}
    assert encoded["O0"].keys == [("p", "f")]
    assert encoded["O0"].vectors.tolist() == [[3.0, 1.0]]
    assert encoded["O0"].vectors.flags.c_contiguous
    assert seen_batches == [
        {"input_ids", "attention_mask", "token_type_ids"},
        {"input_ids", "attention_mask", "token_type_ids"},
    ]
    assert stats == type(stats)(files=2, functions=2, groups=1)


def test_encode_split_rejects_duplicate_identity_before_encoding(tmp_path):
    root = tmp_path / "test" / "p"
    root.mkdir(parents=True)
    function = _function("f", 3)
    torch.save(
        {"project": "p", "opt": "O0", "functions": [function, function]},
        root / "a.pt",
    )

    class UnusedAdapter:
        include_function_features = False
        include_graph = False

        def encode(self, _batch):
            raise AssertionError("duplicates must fail before model inference")

    with pytest.raises(ValueError, match="Duplicate BinaryCorp identity"):
        encode_split(
            tmp_path,
            "test",
            UnusedAdapter(),
            batch_size=2,
            use_edge_features=False,
            use_reverse_edges=False,
        )


def test_encode_split_rejects_order_changes_between_passes(tmp_path, monkeypatch):
    calls = 0

    def changing_artifacts(*_args):
        nonlocal calls
        calls += 1
        functions = [_function("first", 1), _function("second", 2)]
        if calls == 2:
            functions.reverse()
        yield SplitArtifact("p", "O0", functions)

    class UnusedAdapter:
        include_function_features = False
        include_graph = False

        def encode(self, _batch):
            raise AssertionError("changed ordering must fail before model inference")

    monkeypatch.setattr(runner, "iter_split_artifacts", changing_artifacts)
    with pytest.raises(RuntimeError, match="split changed"):
        encode_split(
            tmp_path,
            "test",
            UnusedAdapter(),
            batch_size=2,
            use_edge_features=False,
            use_reverse_edges=False,
        )
