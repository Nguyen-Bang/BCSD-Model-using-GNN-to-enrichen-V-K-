from __future__ import annotations

from pathlib import Path

import pytest
import torch

from preprocessing.III_tokenization.palmtree import augment_with_node_embeddings as palm


class FakeEncoder:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def encode(self, texts, output_option="lst"):
        assert output_option == "lst"
        self.calls.append(list(texts))
        return torch.stack([torch.full((palm.PALMTREE_DIM,), float(len(text))) for text in texts])


def _write_inputs(root: Path) -> tuple[Path, Path, Path]:
    base_path = root / "base" / "train" / "project" / "sample.pt"
    source_path = root / "tokenized" / "project" / "sample.json"
    destination = root / "output" / "train" / "project" / "sample.pt"
    base_path.parent.mkdir(parents=True)
    source_path.parent.mkdir(parents=True)

    function = {
        "function_name": "target",
        "token_ids": torch.tensor([1, 2], dtype=torch.int32),
        "attention_mask": torch.tensor([1, 1], dtype=torch.int8),
        "token_type_ids": torch.tensor([3, 3], dtype=torch.int32),
        "function_features": torch.zeros(132),
        "node_count": 2,
        "edge_count": 1,
        "nodes": [{"id": 10, "size": 2}, {"id": 20, "size": 1}],
        "edges": [[10, 20]],
    }
    torch.save(
        {"binary_name": "sample", "project": "project", "functions": [function]},
        base_path,
    )
    source_path.write_text(
        """{
          "binary_name": "sample",
          "functions": [{
            "function_name": "target",
            "token_ids": [1, 2],
            "node_count": 2,
            "nodes": [
              {"id": 10, "instructions": {"2": "ADD EAX, 1", "1": "MOV EAX, EBX"}},
              {"id": 20, "instructions": {"1": "MOV EAX, EBX"}}
            ]
          }]
        }""",
        encoding="utf-8",
    )
    return base_path, source_path, destination


def _provenance() -> dict[str, int | str]:
    return {
        "schema_version": palm.PALMTREE_SCHEMA_VERSION,
        "embedding_dim": palm.PALMTREE_DIM,
        "base_sha256": "base",
        "tokenized_sha256": "source",
        "model_sha256": "model",
        "vocab_sha256": "vocab",
    }


def test_enrich_file_preserves_base_and_writes_inline_embeddings(tmp_path):
    base_path, source_path, destination = _write_inputs(tmp_path)
    encoder = FakeEncoder()

    nodes, instructions = palm.enrich_file(
        base_path,
        source_path,
        destination,
        encoder,
        str.lower,
        palm.EmbeddingCache(max_entries=1),
        batch_size=1,
        provenance=_provenance(),
    )

    assert (nodes, instructions) == (2, 3)
    output = torch.load(destination, weights_only=True)
    features = output["functions"][0]["node_features"]
    assert features.dtype == torch.float16
    assert features.shape == (2, palm.PALMTREE_DIM)
    assert features[0, 0].item() == pytest.approx((len("mov eax, ebx") + len("add eax, 1")) / 2)
    assert features[1, 0].item() == len("mov eax, ebx")
    assert output["functions"][0]["token_ids"].tolist() == [1, 2]
    assert output["palmtree"] == _provenance()
    assert max(len(call) for call in encoder.calls) == 1


def test_enrich_file_rejects_misaligned_nodes_without_publishing(tmp_path):
    base_path, source_path, destination = _write_inputs(tmp_path)
    source = palm._load_json(source_path)
    source["functions"][0]["nodes"][1]["id"] = 99
    source_path.write_text(__import__("json").dumps(source), encoding="utf-8")

    with pytest.raises(ValueError, match="Node mismatch"):
        palm.enrich_file(
            base_path,
            source_path,
            destination,
            FakeEncoder(),
            str.lower,
            palm.EmbeddingCache(max_entries=10),
            batch_size=4,
            provenance=_provenance(),
        )

    assert not destination.exists()


def test_enrich_file_rejects_invalid_encoder_shape(tmp_path):
    base_path, source_path, destination = _write_inputs(tmp_path)

    class BadEncoder:
        def encode(self, texts, output_option="lst"):
            return torch.zeros((len(texts), 64))

    with pytest.raises(ValueError, match="expected .*128"):
        palm.enrich_file(
            base_path,
            source_path,
            destination,
            BadEncoder(),
            str.lower,
            palm.EmbeddingCache(max_entries=10),
            batch_size=4,
            provenance=_provenance(),
        )

    assert not destination.exists()


def test_main_resumes_only_after_complete_file_validation(tmp_path, monkeypatch):
    base_path, source_path, destination = _write_inputs(tmp_path)
    checkout = tmp_path / "PalmTree"
    weights = checkout / "pre-trained_model" / "palmtree"
    weights.mkdir(parents=True)
    (checkout / "src").mkdir()
    (weights / "transformer.ep19").write_bytes(b"model")
    (weights / "vocab").write_bytes(b"vocab")

    load_calls = 0

    def fake_load(*_args):
        nonlocal load_calls
        load_calls += 1
        return FakeEncoder(), str.lower

    monkeypatch.setattr(palm, "load_palmtree", fake_load)
    arguments = [
        "--base-dir",
        str(base_path.parents[2]),
        "--tokenized-dir",
        str(source_path.parents[1]),
        "--output-dir",
        str(destination.parents[2]),
        "--palmtree-checkout",
        str(checkout),
    ]

    assert palm.main(arguments) == 0
    assert load_calls == 1
    assert palm.main(arguments) == 0
    assert load_calls == 1

    partial = torch.load(destination, weights_only=True)
    partial["functions"][0].pop("node_features")
    torch.save(partial, destination)
    with pytest.raises(ValueError, match="Function fields changed"):
        palm.main(arguments)


def test_cache_is_bounded_and_invalid_cli_sizes_are_rejected(tmp_path):
    cache = palm.EmbeddingCache(max_entries=2)
    for index in range(3):
        cache.put(str(index), torch.zeros(palm.PALMTREE_DIM))
    assert len(cache) == 2
    assert cache.get("0") is None

    with pytest.raises(ValueError, match="encode-batch-size"):
        palm.main(
            [
                "--base-dir",
                str(tmp_path),
                "--tokenized-dir",
                str(tmp_path),
                "--output-dir",
                str(tmp_path / "out"),
                "--palmtree-checkout",
                str(tmp_path),
                "--encode-batch-size",
                "0",
            ]
        )
