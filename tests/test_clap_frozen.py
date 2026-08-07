from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
import transformers
from safetensors.torch import save_model
from transformers import RoFormerConfig, RoFormerModel

from models.clap_frozen import CLAPFrozenBaseline, CLAPFrozenGNN
from models.clap_frozen import backbone as backbone_module
from models.clap_frozen.backbone import _validate_config, load_clap_backbone
from models.clap_frozen.factory import create_model, load_checkpoint_strict
from scripts.gnn.clap_frozen.train import _resolve_output_dir


def tiny_backbone(dropout: float = 0.0) -> RoFormerModel:
    config = RoFormerConfig(
        vocab_size=32,
        embedding_size=16,
        hidden_size=16,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=32,
        hidden_dropout_prob=dropout,
        attention_probs_dropout_prob=dropout,
        max_position_embeddings=32,
        type_vocab_size=2,
    )
    return RoFormerModel(config)


def token_inputs():
    return {
        "input_ids": torch.tensor([[2, 3, 1], [4, 5, 6]]),
        "attention_mask": torch.tensor([[1, 1, 0], [1, 1, 1]]),
        "token_type_ids": torch.zeros(2, 3, dtype=torch.long),
        "function_features": torch.randn(2, 3),
    }


def graph_inputs():
    return {
        "node_features": torch.randn(4, 4),
        "edge_index": torch.tensor([[0, 1, 2, 3, 0, 1, 2, 3], [1, 0, 3, 2, 0, 1, 2, 3]]),
        "edge_types": torch.tensor([0, 5, 1, 6, 10, 10, 10, 10]),
        "graph_batch": torch.tensor([0, 0, 1, 1]),
    }


def gnn_model(prefix_source: str) -> CLAPFrozenGNN:
    return CLAPFrozenGNN(
        func_feat_dim=3,
        embed_dim=8,
        dropout=0.0,
        gnn_input_dim=4,
        gnn_hidden_dim=8,
        gnn_layers=2,
        gnn_gat_heads=2,
        gnn_pool_heads=2,
        gnn_use_edge_features=True,
        prefix_source=prefix_source,
        backbone=tiny_backbone(),
    )


def set_model() -> CLAPFrozenGNN:
    return CLAPFrozenGNN(
        func_feat_dim=3,
        embed_dim=8,
        dropout=0.0,
        gnn_input_dim=4,
        gnn_hidden_dim=8,
        gnn_layers=2,
        gnn_pool_heads=2,
        gnn_encoder_type="set",
        gnn_set_hidden_multiplier=2,
        gnn_use_edge_features=False,
        backbone=tiny_backbone(),
    )


def test_baseline_is_normalized_and_keeps_frozen_backbone_in_eval_mode():
    model = CLAPFrozenBaseline(
        func_feat_dim=3,
        embed_dim=8,
        dropout=0.0,
        backbone=tiny_backbone(dropout=0.8),
    )
    model.train()
    inputs = token_inputs()
    first = model(**inputs)
    second = model(**inputs)

    assert not model.roformer.training
    assert all(not parameter.requires_grad for parameter in model.roformer.parameters())
    assert torch.allclose(first, second)
    assert torch.allclose(first.norm(dim=-1), torch.ones(2), atol=1e-6)


def test_typed_edge_gnn_runs_on_cpu_and_returns_pooled_states():
    model = gnn_model("gnn").eval()
    embedding, pooled = model.forward_with_pooled(**token_inputs(), **graph_inputs())

    assert embedding.shape == (2, 8)
    assert pooled.shape == (2, 16)
    assert torch.isfinite(embedding).all()
    assert torch.allclose(embedding.norm(dim=-1), torch.ones(2), atol=1e-6)


def test_set_encoder_runs_without_edge_inputs():
    inputs = graph_inputs()
    embedding = set_model()(
        **token_inputs(),
        node_features=inputs["node_features"],
        graph_batch=inputs["graph_batch"],
    )
    assert embedding.shape == (2, 8)
    assert torch.isfinite(embedding).all()


def test_factory_builds_explicit_set_encoder():
    config = {
        "model": {
            "variant": "gnn",
            "func_feature_dim": 3,
            "embed_dim": 8,
            "dropout": 0.0,
        },
        "gnn": {
            "encoder_type": "set",
            "input_dim": 4,
            "hidden_dim": 8,
            "num_layers": 2,
            "pool_heads": 2,
            "hidden_multiplier": 2,
            "prefix_source": "gnn",
            "use_edge_features": False,
        },
    }

    model = create_model(config, torch.device("cpu"), backbone=tiny_backbone())

    assert not model.gnn_head.requires_edges


def test_zero_initialized_prefix_channel_matches_native_roformer():
    model = gnn_model("register").eval()
    inputs = token_inputs()
    prefixes = model.register_prefix.unsqueeze(0).expand(2, -1, -1)
    actual = model._transformer(
        inputs["input_ids"],
        inputs["attention_mask"],
        inputs["token_type_ids"],
        prefixes,
    )
    expected = model.roformer(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        token_type_ids=inputs["token_type_ids"],
        return_dict=True,
    ).last_hidden_state

    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-5)


def test_register_variant_has_no_unused_gnn_and_receives_gradients():
    model = gnn_model("register")
    for layer in model.kv_prefix_layers:
        layer.weight.data.fill_(0.2)
    embedding = model(**token_inputs())
    embedding.sum().backward()

    assert model.gnn_head is None
    assert model.register_prefix.grad is not None
    assert torch.isfinite(model.register_prefix.grad).all()


def test_gnn_mode_requires_graph_inputs():
    with pytest.raises(ValueError, match="node_features"):
        gnn_model("gnn")(**token_inputs())


def test_factory_and_checkpoint_loading_are_config_strict(tmp_path):
    config = {
        "model": {
            "variant": "baseline",
            "pretrained_model": "unused-in-test",
            "revision": "test",
            "func_feature_dim": 3,
            "embed_dim": 8,
            "dropout": 0.0,
        },
        "data": {},
        "training": {},
    }
    model = create_model(config, torch.device("cpu"), backbone=tiny_backbone())
    checkpoint = tmp_path / "model.pt"
    torch.save({"config": config, "model_state_dict": model.state_dict()}, checkpoint)
    target = create_model(config, torch.device("cpu"), backbone=tiny_backbone())

    loaded = load_checkpoint_strict(target, checkpoint, torch.device("cpu"), config)
    assert loaded["config"] == config

    changed = {**config, "data": {"data_dir": "different-pool"}}
    with pytest.raises(ValueError, match="Checkpoint config"):
        load_checkpoint_strict(target, checkpoint, torch.device("cpu"), changed)


def test_default_training_output_is_seeded_and_refuses_overwrite(tmp_path):
    config = {"training": {"checkpoint_dir": str(tmp_path)}}
    output = _resolve_output_dir(config, seed=101, resume=None, output_dir=None)
    assert output == tmp_path / "seed_101"
    (output / "best_mrr.pt").touch()
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        _resolve_output_dir(config, seed=101, resume=None, output_dir=None)

    resume = output / "checkpoint_epoch_1.pt"
    with pytest.raises(ValueError, match="must match the resume"):
        _resolve_output_dir(
            config,
            seed=101,
            resume=resume,
            output_dir=tmp_path / "different-run",
        )


def test_official_geometry_is_validated_before_loading_weights():
    config = SimpleNamespace(
        hidden_size=768,
        num_attention_heads=12,
        num_hidden_layers=11,
        vocab_size=33_555,
        max_position_embeddings=2_048,
        use_bias=False,
    )
    with pytest.raises(ValueError, match="num_hidden_layers"):
        _validate_config(config)


def test_official_loader_uses_pinned_remote_code_and_local_strict_weights(tmp_path, monkeypatch):
    class FakeEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.jroformer = nn.Linear(3, 2)
            self.projection = nn.Linear(2, 2)

    source = FakeEncoder()
    with torch.no_grad():
        source.jroformer.weight.fill_(0.25)
    checkpoint = tmp_path / "model.safetensors"
    save_model(source, checkpoint)
    target = FakeEncoder()
    calls = {}

    def fake_config(model_id, **kwargs):
        calls["model_id"] = model_id
        calls.update(kwargs)
        return object()

    monkeypatch.setattr(transformers.AutoConfig, "from_pretrained", fake_config)

    def fake_model(config, **kwargs):
        calls["model_kwargs"] = kwargs
        return target

    monkeypatch.setattr(transformers.AutoModel, "from_config", fake_model)
    monkeypatch.setattr(
        "huggingface_hub.hf_hub_download",
        lambda **kwargs: str(checkpoint),
    )
    monkeypatch.setattr(backbone_module, "_validate_config", lambda config: None)

    loaded = load_clap_backbone("mock/clap", revision="pinned-test-revision")

    assert loaded is target.jroformer
    assert torch.equal(loaded.weight, source.jroformer.weight)
    assert calls == {
        "model_id": "mock/clap",
        "revision": "pinned-test-revision",
        "trust_remote_code": True,
        "local_files_only": False,
        "model_kwargs": {
            "trust_remote_code": True,
            "code_revision": "pinned-test-revision",
        },
    }
