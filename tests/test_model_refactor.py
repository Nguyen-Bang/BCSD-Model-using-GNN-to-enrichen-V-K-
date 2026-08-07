"""Tests for the baseline and structural-prefix model interfaces."""

from unittest.mock import patch

import torch

from models.baseline.baseline_model import BaselineModel
from models.baseline_gnn_model import BaselineGNNModel
from models.structural_prefix import SetEncoder, build_structural_prefix_head


def _tiny_model(model_class=BaselineGNNModel):
    return model_class(
        gnn_input_dim=3,
        gnn_hidden_dim=8,
        gnn_layers=2,
        gnn_pool_heads=2,
        prefix_dim=8,
        gnn_encoder_type="set",
        gnn_set_hidden_multiplier=2,
        gnn_use_edge_features=False,
        num_layers=1,
        hidden_size=16,
        num_heads=4,
        vocab_size=32,
        max_seq_length=8,
        embed_dim=16,
        func_feat_dim=4,
        dropout=0.0,
        clap_embedding_init=False,
    )


def test_set_encoder_has_an_edge_free_api():
    torch.manual_seed(0)
    encoder = SetEncoder(
        input_dim=3,
        hidden_dim=8,
        num_layers=2,
        hidden_multiplier=2,
        pool_heads=2,
        dropout=0.0,
    ).eval()
    nodes = torch.randn(3, 3)
    batch = torch.tensor([0, 0, 1])
    with torch.no_grad():
        output = encoder(nodes, batch)
    assert output.shape == (2, 16)


def test_structural_prefix_head_preserves_checkpoint_key_layout():
    head = build_structural_prefix_head(
        encoder_type="set",
        input_dim=3,
        hidden_dim=8,
        num_layers=2,
        set_hidden_multiplier=2,
        pool_heads=2,
        dropout=0.0,
    )
    keys = set(head.state_dict())
    assert "encoder.convs.0.weight" in keys
    assert "encoder.node_transform.0.weight" in keys
    assert "encoder.attention_queries" in keys
    clone = build_structural_prefix_head(
        encoder_type="set",
        input_dim=3,
        hidden_dim=8,
        num_layers=2,
        set_hidden_multiplier=2,
        pool_heads=2,
        dropout=0.0,
    )
    clone.load_state_dict(head.state_dict(), strict=True)


def test_pooled_output_is_consolidated():
    model = _tiny_model().eval()
    input_ids = torch.randint(0, 32, (2, 5))
    attention_mask = torch.ones(2, 5)
    token_type_ids = torch.zeros(2, 5, dtype=torch.long)
    node_features = torch.randn(3, 3)
    graph_batch = torch.tensor([0, 0, 1])
    edge_index = torch.empty(2, 0, dtype=torch.long)

    with torch.no_grad():
        embedding, pooled = model.forward_with_pooled(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            function_features=torch.randn(2, 4),
            node_features=node_features,
            edge_index=edge_index,
            graph_batch=graph_batch,
        )
    assert embedding.shape == (2, 16)
    assert pooled.shape == (2, 16)
    torch.testing.assert_close(embedding.norm(dim=1), torch.ones(2))


def test_removed_attention_modes_fail_loudly():
    for kwargs in (
        {"attention_mode": "scalar"},
        {"attention_mode": "split"},
        {"multi_token": False},
        {"kv_low_rank": 4},
        {"gnn_prefix_mode": "per_block"},
    ):
        try:
            _tiny_model_with(**kwargs)
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError for {kwargs}")


def _tiny_model_with(**overrides):
    kwargs = {
        "gnn_input_dim": 3,
        "gnn_hidden_dim": 8,
        "gnn_layers": 2,
        "gnn_pool_heads": 2,
        "prefix_dim": 8,
        "gnn_encoder_type": "set",
        "gnn_set_hidden_multiplier": 2,
        "num_layers": 1,
        "hidden_size": 16,
        "num_heads": 4,
        "vocab_size": 32,
        "max_seq_length": 8,
        "embed_dim": 16,
        "func_feat_dim": 4,
        "dropout": 0.0,
        "clap_embedding_init": False,
    }
    kwargs.update(overrides)
    return BaselineGNNModel(**kwargs)


def test_clap_embedding_revision_is_forwarded_and_shape_checked():
    model = BaselineModel(
        num_layers=1,
        hidden_size=8,
        num_heads=2,
        vocab_size=16,
        max_seq_length=8,
        embed_dim=8,
        clap_embedding_init=False,
    )
    source = torch.randn(16, 8)
    with (
        patch("huggingface_hub.hf_hub_download", return_value="weights") as download,
        patch(
            "safetensors.torch.load_file",
            return_value={"jroformer.embeddings.word_embeddings.weight": source},
        ),
    ):
        model._init_embeddings_from_clap("example/clap", "deadbeef")
    download.assert_called_once_with(
        repo_id="example/clap",
        filename="model.safetensors",
        revision="deadbeef",
    )
    torch.testing.assert_close(model.roformer.embeddings.word_embeddings.weight, source)

    with (
        patch("huggingface_hub.hf_hub_download", return_value="weights"),
        patch("safetensors.torch.load_file", return_value={}),
    ):
        try:
            model._init_embeddings_from_clap("example/clap", None)
        except KeyError:
            return
    raise AssertionError("missing CLAP embedding key must raise")
