"""Tests for typed edges in GraphEncoder."""

import torch

from models.structural_prefix import GraphEncoder


def _tiny_graph():
    """4 nodes, 6 forward edges of mixed types, 1 graph in the batch."""
    x = torch.randn(4, 20)
    edge_index = torch.tensor(
        [[0, 0, 1, 2, 2, 3], [1, 2, 3, 1, 3, 0]],
        dtype=torch.long,
    )
    edge_types = torch.tensor([0, 1, 2, 3, 4, 0], dtype=torch.long)
    batch = torch.zeros(4, dtype=torch.long)
    return x, edge_index, edge_types, batch


def test_gatv2_without_edge_features_runs():
    """GATv2 also supports the untyped-edge graph variant."""
    enc = GraphEncoder(
        input_dim=20,
        hidden_dim=32,
        num_layers=3,
        gat_heads=2,
        pool_heads=2,
        dropout=0.0,
        use_edge_features=False,
    )
    enc.eval()
    x, edge_index, _, batch = _tiny_graph()
    with torch.no_grad():
        out = enc(x, edge_index, batch)
    assert out.shape == (1, 2 * 32), out.shape


def test_v2_with_edge_features_runs():
    """Typed GATv2 edges run through the graph encoder."""
    enc = GraphEncoder(
        input_dim=20,
        hidden_dim=32,
        num_layers=3,
        gat_heads=2,
        pool_heads=2,
        dropout=0.0,
        use_edge_features=True,
        num_edge_types=11,
        edge_emb_dim=16,
    )
    enc.eval()
    x, edge_index, edge_types, batch = _tiny_graph()
    with torch.no_grad():
        out = enc(x, edge_index, batch, edge_types=edge_types)
    assert out.shape == (1, 2 * 32)
    # Embedding table shape
    assert enc.edge_embedding.weight.shape == (11, 16)


def test_missing_edge_types_raises():
    """If use_edge_features=True but edge_types is None, raise (no silent drop)."""
    enc = GraphEncoder(
        input_dim=20,
        hidden_dim=32,
        num_layers=3,
        gat_heads=2,
        pool_heads=2,
        dropout=0.0,
        use_edge_features=True,
    )
    enc.eval()
    x, edge_index, _, batch = _tiny_graph()
    raised = False
    try:
        enc(x, edge_index, batch, edge_types=None)
    except ValueError:
        raised = True
    assert raised, "expected error when edge_types is None under use_edge_features=True"


def test_different_edge_types_change_output():
    """Same graph, different edge type labels -> different output. Proves
    edge types actually flow into attention."""
    enc = GraphEncoder(
        input_dim=20,
        hidden_dim=32,
        num_layers=3,
        gat_heads=2,
        pool_heads=2,
        dropout=0.0,
        use_edge_features=True,
    )
    enc.eval()
    x, edge_index, et1, batch = _tiny_graph()
    et2 = (et1 + 5).clamp(max=10)  # different categorical labels
    with torch.no_grad():
        o1 = enc(x, edge_index, batch, edge_types=et1)
        o2 = enc(x, edge_index, batch, edge_types=et2)
    diff = (o1 - o2).abs().max().item()
    assert diff > 1e-5, f"edge types had no effect on output, max diff={diff}"


def test_gradient_flows_into_edge_embedding():
    """Backward pass populates edge_embedding.weight.grad and lin_edge.weight.grad."""
    enc = GraphEncoder(
        input_dim=20,
        hidden_dim=32,
        num_layers=3,
        gat_heads=2,
        pool_heads=2,
        dropout=0.0,
        use_edge_features=True,
    )
    enc.train()
    x, edge_index, edge_types, batch = _tiny_graph()
    out = enc(x, edge_index, batch, edge_types=edge_types)
    loss = out.sum()
    loss.backward()
    assert enc.edge_embedding.weight.grad is not None
    assert enc.edge_embedding.weight.grad.abs().sum().item() > 0
    for i, conv in enumerate(enc.convs):
        assert conv.lin_edge.weight.grad is not None, f"layer {i} lin_edge has no grad"
        assert conv.lin_edge.weight.grad.abs().sum().item() > 0, f"layer {i} lin_edge grad is zero"


def test_attention_dropout_can_be_disabled_separately():
    """Attention dropout can be zero while post-layer dropout stays enabled."""
    enc = GraphEncoder(
        input_dim=20,
        hidden_dim=32,
        num_layers=3,
        gat_heads=2,
        pool_heads=2,
        dropout=0.1,
        attention_dropout=0.0,
        use_edge_features=True,
    )
    assert enc.dropout == 0.1
    assert enc.attention_dropout == 0.0
    assert all(conv.dropout == 0.0 for conv in enc.convs)


def test_attention_dropout_defaults_to_layer_dropout():
    """Existing configurations retain their previous dropout behavior."""
    enc = GraphEncoder(
        input_dim=20,
        hidden_dim=32,
        num_layers=3,
        gat_heads=2,
        pool_heads=2,
        dropout=0.1,
        use_edge_features=True,
    )
    assert enc.dropout == 0.1
    assert enc.attention_dropout == 0.1
    assert all(conv.dropout == 0.1 for conv in enc.convs)
