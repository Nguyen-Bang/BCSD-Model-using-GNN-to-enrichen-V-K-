"""Sanity tests for tanh-per-pair KV-prefix attention."""

import torch
import torch.nn.functional as F

from models.structural_prefix import KVPrefixAttention


def _module(dropout=0.0):
    return KVPrefixAttention(
        hidden_size=768,
        num_heads=12,
        prefix_dim=256,
        num_prefix_tokens=10,
        dropout=dropout,
    )


def _inputs(seed=0):
    generator = torch.Generator().manual_seed(seed)
    query = torch.randn(2, 12, 8, 64, generator=generator)
    key = torch.randn(2, 12, 8, 64, generator=generator)
    value = torch.randn(2, 12, 8, 64, generator=generator)
    prefixes = torch.randn(2, 10, 256, generator=generator)
    mask = torch.ones(2, 8)
    return query, key, value, prefixes, mask


def test_zero_init_is_token_only():
    module = _module().eval()
    query, key, value, prefixes, mask = _inputs()
    actual, weights = module(query, key, value, prefixes, mask)
    additive_mask = (1.0 - mask[:, None, None, :]) * -1e4
    expected = F.scaled_dot_product_attention(
        query, key, value, attn_mask=additive_mask, dropout_p=0.0
    )
    torch.testing.assert_close(actual, expected)
    assert weights is None


def test_state_dict_matches_final_checkpoint_layout():
    expected = {
        "weight",
        "prefix_to_k.weight",
        "prefix_to_k.bias",
        "prefix_to_v.weight",
        "prefix_to_v.bias",
        "prefix_k_norm.weight",
        "prefix_k_norm.bias",
        "prefix_v_norm.weight",
        "prefix_v_norm.bias",
    }
    module = _module()
    assert set(module.state_dict()) == expected
    assert "_ablate_mask" not in module.state_dict()


def test_gate_receives_gradient_at_zero_init():
    module = _module()
    query, key, value, prefixes, mask = _inputs()
    output, _ = module(query, key, value, prefixes, mask)
    output.square().mean().backward()
    assert module.weight.grad is not None
    assert module.weight.grad.abs().sum() > 0


def test_effective_gate_is_scalar():
    assert _module().effective_gate().shape == torch.Size([])
