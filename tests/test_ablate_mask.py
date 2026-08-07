"""Regression tests for per-head, per-prefix causal ablation masks."""

import torch
import torch.nn.functional as F

from models.structural_prefix import KVPrefixAttention

HIDDEN = 16
HEADS = 2
PREFIX_DIM = 8
NUM_PREFIX = 3
SEQ = 5
BATCH = 2
HEAD_DIM = HIDDEN // HEADS


def _inputs(seed=0):
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(BATCH, HEADS, SEQ, HEAD_DIM, generator=g)
    k = torch.randn(BATCH, HEADS, SEQ, HEAD_DIM, generator=g)
    v = torch.randn(BATCH, HEADS, SEQ, HEAD_DIM, generator=g)
    prefixes = torch.randn(BATCH, NUM_PREFIX, PREFIX_DIM, generator=g)
    return q, k, v, prefixes


def _token_only(q, k, v):
    return F.scaled_dot_product_attention(q, k, v, attn_mask=None, is_causal=False)


def _activate_gates(mod):
    with torch.no_grad():
        mod.weight.fill_(0.7)


def _make(cls):
    mod = cls(
        hidden_size=HIDDEN,
        num_heads=HEADS,
        prefix_dim=PREFIX_DIM,
        num_prefix_tokens=NUM_PREFIX,
        dropout=0.0,
    )
    mod.eval()
    _activate_gates(mod)
    return mod


def _check_variant(cls):
    name = cls.__name__
    q, k, v, prefixes = _inputs()
    tok = _token_only(q, k, v)

    mod = _make(cls)

    assert hasattr(mod, "_ablate_mask"), f"{name}: missing _ablate_mask buffer"
    assert tuple(mod._ablate_mask.shape) == (HEADS, NUM_PREFIX)
    assert torch.all(mod._ablate_mask == 1.0), f"{name}: mask must init to ones"

    out_full, _ = mod(q, k, v, prefixes)
    assert not torch.allclose(out_full, tok, atol=1e-5), (
        f"{name}: with active gates + default mask the prefix must change the output"
    )

    with torch.no_grad():
        mod._ablate_mask.zero_()
    out_off, _ = mod(q, k, v, prefixes)
    assert torch.allclose(out_off, tok, atol=1e-5), (
        f"{name}: full-mask ablation must collapse to token-only SDPA output"
    )

    with torch.no_grad():
        mod._ablate_mask.fill_(1.0)
        mod._ablate_mask[0, 0] = 0.0
    out_cell, _ = mod(q, k, v, prefixes)
    assert not torch.allclose(out_cell, out_full, atol=1e-6)
    assert not torch.allclose(out_cell, tok, atol=1e-6)

    sd = mod.state_dict()
    assert "_ablate_mask" not in sd, (
        f"{name}: _ablate_mask must be a non-persistent buffer (absent from state_dict)"
    )
    fresh = _make(cls)
    fresh.load_state_dict(sd, strict=True)
    assert torch.all(fresh._ablate_mask == 1.0), f"{name}: mask must default to ones after load"


def test_kv_prefix_ablate_mask():
    _check_variant(KVPrefixAttention)
