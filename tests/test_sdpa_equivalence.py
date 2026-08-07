"""Compare KV-prefix attention with a manual implementation."""

import torch

from models.structural_prefix import KVPrefixAttention


def manual_token_attention(query, key, value, attention_mask, scale_qk):
    """Reference implementation: the original manual softmax path that the
    SDPA refactor replaces. Used here as the equivalence target."""
    token_scores = torch.matmul(query, key.transpose(-2, -1)) / scale_qk
    if attention_mask is not None:
        mask_4d = attention_mask[:, None, None, :]
        token_scores = token_scores + (1.0 - mask_4d) * -1e4
    token_weights = torch.softmax(token_scores, dim=-1)
    return torch.matmul(token_weights, value)


def _make_inputs(B=2, S=64, P=10, num_heads=12, head_dim=64, seed=0):
    """Reproducible inputs in fp32 (so manual reference is the high-precision target)."""
    torch.manual_seed(seed)
    H = num_heads * head_dim
    Q = torch.randn(B, num_heads, S, head_dim)
    K = torch.randn(B, num_heads, S, head_dim)
    V = torch.randn(B, num_heads, S, head_dim)
    prefixes = torch.randn(B, P, 256)
    # mask: 90% of positions attended, last 10% padded
    mask = torch.ones(B, S)
    mask[:, int(S * 0.9) :] = 0.0
    return Q, K, V, prefixes, mask, H


def test_kv_prefix_attention_equivalence():
    """At zero initialization, SDPA must match token-only manual attention."""
    Q, K, V, prefixes, mask, H = _make_inputs(B=2, S=64, P=10, num_heads=12, head_dim=64)

    kv = KVPrefixAttention(
        hidden_size=H,
        num_heads=12,
        prefix_dim=256,
        num_prefix_tokens=10,
        dropout=0.0,
    )
    kv.eval()

    context_sdpa, weights = kv(Q, K, V, prefixes, attention_mask=mask)

    # Manual reference for the token-only path. With gate=0 (init), the prefix
    # contribution is zero, so context should equal the manual token output exactly.
    scale_qk = (H // 12) ** 0.5
    context_manual = manual_token_attention(Q, K, V, mask, scale_qk)

    diff = (context_sdpa - context_manual).abs().max().item()
    assert diff < 1e-5, f"KVPrefix SDPA mismatch: max abs diff = {diff:.2e}"
    assert weights is None, f"expected weights=None, got {type(weights)}"


def test_equivalence_with_open_gates():
    """The interesting case: gates non-zero, so the prefix path contributes.
    Verify SDPA on the token branch + manual on the prefix branch sums to the
    same total as a fully-manual reference."""
    Q, K, V, prefixes, mask, H = _make_inputs(B=2, S=64, P=10, num_heads=12, head_dim=64)
    num_heads, head_dim = 12, 64
    scale_qk = head_dim**0.5

    kv = KVPrefixAttention(
        hidden_size=H,
        num_heads=num_heads,
        prefix_dim=256,
        num_prefix_tokens=10,
        dropout=0.0,
    )
    kv.eval()
    # Force weights away from zero so the prefix branch contributes
    with torch.no_grad():
        kv.weight.normal_(mean=0.5, std=0.3)

    context_actual, _ = kv(Q, K, V, prefixes, attention_mask=mask)

    # Use manual softmax instead of SDPA on the token path.
    token_ref = manual_token_attention(Q, K, V, mask, scale_qk)

    prefix_k = kv.prefix_k_norm(kv.prefix_to_k(prefixes))
    prefix_v = kv.prefix_v_norm(kv.prefix_to_v(prefixes))
    B = Q.size(0)
    prefix_k = prefix_k.view(B, -1, num_heads, head_dim).transpose(1, 2)
    prefix_v = prefix_v.view(B, -1, num_heads, head_dim).transpose(1, 2)
    prefix_scores = torch.matmul(Q, prefix_k.transpose(-2, -1)) / scale_qk
    prefix_weights_attn = torch.sigmoid(prefix_scores)
    gs = torch.tanh(kv.weight).view(1, num_heads, 1, 10)
    prefix_weights_attn = prefix_weights_attn * gs
    prefix_out_ref = torch.matmul(prefix_weights_attn, prefix_v)

    context_ref = token_ref + prefix_out_ref
    diff = (context_actual - context_ref).abs().max().item()
    assert diff < 1e-5, f"Open-gate SDPA mismatch: max abs diff = {diff:.2e}"


def test_dropout_inactive_in_eval():
    """In eval mode, dropout_p=0 in SDPA — outputs must be deterministic
    across two calls with the same input."""
    Q, K, V, prefixes, mask, H = _make_inputs(B=2, S=64, P=10, num_heads=12, head_dim=64)
    kv = KVPrefixAttention(
        hidden_size=H,
        num_heads=12,
        prefix_dim=256,
        num_prefix_tokens=10,
        dropout=0.5,
    )
    kv.eval()

    out1, _ = kv(Q, K, V, prefixes, attention_mask=mask)
    out2, _ = kv(Q, K, V, prefixes, attention_mask=mask)

    diff = (out1 - out2).abs().max().item()
    assert diff == 0.0, f"eval mode produced non-deterministic output: {diff}"


def test_no_mask_path():
    """attention_mask=None must work (used at inference and some eval paths)."""
    Q, K, V, prefixes, mask, H = _make_inputs(B=2, S=64, P=10, num_heads=12, head_dim=64)
    kv = KVPrefixAttention(
        hidden_size=H,
        num_heads=12,
        prefix_dim=256,
        num_prefix_tokens=10,
        dropout=0.0,
    )
    kv.eval()
    context, weights = kv(Q, K, V, prefixes, attention_mask=None)

    # Reference with no mask
    scale_qk = 64**0.5
    token_scores = (Q @ K.transpose(-2, -1)) / scale_qk
    token_ref = torch.softmax(token_scores, dim=-1) @ V

    diff = (context - token_ref).abs().max().item()
    assert diff < 1e-5, f"no-mask SDPA mismatch: {diff}"
    assert weights is None
