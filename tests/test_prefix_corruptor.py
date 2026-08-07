"""Tests for prefix-content corruption controls."""

import torch

from training.evaluation import (
    make_prefix_corruptor,
    validate_intervention_batches,
)

B, P, D = 8, 10, 256


def test_shuffle_is_derangement_same_values():
    prefixes = torch.randn(B, P, D)
    out = make_prefix_corruptor("shuffle", seed=0)(None, None, (prefixes, None))
    assert isinstance(out, tuple) and len(out) == 2, "tuple in -> tuple out"
    new = out[0]
    assert new.shape == prefixes.shape
    for i in range(B):
        matches = [(new[i] == prefixes[j]).all().item() for j in range(B)]
        assert any(matches), f"row {i} is not any original row"
        assert not matches[i], f"row {i} kept its own prefix (not a derangement)"
    assert torch.allclose(new.sum(), prefixes.sum())


def test_noise_replaces_values():
    prefixes = torch.randn(B, P, D) * 0.05 + 0.1
    new = make_prefix_corruptor("noise_global", seed=0)(None, None, (prefixes, None))[0]
    assert new.shape == prefixes.shape
    assert not torch.allclose(new.sum(), prefixes.sum(), atol=1e-3)
    assert 0.2 < (new.std() / prefixes.std()).item() < 5.0


def test_bare_tensor_passthrough_shape():
    prefixes = torch.randn(B, P, D)
    new = make_prefix_corruptor("shuffle", seed=1)(None, None, prefixes)
    assert torch.is_tensor(new) and new.shape == prefixes.shape


def test_determinism():
    a = make_prefix_corruptor("shuffle", seed=3)(
        None, None, (torch.arange(B * P * D).reshape(B, P, D).float(), None)
    )[0]
    b = make_prefix_corruptor("shuffle", seed=3)(
        None, None, (torch.arange(B * P * D).reshape(B, P, D).float(), None)
    )[0]
    assert torch.equal(a, b), "same seed + first call must be reproducible across runs"


def test_independent_across_consecutive_calls():
    prefixes = torch.randn(B, P, D)
    corr = make_prefix_corruptor("shuffle", seed=5)
    a = corr(None, None, (prefixes, None))[0]  # side a (call 0)
    b = corr(None, None, (prefixes, None))[0]  # side b (call 1)
    assert not torch.equal(a, b), "consecutive calls must use independent permutations"
    corr_n = make_prefix_corruptor("noise_global", seed=5)
    na = corr_n(None, None, (prefixes, None))[0]
    nb = corr_n(None, None, (prefixes, None))[0]
    assert not torch.allclose(na, nb), "consecutive noise draws must be independent"


def test_perdim_matches_batchwise_shape_and_is_finite():
    prefixes = torch.randn(B, P, D)
    new = make_prefix_corruptor("noise_perdim", seed=7)(None, None, (prefixes, None))[0]
    assert new.shape == prefixes.shape
    assert torch.isfinite(new).all()


def test_batchwise_modes_reject_remainder_batch():
    for mode in ("shuffle", "noise_perdim"):
        try:
            validate_intervention_batches(pool_size=1089, batch_size=64, modes=(mode,))
        except ValueError:
            pass
        else:
            raise AssertionError(f"{mode} accepted a singleton remainder batch")


def test_batchwise_modes_accept_divisible_batch():
    validate_intervention_batches(pool_size=1089, batch_size=33, modes=("shuffle", "noise_perdim"))
