"""Tests for modality dropout during collation."""

import torch

from dataset.collate import PAD_TOKEN_ID, apply_modality_dropout

FEAT_DIM = 132


def _fake_batch(batch_size=4, seq_len=16, real_len=None, device="cpu", with_features=True):
    """Build a minimal batch with input_ids_{a,b} + attention_mask_{a,b}."""
    if real_len is None:
        real_len = seq_len
    device = torch.device(device)
    # Use tokens far from PAD_TOKEN_ID so replacement is visible
    input_ids_a = torch.randint(100, 200, (batch_size, seq_len), dtype=torch.long, device=device)
    input_ids_b = torch.randint(100, 200, (batch_size, seq_len), dtype=torch.long, device=device)
    attention_mask_a = torch.zeros(batch_size, seq_len, dtype=torch.long, device=device)
    attention_mask_b = torch.zeros(batch_size, seq_len, dtype=torch.long, device=device)
    attention_mask_a[:, :real_len] = 1
    attention_mask_b[:, :real_len] = 1
    input_ids_a[:, real_len:] = PAD_TOKEN_ID
    input_ids_b[:, real_len:] = PAD_TOKEN_ID
    batch = {
        "input_ids_a": input_ids_a,
        "attention_mask_a": attention_mask_a,
        "input_ids_b": input_ids_b,
        "attention_mask_b": attention_mask_b,
    }
    if with_features:
        # Nonzero features so zeroing is observable
        batch["function_features_a"] = (
            torch.ones(batch_size, FEAT_DIM, dtype=torch.float32, device=device) * 0.5
        )
        batch["function_features_b"] = (
            torch.ones(batch_size, FEAT_DIM, dtype=torch.float32, device=device) * 0.7
        )
    return batch


def test_ratio_zero_is_noop():
    batch = _fake_batch()
    a_before = batch["input_ids_a"].clone()
    b_before = batch["input_ids_b"].clone()
    apply_modality_dropout(batch, ratio=0.0, side="a_only")
    assert torch.equal(batch["input_ids_a"], a_before), "ratio=0 must not change tokens"
    assert torch.equal(batch["input_ids_b"], b_before), "ratio=0 must not change tokens"


def test_a_only_does_not_touch_b():
    batch = _fake_batch(batch_size=8, seq_len=32)
    b_before = batch["input_ids_b"].clone()
    mask_b_before = batch["attention_mask_b"].clone()
    g = torch.Generator().manual_seed(0)
    apply_modality_dropout(batch, ratio=0.5, side="a_only", generator=g)
    assert torch.equal(batch["input_ids_b"], b_before), "side=a_only must not touch b"
    assert torch.equal(batch["attention_mask_b"], mask_b_before), (
        "side=a_only must not touch b mask"
    )


def test_dropped_positions_become_pad_with_zero_mask():
    """For masked positions: input_ids=PAD and attention_mask=0, jointly."""
    batch = _fake_batch(batch_size=4, seq_len=64)
    ids_before = batch["input_ids_a"].clone()
    mask_before = batch["attention_mask_a"].clone()
    g = torch.Generator().manual_seed(42)
    apply_modality_dropout(batch, ratio=0.5, side="a_only", generator=g)

    # Positions that were changed in the ids should now be PAD and have attn=0
    changed = ids_before != batch["input_ids_a"]
    assert (batch["input_ids_a"][changed] == PAD_TOKEN_ID).all(), "changed positions must be PAD"
    assert (batch["attention_mask_a"][changed] == 0).all(), "changed positions must have attn=0"

    # Positions the mask dropped (mask_before=1, attn now=0) == positions input changed
    dropped_by_mask = (mask_before == 1) & (batch["attention_mask_a"] == 0)
    assert torch.equal(dropped_by_mask, changed), "mask drop and id drop must agree"


def test_at_least_one_real_token_preserved():
    """Even at high ratio, every originally-nonempty row keeps ≥1 real token."""
    batch = _fake_batch(batch_size=16, seq_len=8, real_len=3)  # short sequences
    g = torch.Generator().manual_seed(1)
    apply_modality_dropout(batch, ratio=0.99, side="a_only", generator=g)
    remaining = batch["attention_mask_a"].sum(dim=1)
    assert (remaining >= 1).all(), f"every row must keep >=1 real token, got {remaining.tolist()}"


def test_empty_row_stays_empty():
    """Rows that had zero real tokens should not have anything 'rescued' out of padding."""
    batch = _fake_batch(batch_size=4, seq_len=16, real_len=0)
    g = torch.Generator().manual_seed(7)
    apply_modality_dropout(batch, ratio=0.5, side="a_only", generator=g)
    assert batch["attention_mask_a"].sum().item() == 0, "empty rows must stay empty"


def test_pad_positions_never_dropped():
    """Original pad positions (attn=0) must stay untouched."""
    batch = _fake_batch(batch_size=4, seq_len=32, real_len=10)
    pad_positions = batch["attention_mask_a"] == 0
    ids_at_pad = batch["input_ids_a"][pad_positions].clone()
    g = torch.Generator().manual_seed(3)
    apply_modality_dropout(batch, ratio=0.5, side="a_only", generator=g)
    assert torch.equal(batch["input_ids_a"][pad_positions], ids_at_pad), (
        "pad positions must not change"
    )
    assert (batch["attention_mask_a"][pad_positions] == 0).all(), (
        "pad positions must stay 0 in attention_mask"
    )


def test_ratio_approximately_respected():
    """Drop ratio, averaged over many real tokens, should match target."""
    batch = _fake_batch(batch_size=64, seq_len=128, real_len=128)
    total_real_before = batch["attention_mask_a"].sum().item()
    g = torch.Generator().manual_seed(11)
    apply_modality_dropout(batch, ratio=0.4, side="a_only", generator=g)
    total_real_after = batch["attention_mask_a"].sum().item()
    dropped = total_real_before - total_real_after
    empirical_ratio = dropped / total_real_before
    # Permissive tolerance — Bernoulli noise with n=64*128=8192 samples
    assert 0.35 < empirical_ratio < 0.45, (
        f"empirical ratio {empirical_ratio:.3f} far from target 0.4"
    )


def test_both_sides_drops_both():
    batch = _fake_batch(batch_size=8, seq_len=32)
    a_before = batch["attention_mask_a"].sum().item()
    b_before = batch["attention_mask_b"].sum().item()
    g = torch.Generator().manual_seed(99)
    apply_modality_dropout(batch, ratio=0.5, side="both", generator=g)
    a_after = batch["attention_mask_a"].sum().item()
    b_after = batch["attention_mask_b"].sum().item()
    assert a_after < a_before, "side=both must drop a"
    assert b_after < b_before, "side=both must drop b"


def test_invalid_side_raises():
    batch = _fake_batch()
    try:
        apply_modality_dropout(batch, ratio=0.5, side="bogus")
    except ValueError:
        return
    raise AssertionError("expected ValueError for unknown side")


def test_ratio_one_raises():
    batch = _fake_batch()
    try:
        apply_modality_dropout(batch, ratio=1.0, side="a_only")
    except ValueError:
        return
    raise AssertionError("expected ValueError for ratio=1.0")


def test_negative_and_nan_ratios_raise():
    for ratio in (-0.1, float("nan")):
        try:
            apply_modality_dropout(_fake_batch(), ratio=ratio, side="a_only")
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError for ratio={ratio}")


def test_function_features_dropped_on_touched_rows():
    """drop_function_features=True zeros features on rows that lost tokens."""
    batch = _fake_batch(batch_size=8, seq_len=32)
    g = torch.Generator().manual_seed(2)
    feats_before = batch["function_features_a"].clone()
    apply_modality_dropout(
        batch, ratio=0.5, side="a_only", generator=g, drop_function_features=True
    )
    # Any row with any changed token should have its features fully zeroed
    changed = (feats_before != batch["function_features_a"]).any(dim=1)
    feats_after = batch["function_features_a"]
    for i in changed.nonzero(as_tuple=True)[0].tolist():
        assert (feats_after[i] == 0.0).all(), f"row {i} features should be 0"
    # Side B features untouched
    assert (batch["function_features_b"] == 0.7).all(), "b features untouched"


def test_function_features_preserved_when_flag_off():
    batch = _fake_batch(batch_size=8, seq_len=32)
    g = torch.Generator().manual_seed(5)
    feats_a_before = batch["function_features_a"].clone()
    apply_modality_dropout(
        batch, ratio=0.5, side="a_only", generator=g, drop_function_features=False
    )
    assert torch.equal(batch["function_features_a"], feats_a_before), (
        "features must be untouched when drop_function_features=False"
    )


def test_missing_features_key_no_crash():
    """If batch doesn't carry function_features_{a,b}, dropout still works."""
    batch = _fake_batch(batch_size=4, seq_len=16, with_features=False)
    g = torch.Generator().manual_seed(8)
    apply_modality_dropout(
        batch, ratio=0.5, side="a_only", generator=g, drop_function_features=True
    )
    # No crash, and tokens still got dropped on side A
    assert batch["attention_mask_a"].sum() < batch["attention_mask_b"].sum()


def test_cuda_device_no_generator():
    """Regression test: dropout must work when batch is on CUDA."""
    if not torch.cuda.is_available():
        return
    batch = _fake_batch(batch_size=8, seq_len=32, device="cuda")
    apply_modality_dropout(batch, ratio=0.4, side="a_only")
    assert batch["input_ids_a"].is_cuda
    assert batch["attention_mask_a"].is_cuda
    assert batch["attention_mask_a"].sum() < 8 * 32, "some tokens must be dropped"


def test_cuda_device_with_cpu_generator():
    """CPU generator + CUDA batch: must not raise, must produce CUDA output."""
    if not torch.cuda.is_available():
        return
    batch = _fake_batch(batch_size=8, seq_len=32, device="cuda")
    g = torch.Generator(device="cpu").manual_seed(123)
    apply_modality_dropout(batch, ratio=0.4, side="a_only", generator=g)
    assert batch["input_ids_a"].is_cuda
    assert batch["attention_mask_a"].sum() < 8 * 32, "some tokens must be dropped"


def test_cuda_device_with_cuda_generator_rescue():
    """A CUDA generator must also work when single-token rows need rescue."""
    if not torch.cuda.is_available():
        return
    batch = _fake_batch(batch_size=64, seq_len=1, device="cuda")
    generator = torch.Generator(device="cuda").manual_seed(123)
    apply_modality_dropout(batch, ratio=0.99, side="a_only", generator=generator)
    assert (batch["attention_mask_a"].sum(dim=1) == 1).all()


def test_cuda_random_side():
    """side=random on CUDA stresses the row_mask device path."""
    if not torch.cuda.is_available():
        return
    batch = _fake_batch(batch_size=16, seq_len=32, device="cuda")
    a_before = batch["attention_mask_a"].sum().item()
    b_before = batch["attention_mask_b"].sum().item()
    apply_modality_dropout(batch, ratio=0.5, side="random")
    a_after = batch["attention_mask_a"].sum().item()
    b_after = batch["attention_mask_b"].sum().item()
    # Total real tokens must decrease overall; one or both sides got dropped
    assert (a_after + b_after) < (a_before + b_before)
