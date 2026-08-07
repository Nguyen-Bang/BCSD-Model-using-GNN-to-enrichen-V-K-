"""Known-answer tests for the evaluation statistics helpers."""

import numpy as np

from training.evaluation import (
    metrics_from_ranks,
    paired_delta_stats,
)


def test_metrics_known_answer():
    ranks = np.array([1, 1, 2, 4], dtype=np.int32)
    m = metrics_from_ranks(ranks)
    # MRR = mean(1, 1, 1/2, 1/4) = 2.75/4 = 0.6875
    assert abs(m["mrr"] - 0.6875) < 1e-9, m["mrr"]
    assert abs(m["recall_at_1"] - 0.5) < 1e-9  # 2 of 4 at rank 1
    assert abs(m["recall_at_5"] - 1.0) < 1e-9  # all <= 5
    assert m["pool_size"] == 4


def test_paired_delta_identical_is_null():
    rng = np.random.default_rng(2)
    ranks = rng.integers(1, 30, size=1500).astype(np.int32)
    s = paired_delta_stats(ranks, ranks.copy(), n_boot=2000, n_perm=2000, seed=0)
    assert abs(s["delta_mrr"]) < 1e-12, s["delta_mrr"]
    assert s["ci_lo"] <= 0.0 <= s["ci_hi"]
    assert s["perm_p"] > 0.5, f"identical inputs must look non-significant, got p={s['perm_p']}"


def test_paired_delta_systematic_drop_is_significant():
    rng = np.random.default_rng(4)
    ranks_full = rng.integers(1, 10, size=1500).astype(np.int32)
    ranks_abl = ranks_full + rng.integers(1, 5, size=1500).astype(np.int32)  # always worse
    s = paired_delta_stats(ranks_full, ranks_abl, n_boot=2000, n_perm=5000, seed=0)
    assert s["delta_mrr"] < 0, s["delta_mrr"]
    assert s["ci_hi"] < 0, f"CI should be entirely below 0, got [{s['ci_lo']},{s['ci_hi']}]"
    assert s["perm_p"] < 0.01, f"systematic drop must be significant, got p={s['perm_p']}"
