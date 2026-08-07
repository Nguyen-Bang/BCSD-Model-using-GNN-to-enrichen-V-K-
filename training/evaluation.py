"""Shared primitives for fixed-pair ranking evaluations.

This module intentionally has no CLI. Evaluation entry points import it so RNG
resetting, strict checkpoint loading, provenance, corruption, and statistics do
not drift between experiments.
"""

from __future__ import annotations

from contextlib import contextmanager
from functools import partial
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset.code_dataset import ReducedDataset
from dataset.collate import collate_gnn
from models.baseline.factory import create_model
from training.runtime import seed_everything, to_device

INTERVENTION_MODES = ("noise_global", "noise_perdim", "shuffle")


def reset_rng(seed: int) -> None:
    """Reset every RNG that can affect pair materialization or inference."""
    seed_everything(seed)


def resolve_precision(cfg: dict, device: torch.device):
    training = cfg["training"]
    if "precision" in training:
        precision = str(training["precision"]).lower()
    elif training.get("fp16", False):
        precision = "fp16"
    else:
        precision = "fp32"
    if precision not in {"fp16", "bf16", "fp32"}:
        raise ValueError(f"Unsupported precision: {precision}")
    use_amp = precision != "fp32" and device.type == "cuda"
    dtype = {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
    }[precision]
    return precision, use_amp, dtype


def build_model(cfg: dict, device: torch.device):
    """Construct the model used by the public GNN trainer."""
    return create_model(
        cfg,
        "baseline",
        device,
        backbone_checkpoint=None,
        initialize_embeddings=False,
    )


def load_checkpoint_strict(
    model,
    checkpoint_path: str | Path,
    device: torch.device,
    config: dict,
):
    """Load a trained state and reject same-shaped configuration drift."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    checkpoint_config = checkpoint.get("config")
    if checkpoint_config is not None:
        if not isinstance(checkpoint_config, dict):
            raise ValueError("Checkpoint config must be a mapping")
        for section in ("model", "gnn"):
            if checkpoint_config.get(section) != config.get(section):
                raise ValueError(f"Checkpoint {section} config does not match evaluation config")
    state = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state, strict=True)
    model.eval()
    return checkpoint


def gate_layers(model):
    return model.kv_prefix_layers


def reset_masks(layers) -> None:
    for layer in layers:
        layer._ablate_mask.fill_(1.0)


def disable_prefix_channel(layers) -> None:
    for layer in layers:
        layer._ablate_mask.zero_()


def build_split_loader(cfg: dict, split: str, batch_size: int):
    """Build a deterministic ranking loader for an explicit physical split."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    data_cfg = cfg["data"]
    split_dir = Path(data_cfg["data_dir"]) / split
    if not split_dir.is_dir():
        raise FileNotFoundError(
            f"Evaluation split does not exist: {split_dir}. "
            "Refusing ReducedDataset's fallback to a different pool."
        )
    gnn_cfg = cfg.get("gnn", {})
    dataset = ReducedDataset(
        data_dir=data_cfg["data_dir"],
        split=split,
        preload=data_cfg.get("preload", True),
        gnn_sidecar_dir=data_cfg.get("gnn_sidecar_dir"),
    )
    collate_fn = partial(
        collate_gnn,
        use_edge_features=bool(gnn_cfg.get("use_edge_features", False)),
        use_reverse_edges=bool(gnn_cfg.get("use_reverse_edges", False)),
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=True,
    )
    return loader, len(dataset)


def pool_metadata(loader, split: str) -> dict:
    dataset = loader.dataset
    group_keys = getattr(dataset, "_group_keys", ())
    projects = sorted({key[0] for key in group_keys})
    return {"split": split, "n": len(dataset), "projects": projects}


@torch.inference_mode()
def evaluate_ranks(model, loader, device, use_amp, amp_dtype, pair_seed: int | None = None):
    """Run full-pool ranking after resetting compilation-pair sampling."""
    if loader.num_workers != 0:
        raise ValueError("Fixed compilation pairs require DataLoader num_workers=0")
    if pair_seed is not None:
        reset_rng(pair_seed)
    model.eval()
    embeddings_a, embeddings_b = [], []
    for batch in loader:
        batch = to_device(batch, device)
        with torch.amp.autocast(device.type, enabled=use_amp, dtype=amp_dtype):
            emb_a, emb_b = model.get_embedding_pairs(batch)
        embeddings_a.append(emb_a.float())
        embeddings_b.append(emb_b.float())
    if not embeddings_a:
        raise RuntimeError("Evaluation loader was empty")
    matrix_a = torch.cat(embeddings_a)
    matrix_b = torch.cat(embeddings_b)
    similarities = matrix_a @ matrix_b.T
    ranks = (similarities >= similarities.diag().unsqueeze(1)).sum(dim=1)
    return ranks.to(torch.int32).cpu().numpy(), int(matrix_a.shape[0])


def metrics_from_ranks(ranks) -> dict:
    ranks = np.asarray(ranks)
    if ranks.ndim != 1 or ranks.size == 0 or np.any(ranks < 1):
        raise ValueError("ranks must be a non-empty one-dimensional array of positive integers")
    reciprocal = 1.0 / ranks.astype(np.float64)
    metrics = {"pool_size": int(ranks.size), "mrr": float(reciprocal.mean())}
    for k in (1, 5, 10):
        metrics[f"recall_at_{k}"] = float((ranks <= k).mean())
    return metrics


def paired_delta_stats(
    ranks_reference,
    ranks_intervention,
    n_boot: int,
    n_perm: int,
    seed: int,
    alpha: float = 0.05,
):
    reference = 1.0 / np.asarray(ranks_reference, dtype=np.float64)
    intervention = 1.0 / np.asarray(ranks_intervention, dtype=np.float64)
    if reference.shape != intervention.shape:
        raise ValueError("Paired rank vectors must have the same shape")
    delta = intervention - reference
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, delta.size, size=(n_boot, delta.size))
    bootstrap = delta[indices].mean(axis=1)
    lo, hi = np.percentile(bootstrap, [50 * alpha, 100 - 50 * alpha])
    signs = rng.choice(np.array([-1.0, 1.0]), size=(n_perm, delta.size))
    permuted = (signs * delta).mean(axis=1)
    observed = float(delta.mean())
    exceedances = int((np.abs(permuted) >= abs(observed) - 1e-12).sum())
    return {
        "delta_mrr": observed,
        "ci_lo": float(lo),
        "ci_hi": float(hi),
        "perm_p": float((exceedances + 1) / (n_perm + 1)),
    }


def deranged_permutation(size: int, generator, device=None):
    if size < 2:
        raise ValueError("A derangement requires a batch with at least two rows")
    identity = torch.arange(size, device=device)
    while True:
        permutation = torch.randperm(size, generator=generator, device=device)
        if not bool((permutation == identity).any()):
            return permutation


def make_prefix_corruptor(mode: str, seed: int):
    """Create a deterministic GNN-head hook for one supported intervention."""
    if mode not in INTERVENTION_MODES:
        raise ValueError(f"Unknown intervention mode {mode!r}; choose from {INTERVENTION_MODES}")
    state = {"call": 0}

    def hook(module, inputs, output):
        del module, inputs
        prefixes, rest = (output[0], output[1:]) if isinstance(output, tuple) else (output, None)
        batch_size = int(prefixes.shape[0])
        call = state["call"]
        state["call"] += 1
        generator = torch.Generator(device=prefixes.device)
        generator.manual_seed(seed * 1_000_003 + call * 9_973 + batch_size)
        if mode == "shuffle":
            corrupted = prefixes[deranged_permutation(batch_size, generator, prefixes.device)]
        else:
            if mode == "noise_global":
                mean = prefixes.mean()
                std = prefixes.std(unbiased=False)
            else:
                mean = prefixes.mean(dim=0, keepdim=True)
                std = prefixes.std(dim=0, keepdim=True, unbiased=False)
            noise = torch.randn(
                prefixes.shape,
                generator=generator,
                device=prefixes.device,
                dtype=prefixes.dtype,
            )
            corrupted = noise * std + mean
        return (corrupted,) + rest if rest is not None else corrupted

    return hook


@contextmanager
def corrupt_prefixes(model, mode: str, seed: int):
    handle = model.gnn_head.register_forward_hook(make_prefix_corruptor(mode, seed))
    try:
        yield
    finally:
        handle.remove()


def validate_intervention_batches(pool_size: int, batch_size: int, modes) -> None:
    batchwise = set(modes) & {"shuffle", "noise_perdim"}
    if batchwise and (batch_size < 2 or pool_size % batch_size != 0):
        raise ValueError(
            f"Modes {sorted(batchwise)} require batch_size > 1 and pool_size divisible by "
            f"batch_size; got n={pool_size}, batch_size={batch_size}"
        )


def aggregate(values) -> dict:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        raise ValueError("Cannot aggregate an empty sequence")
    return {
        "mean": float(array.mean()),
        "std": float(array.std(ddof=1)) if array.size > 1 else 0.0,
        "min": float(array.min()),
        "max": float(array.max()),
        "per_seed": [float(value) for value in array],
    }
