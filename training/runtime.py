"""Shared runtime utilities for single-process training and ranking evaluation."""

from __future__ import annotations

import math
import random
from collections.abc import Mapping
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from dataset.code_dataset import ReducedDataset
from dataset.collate import collate_gnn
from dataset.sampler import SqrtBalancedSampler
from training.metrics import compute_similarity_accuracy


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open() as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Expected a mapping in config file: {path}")
    return config


def resolve_device(device: str | None = None) -> torch.device:
    return torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def capture_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: Mapping[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def optimizer_steps_per_epoch(num_batches: int, accumulation_steps: int) -> int:
    if accumulation_steps < 1:
        raise ValueError("accumulation_steps must be at least 1")
    return math.ceil(num_batches / accumulation_steps)


def accumulation_window_size(
    step: int,
    num_batches: int,
    accumulation_steps: int,
) -> int:
    """Return the divisor for this batch's complete or final partial window."""
    window_start = (step // accumulation_steps) * accumulation_steps
    return min(accumulation_steps, num_batches - window_start)


def to_device(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def create_gnn_dataloaders(
    config: Mapping[str, Any],
    *,
    seed: int,
) -> tuple[DataLoader, DataLoader, SqrtBalancedSampler]:
    data_config = config["data"]
    training_config = config["training"]
    gnn_config = config.get("gnn", {})
    collate_fn = partial(
        collate_gnn,
        use_edge_features=bool(gnn_config.get("use_edge_features", False)),
        use_reverse_edges=bool(gnn_config.get("use_reverse_edges", False)),
    )

    dataset_kwargs = {
        "data_dir": data_config["data_dir"],
        "preload": data_config.get("preload", True),
        "gnn_sidecar_dir": data_config.get("gnn_sidecar_dir"),
    }
    train_dataset = ReducedDataset(split="train", **dataset_kwargs)
    validation_dataset = ReducedDataset(split="validation", **dataset_kwargs)
    sampler = SqrtBalancedSampler(
        train_dataset,
        scale_factor=float(data_config.get("sampling_scale_factor", 50)),
        seed=seed,
    )
    loader_kwargs = {
        "batch_size": int(training_config["batch_size"]),
        "collate_fn": collate_fn,
        "num_workers": int(data_config.get("num_workers", 0)),
        "pin_memory": torch.cuda.is_available(),
    }
    train_loader = DataLoader(
        train_dataset,
        sampler=sampler,
        drop_last=True,
        **loader_kwargs,
    )
    validation_loader = DataLoader(
        validation_dataset,
        shuffle=False,
        **loader_kwargs,
    )
    return train_loader, validation_loader, sampler


@torch.no_grad()
def validate_pairs(
    model: torch.nn.Module,
    loader: DataLoader,
    loss_fn: torch.nn.Module,
    device: torch.device,
    *,
    pair_seed: int | None = None,
    use_amp: bool = False,
    amp_dtype: torch.dtype = torch.float16,
) -> tuple[float, float]:
    if pair_seed is not None and getattr(loader, "num_workers", 0) != 0:
        raise ValueError("Deterministic pair evaluation requires num_workers=0")
    previous_rng_state = capture_rng_state() if pair_seed is not None else None
    if pair_seed is not None:
        seed_everything(pair_seed)
    try:
        model.eval()
        total_loss = 0.0
        similarities: list[float] = []
        for batch in loader:
            batch = to_device(batch, device)
            with torch.amp.autocast(device.type, enabled=use_amp, dtype=amp_dtype):
                embeddings_a, embeddings_b = model.get_embedding_pairs(batch)
                loss = loss_fn(embeddings_a, embeddings_b)
            total_loss += loss.item()
            similarities.append(
                compute_similarity_accuracy(embeddings_a, embeddings_b)["mean_positive_similarity"]
            )
    finally:
        if previous_rng_state is not None:
            restore_rng_state(previous_rng_state)
    count = len(similarities)
    if count == 0:
        raise ValueError("Cannot validate an empty data loader")
    return total_loss / count, sum(similarities) / count


def ranks_from_similarity(similarity: torch.Tensor) -> torch.Tensor:
    """Use the historical pessimistic policy: ties share the worst tied rank."""
    if similarity.ndim != 2 or similarity.shape[0] != similarity.shape[1]:
        raise ValueError("Expected a square query-candidate similarity matrix")
    positives = similarity.diag().unsqueeze(1)
    return (similarity >= positives).sum(dim=1)


def ranks_from_embeddings(
    queries: torch.Tensor,
    candidates: torch.Tensor,
    *,
    block_size: int = 4096,
) -> torch.Tensor:
    """Compute pessimistic paired ranks without materializing the full matrix."""
    if queries.ndim != 2 or candidates.ndim != 2 or queries.shape != candidates.shape:
        raise ValueError("Expected equally shaped 2D query and candidate embeddings")
    if queries.device != candidates.device:
        raise ValueError("Query and candidate embeddings must be on the same device")
    if block_size <= 0:
        raise ValueError("block_size must be positive")

    pool_size = queries.shape[0]
    ranks = torch.zeros(pool_size, dtype=torch.long, device=queries.device)
    for query_start in range(0, pool_size, block_size):
        query_stop = min(query_start + block_size, pool_size)
        query_block = queries[query_start:query_stop]
        matching_similarity = torch.mm(
            query_block,
            candidates[query_start:query_stop].T,
        )
        positive_similarity = matching_similarity.diag().unsqueeze(1)
        block_ranks = torch.zeros(
            query_stop - query_start,
            dtype=torch.long,
            device=queries.device,
        )

        for candidate_start in range(0, pool_size, block_size):
            candidate_stop = min(candidate_start + block_size, pool_size)
            if candidate_start == query_start:
                similarity = matching_similarity
            else:
                similarity = torch.mm(
                    query_block,
                    candidates[candidate_start:candidate_stop].T,
                )
            block_ranks += (similarity >= positive_similarity).sum(dim=1)
        ranks[query_start:query_stop] = block_ranks
    return ranks


@torch.no_grad()
def evaluate_ranking(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    split: str,
    pair_seed: int,
    use_amp: bool = False,
    amp_dtype: torch.dtype = torch.float16,
    ranking_block_size: int = 4096,
) -> dict[str, Any]:
    """Evaluate one deterministic set of sampled compilation pairs."""
    if getattr(loader, "num_workers", 0) != 0:
        raise ValueError("Deterministic pair evaluation requires num_workers=0")
    previous_rng_state = capture_rng_state()
    seed_everything(pair_seed)
    try:
        model.eval()
        embeddings_a: list[torch.Tensor] = []
        embeddings_b: list[torch.Tensor] = []
        for batch in loader:
            batch = to_device(batch, device)
            with torch.amp.autocast(device.type, enabled=use_amp, dtype=amp_dtype):
                batch_a, batch_b = model.get_embedding_pairs(batch)
            embeddings_a.append(batch_a)
            embeddings_b.append(batch_b)
    finally:
        restore_rng_state(previous_rng_state)

    if not embeddings_a:
        raise ValueError("Cannot rank an empty data loader")
    all_a = torch.cat(embeddings_a)
    all_b = torch.cat(embeddings_b)
    ranks = ranks_from_embeddings(
        all_a,
        all_b,
        block_size=ranking_block_size,
    ).float()
    result: dict[str, Any] = {
        "split": split,
        "pool_size": int(all_a.shape[0]),
        "pair_seed": pair_seed,
        "tie_policy": "pessimistic",
        "mrr": (1.0 / ranks).mean().item(),
    }
    for cutoff in (1, 5, 10, 20):
        result[f"recall_at_{cutoff}"] = (ranks <= cutoff).float().mean().item()
    return result


def checkpoint_state(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: torch.amp.GradScaler | None,
    epoch: int,
    config: Mapping[str, Any],
    seed: int,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    state: dict[str, Any] = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
        "rng_state": capture_rng_state(),
        "config": dict(config),
        "seed": seed,
    }
    if extra:
        state.update(extra)
    return state


def save_checkpoint(path: str | Path, **state_kwargs: Any) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint_state(**state_kwargs), output_path)


def load_training_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: torch.amp.GradScaler | None,
    device: torch.device,
    config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    required = {
        "epoch",
        "model_state_dict",
        "optimizer_state_dict",
        "scheduler_state_dict",
        "scaler_state_dict",
        "rng_state",
        "config",
        "seed",
    }
    missing = required.difference(checkpoint)
    if missing:
        raise KeyError(f"Resume checkpoint is missing fields: {sorted(missing)}")
    if config is not None and checkpoint["config"] != dict(config):
        raise ValueError("Resume config does not match checkpoint config")
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler is not None:
        if checkpoint["scheduler_state_dict"] is None:
            raise KeyError("Resume checkpoint has no scheduler state")
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    elif checkpoint["scheduler_state_dict"] is not None:
        raise ValueError("Checkpoint has scheduler state but this run has no scheduler")
    if scaler is not None:
        if checkpoint["scaler_state_dict"] is None:
            raise KeyError("Resume checkpoint has no GradScaler state")
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
    elif checkpoint["scaler_state_dict"] is not None:
        raise ValueError("Checkpoint has GradScaler state but this run has no GradScaler")
    restore_rng_state(checkpoint["rng_state"])
    return checkpoint
