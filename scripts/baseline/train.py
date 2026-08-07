#!/usr/bin/env python3
"""
Two-stage single-device training script for the baseline model.

Stage 1 (MLM Pretrain): Pure masked language modeling.
Stage 2 (Joint Fine-tune): MLM + Contrastive with differential LR.

Run with:
    python -m scripts.baseline.train --config configs/baseline_config.yaml

Module: scripts.baseline.train
"""

import argparse
import csv
import logging
import time
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from dataset.code_dataset import ReducedDataset
from dataset.collate import collate_baseline
from dataset.sampler import SqrtBalancedSampler
from models.baseline import BaselineModel, BaselineWithFeaturesModel
from training.losses import JointLoss
from training.metrics import compute_similarity_accuracy
from training.runtime import (
    accumulation_window_size,
    capture_rng_state,
    optimizer_steps_per_epoch,
    ranks_from_similarity,
    resolve_device,
    restore_rng_state,
    seed_everything,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("bcsd.train_baseline")


# ── Config / Model / Data ───────────────────────────────────────────────


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def resolve_output_dir(
    config: dict,
    *,
    seed: int,
    output_dir: str | Path | None = None,
) -> Path:
    """Resolve the directory for one baseline training run."""
    if output_dir is not None:
        return Path(output_dir)
    base_dir = Path(config["training"].get("checkpoint_dir", "checkpoints/baseline_with_features"))
    return base_dir / f"seed_{seed}"


def prepare_output_dir(output_dir: Path) -> None:
    """Create a fresh run directory without replacing prior training output."""
    artifacts = (
        "training_log.csv",
        "best_mlm.pt",
        "mlm_final.pt",
        "best_model.pt",
    )
    existing = [name for name in artifacts if (output_dir / name).exists()]
    existing.extend(path.name for path in output_dir.glob("checkpoint_s2_epoch_*.pt"))
    if existing:
        raise FileExistsError(f"Output directory already contains training artifacts: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)


def create_dataloaders(cfg: dict, seed: int):
    data_cfg = cfg["data"]
    train_cfg = cfg["training"]

    train_ds = ReducedDataset(
        data_dir=data_cfg["data_dir"],
        split="train",
        preload=data_cfg.get("preload", True),
    )
    val_ds = ReducedDataset(
        data_dir=data_cfg["data_dir"],
        split="validation",
        preload=data_cfg.get("preload", True),
    )

    scale_factor = data_cfg.get("sampling_scale_factor", 50)
    train_sampler = SqrtBalancedSampler(
        train_ds,
        scale_factor=scale_factor,
        seed=seed,
    )
    logger.info(f"Balanced sampler:\n{train_sampler.summary()}")

    train_loader = DataLoader(
        train_ds,
        batch_size=train_cfg["batch_size"],
        sampler=train_sampler,
        collate_fn=collate_baseline,
        num_workers=data_cfg.get("num_workers", 0),
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=train_cfg["batch_size"],
        shuffle=False,
        collate_fn=collate_baseline,
        num_workers=data_cfg.get("num_workers", 0),
        pin_memory=True,
    )

    return train_loader, val_loader, train_sampler


def create_model(cfg: dict, device: torch.device) -> BaselineModel:
    model_cfg = cfg["model"]
    use_features = model_cfg.get("use_function_features", True)

    base_kwargs = dict(
        num_layers=model_cfg.get("num_layers", 3),
        hidden_size=model_cfg.get("hidden_size", 768),
        num_heads=model_cfg.get("num_heads", 12),
        vocab_size=model_cfg.get("vocab_size", 33555),
        max_seq_length=model_cfg.get("max_seq_length", 1024),
        embed_dim=model_cfg.get("embed_dim", 768),
        dropout=model_cfg.get("dropout", 0.1),
        clap_embedding_init=model_cfg.get("clap_embedding_init", True),
    )

    if use_features:
        model = BaselineWithFeaturesModel(
            func_feat_dim=model_cfg.get("func_feature_dim", 132),
            **base_kwargs,
        )
    else:
        model = BaselineModel(**base_kwargs)

    return model.to(device)


# ── Training loops ───────────────────────────────────────────────────────


def _to_device(batch, device):
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}


def _model_kwargs(batch, suffix, use_features, return_logits):
    """Build kwargs dict for model forward call."""
    input_key = (
        f"masked_input_ids_{suffix}"
        if f"masked_input_ids_{suffix}" in batch
        else f"input_ids_{suffix}"
    )
    kwargs = dict(
        input_ids=batch[input_key],
        attention_mask=batch[f"attention_mask_{suffix}"],
        token_type_ids=batch.get(f"token_type_ids_{suffix}"),
        return_logits=return_logits,
    )
    if use_features:
        kwargs["function_features"] = batch.get(f"function_features_{suffix}")
    return kwargs


def _optimizer_step(model, optimizer, scheduler, scaler, grad_clip):
    optimizer_updated = True
    if scaler is None:
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
    else:
        scaler.unscale_(optimizer)
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        previous_scale = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        optimizer_updated = scaler.get_scale() >= previous_scale
    optimizer.zero_grad(set_to_none=True)
    if scheduler is not None and optimizer_updated:
        scheduler.step()


def _unique_parameters(*modules):
    seen = set()
    parameters = []
    for module in modules:
        for parameter in module.parameters():
            if id(parameter) not in seen:
                seen.add(id(parameter))
                parameters.append(parameter)
    return parameters


def train_epoch_mlm(
    model,
    loader,
    loss_fn,
    optimizer,
    scheduler,
    device,
    grad_clip,
    scaler=None,
    grad_accum_steps=1,
    use_features=True,
):
    """Stage 1: MLM-only (one branch per step)."""
    model.train()
    total_loss = 0.0
    num_batches = 0
    use_amp = scaler is not None

    optimizer.zero_grad()
    loader_steps = len(loader)

    for step, batch in enumerate(loader):
        batch = _to_device(batch, device)

        with torch.amp.autocast("cuda", enabled=use_amp):
            _, logits_a = model(**_model_kwargs(batch, "a", use_features, return_logits=True))
            loss = loss_fn(mlm_logits=logits_a, mlm_labels=batch["mlm_labels_a"])
            window_size = accumulation_window_size(step, loader_steps, grad_accum_steps)
            loss = loss / window_size

        if use_amp:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if (step + 1) % grad_accum_steps == 0 or (step + 1) == len(loader):
            _optimizer_step(model, optimizer, scheduler, scaler, grad_clip)

        total_loss += loss.item() * window_size
        num_batches += 1

    return total_loss / max(num_batches, 1)


def train_epoch_joint(
    model,
    loader,
    loss_fn,
    optimizer,
    scheduler,
    device,
    grad_clip,
    scaler=None,
    grad_accum_steps=1,
    use_features=True,
):
    """Stage 2: Joint MLM + Contrastive. Returns (total, mlm, contrastive)."""
    model.train()
    total_loss = 0.0
    total_mlm = 0.0
    total_contra = 0.0
    num_batches = 0
    use_amp = scaler is not None

    optimizer.zero_grad()
    loader_steps = len(loader)

    for step, batch in enumerate(loader):
        batch = _to_device(batch, device)

        with torch.amp.autocast("cuda", enabled=use_amp):
            emb_a, logits_a = model(**_model_kwargs(batch, "a", use_features, return_logits=True))
            emb_b = model(**_model_kwargs(batch, "b", use_features, return_logits=False))
            loss, mlm_val, contra_val = loss_fn(
                mlm_logits=logits_a,
                mlm_labels=batch["mlm_labels_a"],
                embeddings_a=emb_a,
                embeddings_b=emb_b,
                return_components=True,
            )
            window_size = accumulation_window_size(step, loader_steps, grad_accum_steps)
            loss = loss / window_size

        if use_amp:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if (step + 1) % grad_accum_steps == 0 or (step + 1) == len(loader):
            _optimizer_step(model, optimizer, scheduler, scaler, grad_clip)

        total_loss += loss.item() * window_size
        total_mlm += mlm_val.item() if torch.is_tensor(mlm_val) else mlm_val
        total_contra += contra_val.item() if torch.is_tensor(contra_val) else contra_val
        num_batches += 1

    n = max(num_batches, 1)
    return total_loss / n, total_mlm / n, total_contra / n


def train_epoch_joint_gradcache(
    model,
    loader,
    loss_fn,
    optimizer,
    scheduler,
    device,
    grad_clip,
    scaler=None,
    grad_accum_steps=1,
    use_features=True,
):
    """Stage 2 with GradCache: full-batch contrastive over accumulated mini-batches.

    Three-phase loop per accumulation window:
      Phase 1: Forward all chunks without grad, cache embeddings.
      Phase 2: Compute InfoNCE over all cached embeddings.
      Phase 3: Replay each chunk with grad for MLM + contrastive backward.
    """
    model.train()
    total_loss = 0.0
    total_mlm = 0.0
    total_contra = 0.0
    num_windows = 0
    use_amp = scaler is not None

    optimizer.zero_grad()

    buffer = []
    for step, batch in enumerate(loader):
        buffer.append(_to_device(batch, device))

        if len(buffer) < grad_accum_steps and (step + 1) < len(loader):
            continue

        # ── Phase 1: Cache embeddings (no grad) ─────────────────────
        cached_embs_a = []
        cached_embs_b = []
        forward_rng_states = []
        with torch.no_grad():
            for chunk in buffer:
                rng_a = capture_rng_state()
                with torch.amp.autocast("cuda", enabled=use_amp):
                    emb_a = model(**_model_kwargs(chunk, "a", use_features, return_logits=False))
                rng_b = capture_rng_state()
                with torch.amp.autocast("cuda", enabled=use_amp):
                    emb_b = model(**_model_kwargs(chunk, "b", use_features, return_logits=False))
                cached_embs_a.append(emb_a)
                cached_embs_b.append(emb_b)
                forward_rng_states.append((rng_a, rng_b))

        # ── Phase 2: Full-batch contrastive loss ────────────────────
        all_emb_a = torch.cat(cached_embs_a, dim=0).requires_grad_(True)
        all_emb_b = torch.cat(cached_embs_b, dim=0).requires_grad_(True)

        with torch.amp.autocast("cuda", enabled=use_amp):
            contra_loss = loss_fn.contrastive_loss(all_emb_a, all_emb_b)
            scaled_contra = loss_fn.lambda_contrastive * contra_loss

        if use_amp:
            scaler.scale(scaled_contra).backward()
        else:
            scaled_contra.backward()

        # Per-chunk embedding gradients
        grad_chunks_a = all_emb_a.grad.split([e.shape[0] for e in cached_embs_a])
        grad_chunks_b = all_emb_b.grad.split([e.shape[0] for e in cached_embs_b])

        # ── Phase 3: Chunked backward (replay with grad) ───────────
        window_mlm = 0.0
        n_chunks = len(buffer)
        for i, chunk in enumerate(buffer):
            rng_a, rng_b = forward_rng_states[i]
            restore_rng_state(rng_a)
            with torch.amp.autocast("cuda", enabled=use_amp):
                emb_a, logits_a = model(
                    **_model_kwargs(chunk, "a", use_features, return_logits=True)
                )
            restore_rng_state(rng_b)
            with torch.amp.autocast("cuda", enabled=use_amp):
                emb_b = model(**_model_kwargs(chunk, "b", use_features, return_logits=False))
            mlm_loss = loss_fn.mlm_loss(logits_a, chunk["mlm_labels_a"])
            mlm_scaled = mlm_loss / n_chunks

            # Backward MLM (retain graph — need emb_a graph for contrastive grad)
            if use_amp:
                scaler.scale(mlm_scaled).backward(retain_graph=True)
            else:
                mlm_scaled.backward(retain_graph=True)

            # Backward contrastive gradients through model
            emb_a.backward(grad_chunks_a[i])
            emb_b.backward(grad_chunks_b[i])

            window_mlm += mlm_loss.item()

        # ── Optimizer step ──────────────────────────────────────────
        _optimizer_step(model, optimizer, scheduler, scaler, grad_clip)

        # Logging accumulators
        total_contra += contra_loss.item()
        total_mlm += window_mlm / n_chunks
        total_loss += (window_mlm / n_chunks) + loss_fn.lambda_contrastive * contra_loss.item()
        num_windows += 1

        buffer = []

    n = max(num_windows, 1)
    return total_loss / n, total_mlm / n, total_contra / n


@torch.no_grad()
def validate_joint(
    model,
    loader,
    loss_fn,
    device,
    *,
    pair_seed,
    use_amp=False,
    use_features=True,
):
    """Returns (total_loss, mlm_loss, contrastive_loss, avg_sim)."""
    if getattr(loader, "num_workers", 0) != 0:
        raise ValueError("Deterministic pair evaluation requires num_workers=0")
    rng_state = capture_rng_state()
    seed_everything(pair_seed)
    try:
        model.eval()
        total_loss = 0.0
        total_mlm = 0.0
        total_contra = 0.0
        all_sims = []
        num_batches = 0
        for batch in loader:
            batch = _to_device(batch, device)

            with torch.amp.autocast("cuda", enabled=use_amp):
                emb_a, logits_a = model(
                    **_model_kwargs(batch, "a", use_features, return_logits=True)
                )
                emb_b = model(**_model_kwargs(batch, "b", use_features, return_logits=False))
                loss, mlm_val, contra_val = loss_fn(
                    mlm_logits=logits_a,
                    mlm_labels=batch["mlm_labels_a"],
                    embeddings_a=emb_a,
                    embeddings_b=emb_b,
                    return_components=True,
                )

            total_loss += loss.item()
            total_mlm += mlm_val.item() if torch.is_tensor(mlm_val) else mlm_val
            total_contra += contra_val.item() if torch.is_tensor(contra_val) else contra_val
            num_batches += 1
            metrics = compute_similarity_accuracy(emb_a, emb_b)
            all_sims.append(metrics["mean_positive_similarity"])
    finally:
        restore_rng_state(rng_state)

    n = max(num_batches, 1)
    avg_sim = sum(all_sims) / max(len(all_sims), 1)
    return total_loss / n, total_mlm / n, total_contra / n, avg_sim


@torch.no_grad()
def evaluate_ranking(
    model,
    loader,
    device,
    *,
    pair_seed,
    split="validation",
    use_amp=False,
    use_features=True,
):
    """
    Collect all validation embeddings and compute ranking metrics.

    For each function pair (emb_a_i, emb_b_i), uses emb_a as query and
    emb_b as candidate pool. Measures how often the correct emb_b is
    ranked first among all candidates.

    Returns dict with MRR, Recall@1/5/10, and pool size.
    """
    if getattr(loader, "num_workers", 0) != 0:
        raise ValueError("Deterministic pair evaluation requires num_workers=0")
    rng_state = capture_rng_state()
    seed_everything(pair_seed)
    try:
        model.eval()
        all_emb_a = []
        all_emb_b = []
        for batch in loader:
            batch = _to_device(batch, device)
            with torch.amp.autocast("cuda", enabled=use_amp):
                emb_a = model(**_model_kwargs(batch, "a", use_features, return_logits=False))
                emb_b = model(**_model_kwargs(batch, "b", use_features, return_logits=False))
            all_emb_a.append(emb_a)
            all_emb_b.append(emb_b)
    finally:
        restore_rng_state(rng_state)

    all_emb_a = torch.cat(all_emb_a, dim=0)  # [N, dim]
    all_emb_b = torch.cat(all_emb_b, dim=0)  # [N, dim]
    n = all_emb_a.size(0)

    sim = torch.mm(all_emb_a, all_emb_b.t())  # already L2-normalized
    ranks = ranks_from_similarity(sim).float()

    mrr = (1.0 / ranks).mean().item()
    results = {
        "pool_size": n,
        "split": split,
        "pair_seed": pair_seed,
        "tie_policy": "pessimistic",
        "mrr": mrr,
    }
    for k in [1, 5, 10]:
        results[f"recall_at_{k}"] = (ranks <= k).float().mean().item()

    return results


def save_checkpoint(
    model,
    optimizer,
    scheduler,
    scaler,
    epoch,
    val_loss,
    path,
    *,
    stage,
    config,
    seed,
    pair_seed,
):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "stage": stage,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
            "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
            "rng_state": capture_rng_state(),
            "config": config,
            "seed": seed,
            "pair_seed": pair_seed,
            "val_loss": val_loss,
        },
        path,
    )


# ── Main ─────────────────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train baseline BCSD model")
    parser.add_argument("--config", default="configs/baseline_config.yaml")
    parser.add_argument("--output-dir")
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--pair-seed", type=int, default=0)
    return parser


def main(argv: list[str] | None = None):
    args = build_parser().parse_args(argv)

    device = resolve_device(args.device)
    seed_everything(args.seed)

    cfg = load_config(args.config)
    train_cfg = cfg["training"]
    ckpt_dir = resolve_output_dir(cfg, seed=args.seed, output_dir=args.output_dir)
    prepare_output_dir(ckpt_dir)

    use_features = cfg["model"].get("use_function_features", True)

    logger.info(f"Device: {device}, seed={args.seed}, pair_seed={args.pair_seed}")
    logger.info(f"Function features: {'enabled' if use_features else 'disabled'}")

    logger.info("Creating dataloaders...")
    train_loader, val_loader, train_sampler = create_dataloaders(cfg, args.seed)
    logger.info(
        f"Train: {len(train_loader.dataset)} groups "
        f"({len(train_loader)} batches/epoch), "
        f"Val: {len(val_loader.dataset)} groups"
    )

    logger.info("Creating model...")
    model = create_model(cfg, device)

    use_fp16 = train_cfg.get("fp16", False) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_fp16 else None
    grad_clip = train_cfg.get("gradient_clip", 1.0)
    grad_accum = train_cfg.get("grad_accum_steps", 1)
    eff_batch = train_cfg["batch_size"] * grad_accum
    logger.info(
        f"FP16: {use_fp16}, batch_size: {train_cfg['batch_size']}, "
        f"grad_accum: {grad_accum}, effective batch: {eff_batch}, "
        f"GradCache: {train_cfg.get('use_gradcache', False)}"
    )

    log_path = ckpt_dir / "training_log.csv"

    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(
            [
                "stage",
                "epoch",
                "train_loss",
                "train_mlm",
                "train_contrastive",
                "val_loss",
                "val_mlm",
                "val_contrastive",
                "val_sim",
                "lr",
                "time_s",
            ]
        )

    # ═════════════════════════════════════════════════════════════════════
    # Stage 1: MLM Pretrain
    # ═════════════════════════════════════════════════════════════════════
    mlm_epochs = train_cfg.get("mlm_pretrain_epochs", 10)
    mlm_lr = train_cfg.get("mlm_learning_rate", 5e-5)
    temperature = train_cfg.get("temperature", 0.07)

    loss_fn = JointLoss(lambda_contrastive=0.0, temperature=temperature)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=mlm_lr,
        weight_decay=train_cfg.get("weight_decay", 0.01),
    )

    warmup_steps = train_cfg.get("warmup_steps", 100)
    steps_per_epoch = optimizer_steps_per_epoch(len(train_loader), grad_accum)
    total_steps_s1 = steps_per_epoch * mlm_epochs
    scheduler = None
    if warmup_steps > 0:
        from transformers import get_linear_schedule_with_warmup

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps_s1,
        )

    logger.info(f"=== Stage 1: MLM Pretrain ({mlm_epochs} epochs, lr={mlm_lr}) ===")

    best_mlm_loss = float("inf")
    for epoch in range(1, mlm_epochs + 1):
        train_sampler.set_epoch(epoch)
        t0 = time.time()

        train_loss = train_epoch_mlm(
            model,
            train_loader,
            loss_fn,
            optimizer,
            scheduler,
            device,
            grad_clip,
            scaler=scaler,
            grad_accum_steps=grad_accum,
            use_features=use_features,
        )
        val_loss, val_mlm, _, val_sim = validate_joint(
            model,
            val_loader,
            loss_fn,
            device,
            pair_seed=args.pair_seed,
            use_amp=use_fp16,
            use_features=use_features,
        )
        elapsed = time.time() - t0

        lr = optimizer.param_groups[0]["lr"]
        logger.info(
            f"[S1] Epoch {epoch:3d} | "
            f"train_mlm={train_loss:.4f} | val_mlm={val_mlm:.4f} | "
            f"val_sim={val_sim:.4f} | lr={lr:.2e} | time={elapsed:.1f}s"
        )
        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow(
                [
                    "S1",
                    epoch,
                    train_loss,
                    train_loss,
                    0.0,
                    val_loss,
                    val_mlm,
                    0.0,
                    val_sim,
                    lr,
                    elapsed,
                ]
            )

        if val_loss < best_mlm_loss:
            best_mlm_loss = val_loss
            save_checkpoint(
                model,
                optimizer,
                scheduler,
                scaler,
                epoch,
                val_loss,
                str(ckpt_dir / "best_mlm.pt"),
                stage="mlm",
                config=cfg,
                seed=args.seed,
                pair_seed=args.pair_seed,
            )

    logger.info(f"Stage 1 complete. Best MLM val_loss: {best_mlm_loss:.4f}")
    save_checkpoint(
        model,
        optimizer,
        scheduler,
        scaler,
        mlm_epochs,
        best_mlm_loss,
        str(ckpt_dir / "mlm_final.pt"),
        stage="mlm",
        config=cfg,
        seed=args.seed,
        pair_seed=args.pair_seed,
    )

    # ═════════════════════════════════════════════════════════════════════
    # Stage 2: Joint MLM + Contrastive Fine-tune
    # ═════════════════════════════════════════════════════════════════════
    lambda_c = train_cfg.get("lambda_contrastive", 0.7)
    loss_fn.set_lambda(lambda_c)

    head_lr = train_cfg.get("head_learning_rate", 5e-5)
    body_lr = train_cfg.get("learning_rate", 1e-5)

    transformer_params = _unique_parameters(
        model.roformer,
        model.mlm_head,
    )
    head_params = list(model.projection.parameters())
    if hasattr(model, "fusion"):
        head_params += list(model.fusion.parameters())

    optimizer = torch.optim.AdamW(
        [
            {"params": transformer_params, "lr": body_lr},
            {"params": head_params, "lr": head_lr},
        ],
        weight_decay=train_cfg.get("weight_decay", 0.01),
    )

    max_epochs = train_cfg.get("max_epochs", 50)
    total_steps_s2 = steps_per_epoch * max_epochs
    scheduler = None
    if warmup_steps > 0:
        from transformers import get_linear_schedule_with_warmup

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps_s2,
        )

    best_val_loss = float("inf")
    patience_counter = 0
    patience = train_cfg.get("patience", 10)

    logger.info(
        f"=== Stage 2: Joint Fine-tune (lambda={lambda_c}) ===\n"
        f"LR: transformer={body_lr}, heads={head_lr}, "
        f"max_epochs={max_epochs}, patience={patience}"
    )

    for epoch in range(1, max_epochs + 1):
        train_sampler.set_epoch(mlm_epochs + epoch)
        t0 = time.time()

        if train_cfg.get("use_gradcache", False):
            train_loss, train_mlm, train_contra = train_epoch_joint_gradcache(
                model,
                train_loader,
                loss_fn,
                optimizer,
                scheduler,
                device,
                grad_clip,
                scaler=scaler,
                grad_accum_steps=grad_accum,
                use_features=use_features,
            )
        else:
            train_loss, train_mlm, train_contra = train_epoch_joint(
                model,
                train_loader,
                loss_fn,
                optimizer,
                scheduler,
                device,
                grad_clip,
                scaler=scaler,
                grad_accum_steps=grad_accum,
                use_features=use_features,
            )
        val_loss, val_mlm, val_contra, val_sim = validate_joint(
            model,
            val_loader,
            loss_fn,
            device,
            pair_seed=args.pair_seed,
            use_amp=use_fp16,
            use_features=use_features,
        )
        elapsed = time.time() - t0

        lr = optimizer.param_groups[0]["lr"]
        logger.info(
            f"[S2] Epoch {epoch:3d} | "
            f"train={train_loss:.4f} (mlm={train_mlm:.4f} contra={train_contra:.4f}) | "
            f"val={val_loss:.4f} (mlm={val_mlm:.4f} contra={val_contra:.4f}) | "
            f"val_sim={val_sim:.4f} | lr={lr:.2e} | time={elapsed:.1f}s"
        )
        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow(
                [
                    "S2",
                    epoch,
                    train_loss,
                    train_mlm,
                    train_contra,
                    val_loss,
                    val_mlm,
                    val_contra,
                    val_sim,
                    lr,
                    elapsed,
                ]
            )

        rank_every = train_cfg.get("ranking_eval_every", 5)
        if epoch % rank_every == 0 or epoch == 1:
            rank_metrics = evaluate_ranking(
                model,
                val_loader,
                device,
                pair_seed=args.pair_seed,
                split="validation",
                use_amp=use_fp16,
                use_features=use_features,
            )
            logger.info(
                f"  [Ranking] pool={rank_metrics['pool_size']} | "
                f"MRR={rank_metrics['mrr']:.4f} | "
                f"R@1={rank_metrics['recall_at_1']:.4f} | "
                f"R@5={rank_metrics['recall_at_5']:.4f} | "
                f"R@10={rank_metrics['recall_at_10']:.4f}"
            )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            save_checkpoint(
                model,
                optimizer,
                scheduler,
                scaler,
                epoch,
                val_loss,
                str(ckpt_dir / "best_model.pt"),
                stage="joint",
                config=cfg,
                seed=args.seed,
                pair_seed=args.pair_seed,
            )
            logger.info(f"  New best model saved (val_loss={val_loss:.4f})")
        else:
            patience_counter += 1

        if epoch % 5 == 0:
            save_checkpoint(
                model,
                optimizer,
                scheduler,
                scaler,
                epoch,
                val_loss,
                str(ckpt_dir / f"checkpoint_s2_epoch_{epoch}.pt"),
                stage="joint",
                config=cfg,
                seed=args.seed,
                pair_seed=args.pair_seed,
            )

        if patience_counter >= patience:
            logger.info(f"Early stopping at epoch {epoch}")
            break

    logger.info(f"Training complete. Best val_loss: {best_val_loss:.4f}")
    logger.info(f"Logs: {log_path}, Checkpoints: {ckpt_dir}")


if __name__ == "__main__":
    main()
