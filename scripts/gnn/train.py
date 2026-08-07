#!/usr/bin/env python3
"""Train the frozen three-layer baseline with a GNN KV-prefix."""

from __future__ import annotations

import argparse
import csv
import logging
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from models.baseline.factory import create_model as create_baseline_model
from training.gnn import train_epoch as train_gnn_epoch
from training.losses import InfoNCELoss
from training.runtime import (
    create_gnn_dataloaders,
    evaluate_ranking,
    load_config,
    load_training_checkpoint,
    optimizer_steps_per_epoch,
    resolve_device,
    save_checkpoint,
    seed_everything,
    validate_pairs,
)
from training.struct_heads import StructHeadsOnTransformer

logger = logging.getLogger("bcsd.train_gnn")


def resolve_output_dir(
    config: Mapping[str, Any],
    *,
    seed: int,
    output_dir: str | Path | None = None,
    resume: str | Path | None = None,
) -> Path:
    """Resolve a run directory without changing the checkpointed config."""
    if resume is not None:
        resume_dir = Path(resume).parent
        if output_dir is not None and Path(output_dir).resolve() != resume_dir.resolve():
            raise ValueError("--output-dir must match the resume checkpoint directory")
        return resume_dir
    if output_dir is not None:
        return Path(output_dir)
    base_dir = Path(config["training"].get("checkpoint_dir", "checkpoints/gnn"))
    return base_dir / f"seed_{seed}"


def prepare_output_dir(output_dir: Path, *, resume: bool) -> None:
    """Create a fresh run directory and refuse to replace training artifacts."""
    if resume:
        output_dir.mkdir(parents=True, exist_ok=True)
        return

    artifacts = (
        "training_log.csv",
        "best_mrr.pt",
        "best_model.pt",
        "gate_diagnostics.pt",
    )
    existing = [name for name in artifacts if (output_dir / name).exists()]
    existing.extend(path.name for path in output_dir.glob("checkpoint_epoch_*.pt"))
    if existing:
        raise FileExistsError(f"Output directory already contains training artifacts: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)


def _precision(
    training_config: Mapping[str, Any],
    device: torch.device,
) -> tuple[bool, torch.dtype, torch.amp.GradScaler | None, str]:
    precision = str(
        training_config.get(
            "precision",
            "fp16" if training_config.get("fp16", False) else "fp32",
        )
    ).lower()
    if precision not in {"fp16", "bf16", "fp32"}:
        raise ValueError(f"Unsupported precision: {precision}")
    use_amp = device.type == "cuda" and precision in {"fp16", "bf16"}
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[precision]
    scaler = torch.amp.GradScaler("cuda", enabled=True) if use_amp and precision == "fp16" else None
    return use_amp, dtype, scaler, precision


def _save(
    path: Path,
    *,
    model: torch.nn.Module,
    struct_heads: StructHeadsOnTransformer,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler: torch.amp.GradScaler | None,
    epoch: int,
    config: Mapping[str, Any],
    seed: int,
    pair_seed: int,
    val_loss: float,
    best_val_loss: float,
    best_mrr: float,
    patience_counter: int,
    ranking: Mapping[str, Any] | None = None,
) -> None:
    extra: dict[str, Any] = {
        "struct_heads_state_dict": struct_heads.state_dict(),
        "val_loss": val_loss,
        "best_val_loss": best_val_loss,
        "best_mrr": best_mrr,
        "patience_counter": patience_counter,
        "pair_seed": pair_seed,
    }
    if ranking is not None:
        extra["ranking"] = dict(ranking)
    save_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        epoch=epoch,
        config=config,
        seed=seed,
        extra=extra,
    )


def train(
    config: Mapping[str, Any],
    *,
    device: torch.device,
    seed: int,
    pair_seed: int,
    backbone: str = "baseline",
    backbone_checkpoint: str | None = None,
    resume: str | None = None,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Run single-device training and return the best validation metrics."""
    checkpoint_dir = resolve_output_dir(
        config,
        seed=seed,
        output_dir=output_dir,
        resume=resume,
    )
    prepare_output_dir(checkpoint_dir, resume=resume is not None)
    seed_everything(seed)
    training_config = config["training"]
    train_loader, validation_loader, train_sampler = create_gnn_dataloaders(config, seed=seed)
    model = create_baseline_model(
        config,
        backbone,
        device,
        None if resume else backbone_checkpoint,
        initialize_embeddings=not (resume or backbone_checkpoint),
    )
    struct_heads = StructHeadsOnTransformer(
        hidden_size=int(config["model"].get("hidden_size", 768)),
        dropout=float(config["model"].get("dropout", 0.1)),
    ).to(device)
    trainable_params = [parameter for parameter in model.parameters() if parameter.requires_grad]
    trainable_params.extend(struct_heads.parameters())
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=float(training_config.get("learning_rate", 5e-5)),
        weight_decay=float(training_config.get("weight_decay", 0.01)),
    )
    loss_fn = InfoNCELoss(temperature=float(training_config.get("temperature", 0.07)))
    use_amp, amp_dtype, scaler, precision = _precision(training_config, device)
    grad_accum = int(training_config.get("grad_accum_steps", 1))
    max_epochs = int(training_config.get("max_epochs", 50))
    warmup_steps = int(training_config.get("warmup_steps", 100))
    total_steps = optimizer_steps_per_epoch(len(train_loader), grad_accum) * max_epochs
    scheduler = None
    if warmup_steps > 0:
        from transformers import get_linear_schedule_with_warmup

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

    log_path = checkpoint_dir / "training_log.csv"
    start_epoch = 1
    best_val_loss = float("inf")
    best_mrr = 0.0
    patience_counter = 0
    if resume:
        checkpoint = load_training_checkpoint(
            resume,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            device=device,
            config=config,
        )
        if checkpoint["seed"] != seed:
            raise ValueError(
                f"Resume seed {seed} does not match checkpoint seed {checkpoint['seed']}"
            )
        if checkpoint.get("pair_seed") != pair_seed:
            raise ValueError(
                "Resume pair seed does not match checkpoint pair seed "
                f"{checkpoint.get('pair_seed')}"
            )
        if "struct_heads_state_dict" not in checkpoint:
            raise KeyError("Resume checkpoint has no struct_heads_state_dict")
        struct_heads.load_state_dict(checkpoint["struct_heads_state_dict"], strict=True)
        start_epoch = int(checkpoint["epoch"]) + 1
        best_val_loss = float(checkpoint["best_val_loss"])
        best_mrr = float(checkpoint["best_mrr"])
        patience_counter = int(checkpoint["patience_counter"])
    else:
        gate_count = len(model.kv_prefix_layers)
        with log_path.open("w", newline="") as handle:
            csv.writer(handle).writerow(
                [
                    "epoch",
                    "train_loss",
                    "train_contrastive",
                    "train_struct",
                    "val_loss",
                    "val_sim",
                    "lr",
                    "time_s",
                    "split",
                    "pair_seed",
                    "mrr",
                    "recall_at_1",
                    "recall_at_5",
                    "recall_at_10",
                    "pool_size",
                    *[f"gate_{index}" for index in range(gate_count)],
                    "md_ratio",
                ]
            )

    logger.info(
        "Training on %s with seed=%d pair_seed=%d precision=%s",
        device,
        seed,
        pair_seed,
        precision,
    )
    patience = int(training_config.get("patience", 15))
    ranking_interval = int(training_config.get("ranking_eval_interval", 5))
    lambda_contrastive = float(training_config.get("lambda_contrastive", 1.0))
    lambda_struct = float(training_config.get("lambda_struct", 0.3))
    dropout_config = training_config.get("modality_dropout") or {}
    dropout_enabled = bool(dropout_config.get("enabled", False))
    dropout_ratio = float(dropout_config.get("ratio", 0.0)) if dropout_enabled else 0.0
    dropout_warmup = int(dropout_config.get("warmup_epochs", 0))
    last_val_loss = float("nan")

    for epoch in range(start_epoch, max_epochs + 1):
        train_sampler.set_epoch(epoch)
        started = time.time()
        if dropout_enabled and dropout_warmup > 0 and epoch <= dropout_warmup:
            epoch_dropout = dropout_ratio * epoch / dropout_warmup
        else:
            epoch_dropout = dropout_ratio
        train_loss, train_contrastive, train_struct = train_gnn_epoch(
            model,
            train_loader,
            loss_fn,
            struct_heads,
            optimizer,
            scheduler,
            device,
            trainable_params,
            grad_clip=float(training_config.get("gradient_clip", 1.0)),
            lambda_contrastive=lambda_contrastive,
            lambda_struct=lambda_struct,
            scaler=scaler,
            grad_accum_steps=grad_accum,
            modality_dropout_ratio=epoch_dropout,
            modality_dropout_side=str(dropout_config.get("side", "a_only")),
            modality_dropout_drop_features=bool(dropout_config.get("drop_function_features", True)),
            use_amp=use_amp,
            amp_dtype=amp_dtype,
        )
        val_loss, val_similarity = validate_pairs(
            model,
            validation_loader,
            loss_fn,
            device,
            pair_seed=pair_seed,
            use_amp=use_amp,
            amp_dtype=amp_dtype,
        )
        last_val_loss = val_loss
        ranking = None
        if epoch % ranking_interval == 0 or epoch == 1:
            ranking = evaluate_ranking(
                model,
                validation_loader,
                device,
                split="validation",
                pair_seed=pair_seed,
                use_amp=use_amp,
                amp_dtype=amp_dtype,
            )
        gates = [layer.effective_gate().item() for layer in model.kv_prefix_layers]
        elapsed = time.time() - started
        logger.info(
            "epoch=%d train=%.4f val=%.4f pool=%s mrr=%s",
            epoch,
            train_loss,
            val_loss,
            ranking["pool_size"] if ranking else "-",
            f"{ranking['mrr']:.4f}" if ranking else "-",
        )
        with log_path.open("a", newline="") as handle:
            csv.writer(handle).writerow(
                [
                    epoch,
                    train_loss,
                    train_contrastive,
                    train_struct,
                    val_loss,
                    val_similarity,
                    optimizer.param_groups[0]["lr"],
                    elapsed,
                    ranking["split"] if ranking else "",
                    ranking["pair_seed"] if ranking else "",
                    ranking["mrr"] if ranking else "",
                    ranking["recall_at_1"] if ranking else "",
                    ranking["recall_at_5"] if ranking else "",
                    ranking["recall_at_10"] if ranking else "",
                    ranking["pool_size"] if ranking else "",
                    *gates,
                    epoch_dropout,
                ]
            )

        improved_mrr = bool(ranking and ranking["mrr"] > best_mrr)
        improved_val = val_loss < best_val_loss
        if improved_mrr:
            best_mrr = float(ranking["mrr"])
        if improved_val:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1

        if improved_mrr:
            _save(
                checkpoint_dir / "best_mrr.pt",
                model=model,
                struct_heads=struct_heads,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                epoch=epoch,
                config=config,
                seed=seed,
                pair_seed=pair_seed,
                val_loss=val_loss,
                best_val_loss=best_val_loss,
                best_mrr=best_mrr,
                patience_counter=patience_counter,
                ranking=ranking,
            )
        if improved_val:
            _save(
                checkpoint_dir / "best_model.pt",
                model=model,
                struct_heads=struct_heads,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                epoch=epoch,
                config=config,
                seed=seed,
                pair_seed=pair_seed,
                val_loss=val_loss,
                best_val_loss=best_val_loss,
                best_mrr=best_mrr,
                patience_counter=patience_counter,
                ranking=ranking,
            )
        if epoch % ranking_interval == 0:
            _save(
                checkpoint_dir / f"checkpoint_epoch_{epoch}.pt",
                model=model,
                struct_heads=struct_heads,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                epoch=epoch,
                config=config,
                seed=seed,
                pair_seed=pair_seed,
                val_loss=val_loss,
                best_val_loss=best_val_loss,
                best_mrr=best_mrr,
                patience_counter=patience_counter,
                ranking=ranking,
            )
        if patience_counter >= patience:
            logger.info("Early stopping at epoch %d", epoch)
            break

    diagnostics = {}
    for index, layer in enumerate(model.kv_prefix_layers):
        entry = {}
        for name in ("weight", "gate", "scale"):
            value = getattr(layer, name, None)
            if isinstance(value, torch.Tensor):
                entry[name] = value.detach().cpu()
        entry["effective"] = layer.effective_gate().detach().cpu()
        diagnostics[f"layer_{index}"] = entry
    torch.save(diagnostics, checkpoint_dir / "gate_diagnostics.pt")
    return {
        "best_val_loss": best_val_loss,
        "best_mrr": best_mrr,
        "last_val_loss": last_val_loss,
        "seed": seed,
        "pair_seed": pair_seed,
        "output_dir": str(checkpoint_dir),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/gnn_v8_typed_edges.yaml")
    parser.add_argument("--backbone", choices=["baseline"], default="baseline")
    parser.add_argument("--backbone-checkpoint")
    parser.add_argument("--resume")
    parser.add_argument("--output-dir")
    parser.add_argument("--device")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--pair-seed", type=int, default=0)
    return parser


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    return train(
        config,
        device=resolve_device(args.device),
        seed=args.seed,
        pair_seed=args.pair_seed,
        backbone=args.backbone,
        backbone_checkpoint=args.backbone_checkpoint or config.get("backbone_checkpoint"),
        resume=args.resume,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )
    main()
