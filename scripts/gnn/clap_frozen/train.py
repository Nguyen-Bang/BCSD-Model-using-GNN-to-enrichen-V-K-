#!/usr/bin/env python3
"""Train frozen CLAP baseline, register, or typed-edge GNN variants."""

from __future__ import annotations

import argparse
import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from models.clap_frozen.factory import create_model
from training.gnn import train_epoch as train_gnn_epoch
from training.losses import InfoNCELoss
from training.runtime import (
    accumulation_window_size,
    create_gnn_dataloaders,
    evaluate_ranking,
    load_config,
    load_training_checkpoint,
    optimizer_steps_per_epoch,
    resolve_device,
    save_checkpoint,
    seed_everything,
    to_device,
    validate_pairs,
)
from training.struct_heads import StructHeadsOnTransformer

logger = logging.getLogger("bcsd.clap_frozen.train")


def _precision(training: Mapping[str, Any], device: torch.device):
    precision = str(training.get("precision", "fp32")).lower()
    if precision not in {"fp16", "bf16", "fp32"}:
        raise ValueError(f"Unsupported precision: {precision}")
    use_amp = device.type == "cuda" and precision != "fp32"
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[precision]
    scaler = torch.amp.GradScaler("cuda", enabled=True) if use_amp and precision == "fp16" else None
    return use_amp, dtype, scaler


def train_baseline_epoch(
    model,
    loader,
    loss_fn,
    optimizer,
    scheduler,
    device,
    trainable_params,
    *,
    grad_clip: float,
    scaler,
    grad_accum_steps: int,
    use_amp: bool,
    amp_dtype: torch.dtype,
) -> float:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    total = 0.0
    num_batches = len(loader)
    for step, batch in enumerate(loader):
        batch = to_device(batch, device)
        with torch.amp.autocast(device.type, enabled=use_amp, dtype=amp_dtype):
            embedding_a, embedding_b = model.get_embedding_pairs(batch)
            full_loss = loss_fn(embedding_a, embedding_b)
        window_size = accumulation_window_size(step, num_batches, grad_accum_steps)
        loss = full_loss / window_size
        if scaler is None:
            loss.backward()
        else:
            scaler.scale(loss).backward()

        boundary = (step + 1) % grad_accum_steps == 0 or step + 1 == num_batches
        if boundary:
            optimizer_updated = True
            if scaler is None:
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(trainable_params, grad_clip)
                optimizer.step()
            else:
                scaler.unscale_(optimizer)
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(trainable_params, grad_clip)
                previous_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                optimizer_updated = scaler.get_scale() >= previous_scale
            optimizer.zero_grad(set_to_none=True)
            if scheduler is not None and optimizer_updated:
                scheduler.step()
        total += full_loss.item()
    if num_batches == 0:
        raise ValueError("Cannot train on an empty data loader")
    return total / num_batches


def _save(
    path: Path,
    *,
    model,
    struct_heads,
    optimizer,
    scheduler,
    scaler,
    epoch: int,
    config,
    seed: int,
    pair_seed: int,
    val_loss: float,
    best_val_loss: float,
    best_mrr: float,
    patience_counter: int,
    ranking,
) -> None:
    extra = {
        "pair_seed": pair_seed,
        "val_loss": val_loss,
        "best_val_loss": best_val_loss,
        "best_mrr": best_mrr,
        "patience_counter": patience_counter,
        "ranking": ranking,
    }
    if struct_heads is not None:
        extra["struct_heads_state_dict"] = struct_heads.state_dict()
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


def _resolve_output_dir(
    config: Mapping[str, Any],
    *,
    seed: int,
    resume: str | Path | None,
    output_dir: str | Path | None,
) -> Path:
    if resume is not None:
        resume_dir = Path(resume).parent
        if output_dir is not None and Path(output_dir).resolve() != resume_dir.resolve():
            raise ValueError("--output-dir must match the resume checkpoint directory")
        resolved = resume_dir
    elif output_dir is not None:
        resolved = Path(output_dir)
    else:
        configured = Path(config["training"].get("checkpoint_dir", "checkpoints/clap_frozen"))
        resolved = configured / f"seed_{seed}"
    if resume is None and resolved.is_dir() and any(resolved.glob("*.pt")):
        raise FileExistsError(
            f"Refusing to overwrite existing checkpoints in {resolved}; "
            "use --resume or --output-dir"
        )
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def train(
    config: Mapping[str, Any],
    *,
    device: torch.device,
    seed: int,
    pair_seed: int,
    resume: str | Path | None = None,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    seed_everything(seed)
    training = config["training"]
    output_dir_path = _resolve_output_dir(
        config,
        seed=seed,
        resume=resume,
        output_dir=output_dir,
    )
    train_loader, validation_loader, sampler = create_gnn_dataloaders(config, seed=seed)
    model = create_model(config, device)
    variant = str(config["model"].get("variant", "baseline"))
    struct_heads = None
    if variant != "baseline":
        struct_heads = StructHeadsOnTransformer(
            hidden_size=model.hidden_size,
            dropout=float(config["model"].get("dropout", 0.1)),
        ).to(device)

    trainable_params = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if struct_heads is not None:
        trainable_params.extend(struct_heads.parameters())
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=float(training.get("learning_rate", 5e-5)),
        weight_decay=float(training.get("weight_decay", 0.01)),
    )
    loss_fn = InfoNCELoss(temperature=float(training.get("temperature", 0.07)))
    use_amp, amp_dtype, scaler = _precision(training, device)
    grad_accum_steps = int(training.get("grad_accum_steps", 1))
    max_epochs = int(training.get("max_epochs", 50))
    warmup_steps = int(training.get("warmup_steps", 100))
    total_steps = optimizer_steps_per_epoch(len(train_loader), grad_accum_steps) * max_epochs
    scheduler = None
    if warmup_steps > 0:
        from transformers import get_linear_schedule_with_warmup

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

    start_epoch = 1
    best_val_loss = float("inf")
    best_mrr = -1.0
    patience_counter = 0
    if resume is not None:
        checkpoint = load_training_checkpoint(
            resume,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            device=device,
            config=config,
        )
        if checkpoint["seed"] != seed or checkpoint.get("pair_seed") != pair_seed:
            raise ValueError("Resume seed or pair seed does not match the checkpoint")
        start_epoch = int(checkpoint["epoch"]) + 1
        best_val_loss = float(checkpoint["best_val_loss"])
        best_mrr = float(checkpoint["best_mrr"])
        patience_counter = int(checkpoint["patience_counter"])
        if struct_heads is not None:
            if "struct_heads_state_dict" not in checkpoint:
                raise KeyError("Resume checkpoint has no struct_heads_state_dict")
            struct_heads.load_state_dict(checkpoint["struct_heads_state_dict"], strict=True)

    ranking_interval = int(training.get("ranking_eval_interval", 5))
    if ranking_interval < 1:
        raise ValueError("training.ranking_eval_interval must be positive")
    patience = int(training.get("patience", 15))
    last_val_loss = float("nan")

    for epoch in range(start_epoch, max_epochs + 1):
        sampler.set_epoch(epoch)
        if struct_heads is None:
            train_loss = train_baseline_epoch(
                model,
                train_loader,
                loss_fn,
                optimizer,
                scheduler,
                device,
                trainable_params,
                grad_clip=float(training.get("gradient_clip", 1.0)),
                scaler=scaler,
                grad_accum_steps=grad_accum_steps,
                use_amp=use_amp,
                amp_dtype=amp_dtype,
            )
        else:
            dropout = training.get("modality_dropout") or {}
            ratio = float(dropout.get("ratio", 0.0)) if dropout.get("enabled", False) else 0.0
            warmup = int(dropout.get("warmup_epochs", 0))
            if warmup and epoch <= warmup:
                ratio *= epoch / warmup
            train_loss, _, _ = train_gnn_epoch(
                model,
                train_loader,
                loss_fn,
                struct_heads,
                optimizer,
                scheduler,
                device,
                trainable_params,
                grad_clip=float(training.get("gradient_clip", 1.0)),
                lambda_contrastive=float(training.get("lambda_contrastive", 1.0)),
                lambda_struct=float(training.get("lambda_struct", 0.1)),
                scaler=scaler,
                grad_accum_steps=grad_accum_steps,
                modality_dropout_ratio=ratio,
                modality_dropout_side=str(dropout.get("side", "a_only")),
                modality_dropout_drop_features=bool(dropout.get("drop_function_features", True)),
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
        if epoch == 1 or epoch % ranking_interval == 0:
            ranking = evaluate_ranking(
                model,
                validation_loader,
                device,
                split="validation",
                pair_seed=pair_seed,
                use_amp=use_amp,
                amp_dtype=amp_dtype,
            )
        logger.info(
            "epoch=%d train=%.4f val=%.4f similarity=%.4f mrr=%s",
            epoch,
            train_loss,
            val_loss,
            val_similarity,
            f"{ranking['mrr']:.4f}" if ranking else "-",
        )

        improved_val = val_loss < best_val_loss
        improved_mrr = ranking is not None and ranking["mrr"] > best_mrr
        if improved_val:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
        if improved_mrr:
            best_mrr = float(ranking["mrr"])
        save_args = {
            "model": model,
            "struct_heads": struct_heads,
            "optimizer": optimizer,
            "scheduler": scheduler,
            "scaler": scaler,
            "epoch": epoch,
            "config": config,
            "seed": seed,
            "pair_seed": pair_seed,
            "val_loss": val_loss,
            "best_val_loss": best_val_loss,
            "best_mrr": best_mrr,
            "patience_counter": patience_counter,
            "ranking": ranking,
        }
        if improved_val:
            _save(output_dir_path / "best_model.pt", **save_args)
        if improved_mrr:
            _save(output_dir_path / "best_mrr.pt", **save_args)
        if epoch % ranking_interval == 0:
            _save(output_dir_path / f"checkpoint_epoch_{epoch}.pt", **save_args)
        if patience_counter >= patience:
            logger.info("Early stopping at epoch %d", epoch)
            break

    return {
        "variant": variant,
        "best_val_loss": best_val_loss,
        "best_mrr": best_mrr,
        "last_val_loss": last_val_loss,
        "seed": seed,
        "pair_seed": pair_seed,
    }


def build_parser(default_config: str | None = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=default_config, required=default_config is None)
    parser.add_argument("--resume")
    parser.add_argument("--output-dir")
    parser.add_argument("--device")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--pair-seed", type=int, default=0)
    return parser


def main(argv: list[str] | None = None, *, default_config: str | None = None):
    args = build_parser(default_config).parse_args(argv)
    config = load_config(args.config)
    return train(
        config,
        device=resolve_device(args.device),
        seed=args.seed,
        pair_seed=args.pair_seed,
        resume=args.resume,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
