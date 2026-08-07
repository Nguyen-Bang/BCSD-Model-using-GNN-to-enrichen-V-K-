#!/usr/bin/env python3
"""Evaluate a frozen CLAP experiment on an explicit retrieval pool."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import torch

from evaluation.provenance import provenance
from models.clap_frozen.factory import create_model, load_checkpoint_strict
from training.evaluation import (
    build_split_loader,
    evaluate_ranks,
    metrics_from_ranks,
    pool_metadata,
    resolve_precision,
)
from training.runtime import load_config, resolve_device

logger = logging.getLogger("bcsd.clap_frozen.evaluate")


def evaluate(
    config_path: str | Path,
    checkpoint_path: str | Path,
    *,
    split: str,
    batch_size: int,
    pair_seed: int,
    device: torch.device,
    local_files_only: bool = False,
) -> dict:
    config = load_config(config_path)
    model = create_model(
        config,
        device,
        local_files_only=local_files_only,
    )
    checkpoint = load_checkpoint_strict(model, checkpoint_path, device, config)
    precision, use_amp, amp_dtype = resolve_precision(config, device)
    loader, expected_size = build_split_loader(config, split, batch_size)
    ranks, pool_size = evaluate_ranks(
        model,
        loader,
        device,
        use_amp,
        amp_dtype,
        pair_seed,
    )
    if pool_size != expected_size:
        raise RuntimeError(f"Evaluated n={pool_size}, expected n={expected_size}")
    return {
        "experiment": "clap_frozen_retrieval",
        "variant": config["model"].get("variant", "baseline"),
        **provenance(config_path, checkpoint_path),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "pool": pool_metadata(loader, split),
        "batch_size": batch_size,
        "pair_seed": pair_seed,
        "precision": precision,
        "tie_policy": "pessimistic",
        "metrics": metrics_from_ranks(ranks),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--pair-seed", type=int, default=0)
    parser.add_argument("--device")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: list[str] | None = None) -> dict:
    args = build_parser().parse_args(argv)
    if args.output is not None and args.output.exists():
        raise FileExistsError(f"Refusing to overwrite evaluation output: {args.output}")
    result = evaluate(
        args.config,
        args.checkpoint,
        split=args.split,
        batch_size=args.batch_size,
        pair_seed=args.pair_seed,
        device=resolve_device(args.device),
        local_files_only=args.local_files_only,
    )
    rendered = json.dumps(result, indent=2) + "\n"
    if args.output is None:
        print(rendered, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
        logger.info("Wrote %s", args.output)
    metrics = result["metrics"]
    logger.info(
        "split=%s n=%d MRR=%.4f R@1=%.4f",
        result["pool"]["split"],
        result["pool"]["n"],
        metrics["mrr"],
        metrics["recall_at_1"],
    )
    return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
