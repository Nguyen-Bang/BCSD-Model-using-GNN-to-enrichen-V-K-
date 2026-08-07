#!/usr/bin/env python3
"""Fixed-pair full-pool ranking for one trained checkpoint."""

import argparse
import json
import logging
import time
from pathlib import Path

import torch

from evaluation.provenance import provenance
from training.evaluation import (
    build_model,
    build_split_loader,
    evaluate_ranks,
    load_checkpoint_strict,
    metrics_from_ranks,
    pool_metadata,
    resolve_precision,
)
from training.runtime import load_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("bcsd.fixed_pair_ranking")


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--pair-seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", required=True)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    output = Path(args.output_dir) / "fixed_pair_ranking.json"
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite evaluation output: {output}")
    if (args.split, args.batch_size, args.pair_seed) != ("test", 64, 0):
        raise ValueError("Fixed evaluation requires --split test --batch-size 64 --pair-seed 0")
    device = torch.device(args.device)
    config = load_config(args.config)
    precision, use_amp, amp_dtype = resolve_precision(config, device)

    model = build_model(config, device)
    checkpoint = load_checkpoint_strict(model, args.checkpoint, device, config)
    loader, expected_pool_size = build_split_loader(config, args.split, args.batch_size)
    ranks, pool_size = evaluate_ranks(model, loader, device, use_amp, amp_dtype, args.pair_seed)
    if pool_size != expected_pool_size:
        raise RuntimeError(f"Evaluated n={pool_size}, expected n={expected_pool_size}")

    result = {
        "experiment": "fixed_pair_full_ranking",
        "model_type": "gnn_kv_prefix",
        **provenance(args.config, args.checkpoint),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "pool": pool_metadata(loader, args.split),
        "split": args.split,
        "pool_size": pool_size,
        "batch_size": args.batch_size,
        "pair_seed": args.pair_seed,
        "precision": precision,
        "metrics": metrics_from_ranks(ranks),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    logger.info(
        "split=%s n=%d batch=%d pair_seed=%d MRR=%.4f R@1=%.4f",
        args.split,
        pool_size,
        args.batch_size,
        args.pair_seed,
        result["metrics"]["mrr"],
        result["metrics"]["recall_at_1"],
    )
    logger.info("Wrote %s", output)


if __name__ == "__main__":
    main()
