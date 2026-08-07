#!/usr/bin/env python3
"""Fixed-pair RQ2 prefix-channel interventions with sampling-seed repeats."""

import argparse
import json
import logging
import time
from pathlib import Path

import torch

from evaluation.provenance import provenance
from training.evaluation import (
    INTERVENTION_MODES,
    aggregate,
    build_model,
    build_split_loader,
    corrupt_prefixes,
    disable_prefix_channel,
    evaluate_ranks,
    gate_layers,
    load_checkpoint_strict,
    metrics_from_ranks,
    paired_delta_stats,
    pool_metadata,
    reset_masks,
    resolve_precision,
    validate_intervention_batches,
)
from training.runtime import load_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("bcsd.rq2_interventions")


def _comma_ints(value: str):
    try:
        values = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as error:
        raise argparse.ArgumentTypeError("seeds must be comma-separated integers") from error
    if not values:
        raise argparse.ArgumentTypeError("at least one sampling seed is required")
    if len(values) != len(set(values)):
        raise argparse.ArgumentTypeError("sampling seeds must be unique")
    return values


def _modes(value: str):
    modes = [item.strip() for item in value.split(",") if item.strip()]
    unknown = sorted(set(modes) - set(INTERVENTION_MODES))
    if unknown:
        raise argparse.ArgumentTypeError(
            f"unknown modes {unknown}; choose from {list(INTERVENTION_MODES)}"
        )
    if not modes:
        raise argparse.ArgumentTypeError("at least one intervention mode is required")
    if len(modes) != len(set(modes)):
        raise argparse.ArgumentTypeError("intervention modes must be unique")
    return modes


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--batch-size", type=int, default=33)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--sampling-seeds", type=_comma_ints, default=_comma_ints("0,1,2,3,4"))
    parser.add_argument("--modes", type=_modes, default=_modes("noise_global,noise_perdim,shuffle"))
    parser.add_argument("--pair-seed", type=int, default=0)
    parser.add_argument("--stats-seed", type=int, default=0)
    parser.add_argument("--n-boot", type=int, default=10_000)
    parser.add_argument("--n-perm", type=int, default=10_000)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    output = Path(args.output_dir) / "content_control_hardened.json"
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite evaluation output: {output}")
    if (args.split, args.batch_size, args.pair_seed) != ("test", 33, 0):
        raise ValueError("Control evaluation requires --split test --batch-size 33 --pair-seed 0")
    device = torch.device(args.device)
    config = load_config(args.config)
    precision, use_amp, amp_dtype = resolve_precision(config, device)
    model = build_model(config, device)
    checkpoint = load_checkpoint_strict(model, args.checkpoint, device, config)
    layers = gate_layers(model)
    loader, expected_pool_size = build_split_loader(config, args.split, args.batch_size)
    validate_intervention_batches(expected_pool_size, args.batch_size, args.modes)

    reset_masks(layers)
    ranks_full, pool_size = evaluate_ranks(
        model, loader, device, use_amp, amp_dtype, args.pair_seed
    )
    if pool_size != expected_pool_size:
        raise RuntimeError(f"Evaluated n={pool_size}, expected n={expected_pool_size}")
    metrics_full = metrics_from_ranks(ranks_full)

    disable_prefix_channel(layers)
    ranks_off, _ = evaluate_ranks(model, loader, device, use_amp, amp_dtype, args.pair_seed)
    reset_masks(layers)
    metrics_off = metrics_from_ranks(ranks_off)
    off_stats = paired_delta_stats(ranks_full, ranks_off, args.n_boot, args.n_perm, args.stats_seed)

    controls = {}
    for mode in args.modes:
        metrics_by_seed = []
        for sampling_seed in args.sampling_seeds:
            with corrupt_prefixes(model, mode, sampling_seed):
                ranks, _ = evaluate_ranks(model, loader, device, use_amp, amp_dtype, args.pair_seed)
            metrics_by_seed.append(metrics_from_ranks(ranks))
        controls[mode] = {
            "mrr": aggregate([row["mrr"] for row in metrics_by_seed]),
            "recall_at_1": aggregate([row["recall_at_1"] for row in metrics_by_seed]),
            "delta_vs_full": aggregate(
                [row["mrr"] - metrics_full["mrr"] for row in metrics_by_seed]
            ),
            "delta_vs_off": aggregate([row["mrr"] - metrics_off["mrr"] for row in metrics_by_seed]),
        }

    result = {
        "experiment": "rq2_fixed_pair_prefix_interventions",
        "model_type": "gnn_kv_prefix",
        **provenance(args.config, args.checkpoint),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "pool": pool_metadata(loader, args.split),
        "split": args.split,
        "pool_size": pool_size,
        "batch_size": args.batch_size,
        "pair_seed": args.pair_seed,
        "sampling_seeds": args.sampling_seeds,
        "stats_seed": args.stats_seed,
        "precision": precision,
        "modes": args.modes,
        "full": metrics_full,
        "off": {**metrics_off, **off_stats},
        "controls_multiseed": controls,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    logger.info(
        "split=%s n=%d batch=%d pair_seed=%d full=%.4f off=%.4f",
        args.split,
        pool_size,
        args.batch_size,
        args.pair_seed,
        metrics_full["mrr"],
        metrics_off["mrr"],
    )
    logger.info("Wrote %s", output)


if __name__ == "__main__":
    main()
