#!/usr/bin/env python3
"""Fixed-pair causal ablation of every KV-prefix gate cell."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch

from evaluation.provenance import provenance
from training.diagnostics import (
    ablate,
    benjamini_hochberg,
    effective_gate_grid,
    gate_shape,
    joint_sign_flip_pvalues,
    magnitude_correlations,
    paired_rank_stats,
    save_npy_atomic,
)
from training.evaluation import (
    build_split_loader,
    evaluate_ranks,
    metrics_from_ranks,
    pool_metadata,
    resolve_precision,
)
from training.runtime import load_config

logger = logging.getLogger("bcsd.gate_ablation")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--backbone", choices=("baseline", "clap"), default="baseline")
    parser.add_argument("--split", default="test")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--pair-seed", type=int, default=0)
    parser.add_argument("--stats-seed", type=int, default=0)
    parser.add_argument("--n-boot", type=int, default=10_000)
    parser.add_argument("--n-sign-flips", type=int, default=100_000)
    parser.add_argument("--sign-flip-device", default="cpu")
    parser.add_argument("--sign-flip-chunk", type=int, default=1_000)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--skip-per-cell", action="store_true")
    parser.add_argument("--max-cells", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--resume", action="store_true")
    return parser


def _build_and_load(config, backbone, checkpoint_path, device):
    if backbone == "baseline":
        from training.evaluation import build_model, load_checkpoint_strict
    else:
        from models.clap_frozen.factory import create_model as build_model
        from models.clap_frozen.factory import load_checkpoint_strict

    model = build_model(config, device)
    checkpoint = load_checkpoint_strict(model, checkpoint_path, device, config)
    return model, checkpoint


def _cell_groups(n_layers: int, n_heads: int, n_prefix: int):
    for layer in range(n_layers):
        yield (
            "layer",
            layer,
            [(layer, head, prefix) for head in range(n_heads) for prefix in range(n_prefix)],
        )
    for head in range(n_heads):
        yield (
            "head",
            head,
            [(layer, head, prefix) for layer in range(n_layers) for prefix in range(n_prefix)],
        )
    for prefix in range(n_prefix):
        yield (
            "prefix",
            prefix,
            [(layer, head, prefix) for layer in range(n_layers) for head in range(n_heads)],
        )


def _json_dump(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.batch_size <= 0 or args.max_cells < 0:
        raise ValueError("batch-size must be positive and max-cells must be non-negative")
    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()) and not args.resume:
        raise FileExistsError(
            f"Refusing to overwrite non-empty output directory: {output_dir}; pass --resume"
        )
    device = torch.device(args.device)
    config = load_config(args.config)
    precision, use_amp, amp_dtype = resolve_precision(config, device)
    model, checkpoint = _build_and_load(config, args.backbone, args.checkpoint, device)
    layers = model.kv_prefix_layers
    n_layers, n_heads, n_prefix = gate_shape(layers)
    if any(not bool(torch.all(layer._ablate_mask == 1)) for layer in layers):
        raise ValueError("Gate ablation requires a checkpoint with an enabled prefix channel")
    effective = effective_gate_grid(layers).numpy()

    loader, expected_pool_size = build_split_loader(config, args.split, args.batch_size)
    progress_dir = output_dir / ".ablation_progress"
    output_dir.mkdir(parents=True, exist_ok=True)
    identity = {
        **provenance(args.config, args.checkpoint),
        "backbone": args.backbone,
        "split": args.split,
        "batch_size": args.batch_size,
        "pair_seed": args.pair_seed,
        "gate_shape": [n_layers, n_heads, n_prefix],
    }
    identity_path = progress_dir / "identity.json"
    if progress_dir.exists():
        if not args.resume:
            raise FileExistsError(f"Progress exists at {progress_dir}; pass --resume or remove it")
        if not identity_path.is_file() or json.loads(identity_path.read_text()) != identity:
            raise ValueError("Existing ablation progress belongs to a different run")
    else:
        progress_dir.mkdir()
        _json_dump(identity_path, identity)

    def evaluate(name, cells="active"):
        path = progress_dir / f"{name}.npy"
        if args.resume and path.is_file():
            ranks = np.load(path, allow_pickle=False)
        else:
            intervention = nullcontext() if cells == "active" else ablate(layers, cells)
            with intervention:
                ranks, _ = evaluate_ranks(
                    model,
                    loader,
                    device,
                    use_amp,
                    amp_dtype,
                    args.pair_seed,
                )
            save_npy_atomic(path, ranks.astype(np.int32))
        if ranks.shape != (expected_pool_size,) or np.any(ranks < 1):
            raise ValueError(f"Invalid cached ranks in {path}")
        return ranks.astype(np.int32, copy=False)

    started = time.time()
    ranks_full = evaluate("full")
    ranks_off = evaluate("full_prefix_off", None)
    metrics_full = metrics_from_ranks(ranks_full)
    metrics_off = metrics_from_ranks(ranks_off)
    headline_stats = paired_rank_stats(
        ranks_full,
        ranks_off,
        n_boot=args.n_boot,
        n_draws=args.n_sign_flips,
        seed=args.stats_seed,
        sign_flip_device=args.sign_flip_device,
        chunk_size=args.sign_flip_chunk,
    )

    coarse = []
    coarse_ranks = []
    for kind, index, cells in _cell_groups(n_layers, n_heads, n_prefix):
        ranks = evaluate(f"{kind}_{index}", cells)
        metrics = metrics_from_ranks(ranks)
        statistics = paired_rank_stats(
            ranks_full,
            ranks,
            n_boot=args.n_boot,
            n_draws=args.n_sign_flips,
            seed=args.stats_seed,
            sign_flip_device=args.sign_flip_device,
            chunk_size=args.sign_flip_chunk,
        )
        coarse.append({"kind": kind, "index": index, **metrics, **statistics})
        coarse_ranks.append(ranks)
        logger.info("%s=%d delta_mrr=%+.5f", kind, index, statistics["delta_mrr"])

    all_cells = [
        (layer, head, prefix)
        for layer in range(n_layers)
        for head in range(n_heads)
        for prefix in range(n_prefix)
    ]
    cells = all_cells[: args.max_cells or None]
    per_cell = []
    cell_ranks = []
    if not args.skip_per_cell:
        for cell_index, (layer, head, prefix) in enumerate(cells):
            ranks = evaluate(f"cell_{layer}_{head}_{prefix}", [(layer, head, prefix)])
            cell_ranks.append(ranks)
            delta_mrr = metrics_from_ranks(ranks)["mrr"] - metrics_full["mrr"]
            per_cell.append(
                {
                    "cell_index": cell_index,
                    "layer": layer,
                    "head": head,
                    "prefix": prefix,
                    "effective_gate": float(effective[layer, head, prefix]),
                    "abs_effective_gate": abs(float(effective[layer, head, prefix])),
                    "delta_mrr": delta_mrr,
                }
            )
            if (cell_index + 1) % 20 == 0 or cell_index + 1 == len(cells):
                logger.info("per-cell %d/%d", cell_index + 1, len(cells))

        reciprocal_full = 1.0 / ranks_full.astype(np.float64)
        reciprocal_cells = 1.0 / np.stack(cell_ranks).astype(np.float64)
        deltas = reciprocal_cells - reciprocal_full[None, :]
        pvalues = joint_sign_flip_pvalues(
            deltas,
            args.n_sign_flips,
            args.stats_seed,
            device=args.sign_flip_device,
            chunk_size=args.sign_flip_chunk,
        )
        qvalues = benjamini_hochberg(pvalues)
        for row, pvalue, qvalue in zip(per_cell, pvalues, qvalues, strict=True):
            row["sign_flip_p"] = float(pvalue)
            row["bh_q"] = float(qvalue)
        correlations = (
            magnitude_correlations(
                [row["effective_gate"] for row in per_cell],
                [row["delta_mrr"] for row in per_cell],
            )
            if len(per_cell) > 1
            else None
        )
    else:
        correlations = None

    final_layer = next(
        row for row in coarse if row["kind"] == "layer" and row["index"] == n_layers - 1
    )
    metadata = {
        "experiment": "fixed_pair_gate_cell_ablation",
        **identity,
        "checkpoint_epoch": checkpoint.get("epoch"),
        "pool": pool_metadata(loader, args.split),
        "pool_size": expected_pool_size,
        "precision": precision,
        "n_boot": args.n_boot,
        "n_sign_flips": args.n_sign_flips,
        "stats_seed": args.stats_seed,
        "monte_carlo_p": "(exceedances + 1) / (draws + 1)",
        "multiple_testing": "Benjamini-Hochberg over all evaluated gate cells",
        "elapsed_seconds": round(time.time() - started, 2),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    result = {
        "metadata": metadata,
        "pool_size": expected_pool_size,
        "full": metrics_full,
        "full_prefix_off": {**metrics_off, **headline_stats},
        "final_layer_group_removal": final_layer,
        "coarse": coarse,
        "per_cell": per_cell,
        "magnitude_correlations": correlations,
    }
    _json_dump(output_dir / "metadata.json", metadata)
    _json_dump(output_dir / "gate_ablation.json", result)

    with (output_dir / "coarse.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=coarse[0].keys())
        writer.writeheader()
        writer.writerows(coarse)
    if per_cell:
        with (output_dir / "per_cell.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=per_cell[0].keys())
            writer.writeheader()
            writer.writerows(per_cell)
    arrays = {
        "ranks_full": ranks_full,
        "ranks_full_prefix_off": ranks_off,
        "ranks_coarse": np.stack(coarse_ranks),
        "effective_gate": effective,
    }
    if cell_ranks:
        arrays["ranks_per_cell"] = np.stack(cell_ranks)
        arrays["cells"] = np.asarray(cells, dtype=np.int16)
    np.savez_compressed(output_dir / "gate_ablation_ranks.npz", **arrays)
    logger.info(
        "Wrote %s: full MRR %.4f, prefix-off delta %+.4f, cells %d",
        output_dir,
        metrics_full["mrr"],
        headline_stats["delta_mrr"],
        len(per_cell),
    )
    return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()
