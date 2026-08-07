#!/usr/bin/env python3
"""Fixed-pair matched-versus-deranged GNN-prefix cosine diagnostic."""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import torch

from evaluation.provenance import provenance
from models.structural_prefix import structural_edges
from training.diagnostics import prefix_cosines, summarize
from training.evaluation import build_split_loader, pool_metadata, reset_rng
from training.runtime import load_config, to_device

logger = logging.getLogger("bcsd.prefix_cosine")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--backbone", choices=("baseline", "clap"), default="baseline")
    parser.add_argument("--split", default="test")
    parser.add_argument("--batch-size", type=int, default=33)
    parser.add_argument("--pair-seed", type=int, default=0)
    parser.add_argument("--negative-seed", type=int, default=17)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", required=True)
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


def _make_side_prefixes(model, batch, side: str) -> torch.Tensor:
    node_features = batch.get(f"node_features_{side}")
    edge_index, edge_types = structural_edges(batch, side)
    if node_features is not None:
        node_features = node_features.float()
    if edge_index is not None:
        edge_index = edge_index.long()
    batch_size = batch[f"input_ids_{side}"].shape[0]
    graph_batch = batch.get(f"graph_batch_{side}")
    if hasattr(model, "_make_prefixes"):
        return model._make_prefixes(
            node_features,
            edge_index,
            graph_batch,
            edge_types,
            batch_size,
            batch[f"input_ids_{side}"].device,
        )
    if hasattr(model, "_prefixes"):
        return model._prefixes(
            batch_size,
            node_features,
            edge_index,
            edge_types,
            graph_batch,
        )
    raise TypeError("The selected model does not expose a prefix-construction method")


@torch.inference_mode()
def collect_prefixes(model, loader, device, pair_seed: int):
    if loader.num_workers != 0:
        raise ValueError("Fixed compilation pairs require DataLoader num_workers=0")
    reset_rng(pair_seed)
    model.eval()
    prefixes_a, prefixes_b, batch_sizes = [], [], []
    for batch in loader:
        batch = to_device(batch, device)
        with torch.amp.autocast(device.type, enabled=False):
            side_a = _make_side_prefixes(model, batch, "a")
            side_b = _make_side_prefixes(model, batch, "b")
        if side_a.shape != side_b.shape or side_a.ndim != 3:
            raise ValueError("Model prefixes must be paired [batch, prefix, dimension] tensors")
        prefixes_a.append(side_a.detach().float().cpu())
        prefixes_b.append(side_b.detach().float().cpu())
        batch_sizes.append(int(side_a.shape[0]))
    if not prefixes_a:
        raise RuntimeError("Evaluation loader was empty")
    return torch.cat(prefixes_a), torch.cat(prefixes_b), batch_sizes


def _comparison(matched, deranged):
    matched_summary = summarize(matched)
    deranged_summary = summarize(deranged)
    return {
        "matched": matched_summary,
        "deranged": deranged_summary,
        "mean_gap_matched_minus_deranged": matched_summary["mean"] - deranged_summary["mean"],
    }


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.batch_size < 2:
        raise ValueError("batch-size must be at least two")
    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {output_dir}")
    device = torch.device(args.device)
    config = load_config(args.config)
    model, checkpoint = _build_and_load(config, args.backbone, args.checkpoint, device)
    if getattr(model, "prefix_source", "gnn") in {"none", "register"}:
        raise ValueError("Prefix cosine requires function-dependent GNN prefixes")
    loader, expected_pool_size = build_split_loader(config, args.split, args.batch_size)
    prefixes_a, prefixes_b, batch_sizes = collect_prefixes(model, loader, device, args.pair_seed)
    if prefixes_a.shape[0] != expected_pool_size:
        raise RuntimeError(
            f"Collected n={prefixes_a.shape[0]} prefixes, expected n={expected_pool_size}"
        )
    values = prefix_cosines(
        prefixes_a,
        prefixes_b,
        batch_sizes,
        negative_seed=args.negative_seed,
    )
    identity = torch.arange(expected_pool_size).numpy()
    if np.any(values["permutation"] == identity):
        raise RuntimeError("Negative pairing was not a true derangement")

    result = {
        "experiment": "fixed_pair_prefix_cosine",
        "backbone": args.backbone,
        **provenance(args.config, args.checkpoint),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "pool": pool_metadata(loader, args.split),
        "pool_size": expected_pool_size,
        "batch_size": args.batch_size,
        "batch_sizes": batch_sizes,
        "pair_seed": args.pair_seed,
        "negative_seed": args.negative_seed,
        "negative_pairing": (
            "derangement within each evaluation batch; singleton batches are paired "
            "across batch boundaries"
        ),
        "raw": _comparison(values["raw_matched"], values["raw_deranged"]),
        "batch_mean_centered": {
            "definition": (
                "subtract each side's batch mean at every (prefix-token, dimension), "
                "then flatten; multiple singleton batches are centered together, while "
                "one singleton uses the final non-singleton batch mean"
            ),
            **_comparison(values["centered_matched"], values["centered_deranged"]),
        },
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / "prefix_cosine.json"
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    np.savez_compressed(
        output_dir / "prefix_cosine_values.npz",
        raw_matched=values["raw_matched"],
        raw_deranged=values["raw_deranged"],
        centered_matched=values["centered_matched"],
        centered_deranged=values["centered_deranged"],
        permutation=values["permutation"],
        batch_sizes=np.asarray(batch_sizes, dtype=np.int32),
    )
    logger.info(
        "Wrote %s: raw gap %+.4f, centered gap %+.4f, n=%d",
        output,
        result["raw"]["mean_gap_matched_minus_deranged"],
        result["batch_mean_centered"]["mean_gap_matched_minus_deranged"],
        expected_pool_size,
    )
    return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()
