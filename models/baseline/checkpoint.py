"""Checkpoint loading for the baseline backbone."""

from __future__ import annotations

from pathlib import Path

import torch


def load_backbone_checkpoint(
    model: torch.nn.Module,
    path: str | Path,
) -> int:
    """Warm-start matching baseline weights and reject incompatible shapes."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if "model_state_dict" not in checkpoint:
        raise KeyError(f"Backbone checkpoint has no model_state_dict: {path}")

    source = checkpoint["model_state_dict"]
    target = model.state_dict()
    graph_prefixes = ("gnn_head.", "kv_prefix_layers.")
    expected = {
        key
        for key in target
        if not key.startswith(graph_prefixes)
        and key not in {"register_prefix", "_rand_node_primes"}
    }
    missing = expected.difference(source)
    unexpected = set(source).difference(expected)
    if missing or unexpected:
        raise RuntimeError(
            "Backbone checkpoint key mismatch: "
            f"missing={sorted(missing)[:5]}, unexpected={sorted(unexpected)[:5]}"
        )

    mismatched = [key for key in expected if source[key].shape != target[key].shape]
    if mismatched:
        raise RuntimeError(f"Backbone checkpoint shape mismatch: {mismatched[:5]}")

    target.update({key: source[key] for key in expected})
    model.load_state_dict(target, strict=True)
    return len(expected)
