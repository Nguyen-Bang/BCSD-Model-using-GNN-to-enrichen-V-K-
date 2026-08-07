"""Construct frozen CLAP models and load their checkpoints."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from models.clap_frozen import (
    CLAP_MODEL_ID,
    CLAP_REVISION,
    CLAPFrozenBaseline,
    CLAPFrozenGNN,
)


def create_model(
    config: Mapping[str, Any],
    device: torch.device,
    *,
    backbone: nn.Module | None = None,
    local_files_only: bool = False,
) -> nn.Module:
    model_config = config["model"]
    variant = str(model_config.get("variant", "baseline"))
    common = {
        "pretrained_model": str(model_config.get("pretrained_model", CLAP_MODEL_ID)),
        "revision": model_config.get("revision", CLAP_REVISION),
        "func_feat_dim": int(model_config.get("func_feature_dim", 132)),
        "embed_dim": int(model_config.get("embed_dim", 768)),
        "dropout": float(model_config.get("dropout", 0.1)),
        "backbone": backbone,
        "local_files_only": local_files_only,
    }
    if variant == "baseline":
        if "gnn" in config:
            raise ValueError("Baseline CLAP config must not contain a gnn section")
        model = CLAPFrozenBaseline(**common)
    elif variant in {"gnn", "register"}:
        gnn = config["gnn"]
        if gnn.get("attention_mode", "tanh_per_pair") != "tanh_per_pair":
            raise ValueError("Only tanh_per_pair prefix attention is supported")
        if not bool(gnn.get("multi_token", True)):
            raise ValueError("Single-token prefixes are not supported")
        hidden_dim = int(gnn.get("hidden_dim", 256))
        if int(gnn.get("prefix_dim", hidden_dim)) != hidden_dim:
            raise ValueError("gnn.prefix_dim must match gnn.hidden_dim")
        prefix_source = str(gnn.get("prefix_source", variant))
        if prefix_source != variant:
            raise ValueError(f"model.variant={variant!r} requires gnn.prefix_source={variant!r}")
        model = CLAPFrozenGNN(
            **common,
            gnn_input_dim=int(gnn.get("input_dim", 20)),
            gnn_hidden_dim=hidden_dim,
            gnn_layers=int(gnn.get("num_layers", 3)),
            gnn_gat_heads=int(gnn.get("gat_heads", 4)),
            gnn_pool_heads=int(gnn.get("pool_heads", 10)),
            gnn_encoder_type=str(gnn.get("encoder_type", "graph")),
            gnn_set_hidden_multiplier=int(gnn.get("hidden_multiplier", 4)),
            gnn_use_edge_features=bool(gnn.get("use_edge_features", True)),
            gnn_num_edge_types=int(gnn.get("num_edge_types", 11)),
            gnn_edge_emb_dim=int(gnn.get("edge_emb_dim", 16)),
            gnn_pool_query_init=str(gnn.get("pool_query_init", "ones")),
            gnn_attention_dropout=(
                float(gnn["attention_dropout"]) if "attention_dropout" in gnn else None
            ),
            prefix_source=prefix_source,
        )
    else:
        raise ValueError("model.variant must be 'baseline', 'register', or 'gnn'")
    declared_hidden_size = model_config.get("hidden_size")
    if declared_hidden_size is not None and int(declared_hidden_size) != model.hidden_size:
        raise ValueError(
            f"model.hidden_size={declared_hidden_size} does not match "
            f"the CLAP backbone ({model.hidden_size})"
        )
    return model.to(device)


def load_checkpoint_strict(
    model: nn.Module,
    checkpoint_path: str | Path,
    device: torch.device,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    del device
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise KeyError("CLAP checkpoint must contain model_state_dict")
    checkpoint_config = checkpoint.get("config")
    if not isinstance(checkpoint_config, dict):
        raise KeyError("CLAP checkpoint must contain its resolved config")
    if checkpoint_config != dict(config):
        raise ValueError("Checkpoint config does not match evaluation config")
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    return checkpoint


def load_model(
    config: Mapping[str, Any],
    checkpoint: str | Path,
    device: torch.device,
    *,
    local_files_only: bool = False,
) -> nn.Module:
    model = create_model(config, device, local_files_only=local_files_only)
    load_checkpoint_strict(model, checkpoint, device, config)
    return model
