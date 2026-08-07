"""Factory for the baseline structural-prefix model."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

import torch

from models.baseline_gnn_model import BaselineGNNModel

from .checkpoint import load_backbone_checkpoint

logger = logging.getLogger("bcsd.models.baseline")


def create_model(
    config: Mapping[str, Any],
    backbone: str,
    device: torch.device,
    backbone_checkpoint: str | None = None,
    *,
    initialize_embeddings: bool | None = None,
) -> BaselineGNNModel:
    """Build the canonical three-layer baseline with structural prefixes."""
    if backbone != "baseline":
        raise ValueError(f"Unsupported backbone: {backbone}")

    if initialize_embeddings is None:
        initialize_embeddings = backbone_checkpoint is None
    model_config = config["model"]
    gnn_config = config["gnn"]
    if gnn_config.get("attention_mode", "tanh_per_pair") != "tanh_per_pair":
        raise ValueError("Only tanh_per_pair prefix attention is supported")
    if not bool(gnn_config.get("multi_token", True)):
        raise ValueError("Single-token prefixes are not supported")
    if gnn_config.get("prefix_mode", "pool") != "pool":
        raise ValueError("Only pooled graph prefixes are supported")
    if int(model_config.get("kv_low_rank", 0)) != 0:
        raise ValueError("Low-rank prefix projections are not supported")
    if int(gnn_config.get("prefix_dim", gnn_config.get("hidden_dim", 256))) != int(
        gnn_config.get("hidden_dim", 256)
    ):
        raise ValueError("gnn.prefix_dim must match gnn.hidden_dim")

    model = BaselineGNNModel(
        num_layers=int(model_config.get("num_layers", 3)),
        hidden_size=int(model_config.get("hidden_size", 768)),
        num_heads=int(model_config.get("num_heads", 12)),
        vocab_size=int(model_config.get("vocab_size", 33555)),
        max_seq_length=int(model_config.get("max_seq_length", 1024)),
        clap_embedding_init=(
            bool(model_config.get("clap_embedding_init", True)) and initialize_embeddings
        ),
        func_feat_dim=int(model_config.get("func_feature_dim", 132)),
        embed_dim=int(model_config.get("embed_dim", 768)),
        dropout=float(model_config.get("dropout", 0.1)),
        gnn_input_dim=int(gnn_config.get("input_dim", 20)),
        gnn_hidden_dim=int(gnn_config.get("hidden_dim", 256)),
        gnn_layers=int(gnn_config.get("num_layers", 3)),
        gnn_gat_heads=int(gnn_config.get("gat_heads", 4)),
        gnn_pool_heads=int(gnn_config.get("pool_heads", 10)),
        freeze_backbone=False,
        gnn_encoder_type=str(gnn_config.get("encoder_type", "graph")),
        gnn_set_hidden_multiplier=int(gnn_config.get("hidden_multiplier", 4)),
        gnn_use_edge_features=bool(gnn_config.get("use_edge_features", False)),
        gnn_num_edge_types=int(gnn_config.get("num_edge_types", 11)),
        gnn_edge_emb_dim=int(gnn_config.get("edge_emb_dim", 16)),
        gnn_pool_query_init=str(gnn_config.get("pool_query_init", "ones")),
        gnn_attention_dropout=(
            float(gnn_config["attention_dropout"]) if "attention_dropout" in gnn_config else None
        ),
        prefix_source=str(gnn_config.get("prefix_source", "gnn")),
    )
    if backbone_checkpoint:
        loaded = load_backbone_checkpoint(model, backbone_checkpoint)
        logger.info("Loaded %d matching baseline parameters", loaded)
    elif initialize_embeddings:
        logger.warning("No baseline checkpoint; the frozen backbone may be random")
    model._freeze_backbone()
    return model.to(device)
