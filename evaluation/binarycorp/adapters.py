"""Model adapters for BinaryCorp single-function encoding."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol

import torch
import torch.nn.functional as F
import yaml

MODEL_KINDS = (
    "clap-zero-shot",
    "projection-fusion",
    "typed-gnn",
    "set-encoder",
)
DEFAULT_CLAP_MODEL = "hustcw/clap-asm"
DEFAULT_CLAP_REVISION = "620f4beba2edce172e8f35e263399716494950c9"


class EmbeddingAdapter(Protocol):
    """Encode a batch of independent functions."""

    include_function_features: bool
    include_graph: bool

    def encode(self, batch: dict[str, Any]) -> torch.Tensor: ...


def load_model_config(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    path = Path(path)
    with path.open(encoding="utf-8") as handle:
        config = json.load(handle) if path.suffix.lower() == ".json" else yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Model config must be a mapping: {path}")
    return config


def _to_device(
    batch: Mapping[str, Any],
    keys: Sequence[str],
    device: torch.device,
) -> dict[str, Any]:
    moved = {}
    for key in keys:
        value = batch.get(key)
        if isinstance(value, torch.Tensor):
            moved[key] = value.to(device)
    return moved


class ZeroShotClapAdapter:
    include_function_features = False
    include_graph = False

    def __init__(self, encoder: torch.nn.Module, device: torch.device):
        self.encoder = encoder.eval().to(device)
        self.device = device

    @torch.inference_mode()
    def encode(self, batch: dict[str, Any]) -> torch.Tensor:
        batch = _to_device(
            batch,
            ("input_ids", "attention_mask", "token_type_ids"),
            self.device,
        )
        output = self.encoder(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            token_type_ids=batch.get("token_type_ids"),
        )
        if not isinstance(output, torch.Tensor) or output.ndim != 2:
            raise TypeError("Official CLAP encoder must return [batch, embedding]")
        return F.normalize(output.float(), dim=1)


class ModelAdapter:
    """Call a model's direct single-function forward API."""

    def __init__(self, model: torch.nn.Module, model_kind: str, device: torch.device):
        self.model = model.eval()
        self.model_kind = model_kind
        self.device = device
        self.include_function_features = True
        self.include_graph = model_kind in {"typed-gnn", "set-encoder"}

    @torch.inference_mode()
    def encode(self, batch: dict[str, Any]) -> torch.Tensor:
        keys = ["input_ids", "attention_mask", "token_type_ids", "function_features"]
        if self.include_graph:
            keys.extend(("node_features", "batch"))
        if self.model_kind == "typed-gnn":
            keys.extend(("gnn_edge_index", "gnn_edge_types"))
        batch = _to_device(batch, keys, self.device)
        common = {
            "input_ids": batch["input_ids"],
            "attention_mask": batch["attention_mask"],
            "token_type_ids": batch.get("token_type_ids"),
            "function_features": batch.get("function_features"),
        }
        if self.model_kind in {"typed-gnn", "set-encoder"}:
            common.update(
                {
                    "node_features": batch.get("node_features"),
                    "graph_batch": batch.get("batch"),
                }
            )
        if self.model_kind == "typed-gnn":
            common.update(
                {
                    "edge_index": batch["gnn_edge_index"],
                    "edge_types": batch["gnn_edge_types"],
                }
            )
        output = self.model(**common)
        if isinstance(output, tuple):
            output = output[0]
        if not isinstance(output, torch.Tensor) or output.ndim != 2:
            raise TypeError("Model adapter expected a [batch, embedding] tensor")
        return F.normalize(output.float(), dim=1)


def _validate_kind_config(model_kind: str, config: Mapping[str, Any]) -> None:
    model_config = config.get("model", {})
    gnn_config = config.get("gnn", {})
    if not isinstance(model_config, Mapping) or not isinstance(gnn_config, Mapping):
        raise ValueError("model and gnn config sections must be mappings")
    variant = model_config.get("variant")
    if model_kind == "projection-fusion" and variant != "baseline":
        raise ValueError("projection-fusion requires model.variant: baseline")
    if model_kind == "typed-gnn":
        if variant != "gnn":
            raise ValueError("typed-gnn requires model.variant: gnn")
        if gnn_config.get("encoder_type") != "graph":
            raise ValueError("typed-gnn requires gnn.encoder_type: graph")
        if not gnn_config.get("use_edge_features") or not gnn_config.get("use_reverse_edges"):
            raise ValueError("typed-gnn requires typed edge features and reverse edges")
    if model_kind == "set-encoder":
        if variant != "gnn" or gnn_config.get("encoder_type") != "set":
            raise ValueError("set-encoder requires model.variant: gnn and gnn.encoder_type: set")
        if gnn_config.get("use_edge_features", False):
            raise ValueError("set-encoder cannot use edge features")


def create_adapter(
    *,
    model_kind: str,
    checkpoint: str | Path | None,
    config: Mapping[str, Any],
    device: torch.device,
    pretrained: str = DEFAULT_CLAP_MODEL,
    revision: str = DEFAULT_CLAP_REVISION,
    local_files_only: bool = False,
) -> EmbeddingAdapter:
    """Create a built-in BinaryCorp model adapter."""
    if model_kind not in MODEL_KINDS:
        raise ValueError(f"Unknown BinaryCorp model kind: {model_kind}")

    if model_kind == "clap-zero-shot":
        from models.clap_frozen import load_clap_encoder

        encoder = load_clap_encoder(
            pretrained,
            revision,
            local_files_only=local_files_only,
        )
        return ZeroShotClapAdapter(encoder, device)

    if checkpoint is None:
        raise ValueError(f"{model_kind} requires a checkpoint")
    _validate_kind_config(model_kind, config)
    from models.clap_frozen.factory import load_model

    model = load_model(
        config,
        checkpoint,
        device,
        local_files_only=local_files_only,
    )
    return ModelAdapter(model, model_kind, device)


def collation_options(model_kind: str, config: Mapping[str, Any]) -> tuple[bool, bool]:
    if model_kind not in {"typed-gnn", "set-encoder"}:
        return False, False
    gnn_config = config.get("gnn", {})
    if not isinstance(gnn_config, Mapping):
        raise ValueError("gnn config section must be a mapping")
    return (
        bool(gnn_config.get("use_edge_features", False)),
        bool(gnn_config.get("use_reverse_edges", False)),
    )
