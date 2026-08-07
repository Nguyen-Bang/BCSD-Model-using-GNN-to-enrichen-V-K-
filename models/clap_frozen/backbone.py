"""Strict loading utilities for the official CLAP assembly encoder."""

from __future__ import annotations

from pathlib import Path

CLAP_MODEL_ID = "hustcw/clap-asm"
CLAP_REVISION = "620f4beba2edce172e8f35e263399716494950c9"
CLAP_HIDDEN_SIZE = 768
CLAP_NUM_HEADS = 12
CLAP_NUM_LAYERS = 12
CLAP_VOCAB_SIZE = 33_555
CLAP_MAX_POSITIONS = 2_048


def _validate_config(config) -> None:
    expected = {
        "hidden_size": CLAP_HIDDEN_SIZE,
        "num_attention_heads": CLAP_NUM_HEADS,
        "num_hidden_layers": CLAP_NUM_LAYERS,
        "vocab_size": CLAP_VOCAB_SIZE,
        "max_position_embeddings": CLAP_MAX_POSITIONS,
        "use_bias": False,
    }
    mismatches = {
        name: (getattr(config, name, None), value)
        for name, value in expected.items()
        if getattr(config, name, None) != value
    }
    if mismatches:
        raise ValueError(f"Unexpected CLAP backbone geometry: {mismatches}")


def load_clap_encoder(
    pretrained_model: str | Path = CLAP_MODEL_ID,
    revision: str | None = CLAP_REVISION,
    *,
    local_files_only: bool = False,
):
    """Load CLAP's pinned custom encoder and weights."""
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_model
    from transformers import AutoConfig, AutoModel

    model_id = str(pretrained_model)
    config = AutoConfig.from_pretrained(
        model_id,
        revision=revision,
        trust_remote_code=True,
        local_files_only=local_files_only,
    )
    _validate_config(config)
    encoder = AutoModel.from_config(
        config,
        trust_remote_code=True,
        code_revision=revision,
    )
    model_path = Path(model_id)
    if model_path.is_dir():
        checkpoint = model_path / "model.safetensors"
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Missing CLAP checkpoint: {checkpoint}")
    else:
        checkpoint = Path(
            hf_hub_download(
                repo_id=model_id,
                filename="model.safetensors",
                revision=revision,
                local_files_only=local_files_only,
            )
        )
    missing, unexpected = load_model(
        encoder,
        checkpoint,
        strict=True,
        device="cpu",
    )
    if missing or unexpected:
        raise RuntimeError(f"CLAP checkpoint mismatch: missing={missing}, unexpected={unexpected}")
    if not hasattr(encoder, "jroformer"):
        raise AttributeError("Official CLAP encoder has no jroformer backbone")
    return encoder


def load_clap_backbone(
    pretrained_model: str | Path = CLAP_MODEL_ID,
    revision: str | None = CLAP_REVISION,
    *,
    local_files_only: bool = False,
):
    """Load the frozen token backbone used by the trainable experiment heads."""
    encoder = load_clap_encoder(
        pretrained_model,
        revision,
        local_files_only=local_files_only,
    )
    return encoder.jroformer
