"""
Baseline Model: 3-layer RoFormer with CLAP embedding initialization.

A small transformer trained from scratch for binary code similarity
detection. Uses CLAP's pretrained word embeddings as initialization
for faster convergence, but all transformer layers are randomly
initialized and trained from scratch.

Architecture:
    rebased_instructions -> ClapASMTokenizer
        -> input_ids + attention_mask + token_type_ids
        -> 3-layer RoFormer (random init, all trainable)
        -> masked mean pooling
        -> projection [768]
        -> L2 normalize
        -> embedding [768]

The token_type_embeddings share weights with word_embeddings,
matching CLAP's JRoFormerEmbeddings design. This means INSTR{N}
tokens in token_type_ids get looked up in the same 33k vocabulary
embedding table, providing instruction-boundary awareness.

Module: models.baseline.baseline_model
"""

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import RoFormerConfig, RoFormerModel

logger = logging.getLogger("bcsd.models")

CLAP_VOCAB_SIZE = 33555
CLAP_HIDDEN_SIZE = 768
CLAP_PAD_TOKEN_ID = 1
CLAP_MODEL_ID = "hustcw/clap-asm"
CLAP_REVISION = "620f4beba2edce172e8f35e263399716494950c9"


class BaselineModel(nn.Module):
    """
    From-scratch 3-layer RoFormer baseline for BCSD.

    Only the word embedding weights are initialized from CLAP's pretrained
    model. All transformer layers and the projection head are randomly
    initialized and fully trainable.

    This is the pure transformer baseline — no hand-engineered function
    features. See BaselineWithFeaturesModel for the feature-augmented variant.

    Args:
        num_layers: Number of transformer layers (default: 3).
        hidden_size: Hidden dimension (default: 768, must match CLAP).
        num_heads: Number of attention heads (default: 12).
        vocab_size: Vocabulary size (default: 33555, CLAP vocab).
        max_seq_length: Max sequence length (default: 1024).
        embed_dim: Output embedding dimension (default: 768).
        dropout: Dropout probability (default: 0.1).
        clap_embedding_init: Whether to load CLAP pretrained embeddings.
    """

    def __init__(
        self,
        num_layers: int = 3,
        hidden_size: int = CLAP_HIDDEN_SIZE,
        num_heads: int = 12,
        vocab_size: int = CLAP_VOCAB_SIZE,
        max_seq_length: int = 1024,
        embed_dim: int = CLAP_HIDDEN_SIZE,
        dropout: float = 0.1,
        clap_embedding_init: bool = True,
        clap_model_id: str = CLAP_MODEL_ID,
        clap_revision: str | None = CLAP_REVISION,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.embed_dim = embed_dim

        config = RoFormerConfig(
            vocab_size=vocab_size,
            embedding_size=hidden_size,
            hidden_size=hidden_size,
            num_hidden_layers=num_layers,
            num_attention_heads=num_heads,
            intermediate_size=hidden_size * 4,
            max_position_embeddings=max_seq_length,
            rotary_value=False,
            pad_token_id=CLAP_PAD_TOKEN_ID,
            hidden_dropout_prob=dropout,
            attention_probs_dropout_prob=dropout,
        )
        self.roformer = RoFormerModel(config)

        # Replace token_type_embeddings with a full-vocab tied view of
        # word_embeddings (CLAP's JRoFormerEmbeddings design: INSTR{N}
        # token_type_ids index the same 33k vocabulary). DDP supports
        # shared parameters, so we tie the tensor directly instead of
        # keeping an independent copy.
        self.roformer.embeddings.token_type_embeddings = nn.Embedding(vocab_size, hidden_size)
        self.roformer.embeddings.token_type_embeddings.weight = (
            self.roformer.embeddings.word_embeddings.weight
        )

        if clap_embedding_init:
            self._init_embeddings_from_clap(clap_model_id, clap_revision)

        self.projection = nn.Linear(hidden_size, embed_dim)

        # MLM head weight-tied to word embeddings (standard BERT/RoBERTa/CLAP
        # practice). Saves ~25.8M params vs. an independent copy.
        self.mlm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        self.mlm_head.weight = self.roformer.embeddings.word_embeddings.weight

        param_count = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(
            f"BaselineModel: {num_layers}L RoFormer, "
            f"{param_count / 1e6:.1f}M params ({trainable / 1e6:.1f}M trainable), "
            f"CLAP embedding init: {clap_embedding_init}"
        )

    def _init_embeddings_from_clap(self, model_id: str, revision: str | None) -> None:
        """Load the CLAP embedding table, optionally at a pinned Hub revision."""
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file

        path = hf_hub_download(
            repo_id=model_id,
            filename="model.safetensors",
            revision=revision,
        )
        state_dict = load_file(path)
        embedding_key = "jroformer.embeddings.word_embeddings.weight"
        if embedding_key not in state_dict:
            raise KeyError(f"{embedding_key!r} not found in {model_id!r}")

        source = state_dict[embedding_key]
        target = self.roformer.embeddings.word_embeddings.weight
        if source.shape != target.shape:
            raise ValueError(
                f"CLAP embedding shape {tuple(source.shape)} does not match "
                f"baseline shape {tuple(target.shape)}"
            )
        with torch.no_grad():
            target.copy_(source)
        logger.info(
            "Loaded CLAP embeddings from %s at revision %s",
            model_id,
            revision or "main",
        )

    def _encode_hidden(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor | None = None,
        return_logits: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Encode inputs through transformer, pool, and project.

        Returns the projected embedding (before L2 normalization) and
        optional MLM logits. Subclasses can override forward() to insert
        additional processing (e.g. feature fusion) between projection
        and normalization.

        Returns:
            (projected, mlm_logits_or_None)
        """
        outputs = self.roformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        hidden = outputs.last_hidden_state  # [B, seq_len, hidden_size]

        # MLM logits (before pooling destroys per-token info)
        mlm_logits = self.mlm_head(hidden) if return_logits else None

        # Masked mean pooling (matching CLAP's AsmEncoder)
        mask = attention_mask.unsqueeze(-1).expand(hidden.size()).to(hidden.dtype)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)

        projected = self.projection(pooled)  # [B, embed_dim]
        return projected, mlm_logits

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor | None = None,
        return_logits: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass producing L2-normalized embeddings and optional MLM logits.

        Args:
            input_ids: [B, seq_len] token IDs.
            attention_mask: [B, seq_len] 1 for real tokens, 0 for padding.
            token_type_ids: [B, seq_len] INSTR{N} token IDs (optional).
            return_logits: If True, also return MLM logits.

        Returns:
            If return_logits=False: [B, embed_dim] L2-normalized embeddings.
            If return_logits=True: (embeddings, mlm_logits) where
                mlm_logits is [B, seq_len, vocab_size].
        """
        projected, mlm_logits = self._encode_hidden(
            input_ids, attention_mask, token_type_ids, return_logits
        )
        embedding = F.normalize(projected, p=2, dim=1)

        if return_logits:
            return embedding, mlm_logits
        return embedding

    def get_embedding_pairs(
        self,
        batch: dict[str, torch.Tensor],
        return_logits: bool = False,
    ):
        """
        Process a Siamese batch, returning (emb_a, emb_b) or
        (emb_a, emb_b, logits_a, logits_b) if return_logits=True.

        Expects batch keys: input_ids_a/b (or masked_input_ids_a/b),
        attention_mask_a/b, token_type_ids_a/b.
        """
        input_key_a = "masked_input_ids_a" if "masked_input_ids_a" in batch else "input_ids_a"
        input_key_b = "masked_input_ids_b" if "masked_input_ids_b" in batch else "input_ids_b"

        out_a = self(
            input_ids=batch[input_key_a],
            attention_mask=batch["attention_mask_a"],
            token_type_ids=batch.get("token_type_ids_a"),
            return_logits=return_logits,
        )
        out_b = self(
            input_ids=batch[input_key_b],
            attention_mask=batch["attention_mask_b"],
            token_type_ids=batch.get("token_type_ids_b"),
            return_logits=return_logits,
        )

        if return_logits:
            emb_a, logits_a = out_a
            emb_b, logits_b = out_b
            return emb_a, emb_b, logits_a, logits_b
        return out_a, out_b
