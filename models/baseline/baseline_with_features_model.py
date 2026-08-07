"""
Baseline + Function Features Model.

Extends BaselineModel with a fusion MLP that concatenates 132-dim
hand-engineered function features (4 log-scaled scalars + 128-dim
opcode MinHash) into the final embedding.

Architecture:
    rebased_instructions -> ClapASMTokenizer
        -> input_ids + attention_mask + token_type_ids
        -> 3-layer RoFormer (random init, all trainable)
        -> masked mean pooling
        -> projection [768]
        -> concat function_features [132]
        -> fusion MLP [768]
        -> L2 normalize
        -> embedding [768]

Module: models.baseline.baseline_with_features_model
"""

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from .baseline_model import BaselineModel

logger = logging.getLogger("bcsd.models")


class BaselineWithFeaturesModel(BaselineModel):
    """
    BaselineModel + hand-engineered function feature fusion.

    Inherits all transformer + MLM components from BaselineModel and adds
    a 2-layer fusion MLP that concatenates function-level features into
    the embedding before L2 normalization.

    Args:
        func_feat_dim: Dimension of function-level feature vector (default: 132).
        **kwargs: All BaselineModel arguments (num_layers, hidden_size, etc.).
    """

    def __init__(self, func_feat_dim: int = 132, **kwargs):
        super().__init__(**kwargs)

        self.func_feat_dim = func_feat_dim
        self.fusion = nn.Sequential(
            nn.Linear(self.embed_dim + func_feat_dim, self.embed_dim),
            nn.GELU(),
            nn.Linear(self.embed_dim, self.embed_dim),
        )

        param_count = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(
            f"BaselineWithFeaturesModel: +fusion MLP (feat_dim={func_feat_dim}), "
            f"{param_count / 1e6:.1f}M params ({trainable / 1e6:.1f}M trainable)"
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor | None = None,
        function_features: torch.Tensor | None = None,
        return_logits: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with optional function feature fusion.

        Args:
            input_ids: [B, seq_len] token IDs.
            attention_mask: [B, seq_len] 1 for real tokens, 0 for padding.
            token_type_ids: [B, seq_len] INSTR{N} token IDs (optional).
            function_features: [B, func_feat_dim] function-level features (optional).
            return_logits: If True, also return MLM logits.

        Returns:
            If return_logits=False: [B, embed_dim] L2-normalized embeddings.
            If return_logits=True: (embeddings, mlm_logits).
        """
        projected, mlm_logits = self._encode_hidden(
            input_ids, attention_mask, token_type_ids, return_logits
        )

        if function_features is not None:
            combined = torch.cat([projected, function_features], dim=-1)
            projected = self.fusion(combined)

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
        Process a Siamese batch with function features.

        Expects batch keys: input_ids_a/b (or masked_input_ids_a/b),
        attention_mask_a/b, token_type_ids_a/b, function_features_a/b.
        """
        input_key_a = "masked_input_ids_a" if "masked_input_ids_a" in batch else "input_ids_a"
        input_key_b = "masked_input_ids_b" if "masked_input_ids_b" in batch else "input_ids_b"

        out_a = self(
            input_ids=batch[input_key_a],
            attention_mask=batch["attention_mask_a"],
            token_type_ids=batch.get("token_type_ids_a"),
            function_features=batch.get("function_features_a"),
            return_logits=return_logits,
        )
        out_b = self(
            input_ids=batch[input_key_b],
            attention_mask=batch["attention_mask_b"],
            token_type_ids=batch.get("token_type_ids_b"),
            function_features=batch.get("function_features_b"),
            return_logits=return_logits,
        )

        if return_logits:
            emb_a, logits_a = out_a
            emb_b, logits_b = out_b
            return emb_a, emb_b, logits_a, logits_b
        return out_a, out_b
