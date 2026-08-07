"""Frozen CLAP with a trainable retrieval readout."""

from __future__ import annotations

from collections.abc import Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.clap_frozen.backbone import CLAP_MODEL_ID, CLAP_REVISION, load_clap_backbone


def masked_mean(hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
    return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1e-9)


class CLAPFrozenBaseline(nn.Module):
    """Official frozen CLAP backbone plus projection and feature fusion."""

    def __init__(
        self,
        pretrained_model: str = CLAP_MODEL_ID,
        revision: str | None = CLAP_REVISION,
        func_feat_dim: int = 132,
        embed_dim: int = 768,
        dropout: float = 0.1,
        *,
        backbone: nn.Module | None = None,
        local_files_only: bool = False,
    ):
        super().__init__()
        self.pretrained_model = pretrained_model
        self.revision = revision
        self.func_feat_dim = func_feat_dim
        self.roformer = backbone or load_clap_backbone(
            pretrained_model,
            revision,
            local_files_only=local_files_only,
        )
        self.config = self.roformer.config
        self.hidden_size = int(self.config.hidden_size)
        self.num_layers = int(self.config.num_hidden_layers)
        self.roformer.requires_grad_(False)
        self.roformer.eval()

        self.projection = nn.Linear(self.hidden_size, embed_dim)
        self.fusion = nn.Sequential(
            nn.Linear(embed_dim + func_feat_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim),
        )

    def train(self, mode: bool = True):
        super().train(mode)
        self.roformer.eval()
        return self

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor | None = None,
        function_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        output = self.roformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True,
        )
        pooled = masked_mean(output.last_hidden_state, attention_mask)
        embedding = self.projection(pooled)
        if function_features is not None:
            embedding = self.fusion(torch.cat((embedding, function_features), dim=-1))
        return F.normalize(embedding, dim=-1)

    def get_embedding_pairs(
        self,
        batch: Mapping[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self._forward_side(batch, "a"), self._forward_side(batch, "b")

    def _forward_side(self, batch: Mapping[str, torch.Tensor], side: str) -> torch.Tensor:
        return self(
            input_ids=batch[f"input_ids_{side}"],
            attention_mask=batch[f"attention_mask_{side}"],
            token_type_ids=batch.get(f"token_type_ids_{side}"),
            function_features=batch.get(f"function_features_{side}"),
        )
