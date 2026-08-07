"""Typed-edge graph prefixes for a frozen CLAP backbone."""

from __future__ import annotations

from collections.abc import Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.clap_frozen.backbone import CLAP_MODEL_ID, CLAP_REVISION, load_clap_backbone
from models.clap_frozen.baseline import masked_mean
from models.structural_prefix import (
    KVPrefixAttention,
    StructuralPrefixHead,
    build_structural_prefix_head,
    structural_edges,
)


class CLAPFrozenGNN(nn.Module):
    """Frozen CLAP with either graph-conditioned or shared register prefixes."""

    def __init__(
        self,
        pretrained_model: str = CLAP_MODEL_ID,
        revision: str | None = CLAP_REVISION,
        func_feat_dim: int = 132,
        embed_dim: int = 768,
        dropout: float = 0.1,
        gnn_input_dim: int = 20,
        gnn_hidden_dim: int = 256,
        gnn_layers: int = 3,
        gnn_gat_heads: int = 4,
        gnn_pool_heads: int = 10,
        gnn_encoder_type: str = "graph",
        gnn_set_hidden_multiplier: int = 4,
        gnn_use_edge_features: bool = True,
        gnn_num_edge_types: int = 11,
        gnn_edge_emb_dim: int = 16,
        gnn_pool_query_init: str = "ones",
        gnn_attention_dropout: float | None = None,
        prefix_source: str = "gnn",
        *,
        backbone: nn.Module | None = None,
        local_files_only: bool = False,
    ):
        super().__init__()
        if prefix_source not in {"gnn", "register"}:
            raise ValueError("prefix_source must be 'gnn' or 'register'")
        self.pretrained_model = pretrained_model
        self.revision = revision
        self.prefix_source = prefix_source
        self.func_feat_dim = func_feat_dim
        self.num_prefix_tokens = gnn_pool_heads
        self.prefix_dim = gnn_hidden_dim
        self.roformer = backbone or load_clap_backbone(
            pretrained_model,
            revision,
            local_files_only=local_files_only,
        )
        self.config = self.roformer.config
        self.hidden_size = int(self.config.hidden_size)
        self.num_heads = int(self.config.num_attention_heads)
        self.num_layers = int(self.config.num_hidden_layers)
        if self.hidden_size % self.num_heads:
            raise ValueError("Backbone hidden size must be divisible by its attention heads")
        self.head_dim = self.hidden_size // self.num_heads
        self.roformer.requires_grad_(False)
        self.roformer.eval()

        self.kv_prefix_layers = nn.ModuleList(
            KVPrefixAttention(
                hidden_size=self.hidden_size,
                num_heads=self.num_heads,
                prefix_dim=gnn_hidden_dim,
                num_prefix_tokens=gnn_pool_heads,
                dropout=dropout,
            )
            for _ in range(self.num_layers)
        )
        self.gnn_head: StructuralPrefixHead | None
        if prefix_source == "gnn":
            self.gnn_head = build_structural_prefix_head(
                encoder_type=gnn_encoder_type,
                input_dim=gnn_input_dim,
                hidden_dim=gnn_hidden_dim,
                num_layers=gnn_layers,
                gat_heads=gnn_gat_heads,
                pool_heads=gnn_pool_heads,
                dropout=dropout,
                attention_dropout=gnn_attention_dropout,
                use_edge_features=gnn_use_edge_features,
                num_edge_types=gnn_num_edge_types,
                edge_emb_dim=gnn_edge_emb_dim,
                pool_query_init=gnn_pool_query_init,
                set_hidden_multiplier=gnn_set_hidden_multiplier,
            )
            self.register_parameter("register_prefix", None)
        else:
            self.gnn_head = None
            self.register_prefix = nn.Parameter(
                torch.empty(gnn_pool_heads, gnn_hidden_dim).normal_(std=0.02)
            )

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

    def _prefixes(
        self,
        batch_size: int,
        node_features: torch.Tensor | None,
        edge_index: torch.Tensor | None,
        edge_types: torch.Tensor | None,
        graph_batch: torch.Tensor | None,
    ) -> torch.Tensor:
        if self.prefix_source == "register":
            return self.register_prefix.unsqueeze(0).expand(batch_size, -1, -1)
        if node_features is None or graph_batch is None:
            raise ValueError("node_features and graph_batch are required")
        if self.gnn_head.requires_edges and edge_index is None:
            raise ValueError("edge_index is required by the graph encoder")
        with torch.amp.autocast(node_features.device.type, enabled=False):
            prefixes, _ = self.gnn_head(
                node_features.float(),
                graph_batch.long(),
                edge_index=edge_index.long() if edge_index is not None else None,
                edge_types=edge_types.long() if edge_types is not None else None,
            )
        return prefixes

    def _transformer(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor | None,
        prefixes: torch.Tensor,
    ) -> torch.Tensor:
        hidden = self.roformer.embeddings(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
        )
        positions = self.roformer.encoder.embed_positions(hidden.shape)
        batch_size, sequence_length, _ = hidden.shape
        for layer, prefix_attention in zip(
            self.roformer.encoder.layer,
            self.kv_prefix_layers,
            strict=True,
        ):
            attention = layer.attention.self
            shape = (batch_size, sequence_length, self.num_heads, self.head_dim)
            query = attention.query(hidden).view(shape).transpose(1, 2)
            key = attention.key(hidden).view(shape).transpose(1, 2)
            value = attention.value(hidden).view(shape).transpose(1, 2)
            query, key = attention.apply_rotary_position_embeddings(positions, query, key)
            context, _ = prefix_attention(query, key, value, prefixes, attention_mask)
            context = context.transpose(1, 2).reshape(batch_size, sequence_length, -1)

            attention_output = layer.attention.output.dense(context)
            attention_output = layer.attention.output.dropout(attention_output)
            attention_output = layer.attention.output.LayerNorm(attention_output + hidden)
            output = layer.intermediate(attention_output)
            output = layer.output.dense(output)
            output = layer.output.dropout(output)
            hidden = layer.output.LayerNorm(output + attention_output)
        return hidden

    def forward_with_pooled(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor | None = None,
        function_features: torch.Tensor | None = None,
        node_features: torch.Tensor | None = None,
        edge_index: torch.Tensor | None = None,
        edge_types: torch.Tensor | None = None,
        graph_batch: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        prefixes = self._prefixes(
            input_ids.shape[0], node_features, edge_index, edge_types, graph_batch
        )
        hidden = self._transformer(input_ids, attention_mask, token_type_ids, prefixes)
        pooled = masked_mean(hidden, attention_mask)
        embedding = self.projection(pooled)
        if function_features is not None:
            embedding = self.fusion(torch.cat((embedding, function_features), dim=-1))
        return F.normalize(embedding, dim=-1), pooled

    def forward(self, *args, **kwargs) -> torch.Tensor:
        embedding, _ = self.forward_with_pooled(*args, **kwargs)
        return embedding

    def get_embedding_pairs(
        self,
        batch: Mapping[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        embedding_a, _ = self._forward_side(batch, "a")
        embedding_b, _ = self._forward_side(batch, "b")
        return embedding_a, embedding_b

    def get_embedding_pairs_with_pooled(
        self,
        batch: Mapping[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        embedding_a, pooled_a = self._forward_side(batch, "a")
        embedding_b, pooled_b = self._forward_side(batch, "b")
        return embedding_a, embedding_b, pooled_a, pooled_b

    def _forward_side(
        self,
        batch: Mapping[str, torch.Tensor],
        side: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        edge_index, edge_types = structural_edges(batch, side)
        return self.forward_with_pooled(
            input_ids=batch[f"input_ids_{side}"],
            attention_mask=batch[f"attention_mask_{side}"],
            token_type_ids=batch.get(f"token_type_ids_{side}"),
            function_features=batch.get(f"function_features_{side}"),
            node_features=batch.get(f"node_features_{side}"),
            edge_index=edge_index,
            edge_types=edge_types,
            graph_batch=batch.get(f"graph_batch_{side}"),
        )
