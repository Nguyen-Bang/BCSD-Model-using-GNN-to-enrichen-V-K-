"""Three-layer BCSD model with graph-conditioned KV-prefix attention."""

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.roformer.modeling_roformer import RoFormerSelfAttention

from models.baseline.baseline_with_features_model import BaselineWithFeaturesModel
from models.structural_prefix import (
    KVPrefixAttention,
    build_structural_prefix_head,
    structural_edges,
)

logger = logging.getLogger("bcsd.models")


class BaselineGNNModel(BaselineWithFeaturesModel):
    """Add a graph or set encoder prefix to the three-layer baseline."""

    def __init__(
        self,
        gnn_input_dim: int = 20,
        gnn_hidden_dim: int = 256,
        gnn_layers: int = 3,
        gnn_gat_heads: int = 4,
        gnn_pool_heads: int = 10,
        prefix_dim: int = 256,
        freeze_backbone: bool = False,
        multi_token: bool = True,
        attention_mode: str = "tanh_per_pair",
        gnn_encoder_type: str = "graph",
        gnn_set_hidden_multiplier: int = 4,
        gnn_use_edge_features: bool = False,
        gnn_num_edge_types: int = 11,
        gnn_edge_emb_dim: int = 16,
        gnn_pool_query_init: str = "ones",
        gnn_prefix_mode: str = "pool",
        gnn_attention_dropout: float | None = None,
        kv_low_rank: int = 0,
        prefix_source: str = "gnn",
        **kwargs,
    ):
        if not multi_token:
            raise ValueError("single-token prefixes are no longer supported")
        if attention_mode != "tanh_per_pair":
            raise ValueError("attention_mode must be 'tanh_per_pair'")
        if kv_low_rank:
            raise ValueError("low-rank prefix projections are no longer supported")
        if gnn_prefix_mode != "pool":
            raise ValueError("gnn_prefix_mode must be 'pool'")
        if prefix_dim != gnn_hidden_dim:
            raise ValueError("prefix_dim must equal gnn_hidden_dim")

        super().__init__(**kwargs)
        config = self.roformer.config
        num_layers = config.num_hidden_layers
        hidden_size = config.hidden_size
        num_heads = config.num_attention_heads
        dropout = config.hidden_dropout_prob

        self.num_heads = num_heads
        self.num_layers = num_layers
        self.head_dim = hidden_size // num_heads

        self.multi_token = True
        self.attention_mode = attention_mode
        effective_prefix_dim = gnn_hidden_dim
        num_prefix_tokens = gnn_pool_heads

        self.kv_prefix_layers = nn.ModuleList(
            [
                KVPrefixAttention(
                    hidden_size=hidden_size,
                    num_heads=num_heads,
                    prefix_dim=effective_prefix_dim,
                    num_prefix_tokens=num_prefix_tokens,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )

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

        self.prefix_source = prefix_source
        self.num_prefix_tokens = num_prefix_tokens
        self.effective_prefix_dim = effective_prefix_dim
        if prefix_source == "register":
            self.register_prefix = nn.Parameter(
                torch.randn(num_prefix_tokens, effective_prefix_dim) * 0.02
            )
        elif prefix_source == "none":
            for kv in self.kv_prefix_layers:
                kv._ablate_mask.zero_()  # prefix term is zeroed for the whole run
        elif prefix_source == "gnn_randomnodes":
            K = 1 << 20
            gen = torch.Generator().manual_seed(1234567)
            self.register_buffer(
                "_rand_node_table",
                torch.randn(K, gnn_input_dim, generator=gen),
                persistent=False,
            )
            primes = [
                73856093,
                19349663,
                83492791,
                49979687,
                86028121,
                15485863,
                32452843,
                67867967,
                49979693,
                86028157,
                15485867,
                32452867,
                67867979,
                73856081,
                19349669,
                83492803,
                49979671,
                86028139,
                15485849,
                32452877,
            ]
            primes = (primes * ((gnn_input_dim + len(primes) - 1) // len(primes)))[:gnn_input_dim]
            self.register_buffer(
                "_rand_node_primes",
                torch.tensor(primes, dtype=torch.int64),
                persistent=False,
            )
            self._rand_node_K = K
        elif prefix_source not in ("gnn", "gnn_noedges", "gnn_nonodes"):
            raise ValueError(
                f"prefix_source must be one of "
                f"gnn|register|none|gnn_noedges|gnn_nonodes|gnn_randomnodes; "
                f"got {prefix_source!r}"
            )

        if freeze_backbone:
            self._freeze_backbone()

        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(
            f"BaselineGNNModel: {num_layers}L RoFormer + GNN Head "
            f"({gnn_pool_heads} prefix tokens, attention_mode={attention_mode}), "
            f"{total / 1e6:.1f}M total, {trainable / 1e6:.1f}M trainable"
        )

    def _freeze_backbone(self):
        """Freeze all RoFormer parameters (embeddings + encoder layers)."""
        for param in self.roformer.parameters():
            param.requires_grad = False
        logger.info("Backbone frozen: only GNN Head + KV-prefix + fusion are trainable")

    def _make_prefixes(
        self, node_features, edge_index, graph_batch, edge_types, batch_size, device
    ) -> torch.Tensor:
        """Produce [B, num_prefix, prefix_dim] prefixes per ``self.prefix_source``.

        'register'/'none' ignore the graph entirely (content-free controls); 'gnn' runs the
        GAT as before. Returned in fp32 to match the GAT path (caller is under autocast-off).
        """
        if self.prefix_source == "register":
            return self.register_prefix.unsqueeze(0).expand(batch_size, -1, -1)
        if self.prefix_source == "none":
            return torch.zeros(
                batch_size,
                self.num_prefix_tokens,
                self.effective_prefix_dim,
                device=device,
                dtype=torch.float32,
            )
        if self.prefix_source == "gnn_nonodes":
            # keep CFG topology, zero the node content -> "does graph structure alone help?"
            nf = torch.zeros_like(node_features) if node_features is not None else node_features
            prefixes, _ = self.gnn_head(
                nf,
                graph_batch,
                edge_index=edge_index,
                edge_types=edge_types,
            )
            return prefixes
        if self.prefix_source == "gnn_noedges":
            # keep node features, replace edges with self-loops only (no message passing) ->
            # same GAT params/capacity but graph TOPOLOGY removed (a generic per-node encoder).
            if node_features is not None and node_features.numel() > 0:
                n = node_features.shape[0]
                idx = torch.arange(n, device=node_features.device)
                ei = torch.stack([idx, idx], dim=0)
                et = torch.full((n,), 10, dtype=torch.long, device=node_features.device)
            else:
                ei, et = edge_index, edge_types
            prefixes, _ = self.gnn_head(
                node_features,
                graph_batch,
                edge_index=ei,
                edge_types=et,
            )
            return prefixes
        if self.prefix_source == "gnn_randomnodes":
            # keep topology + edges; swap node-feature values for a fixed random codebook
            # keyed by node identity -> destroys node-feature content, preserves alignment.
            if node_features is not None and node_features.numel() > 0:
                q = torch.round(node_features.float() * 100.0).to(torch.int64)
                key = torch.remainder((q * self._rand_node_primes).sum(dim=1), self._rand_node_K)
                nf = self._rand_node_table[key].to(node_features.dtype)
            else:
                nf = node_features
            prefixes, _ = self.gnn_head(
                nf,
                graph_batch,
                edge_index=edge_index,
                edge_types=edge_types,
            )
            return prefixes
        prefixes, _ = self.gnn_head(
            node_features,
            graph_batch,
            edge_index=edge_index,
            edge_types=edge_types,
        )
        return prefixes

    def _forward_transformer(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor | None,
        prefixes: torch.Tensor,
    ) -> torch.Tensor:
        """Run RoFormer layers with an additive structural attention channel."""
        hidden = self.roformer.embeddings(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
        )

        embed_positions = self.roformer.encoder.embed_positions(hidden.shape)

        batch_size, seq_len, hidden_size = hidden.shape

        for layer_idx in range(self.num_layers):
            layer = self.roformer.encoder.layer[layer_idx]
            attn_self = layer.attention.self

            query = (
                attn_self.query(hidden)
                .view(batch_size, seq_len, self.num_heads, self.head_dim)
                .transpose(1, 2)
            )
            key = (
                attn_self.key(hidden)
                .view(batch_size, seq_len, self.num_heads, self.head_dim)
                .transpose(1, 2)
            )
            value = (
                attn_self.value(hidden)
                .view(batch_size, seq_len, self.num_heads, self.head_dim)
                .transpose(1, 2)
            )

            query, key = RoFormerSelfAttention.apply_rotary_position_embeddings(
                embed_positions, query, key
            )

            context, _ = self.kv_prefix_layers[layer_idx](
                query=query,
                key=key,
                value=value,
                graph_prefixes=prefixes,
                attention_mask=attention_mask,
            )

            context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, hidden_size)

            attn_out = layer.attention.output.dense(context)
            attn_out = layer.attention.output.dropout(attn_out)
            attn_out = layer.attention.output.LayerNorm(attn_out + hidden)

            ff_out = layer.intermediate(attn_out)
            ff_out = layer.output.dense(ff_out)
            ff_out = layer.output.dropout(ff_out)
            hidden = layer.output.LayerNorm(ff_out + attn_out)

        return hidden

    def _embedding_from_hidden(
        self,
        hidden: torch.Tensor,
        attention_mask: torch.Tensor,
        function_features: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        projected = self.projection(pooled)
        if function_features is not None:
            projected = self.fusion(torch.cat([projected, function_features], dim=-1))
        return F.normalize(projected, p=2, dim=1), pooled

    def _encode(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor | None,
        function_features: torch.Tensor | None,
        node_features: torch.Tensor | None,
        edge_index: torch.Tensor | None,
        edge_types: torch.Tensor | None,
        graph_batch: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        with torch.amp.autocast("cuda", enabled=False):
            node_features = node_features.float() if node_features is not None else None
            edge_index = edge_index.long() if edge_index is not None else None
            prefixes = self._make_prefixes(
                node_features,
                edge_index,
                graph_batch,
                edge_types,
                input_ids.shape[0],
                input_ids.device,
            )

        hidden = self._forward_transformer(input_ids, attention_mask, token_type_ids, prefixes)
        embedding, pooled = self._embedding_from_hidden(hidden, attention_mask, function_features)
        return embedding, pooled, hidden

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor | None = None,
        function_features: torch.Tensor | None = None,
        node_features: torch.Tensor | None = None,
        edge_index: torch.Tensor | None = None,
        edge_types: torch.Tensor | None = None,
        graph_batch: torch.Tensor | None = None,
        return_logits: bool = False,
    ) -> torch.Tensor:
        embedding, _, hidden = self._encode(
            input_ids,
            attention_mask,
            token_type_ids,
            function_features,
            node_features,
            edge_index,
            edge_types,
            graph_batch,
        )
        if return_logits:
            return embedding, self.mlm_head(hidden)
        return embedding

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
        """Return the embedding and pre-projection pooled transformer output."""
        embedding, pooled, _ = self._encode(
            input_ids,
            attention_mask,
            token_type_ids,
            function_features,
            node_features,
            edge_index,
            edge_types,
            graph_batch,
        )
        return embedding, pooled

    def forward_with_node_embeddings(
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
        """Forward pass that also returns GNN node embeddings for edge prediction."""
        with torch.amp.autocast("cuda", enabled=False):
            if node_features is not None:
                node_features = node_features.float()
            if edge_index is not None:
                edge_index = edge_index.long()
            prefixes, _, node_emb = self.gnn_head(
                node_features,
                graph_batch,
                edge_index=edge_index,
                edge_types=edge_types,
                return_node_embeddings=True,
            )

        hidden = self._forward_transformer(input_ids, attention_mask, token_type_ids, prefixes)

        embedding, _ = self._embedding_from_hidden(hidden, attention_mask, function_features)
        return embedding, node_emb

    def get_embedding_pairs(
        self,
        batch: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Process a Siamese batch, returning (emb_a, emb_b)."""
        ei_a, et_a = structural_edges(batch, "a")
        ei_b, et_b = structural_edges(batch, "b")
        emb_a = self(
            input_ids=batch["input_ids_a"],
            attention_mask=batch["attention_mask_a"],
            token_type_ids=batch.get("token_type_ids_a"),
            function_features=batch.get("function_features_a"),
            node_features=batch.get("node_features_a"),
            edge_index=ei_a,
            edge_types=et_a,
            graph_batch=batch.get("graph_batch_a"),
        )
        emb_b = self(
            input_ids=batch["input_ids_b"],
            attention_mask=batch["attention_mask_b"],
            token_type_ids=batch.get("token_type_ids_b"),
            function_features=batch.get("function_features_b"),
            node_features=batch.get("node_features_b"),
            edge_index=ei_b,
            edge_types=et_b,
            graph_batch=batch.get("graph_batch_b"),
        )
        return emb_a, emb_b

    def get_embedding_pairs_with_pooled(
        self,
        batch: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode a Siamese batch and expose both pooled transformer outputs."""
        ei_a, et_a = structural_edges(batch, "a")
        ei_b, et_b = structural_edges(batch, "b")
        emb_a, pooled_a = self.forward_with_pooled(
            input_ids=batch["input_ids_a"],
            attention_mask=batch["attention_mask_a"],
            token_type_ids=batch.get("token_type_ids_a"),
            function_features=batch.get("function_features_a"),
            node_features=batch.get("node_features_a"),
            edge_index=ei_a,
            edge_types=et_a,
            graph_batch=batch.get("graph_batch_a"),
        )
        emb_b, pooled_b = self.forward_with_pooled(
            input_ids=batch["input_ids_b"],
            attention_mask=batch["attention_mask_b"],
            token_type_ids=batch.get("token_type_ids_b"),
            function_features=batch.get("function_features_b"),
            node_features=batch.get("node_features_b"),
            edge_index=ei_b,
            edge_types=et_b,
            graph_batch=batch.get("graph_batch_b"),
        )
        return emb_a, emb_b, pooled_a, pooled_b

    def get_embedding_pairs_with_nodes(
        self,
        batch: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Process a Siamese batch, returning (emb_a, emb_b, node_emb_a, node_emb_b)."""
        ei_a, et_a = structural_edges(batch, "a")
        ei_b, et_b = structural_edges(batch, "b")
        emb_a, node_emb_a = self.forward_with_node_embeddings(
            input_ids=batch["input_ids_a"],
            attention_mask=batch["attention_mask_a"],
            token_type_ids=batch.get("token_type_ids_a"),
            function_features=batch.get("function_features_a"),
            node_features=batch.get("node_features_a"),
            edge_index=ei_a,
            edge_types=et_a,
            graph_batch=batch.get("graph_batch_a"),
        )
        emb_b, node_emb_b = self.forward_with_node_embeddings(
            input_ids=batch["input_ids_b"],
            attention_mask=batch["attention_mask_b"],
            token_type_ids=batch.get("token_type_ids_b"),
            function_features=batch.get("function_features_b"),
            node_features=batch.get("node_features_b"),
            edge_index=ei_b,
            edge_types=et_b,
            graph_batch=batch.get("graph_batch_b"),
        )
        return emb_a, emb_b, node_emb_a, node_emb_b
