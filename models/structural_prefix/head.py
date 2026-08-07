"""Structural encoder head and construction helper."""

import logging

import torch
import torch.nn as nn

from models.structural_prefix.graph_encoder import GraphEncoder
from models.structural_prefix.set_encoder import SetEncoder

logger = logging.getLogger("bcsd.models")


class StructuralPrefixHead(nn.Module):
    """Convert graph or set summaries into structural prefix tokens."""

    def __init__(self, encoder: GraphEncoder | SetEncoder) -> None:
        super().__init__()
        self.encoder = encoder
        self.pool_heads = encoder.pool_heads
        self.hidden_dim = encoder.hidden_dim

    @property
    def requires_edges(self) -> bool:
        return isinstance(self.encoder, GraphEncoder)

    def forward(
        self,
        node_features: torch.Tensor,
        batch: torch.Tensor,
        *,
        edge_index: torch.Tensor | None = None,
        edge_types: torch.Tensor | None = None,
        return_node_embeddings: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if isinstance(self.encoder, GraphEncoder):
            if edge_index is None:
                raise ValueError("edge_index is required by the graph encoder")
            encoded = self.encoder(
                node_features,
                edge_index,
                batch,
                edge_types=edge_types,
                return_node_embeddings=return_node_embeddings,
            )
        else:
            encoded = self.encoder(
                node_features,
                batch,
                return_node_embeddings=return_node_embeddings,
            )

        if return_node_embeddings:
            summary, node_embeddings = encoded
        else:
            summary = encoded
        prefixes = summary.view(summary.size(0), self.pool_heads, self.hidden_dim)
        if return_node_embeddings:
            return prefixes, summary, node_embeddings
        return prefixes, summary


def build_structural_prefix_head(
    *,
    encoder_type: str,
    input_dim: int,
    hidden_dim: int,
    num_layers: int,
    pool_heads: int,
    dropout: float,
    pool_query_init: str = "ones",
    gat_heads: int = 4,
    attention_dropout: float | None = None,
    use_edge_features: bool = False,
    num_edge_types: int = 11,
    edge_emb_dim: int = 16,
    set_hidden_multiplier: int = 4,
) -> StructuralPrefixHead:
    """Build one explicit structural encoder behind the shared prefix head."""
    if encoder_type == "graph":
        encoder = GraphEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            gat_heads=gat_heads,
            pool_heads=pool_heads,
            dropout=dropout,
            attention_dropout=attention_dropout,
            use_edge_features=use_edge_features,
            num_edge_types=num_edge_types,
            edge_emb_dim=edge_emb_dim,
            pool_query_init=pool_query_init,
        )
    elif encoder_type == "set":
        if use_edge_features:
            raise ValueError("the set encoder does not support edge features")
        encoder = SetEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            hidden_multiplier=set_hidden_multiplier,
            pool_heads=pool_heads,
            dropout=dropout,
            pool_query_init=pool_query_init,
        )
    else:
        raise ValueError("encoder_type must be 'graph' or 'set'")

    head = StructuralPrefixHead(encoder)
    logger.info(
        "StructuralPrefixHead: %s encoder, %.1fM parameters, %d prefix tokens",
        encoder_type,
        sum(parameter.numel() for parameter in head.parameters()) / 1e6,
        pool_heads,
    )
    return head
