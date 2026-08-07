"""Edge-free encoder for sets of structural node features."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.structural_prefix.pooling import AttentionPoolingEncoder


class SetEncoder(AttentionPoolingEncoder):
    """Encode an unordered node set without accepting graph edges."""

    def __init__(
        self,
        input_dim: int = 20,
        hidden_dim: int = 256,
        num_layers: int = 3,
        hidden_multiplier: int = 4,
        pool_heads: int = 4,
        dropout: float = 0.2,
        pool_query_init: str = "ones",
    ) -> None:
        super().__init__(
            hidden_dim=hidden_dim,
            pool_heads=pool_heads,
            pool_query_init=pool_query_init,
        )
        self.dropout = dropout
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        layer_input_dim = input_dim
        for layer_index in range(num_layers):
            is_last = layer_index == num_layers - 1
            layer_output_dim = hidden_dim if is_last else hidden_dim * hidden_multiplier
            self.convs.append(nn.Linear(layer_input_dim, layer_output_dim))
            if not is_last:
                self.batch_norms.append(nn.BatchNorm1d(layer_output_dim))
            layer_input_dim = layer_output_dim

    def forward(
        self,
        nodes: torch.Tensor,
        batch: torch.Tensor,
        *,
        return_node_embeddings: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Encode a batch of node sets."""
        for index, layer in enumerate(self.convs):
            is_last = index == len(self.convs) - 1
            nodes = layer(nodes)
            if not is_last:
                nodes = self.batch_norms[index](nodes)
                nodes = F.leaky_relu(nodes, 0.2)
                nodes = F.dropout(nodes, p=self.dropout, training=self.training)
            else:
                nodes = F.leaky_relu(nodes, 0.2)

        summary = self._pool(nodes, batch)
        return (summary, nodes) if return_node_embeddings else summary
