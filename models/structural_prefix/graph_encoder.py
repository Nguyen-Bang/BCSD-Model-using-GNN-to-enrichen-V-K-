"""GATv2 encoder for control-flow graphs."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv

from models.structural_prefix.pooling import AttentionPoolingEncoder


class GraphEncoder(AttentionPoolingEncoder):
    """Encode graph nodes with GATv2 layers and attention pooling."""

    def __init__(
        self,
        input_dim: int = 20,
        hidden_dim: int = 256,
        num_layers: int = 3,
        gat_heads: int = 4,
        pool_heads: int = 4,
        dropout: float = 0.2,
        attention_dropout: float | None = None,
        use_edge_features: bool = False,
        num_edge_types: int = 11,
        edge_emb_dim: int = 16,
        pool_query_init: str = "ones",
    ) -> None:
        super().__init__(
            hidden_dim=hidden_dim,
            pool_heads=pool_heads,
            pool_query_init=pool_query_init,
        )
        self.dropout = dropout
        self.attention_dropout = dropout if attention_dropout is None else attention_dropout
        self.use_edge_features = use_edge_features
        self.num_edge_types = num_edge_types
        self.edge_emb_dim = edge_emb_dim
        self.edge_embedding = (
            nn.Embedding(num_edge_types, edge_emb_dim) if use_edge_features else None
        )

        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        layer_input_dim = input_dim
        for layer_index in range(num_layers):
            is_last = layer_index == num_layers - 1
            heads = 1 if is_last else gat_heads
            concat = not is_last
            kwargs = {
                "heads": heads,
                "dropout": self.attention_dropout,
                "concat": concat,
            }
            if use_edge_features:
                kwargs.update(edge_dim=edge_emb_dim, add_self_loops=False)
            self.convs.append(GATv2Conv(layer_input_dim, hidden_dim, **kwargs))
            if not is_last:
                self.batch_norms.append(nn.BatchNorm1d(hidden_dim * heads))
            layer_input_dim = hidden_dim * heads if concat else hidden_dim

    def forward(
        self,
        nodes: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        *,
        edge_types: torch.Tensor | None = None,
        return_node_embeddings: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Encode a batch of graphs."""
        edge_attributes = None
        if self.use_edge_features:
            if edge_types is None:
                raise ValueError("edge_types are required when use_edge_features is enabled")
            if edge_index.size(1) != edge_types.size(0):
                raise ValueError("edge_index and edge_types must describe the same number of edges")
            edge_attributes = self.edge_embedding(edge_types)

        for index, conv in enumerate(self.convs):
            is_last = index == len(self.convs) - 1
            if edge_attributes is None:
                nodes = conv(nodes, edge_index)
            else:
                nodes = conv(nodes, edge_index, edge_attributes)
            if not is_last:
                nodes = self.batch_norms[index](nodes)
                nodes = F.leaky_relu(nodes, 0.2)
                nodes = F.dropout(nodes, p=self.dropout, training=self.training)
            else:
                nodes = F.leaky_relu(nodes, 0.2)

        summary = self._pool(nodes, batch)
        return (summary, nodes) if return_node_embeddings else summary
