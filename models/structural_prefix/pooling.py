"""Attention pooling shared by structural encoders."""

import torch
import torch.nn as nn
from torch_geometric.utils import scatter, softmax


class AttentionPoolingEncoder(nn.Module):
    """Base class for node encoders that produce fixed-size graph summaries."""

    def __init__(
        self,
        *,
        hidden_dim: int,
        pool_heads: int,
        pool_query_init: str,
    ) -> None:
        super().__init__()
        if pool_query_init not in {"ones", "orthogonal", "normal"}:
            raise ValueError("pool_query_init must be 'ones', 'orthogonal', or 'normal'")

        self.hidden_dim = hidden_dim
        self.pool_heads = pool_heads
        self.output_dim = pool_heads * hidden_dim
        self.pool_query_init = pool_query_init
        self.node_transform = nn.Sequential(
            nn.Linear(hidden_dim, pool_heads * hidden_dim),
            nn.ReLU(),
        )
        self.attention_queries = nn.Parameter(torch.empty(pool_heads, hidden_dim))
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self._init_pool_queries()

    def _init_pool_queries(self) -> None:
        if self.pool_query_init == "ones":
            nn.init.ones_(self.attention_queries)
        elif self.pool_query_init == "orthogonal":
            nn.init.orthogonal_(self.attention_queries)
        else:
            nn.init.normal_(self.attention_queries, mean=0.0, std=0.02)

    def _pool(self, nodes: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        """Pool node embeddings into one summary per graph."""
        batch_size = int(batch.max()) + 1
        transformed = self.node_transform(nodes).view(-1, self.pool_heads, self.hidden_dim)
        scores = (self.layer_norm(transformed) * self.attention_queries).sum(dim=-1)

        head_ids = torch.arange(self.pool_heads, device=nodes.device)
        flat_batch = (batch.unsqueeze(1) * self.pool_heads + head_ids).reshape(-1)
        weights = softmax(scores.reshape(-1), flat_batch).view(-1, self.pool_heads)
        weighted = transformed * weights.unsqueeze(-1)
        return scatter(
            weighted.view(-1, self.output_dim),
            batch,
            dim=0,
            dim_size=batch_size,
            reduce="sum",
        )
