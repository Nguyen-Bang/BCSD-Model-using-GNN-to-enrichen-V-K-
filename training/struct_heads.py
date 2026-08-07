"""Auxiliary structural prediction heads for the pooled transformer output."""

import torch
import torch.nn as nn
import torch.nn.functional as F

NUM_EDGE_TYPES = 5  # cond_jump, uncond_jump, fallthrough, switch, indirect
NUM_NODE_COUNT_BUCKETS = 10  # log-2 buckets: [1], [2-3], [4-7], ..., [512+]


class StructHeadsOnTransformer(nn.Module):
    """Two small MLPs on the transformer's pooled output."""

    def __init__(
        self,
        hidden_size: int = 768,
        num_edge_types: int = NUM_EDGE_TYPES,
        num_node_buckets: int = NUM_NODE_COUNT_BUCKETS,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.edge_type_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Linear(hidden_size // 2, num_edge_types),
        )
        self.node_count_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Linear(hidden_size // 2, num_node_buckets),
        )

    def forward(self, pooled: torch.Tensor) -> dict[str, torch.Tensor]:
        return {
            "edge_type_logits": self.edge_type_head(pooled),
            "node_count_logits": self.node_count_head(pooled),
        }


def edge_type_histogram_target(
    edge_index: torch.Tensor,
    edge_types: torch.Tensor,
    graph_batch: torch.Tensor,
    batch_size: int,
    num_types: int = NUM_EDGE_TYPES,
) -> torch.Tensor:
    """Normalized per-graph edge-type distribution. Returns [B, num_types]."""
    device = edge_types.device
    if edge_index.numel() == 0 or edge_types.numel() == 0:
        return torch.zeros(batch_size, num_types, device=device)
    # Each edge -> graph via its source node's graph assignment
    edge_graph = graph_batch[edge_index[0]]
    flat = edge_graph * num_types + edge_types
    bins = torch.bincount(flat, minlength=batch_size * num_types)
    bins = bins.view(batch_size, num_types).float()
    denom = bins.sum(dim=1, keepdim=True).clamp(min=1.0)
    return bins / denom


def node_count_bucket_target(
    graph_batch: torch.Tensor,
    batch_size: int,
    num_buckets: int = NUM_NODE_COUNT_BUCKETS,
) -> torch.Tensor:
    """
    Log-bucketed node-count target per graph. Returns [B] long.

    Bucket i corresponds to counts in [2^i, 2^(i+1)); bucket num_buckets-1
    catches all counts >= 2^(num_buckets-1). A graph with 0 nodes falls in
    bucket 0 (clamped at count=1 before log).
    """
    counts = torch.bincount(graph_batch, minlength=batch_size).float()
    log_counts = torch.log2(counts.clamp(min=1.0))
    return log_counts.floor().long().clamp(max=num_buckets - 1)


def struct_loss(
    predictions: dict[str, torch.Tensor],
    edge_index: torch.Tensor,
    edge_types: torch.Tensor,
    graph_batch: torch.Tensor,
    batch_size: int,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    Combined structural loss: KL on edge-type histogram + CE on node-count.

    The loss is computed in FP32 for stable KL divergence.
    """
    edge_logits = predictions["edge_type_logits"].float()
    node_logits = predictions["node_count_logits"].float()
    edge_log_pred = F.log_softmax(edge_logits, dim=-1)
    edge_target = edge_type_histogram_target(edge_index, edge_types, graph_batch, batch_size)
    edge_loss = F.kl_div(edge_log_pred, edge_target, reduction="batchmean")
    node_target = node_count_bucket_target(graph_batch, batch_size)
    node_loss = F.cross_entropy(node_logits, node_target)
    total = edge_loss + node_loss
    return total, {
        "edge_type": edge_loss.detach(),
        "node_count": node_loss.detach(),
    }
