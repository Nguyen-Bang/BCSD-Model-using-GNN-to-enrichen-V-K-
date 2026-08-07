"""Attention layer that adds structural key-value prefixes."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class KVPrefixAttention(nn.Module):
    """Add a tanh-gated structural attention channel to token attention."""

    def __init__(
        self,
        hidden_size: int = 768,
        num_heads: int = 12,
        prefix_dim: int = 256,
        num_prefix_tokens: int = 10,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if hidden_size % num_heads:
            raise ValueError("hidden_size must be divisible by num_heads")

        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale_qk = math.sqrt(self.head_dim)
        self.num_prefix_tokens = num_prefix_tokens

        self.prefix_to_k = nn.Linear(prefix_dim, hidden_size)
        self.prefix_to_v = nn.Linear(prefix_dim, hidden_size)
        self.prefix_k_norm = nn.LayerNorm(hidden_size)
        self.prefix_v_norm = nn.LayerNorm(hidden_size)
        self.weight = nn.Parameter(torch.zeros(num_heads, num_prefix_tokens))
        self.register_buffer(
            "_ablate_mask",
            torch.ones(num_heads, num_prefix_tokens),
            persistent=False,
        )
        self.attention_dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        graph_prefixes: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, None]:
        batch_size = query.size(0)
        token_mask = None
        if attention_mask is not None:
            token_mask = (1.0 - attention_mask[:, None, None, :]) * -1e4

        token_output = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=token_mask,
            dropout_p=self.attention_dropout.p if self.training else 0.0,
            is_causal=False,
        )

        prefix_k = self.prefix_k_norm(self.prefix_to_k(graph_prefixes))
        prefix_v = self.prefix_v_norm(self.prefix_to_v(graph_prefixes))
        prefix_k = prefix_k.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        prefix_v = prefix_v.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        prefix_scores = query @ prefix_k.transpose(-2, -1) / self.scale_qk
        prefix_weights = torch.sigmoid(prefix_scores)
        pair_weights = (torch.tanh(self.weight) * self._ablate_mask).view(
            1, self.num_heads, 1, self.num_prefix_tokens
        )
        prefix_output = (prefix_weights * pair_weights) @ prefix_v
        return token_output + prefix_output, None

    def effective_gate(self) -> torch.Tensor:
        """Return the mean absolute effective gate for logging."""
        return torch.tanh(self.weight).detach().abs().mean()
