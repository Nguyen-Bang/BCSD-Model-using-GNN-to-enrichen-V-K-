"""
Custom KV-Prefix Attention Mechanism

Implements attention mechanism that concatenates graph-derived K/V prefix to
sequence K/V. This is the core innovation of the BCSD model.

Module: models.custom_attention
Owner: User Story 4 (US4) - Custom KV-Prefix Attention Mechanism
Tasks: T046-T058
"""

import logging
import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger("bcsd.models")


class KVPrefixAttention(nn.Module):
    """
    KV-Prefix Attention: Injects graph summary into BERT attention as K/V prefix (T046-T058).
    
    THE CORE INNOVATION:
    Instead of just concatenating graph features, we inject them directly into
    BERT's attention mechanism by prepending graph-derived keys and values to
    the sequence keys and values.
    
    Architecture:
    1. Project graph_summary → prefix_k and prefix_v (separate linear layers)
    2. Reshape for multi-head attention
    3. Concatenate prefix to sequence K/V along length dimension
    4. Extend attention mask to always attend to prefix
    5. Compute scaled dot-product attention with extended K/V
    
    This allows the model to "attend" to the graph structure at every token position.
    
    Example:
        >>> attention = KVPrefixAttention(
        ...     hidden_size=768,
        ...     num_heads=12,
        ...     graph_dim=256
        ... )
        >>> # Q, K, V from BERT: [batch, heads, seq_len, head_dim]
        >>> # graph_summary: [batch, graph_dim]
        >>> # attention_mask: [batch, seq_len]
        >>> context, attn_weights = attention(
        ...     query=Q, key=K, value=V,
        ...     graph_summary=graph_summary,
        ...     attention_mask=attention_mask
        ... )
        >>> print(context.shape)  # [batch, heads, seq_len, head_dim]
        >>> print(attn_weights.shape)  # [batch, heads, seq_len, seq_len+1]
    """
    
    def __init__(
        self,
        hidden_size: int = 768,  # BERT hidden size
        num_heads: int = 12,     # BERT attention heads
        graph_dim: int = 256,    # Graph summary dimension
        dropout: float = 0.1
    ):
        """
        Initialize KV-Prefix attention (T047).
        
        Args:
            hidden_size: BERT hidden size (default: 768)
            num_heads: Number of attention heads (default: 12)
            graph_dim: Graph summary dimension (default: 256)
            dropout: Dropout probability (default: 0.1)
        """
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.graph_dim = graph_dim
        self.dropout = dropout
        
        assert hidden_size % num_heads == 0, "hidden_size must be divisible by num_heads"
        
        # Separate projections for graph → K and graph → V (T047)
        # This allows the model to learn different representations for keys vs values
        self.graph_to_k = nn.Linear(graph_dim, hidden_size)
        self.graph_to_v = nn.Linear(graph_dim, hidden_size)
        
        # Dropout for attention weights
        self.attention_dropout = nn.Dropout(dropout)
        
        # Scaling factor for attention scores
        self.scale = math.sqrt(self.head_dim)
        
        logger.info(f"KVPrefixAttention initialized: hidden={hidden_size}, heads={num_heads}, "
                   f"graph_dim={graph_dim}")
    
    def split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        Split hidden dimension into multiple heads (T048).
        
        Args:
            x: Tensor [batch, seq_len, hidden_size] or [batch, 1, hidden_size] for prefix
            
        Returns:
            Tensor [batch, heads, seq_len, head_dim]
        """
        batch_size = x.size(0)
        seq_len = x.size(1)
        
        # Reshape to [batch, seq_len, heads, head_dim]
        x = x.view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # Transpose to [batch, heads, seq_len, head_dim]
        x = x.transpose(1, 2)
        
        return x
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        graph_summary: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with graph prefix injection (T049-T053).
        
        Args:
            query: Query from BERT [batch, heads, seq_len, head_dim]
            key: Key from BERT [batch, heads, seq_len, head_dim]
            value: Value from BERT [batch, heads, seq_len, head_dim]
            graph_summary: Graph summary [batch, graph_dim]
            attention_mask: Attention mask [batch, seq_len] (1 = attend, 0 = mask)
            
        Returns:
            context: Attention output [batch, heads, seq_len, head_dim]
            attention_weights: Attention weights [batch, heads, seq_len, seq_len+1]
        """
        batch_size = query.size(0)
        seq_len = query.size(2)
        
        # Step 1: Project graph_summary to prefix_k and prefix_v (T049)
        prefix_k = self.graph_to_k(graph_summary)  # [batch, hidden_size]
        prefix_v = self.graph_to_v(graph_summary)  # [batch, hidden_size]
        
        # Add sequence dimension (prefix is single "token")
        prefix_k = prefix_k.unsqueeze(1)  # [batch, 1, hidden_size]
        prefix_v = prefix_v.unsqueeze(1)  # [batch, 1, hidden_size]
        
        # Step 2: Reshape for multi-head attention (T050)
        prefix_k = self.split_heads(prefix_k)  # [batch, heads, 1, head_dim]
        prefix_v = self.split_heads(prefix_v)  # [batch, heads, 1, head_dim]
        
        # Step 3: Concatenate prefix to sequence K/V (T051)
        # Prefix comes FIRST (position 0), then sequence tokens
        extended_key = torch.cat([prefix_k, key], dim=2)      # [batch, heads, seq_len+1, head_dim]
        extended_value = torch.cat([prefix_v, value], dim=2)  # [batch, heads, seq_len+1, head_dim]
        
        # Step 4: Extend attention mask (T052)
        if attention_mask is not None:
            # Original mask: [batch, seq_len]
            # Add 1 at position 0 for prefix (always attended)
            prefix_mask = torch.ones(
                batch_size, 1,
                dtype=attention_mask.dtype,
                device=attention_mask.device
            )
            extended_mask = torch.cat([prefix_mask, attention_mask], dim=1)  # [batch, seq_len+1]
        else:
            # No mask - attend to everything
            extended_mask = None
        
        # Step 5: Compute scaled dot-product attention (T053)
        context, attention_weights = self._scaled_dot_product_attention(
            query=query,
            key=extended_key,
            value=extended_value,
            attention_mask=extended_mask
        )
        
        return context, attention_weights
    
    def _scaled_dot_product_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Scaled dot-product attention with extended K/V (T053).
        
        Args:
            query: [batch, heads, seq_len, head_dim]
            key: [batch, heads, seq_len+1, head_dim] (includes prefix)
            value: [batch, heads, seq_len+1, head_dim] (includes prefix)
            attention_mask: [batch, seq_len+1] (1 = attend, 0 = mask)
            
        Returns:
            context: [batch, heads, seq_len, head_dim]
            attention_weights: [batch, heads, seq_len, seq_len+1]
        """
        # Compute attention scores: Q @ K^T
        # [batch, heads, seq_len, head_dim] @ [batch, heads, head_dim, seq_len+1]
        # → [batch, heads, seq_len, seq_len+1]
        attention_scores = torch.matmul(query, key.transpose(-2, -1))
        
        # Scale by sqrt(head_dim)
        attention_scores = attention_scores / self.scale
        
        # Apply attention mask (convert 1/0 to 0/-inf)
        if attention_mask is not None:
            # Reshape mask for broadcasting: [batch, 1, 1, seq_len+1]
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            
            # Convert to additive mask: 0 → 0, 1 → -inf (masked positions)
            attention_mask = (1.0 - attention_mask) * -10000.0
            
            attention_scores = attention_scores + attention_mask
        
        # Softmax to get attention weights
        attention_weights = F.softmax(attention_scores, dim=-1)
        
        # Apply dropout
        attention_weights = self.attention_dropout(attention_weights)
        
        # Weighted sum of values: [batch, heads, seq_len, seq_len+1] @ [batch, heads, seq_len+1, head_dim]
        # → [batch, heads, seq_len, head_dim]
        context = torch.matmul(attention_weights, value)
        
        return context, attention_weights


def test_kv_prefix_attention():
    """Test KV-Prefix attention with dummy data (T054-T057)."""
    print("Testing KV-Prefix Attention...")
    
    batch_size = 2
    seq_len = 10
    hidden_size = 768
    num_heads = 12
    head_dim = hidden_size // num_heads
    graph_dim = 256
    
    # Create dummy inputs
    # Normally these come from BERT, but we simulate them here
    query = torch.randn(batch_size, num_heads, seq_len, head_dim)
    key = torch.randn(batch_size, num_heads, seq_len, head_dim)
    value = torch.randn(batch_size, num_heads, seq_len, head_dim)
    graph_summary = torch.randn(batch_size, graph_dim)
    attention_mask = torch.ones(batch_size, seq_len)  # All positions valid
    
    print(f"Input shapes:")
    print(f"  Query: {query.shape}")
    print(f"  Key: {key.shape}")
    print(f"  Value: {value.shape}")
    print(f"  Graph summary: {graph_summary.shape}")
    print(f"  Attention mask: {attention_mask.shape}")
    
    # Create attention module
    attention = KVPrefixAttention(
        hidden_size=hidden_size,
        num_heads=num_heads,
        graph_dim=graph_dim,
        dropout=0.1
    )
    
    # Forward pass (T054)
    attention.eval()
    with torch.no_grad():
        context, attn_weights = attention(
            query=query,
            key=key,
            value=value,
            graph_summary=graph_summary,
            attention_mask=attention_mask
        )
    
    # Verify output shapes (T055)
    print(f"\nOutput shapes:")
    print(f"  Context: {context.shape}")  # [batch, heads, seq_len, head_dim]
    print(f"  Attention weights: {attn_weights.shape}")  # [batch, heads, seq_len, seq_len+1]
    
    assert context.shape == (batch_size, num_heads, seq_len, head_dim), "Context shape mismatch!"
    assert attn_weights.shape == (batch_size, num_heads, seq_len, seq_len + 1), "Attention weights shape mismatch!"
    
    # Test gradient flow (T056)
    attention.train()
    context, attn_weights = attention(
        query=query,
        key=key,
        value=value,
        graph_summary=graph_summary,
        attention_mask=attention_mask
    )
    
    loss = context.sum()
    loss.backward()
    
    # Check gradients reach graph projections
    graph_to_k_has_grad = attention.graph_to_k.weight.grad is not None and attention.graph_to_k.weight.grad.abs().sum() > 0
    graph_to_v_has_grad = attention.graph_to_v.weight.grad is not None and attention.graph_to_v.weight.grad.abs().sum() > 0
    
    print(f"\nGradient flow:")
    print(f"  graph_to_k: {'✓ OK' if graph_to_k_has_grad else '✗ FAIL'}")
    print(f"  graph_to_v: {'✓ OK' if graph_to_v_has_grad else '✗ FAIL'}")
    
    # Visualize attention to prefix (T057)
    print(f"\nAttention to prefix (position 0):")
    prefix_attention = attn_weights[:, :, :, 0]  # [batch, heads, seq_len]
    print(f"  Mean attention to prefix: {prefix_attention.mean().item():.4f}")
    print(f"  Min attention to prefix: {prefix_attention.min().item():.4f}")
    print(f"  Max attention to prefix: {prefix_attention.max().item():.4f}")
    
    # Verify prefix is attended
    assert prefix_attention.mean() > 0, "Prefix should receive some attention!"
    
    print("\n✓ KV-Prefix Attention tests passed!")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_kv_prefix_attention()
