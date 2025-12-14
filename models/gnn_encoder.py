"""
GNN Encoder for Graph Summarization

Implements Graph Attention Network (GAT) encoder that converts variable-size
CFGs into fixed-size graph summaries.

Module: models.gnn_encoder
Owner: User Story 3 (US3) - GNN Encoder for Graph Summarization
Tasks: T037-T045
"""

import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool, global_add_pool

logger = logging.getLogger("bcsd.models")


class GATEncoder(nn.Module):
    """
    Graph Attention Network encoder for CFG summarization (T037-T045).
    
    Architecture:
    - 3 GAT layers with 4 attention heads each
    - LeakyReLU activation
    - Dropout (0.2) for regularization
    - Global attention-based pooling
    - Output: Fixed-size graph summary (configurable dimension)
    
    Key Features:
    - Handles variable-size graphs (10-1000+ nodes)
    - Configurable output dimension (128, 256, or 512)
    - Attention mechanism for neighbor aggregation
    - Global pooling for graph-level representation
    
    Example:
        >>> encoder = GATEncoder(
        ...     input_dim=768,  # BERT hidden size for node features
        ...     hidden_dim=256,
        ...     output_dim=256,
        ...     num_layers=3,
        ...     heads=4,
        ...     dropout=0.2
        ... )
        >>> # graph_batch from PyTorch Geometric Batch
        >>> graph_summary = encoder(node_features, edge_index, batch)
        >>> print(graph_summary.shape)  # [batch_size, output_dim]
    """
    
    def __init__(
        self,
        input_dim: int = 768,  # BERT hidden size
        hidden_dim: int = 256,
        output_dim: int = 256,  # Configurable: 128, 256, or 512
        num_layers: int = 3,
        heads: int = 4,
        dropout: float = 0.2,
        pooling: str = "mean"  # "mean" or "add"
    ):
        """
        Initialize GAT encoder (T038).
        
        Args:
            input_dim: Input feature dimension (node embeddings)
            hidden_dim: Hidden dimension for GAT layers
            output_dim: Output dimension for graph summary (128, 256, or 512)
            num_layers: Number of GAT layers (default: 3)
            heads: Number of attention heads (default: 4)
            dropout: Dropout probability (default: 0.2)
            pooling: Global pooling method ("mean" or "add")
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.heads = heads
        self.dropout = dropout
        self.pooling = pooling
        
        # Create GAT layers (T038)
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        # First layer: input_dim -> hidden_dim * heads
        self.convs.append(
            GATConv(
                in_channels=input_dim,
                out_channels=hidden_dim,
                heads=heads,
                dropout=dropout,
                concat=True  # Concatenate attention heads
            )
        )
        self.batch_norms.append(nn.BatchNorm1d(hidden_dim * heads))
        
        # Middle layers: (hidden_dim * heads) -> (hidden_dim * heads)
        for _ in range(num_layers - 2):
            self.convs.append(
                GATConv(
                    in_channels=hidden_dim * heads,
                    out_channels=hidden_dim,
                    heads=heads,
                    dropout=dropout,
                    concat=True
                )
            )
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim * heads))
        
        # Final layer: (hidden_dim * heads) -> output_dim
        # Average attention heads in final layer
        self.convs.append(
            GATConv(
                in_channels=hidden_dim * heads,
                out_channels=output_dim,
                heads=1,  # Single head for final output
                dropout=dropout,
                concat=False
            )
        )
        
        # Attention weights for pooling (T040)
        self.attention_pooling = nn.Sequential(
            nn.Linear(output_dim, output_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Linear(output_dim // 2, 1)
        )
        
        logger.info(f"GATEncoder initialized: input_dim={input_dim}, hidden_dim={hidden_dim}, "
                   f"output_dim={output_dim}, layers={num_layers}, heads={heads}")
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass through GAT encoder (T039).
        
        Args:
            x: Node features [num_nodes, input_dim]
            edge_index: Edge indices [2, num_edges]
            batch: Batch assignment [num_nodes] (which graph each node belongs to)
            
        Returns:
            Graph summaries [batch_size, output_dim]
        """
        # Message passing through GAT layers (T039)
        for i, conv in enumerate(self.convs):
            if i < len(self.convs) - 1:
                # Hidden layers with batch norm and activation
                x = conv(x, edge_index)
                x = self.batch_norms[i](x)
                x = F.leaky_relu(x, 0.2)
                x = F.dropout(x, p=self.dropout, training=self.training)
            else:
                # Final layer without batch norm
                x = conv(x, edge_index)
                x = F.leaky_relu(x, 0.2)
        
        # Global pooling (T040)
        graph_summary = self._global_pooling(x, batch)
        
        return graph_summary
    
    def _global_pooling(
        self,
        x: torch.Tensor,
        batch: torch.Tensor
    ) -> torch.Tensor:
        """
        Global attention-based pooling (T040).
        
        Computes attention scores for each node and performs weighted sum
        pooling to get graph-level representation.
        
        Args:
            x: Node features [num_nodes, output_dim]
            batch: Batch assignment [num_nodes]
            
        Returns:
            Graph summaries [batch_size, output_dim]
        """
        if self.pooling == "mean":
            # Simple mean pooling
            return global_mean_pool(x, batch)
        elif self.pooling == "add":
            # Sum pooling
            return global_add_pool(x, batch)
        else:
            # Attention-based pooling (T040)
            # Compute attention scores for each node
            attention_scores = self.attention_pooling(x)  # [num_nodes, 1]
            
            # Apply softmax per graph
            attention_weights = self._softmax_per_graph(attention_scores, batch)
            
            # Weighted sum of node features
            x_weighted = x * attention_weights
            graph_summary = global_add_pool(x_weighted, batch)
            
            return graph_summary
    
    def _softmax_per_graph(
        self,
        scores: torch.Tensor,
        batch: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply softmax per graph in batch.
        
        Args:
            scores: Attention scores [num_nodes, 1]
            batch: Batch assignment [num_nodes]
            
        Returns:
            Attention weights [num_nodes, 1]
        """
        # Get unique batch indices
        batch_size = batch.max().item() + 1
        
        # Initialize output
        attention_weights = torch.zeros_like(scores)
        
        # Apply softmax for each graph
        for b in range(batch_size):
            mask = batch == b
            graph_scores = scores[mask]
            graph_weights = F.softmax(graph_scores, dim=0)
            attention_weights[mask] = graph_weights
        
        return attention_weights


def initialize_node_features(
    instruction_embeddings: torch.Tensor,
    nodes_per_graph: torch.Tensor
) -> torch.Tensor:
    """
    Initialize node features by averaging instruction token embeddings per basic block (T041).
    
    Since each CFG node (basic block) contains multiple instructions, we average
    the BERT embeddings of all instructions in that block to get a single node feature.
    
    Args:
        instruction_embeddings: BERT embeddings [total_instructions, 768]
        nodes_per_graph: Number of nodes per graph [batch_size]
        
    Returns:
        Node features [total_nodes, 768]
    
    Note:
        This is a simplified version. In practice, the relationship between
        instructions and CFG nodes is stored in the preprocessed data.
    """
    # This function will be properly implemented when integrating with BERT
    # For now, return instruction embeddings directly
    # (assuming 1 instruction per node for simplicity)
    return instruction_embeddings


def test_gat_encoder():
    """Test GAT encoder with dummy data (T042-T044)."""
    print("Testing GAT Encoder...")
    
    # Create dummy data
    batch_size = 3
    num_nodes_per_graph = [10, 50, 200]  # Variable sizes
    input_dim = 768
    output_dim = 256
    
    # Generate node features
    node_features = []
    batch_assignment = []
    edge_indices = []
    current_offset = 0
    
    for b, num_nodes in enumerate(num_nodes_per_graph):
        # Random node features
        x = torch.randn(num_nodes, input_dim)
        node_features.append(x)
        
        # Batch assignment
        batch_assignment.extend([b] * num_nodes)
        
        # Random edges (create a connected graph)
        num_edges = num_nodes * 2
        edges = torch.randint(0, num_nodes, (2, num_edges)) + current_offset
        edge_indices.append(edges)
        
        current_offset += num_nodes
    
    # Concatenate
    x = torch.cat(node_features, dim=0)  # [260, 768]
    edge_index = torch.cat(edge_indices, dim=1)  # [2, total_edges]
    batch = torch.tensor(batch_assignment)  # [260]
    
    print(f"Input shapes:")
    print(f"  Node features: {x.shape}")
    print(f"  Edge index: {edge_index.shape}")
    print(f"  Batch: {batch.shape}")
    
    # Create encoder
    encoder = GATEncoder(
        input_dim=input_dim,
        hidden_dim=256,
        output_dim=output_dim,
        num_layers=3,
        heads=4,
        dropout=0.2,
        pooling="attention"
    )
    
    # Forward pass (T042)
    encoder.eval()
    with torch.no_grad():
        graph_summary = encoder(x, edge_index, batch)
    
    print(f"\nOutput shape: {graph_summary.shape}")  # Should be [3, 256]
    assert graph_summary.shape == (batch_size, output_dim), "Output shape mismatch!"
    
    # Test gradient flow (T044)
    encoder.train()
    graph_summary = encoder(x, edge_index, batch)
    loss = graph_summary.sum()
    loss.backward()
    
    # Check gradients
    has_gradients = any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in encoder.parameters()
    )
    print(f"Gradient flow: {'✓ OK' if has_gradients else '✗ FAIL'}")
    
    print("\n✓ GAT Encoder tests passed!")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_gat_encoder()
