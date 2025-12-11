"""
Graph Neural Network Module for BCSD Model

This module implements a Graph Neural Network (GNN) for processing control flow graphs
and function call graphs. It enriches the BERT embeddings with graph structure information
for binary code similarity detection.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool
from torch_geometric.data import Data, Batch
import logging
from typing import Dict, List, Any, Optional, Tuple
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GNNModel(nn.Module):
    """
    Graph Neural Network model for binary code similarity detection.
    Uses Graph Convolutional Networks (GCN) or Graph Attention Networks (GAT).
    """
    
    def __init__(self, 
                 input_dim: int = 768,
                 hidden_dim: int = 256,
                 output_dim: int = 128,
                 num_layers: int = 3,
                 dropout: float = 0.1,
                 gnn_type: str = 'GCN'):
        """
        Initialize the GNN model.
        
        Args:
            input_dim: Dimension of input node features (BERT embedding size)
            hidden_dim: Dimension of hidden layers
            output_dim: Dimension of output embeddings
            num_layers: Number of GNN layers
            dropout: Dropout rate
            gnn_type: Type of GNN layer ('GCN' or 'GAT')
        """
        super(GNNModel, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.gnn_type = gnn_type
        
        # Build GNN layers
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        # First layer
        if gnn_type == 'GCN':
            self.convs.append(GCNConv(input_dim, hidden_dim))
        elif gnn_type == 'GAT':
            self.convs.append(GATConv(input_dim, hidden_dim, heads=4, concat=True))
            hidden_dim = hidden_dim * 4  # Adjust for concatenated attention heads
        else:
            raise ValueError(f"Unsupported GNN type: {gnn_type}")
        
        self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            if gnn_type == 'GCN':
                self.convs.append(GCNConv(hidden_dim, hidden_dim))
            elif gnn_type == 'GAT':
                self.convs.append(GATConv(hidden_dim, hidden_dim // 4, heads=4, concat=True))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        
        # Output layer
        if gnn_type == 'GCN':
            self.convs.append(GCNConv(hidden_dim, output_dim))
        elif gnn_type == 'GAT':
            self.convs.append(GATConv(hidden_dim, output_dim, heads=1, concat=False))
        
        self.dropout_layer = nn.Dropout(dropout)
        
        logger.info(f"Initialized {gnn_type} model with {num_layers} layers")
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, 
                batch: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through the GNN.
        
        Args:
            x: Node feature matrix [num_nodes, input_dim]
            edge_index: Graph connectivity [2, num_edges]
            batch: Batch vector [num_nodes] (optional, for batched graphs)
            
        Returns:
            Graph-level embeddings [num_graphs, output_dim]
        """
        # Apply GNN layers
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = self.batch_norms[i](x)
            x = F.relu(x)
            x = self.dropout_layer(x)
        
        # Final layer (no activation)
        x = self.convs[-1](x, edge_index)
        
        # Global pooling to get graph-level representation
        if batch is not None:
            x = global_mean_pool(x, batch)
        else:
            x = x.mean(dim=0, keepdim=True)
        
        return x


class GNNProcessor:
    """
    Processes binary code graphs using GNN for similarity detection.
    """
    
    def __init__(self,
                 model_config: Optional[Dict[str, Any]] = None,
                 device: Optional[str] = None):
        """
        Initialize the GNN processor.
        
        Args:
            model_config: Configuration dictionary for the GNN model
            device: Device to run the model on ('cuda' or 'cpu')
        """
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Default model configuration
        default_config = {
            'input_dim': 768,
            'hidden_dim': 256,
            'output_dim': 128,
            'num_layers': 3,
            'dropout': 0.1,
            'gnn_type': 'GCN'
        }
        
        self.model_config = {**default_config, **(model_config or {})}
        self.model = GNNModel(**self.model_config).to(self.device)
        self.model.eval()
        
        logger.info(f"GNN processor initialized on device: {self.device}")
    
    def create_graph_from_cfg(self, 
                             cfg_data: Dict[str, Any],
                             node_embeddings: Dict[int, np.ndarray]) -> Data:
        """
        Create a PyTorch Geometric graph from CFG data and node embeddings.
        
        Args:
            cfg_data: Control flow graph data
            node_embeddings: Dictionary mapping node addresses to embeddings
            
        Returns:
            PyTorch Geometric Data object
        """
        # Extract nodes and edges
        nodes = list(node_embeddings.keys())
        node_to_idx = {node: idx for idx, node in enumerate(nodes)}
        
        # Create feature matrix
        features = []
        for node in nodes:
            features.append(node_embeddings[node])
        x = torch.tensor(np.array(features), dtype=torch.float32)
        
        # Create edge index (placeholder - should be extracted from actual CFG)
        # This is a simplified version - in practice, extract from angr CFG
        edge_index = self._extract_edges(cfg_data, node_to_idx)
        
        # Create PyTorch Geometric Data object
        data = Data(x=x, edge_index=edge_index)
        
        return data
    
    def _extract_edges(self, cfg_data: Dict[str, Any], 
                      node_to_idx: Dict[int, int]) -> torch.Tensor:
        """
        Extract edges from CFG data.
        
        Args:
            cfg_data: Control flow graph data
            node_to_idx: Mapping from node addresses to indices
            
        Returns:
            Edge index tensor [2, num_edges]
            
        Note:
            This is a placeholder implementation that creates sequential edges.
            In a production system, this should extract actual control flow edges
            from the angr CFG by analyzing successor/predecessor relationships
            between basic blocks.
        """
        # Placeholder implementation
        # TODO: Extract actual CFG edges from angr's control flow graph
        # by accessing cfg.graph.edges() or node.successors/predecessors
        edges = []
        
        # Example: create edges between consecutive basic blocks
        nodes = list(node_to_idx.keys())
        for i in range(len(nodes) - 1):
            src_idx = node_to_idx[nodes[i]]
            dst_idx = node_to_idx[nodes[i + 1]]
            edges.append([src_idx, dst_idx])
        
        if edges:
            edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        else:
            # Empty graph - create self-loops
            edge_index = torch.zeros((2, 0), dtype=torch.long)
        
        return edge_index
    
    def create_graph_from_function(self,
                                   function_data: Dict[str, Any],
                                   block_embeddings: Dict[int, np.ndarray]) -> Data:
        """
        Create a graph from function data and basic block embeddings.
        
        Args:
            function_data: Function information including basic blocks
            block_embeddings: Dictionary mapping block addresses to embeddings
            
        Returns:
            PyTorch Geometric Data object
        """
        block_addrs = function_data.get('blocks', [])
        
        if not block_addrs:
            # Empty function - create dummy graph
            x = torch.zeros((1, 768), dtype=torch.float32)
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            return Data(x=x, edge_index=edge_index)
        
        # Filter to blocks that have embeddings
        valid_blocks = [addr for addr in block_addrs if addr in block_embeddings]
        
        if not valid_blocks:
            x = torch.zeros((1, 768), dtype=torch.float32)
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            return Data(x=x, edge_index=edge_index)
        
        # Create node features
        features = [block_embeddings[addr] for addr in valid_blocks]
        x = torch.tensor(np.array(features), dtype=torch.float32)
        
        # Create edges (sequential for now, should use actual CFG edges)
        edges = []
        for i in range(len(valid_blocks) - 1):
            edges.append([i, i + 1])
        
        if edges:
            edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
        
        return Data(x=x, edge_index=edge_index)
    
    def process_function(self, function_data: Dict[str, Any],
                        block_embeddings: Dict[int, np.ndarray]) -> np.ndarray:
        """
        Process a function through the GNN to get enriched embedding.
        
        Args:
            function_data: Function information
            block_embeddings: Dictionary of block embeddings
            
        Returns:
            Enriched function embedding
        """
        # Create graph
        graph = self.create_graph_from_function(function_data, block_embeddings)
        graph = graph.to(self.device)
        
        # Process through GNN
        with torch.no_grad():
            embedding = self.model(graph.x, graph.edge_index)
        
        return embedding.cpu().numpy().squeeze()
    
    def process_binary(self, encoded_data: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """
        Process all functions in a binary through the GNN.
        
        Args:
            encoded_data: Output from bert_encoder module
            
        Returns:
            Dictionary mapping function addresses to enriched embeddings
        """
        logger.info(f"Processing binary: {encoded_data.get('binary_path')}")
        
        block_embeddings = encoded_data.get('block_embeddings', {})
        function_embeddings = encoded_data.get('function_embeddings', {})
        
        # Get original disassembly data if available to get proper function-to-block mapping
        # For now, we'll process each function's embedding through a simple transformation
        # since we don't have the actual CFG edge information in the encoded data
        enriched_functions = {}
        
        for func_addr, func_emb in function_embeddings.items():
            # Create a single-node graph with the function embedding
            # In a full implementation, this would use the actual CFG edges from disassembly
            x = torch.tensor(func_emb.reshape(1, -1), dtype=torch.float32).to(self.device)
            edge_index = torch.zeros((2, 0), dtype=torch.long).to(self.device)
            
            with torch.no_grad():
                # Process through a simple transformation since we don't have graph structure
                enriched_emb = self.model(x, edge_index)
            
            enriched_functions[func_addr] = enriched_emb.cpu().numpy().squeeze()
        
        logger.info(f"Processed {len(enriched_functions)} functions")
        
        return enriched_functions
    
    def compute_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Compute cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            Similarity score between 0 and 1
        """
        # Normalize vectors
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        # Cosine similarity
        similarity = np.dot(embedding1, embedding2) / (norm1 * norm2)
        
        # Convert to [0, 1] range
        return (similarity + 1) / 2


def process_encoded_binary(encoded_data: Dict[str, Any],
                          model_config: Optional[Dict[str, Any]] = None) -> Dict[str, np.ndarray]:
    """
    Convenience function to process encoded binary data through GNN.
    
    Args:
        encoded_data: Output from bert_encoder module
        model_config: Optional GNN model configuration
        
    Returns:
        Dictionary of enriched function embeddings
    """
    processor = GNNProcessor(model_config=model_config)
    return processor.process_binary(encoded_data)


if __name__ == "__main__":
    logger.info("Testing GNN module...")
    
    # Test model initialization
    processor = GNNProcessor()
    
    # Test with dummy data
    dummy_block_embeddings = {
        0x401000: np.random.randn(768),
        0x401100: np.random.randn(768),
        0x401200: np.random.randn(768),
    }
    
    dummy_function_data = {
        'name': 'test_function',
        'blocks': list(dummy_block_embeddings.keys())
    }
    
    # Process function
    enriched_embedding = processor.process_function(dummy_function_data, dummy_block_embeddings)
    print(f"Enriched embedding shape: {enriched_embedding.shape}")
    print(f"Enriched embedding preview: {enriched_embedding[:5]}")
    
    # Test similarity computation
    emb1 = np.random.randn(128)
    emb2 = np.random.randn(128)
    similarity = processor.compute_similarity(emb1, emb2)
    print(f"\nSimilarity between random embeddings: {similarity:.4f}")
