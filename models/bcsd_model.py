"""
BCSD Model: Complete Binary Code Similarity Detection System

Integrates GNN encoder, graph prefix projection, and BERT with custom attention
into a single unified model for binary code similarity detection.

Module: models.bcsd_model
Owner: User Story 5 (US5) - BERT Integration & Siamese Training
Tasks: T063-T070
"""

import logging
from typing import Optional, Tuple, Dict

import torch
import torch.nn as nn

from models.gnn_encoder import GATEncoder
from models.bert_encoder import BERTWithGraphPrefix

logger = logging.getLogger("bcsd.models")


class BCSModel(nn.Module):
    """
    Binary Code Similarity Detection Model - The "God Class" (T063-T070).
    
    THE COMPLETE PIPELINE:
    This is the main model that combines all components:
    1. GNN Encoder: CFG → graph summary (fixed-size vector)
    2. BERT with Graph Prefix: Assembly tokens + graph summary → contextualized embeddings
    3. Pooling: Extract [CLS] token embedding for similarity learning
    
    Architecture Flow:
    
    Input:
        - node_features: [total_nodes, 768] - Node embeddings from tokens
        - edge_index: [2, total_edges] - Graph edges
        - batch: [total_nodes] - Graph batch mapping
        - input_ids: [batch, seq_len] - Token IDs
        - attention_mask: [batch, seq_len] - Attention mask
    
    Processing:
        1. GNN: (node_features, edge_index, batch) → graph_summary [batch, graph_dim]
        2. BERT: (input_ids, attention_mask, graph_summary) → hidden_states [batch, seq_len, 768]
        3. Pool: hidden_states[:, 0, :] → embeddings [batch, 768]
    
    Output:
        - embeddings: [batch, 768] - Function-level embeddings for similarity
        - mlm_logits: [batch, seq_len, vocab_size] - For MLM training (optional)
    
    Example (Single Sample):
        >>> model = BCSModel(graph_dim=256)
        >>> 
        >>> # Single graph
        >>> node_features = torch.randn(50, 768)  # 50 nodes
        >>> edge_index = torch.randint(0, 50, (2, 80))  # 80 edges
        >>> batch = torch.zeros(50, dtype=torch.long)  # Single graph
        >>> 
        >>> # Single sequence
        >>> input_ids = torch.randint(0, 30000, (1, 100))  # 1 sample, 100 tokens
        >>> attention_mask = torch.ones(1, 100)
        >>> 
        >>> outputs = model(
        ...     node_features=node_features,
        ...     edge_index=edge_index,
        ...     batch=batch,
        ...     input_ids=input_ids,
        ...     attention_mask=attention_mask
        ... )
        >>> print(outputs["embeddings"].shape)  # [1, 768]
    
    Example (Siamese Training - Batch of Pairs):
        >>> # Process binary 1
        >>> outputs_1 = model(
        ...     node_features=node_features_1,
        ...     edge_index=edge_index_1,
        ...     batch=batch_1,
        ...     input_ids=input_ids_1,
        ...     attention_mask=attention_mask_1
        ... )
        >>> 
        >>> # Process binary 2 (same model, same weights)
        >>> outputs_2 = model(
        ...     node_features=node_features_2,
        ...     edge_index=edge_index_2,
        ...     batch=batch_2,
        ...     input_ids=input_ids_2,
        ...     attention_mask=attention_mask_2
        ... )
        >>> 
        >>> # Compute contrastive loss
        >>> embeddings_1 = outputs_1["embeddings"]  # [batch, 768]
        >>> embeddings_2 = outputs_2["embeddings"]  # [batch, 768]
        >>> similarity = F.cosine_similarity(embeddings_1, embeddings_2)
    """
    
    def __init__(
        self,
        graph_dim: int = 256,
        gnn_hidden_dim: int = 256,
        gnn_layers: int = 3,
        gnn_heads: int = 4,
        gnn_dropout: float = 0.2,
        bert_model: str = "bert-base-uncased",
        bert_dropout: float = 0.1,
        freeze_bert_embeddings: bool = False
    ):
        """
        Initialize BCSD model (T064).
        
        Args:
            graph_dim: Graph summary dimension (output of GNN)
            gnn_hidden_dim: Hidden dimension for GNN layers
            gnn_layers: Number of GNN layers
            gnn_heads: Number of attention heads in GNN
            gnn_dropout: Dropout for GNN
            bert_model: Pre-trained BERT model name
            bert_dropout: Dropout for BERT
            freeze_bert_embeddings: Whether to freeze BERT embeddings
        """
        super().__init__()
        
        self.graph_dim = graph_dim
        
        logger.info("="*80)
        logger.info("Initializing BCSD Model (Binary Code Similarity Detection)")
        logger.info("="*80)
        
        # Component 1: GNN Encoder (T064)
        logger.info("Component 1: GNN Encoder")
        self.gnn_encoder = GATEncoder(
            input_dim=768,  # BERT hidden size for node features
            hidden_dim=gnn_hidden_dim,
            output_dim=graph_dim,
            num_layers=gnn_layers,
            heads=gnn_heads,
            dropout=gnn_dropout,
            pooling="mean"
        )
        logger.info(f"  ✓ GNN Encoder: {gnn_layers} layers, {gnn_heads} heads, output_dim={graph_dim}")
        
        # Component 2: BERT with Graph Prefix (T064)
        logger.info("Component 2: BERT with Graph Prefix")
        self.bert_encoder = BERTWithGraphPrefix(
            graph_dim=graph_dim,
            pretrained_model=bert_model,
            freeze_embeddings=freeze_bert_embeddings,
            dropout=bert_dropout
        )
        logger.info(f"  ✓ BERT Encoder: {bert_model}, graph_dim={graph_dim}")
        
        # MLM head for masked language modeling (optional, for training)
        self.mlm_head = nn.Linear(768, 30522)  # BERT vocab size
        
        logger.info("="*80)
        logger.info("BCSD Model initialized successfully")
        logger.info(f"  - Graph dimension: {graph_dim}")
        logger.info(f"  - GNN layers: {gnn_layers}")
        logger.info(f"  - BERT model: {bert_model}")
        logger.info(f"  - Total parameters: {sum(p.numel() for p in self.parameters()):,}")
        logger.info(f"  - Trainable parameters: {sum(p.numel() for p in self.parameters() if p.requires_grad):,}")
        logger.info("="*80)
    
    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        return_mlm_logits: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through complete model (T065).
        
        Args:
            node_features: Node features [total_nodes, 768]
            edge_index: Graph edges [2, total_edges]
            batch: Graph batch mapping [total_nodes]
            input_ids: Token IDs [batch, seq_len]
            attention_mask: Attention mask [batch, seq_len]
            return_mlm_logits: Whether to return MLM logits for training
        
        Returns:
            Dictionary containing:
                - embeddings: [batch, 768] - Function-level embeddings
                - graph_summary: [batch, graph_dim] - Graph representations
                - mlm_logits: [batch, seq_len, vocab_size] - MLM predictions (if requested)
        """
        # Step 1: Process graph through GNN (T065)
        graph_summary = self.gnn_encoder(
            x=node_features,
            edge_index=edge_index,
            batch=batch
        )  # [batch, graph_dim]
        
        # Step 2: Process tokens through BERT with graph prefix (T065)
        bert_outputs = self.bert_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            graph_summary=graph_summary,
            return_dict=True
        )
        
        # Step 3: Extract [CLS] embedding for similarity learning (T065)
        embeddings = bert_outputs.pooler_output  # [batch, 768]
        
        # Prepare output dictionary
        outputs = {
            "embeddings": embeddings,
            "graph_summary": graph_summary,
            "last_hidden_state": bert_outputs.last_hidden_state
        }
        
        # Optionally compute MLM logits for training (T065)
        if return_mlm_logits:
            mlm_logits = self.mlm_head(bert_outputs.last_hidden_state)
            outputs["mlm_logits"] = mlm_logits
        
        return outputs
    
    def get_embeddings(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Convenience method to get embeddings only (T066).
        
        Args:
            node_features: Node features [total_nodes, 768]
            edge_index: Graph edges [2, total_edges]
            batch: Graph batch mapping [total_nodes]
            input_ids: Token IDs [batch, seq_len]
            attention_mask: Attention mask [batch, seq_len]
        
        Returns:
            embeddings: [batch, 768] - Function-level embeddings
        """
        with torch.no_grad():
            outputs = self.forward(
                node_features=node_features,
                edge_index=edge_index,
                batch=batch,
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_mlm_logits=False
            )
        return outputs["embeddings"]
    
    def compute_similarity(
        self,
        embeddings_1: torch.Tensor,
        embeddings_2: torch.Tensor,
        metric: str = "cosine"
    ) -> torch.Tensor:
        """
        Compute similarity between two sets of embeddings.
        
        Args:
            embeddings_1: First set of embeddings [batch, 768]
            embeddings_2: Second set of embeddings [batch, 768]
            metric: Similarity metric ("cosine" or "euclidean")
        
        Returns:
            similarity: [batch] - Similarity scores
        """
        if metric == "cosine":
            return torch.nn.functional.cosine_similarity(embeddings_1, embeddings_2, dim=1)
        elif metric == "euclidean":
            return -torch.norm(embeddings_1 - embeddings_2, dim=1)
        else:
            raise ValueError(f"Unknown metric: {metric}")
    
    def save_checkpoint(self, path: str, epoch: int, optimizer_state: Optional[dict] = None):
        """
        Save model checkpoint.
        
        Args:
            path: Path to save checkpoint
            epoch: Current epoch
            optimizer_state: Optimizer state dict (optional)
        """
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.state_dict(),
            "config": {
                "graph_dim": self.graph_dim
            }
        }
        
        if optimizer_state is not None:
            checkpoint["optimizer_state_dict"] = optimizer_state
        
        torch.save(checkpoint, path)
        logger.info(f"Checkpoint saved to {path}")
    
    def load_checkpoint(self, path: str, load_optimizer: bool = False):
        """
        Load model checkpoint.
        
        Args:
            path: Path to checkpoint
            load_optimizer: Whether to load optimizer state
        
        Returns:
            Checkpoint dict (contains epoch, optimizer_state if available)
        """
        checkpoint = torch.load(path)
        self.load_state_dict(checkpoint["model_state_dict"])
        logger.info(f"Model loaded from {path} (epoch {checkpoint['epoch']})")
        
        if load_optimizer:
            return checkpoint
        else:
            return {"epoch": checkpoint["epoch"]}
