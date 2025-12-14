"""
BERT Encoder with Graph Prefix Integration

Implements BERT with custom KV-Prefix attention mechanism for deep graph
structure injection across all 12 transformer layers.

Module: models.bert_encoder
Owner: User Story 5 (US5) - BERT Integration & Siamese Training
Tasks: T059-T062
"""

import logging
from typing import Optional, Tuple

import torch
import torch.nn as nn
from transformers import BertModel, BertConfig

from models.custom_attention import KVPrefixAttention

logger = logging.getLogger("bcsd.models")


class BERTWithGraphPrefix(nn.Module):
    """
    BERT with Graph Prefix: Replaces standard attention with KV-Prefix attention (T059-T062).
    
    THE INTEGRATION:
    This module takes a pre-trained BERT model and replaces its standard self-attention
    mechanism with our custom KVPrefixAttention in all 12 layers. This allows the
    graph summary to influence token representations at every layer of BERT.
    
    Architecture:
    1. Load pre-trained BERT (bert-base-uncased)
    2. Replace self-attention in all 12 layers with KVPrefixAttention
    3. Keep all other BERT components unchanged (embeddings, FFN, layer norm)
    4. Forward pass injects graph_summary into every attention layer
    
    This "deep prefix injection" is more powerful than simple concatenation because:
    - Graph structure influences all layers, not just input
    - Each layer can learn different graph-text interactions
    - Preserves BERT's pre-trained representations while adding graph context
    
    Example:
        >>> bert = BERTWithGraphPrefix(
        ...     graph_dim=256,
        ...     pretrained_model="bert-base-uncased"
        ... )
        >>> # input_ids: [batch, seq_len]
        >>> # attention_mask: [batch, seq_len]
        >>> # graph_summary: [batch, graph_dim]
        >>> outputs = bert(
        ...     input_ids=input_ids,
        ...     attention_mask=attention_mask,
        ...     graph_summary=graph_summary
        ... )
        >>> print(outputs.last_hidden_state.shape)  # [batch, seq_len, 768]
        >>> print(outputs.pooler_output.shape)  # [batch, 768] - [CLS] embedding
    """
    
    def __init__(
        self,
        graph_dim: int = 256,
        pretrained_model: str = "bert-base-uncased",
        freeze_embeddings: bool = False,
        dropout: float = 0.1
    ):
        """
        Initialize BERT with graph prefix (T060).
        
        Args:
            graph_dim: Graph summary dimension (must match GNN output)
            pretrained_model: HuggingFace model name
            freeze_embeddings: Whether to freeze BERT embeddings
            dropout: Dropout probability
        """
        super().__init__()
        
        logger.info(f"Loading pre-trained BERT: {pretrained_model}")
        
        # Load pre-trained BERT
        self.bert = BertModel.from_pretrained(pretrained_model)
        self.config = self.bert.config
        
        # Store dimensions
        self.graph_dim = graph_dim
        self.hidden_size = self.config.hidden_size  # 768 for bert-base
        self.num_heads = self.config.num_attention_heads  # 12 for bert-base
        self.num_layers = self.config.num_hidden_layers  # 12 for bert-base
        
        # Replace self-attention in all layers with KVPrefixAttention (T060)
        logger.info(f"Replacing attention in {self.num_layers} layers with KVPrefixAttention")
        
        for layer_idx in range(self.num_layers):
            # Get the attention module
            attention_module = self.bert.encoder.layer[layer_idx].attention.self
            
            # Create custom attention with same dimensions
            custom_attention = KVPrefixAttention(
                hidden_size=self.hidden_size,
                num_heads=self.num_heads,
                graph_dim=graph_dim,
                dropout=dropout
            )
            
            # Store original projections (we'll reuse them)
            custom_attention.query = attention_module.query
            custom_attention.key = attention_module.key
            custom_attention.value = attention_module.value
            
            # Replace the attention module
            self.bert.encoder.layer[layer_idx].attention.self = custom_attention
            
            logger.debug(f"  Layer {layer_idx}: Replaced attention with KVPrefixAttention")
        
        # Optionally freeze embeddings
        if freeze_embeddings:
            logger.info("Freezing BERT embeddings")
            for param in self.bert.embeddings.parameters():
                param.requires_grad = False
        
        logger.info(f"BERTWithGraphPrefix initialized: "
                   f"hidden={self.hidden_size}, heads={self.num_heads}, "
                   f"layers={self.num_layers}, graph_dim={graph_dim}")
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        graph_summary: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        return_dict: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with deep graph prefix injection (T061).
        
        Args:
            input_ids: Token IDs [batch, seq_len]
            attention_mask: Attention mask [batch, seq_len]
            graph_summary: Graph summary from GNN [batch, graph_dim]
            token_type_ids: Token type IDs (optional)
            position_ids: Position IDs (optional)
            return_dict: Whether to return dict or tuple
        
        Returns:
            If return_dict=True:
                BertModel outputs with:
                - last_hidden_state: [batch, seq_len, hidden_size]
                - pooler_output: [batch, hidden_size] - [CLS] embedding
            If return_dict=False:
                Tuple of (last_hidden_state, pooler_output)
        """
        # Get embeddings from BERT
        embedding_output = self.bert.embeddings(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids
        )
        
        # Extended attention mask for BERT (add broadcasting dimensions)
        extended_attention_mask = attention_mask[:, None, None, :]  # [batch, 1, 1, seq_len]
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        
        # Pass through encoder layers with graph prefix injection (T061)
        hidden_states = embedding_output
        
        for layer_idx, layer_module in enumerate(self.bert.encoder.layer):
            # Get the custom attention module
            attention_module = layer_module.attention.self
            
            # Compute Q, K, V from hidden states
            batch_size, seq_len, hidden_size = hidden_states.shape
            
            query = attention_module.query(hidden_states)
            key = attention_module.key(hidden_states)
            value = attention_module.value(hidden_states)
            
            # Reshape for multi-head attention
            query = query.view(batch_size, seq_len, self.num_heads, self.hidden_size // self.num_heads)
            query = query.transpose(1, 2)  # [batch, heads, seq_len, head_dim]
            
            key = key.view(batch_size, seq_len, self.num_heads, self.hidden_size // self.num_heads)
            key = key.transpose(1, 2)
            
            value = value.view(batch_size, seq_len, self.num_heads, self.hidden_size // self.num_heads)
            value = value.transpose(1, 2)
            
            # Apply custom attention with graph prefix
            attention_output, attention_weights = attention_module(
                query=query,
                key=key,
                value=value,
                graph_summary=graph_summary,
                attention_mask=attention_mask  # Original mask, will be extended in KVPrefixAttention
            )
            
            # Reshape back to [batch, seq_len, hidden_size]
            attention_output = attention_output.transpose(1, 2).contiguous()
            attention_output = attention_output.view(batch_size, seq_len, hidden_size)
            
            # Apply attention output layer (dense + dropout + layer norm)
            attention_output = layer_module.attention.output.dense(attention_output)
            attention_output = layer_module.attention.output.dropout(attention_output)
            attention_output = layer_module.attention.output.LayerNorm(attention_output + hidden_states)
            
            # Apply feed-forward network
            intermediate_output = layer_module.intermediate(attention_output)
            layer_output = layer_module.output.dense(intermediate_output)
            layer_output = layer_module.output.dropout(layer_output)
            layer_output = layer_module.output.LayerNorm(layer_output + attention_output)
            
            # Update hidden states for next layer
            hidden_states = layer_output
        
        # Apply final pooler to get [CLS] embedding
        pooled_output = self.bert.pooler(hidden_states)
        
        if return_dict:
            from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions
            return BaseModelOutputWithPoolingAndCrossAttentions(
                last_hidden_state=hidden_states,
                pooler_output=pooled_output
            )
        else:
            return (hidden_states, pooled_output)
    
    def get_cls_embedding(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        graph_summary: torch.Tensor
    ) -> torch.Tensor:
        """
        Get [CLS] token embedding (convenience method).
        
        Args:
            input_ids: Token IDs [batch, seq_len]
            attention_mask: Attention mask [batch, seq_len]
            graph_summary: Graph summary [batch, graph_dim]
        
        Returns:
            [CLS] embeddings [batch, hidden_size]
        """
        outputs = self.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            graph_summary=graph_summary,
            return_dict=True
        )
        return outputs.pooler_output
