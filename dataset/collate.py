"""
Custom Collate Function for Heterogeneous Batching

Handles variable-length sequences (instructions) and variable-size graphs (CFGs)
in the same batch for Siamese training.

Module: dataset.collate
Owner: User Story 2 (US2) - PyTorch Dataset Implementation
Tasks: T030-T033
"""

import logging
from typing import List, Dict, Any

import torch
from torch.nn.utils.rnn import pad_sequence

logger = logging.getLogger("bcsd.dataset")


def collate_heterogeneous(batch: List[Dict]) -> Dict[str, Any]:
    """
    Custom collate function for heterogeneous batching (T030-T033).
    
    Handles:
    1. Variable-length token sequences → pad to max length in batch (T031)
    2. Variable-size graphs → separate lists, indices for batching (T032)
    3. Create attention masks for padded sequences (T031)
    4. Construct batch dictionary (T033)
    
    Note: For Siamese training, each sample contains 2 binaries (binary_1, binary_2).
    We process them separately to maintain independence during forward pass.
    
    Args:
        batch: List of samples from BinaryCodeDataset.__getitem__
            Each sample: {
                "binary_1": {"tokens": [...], "edges": [...], ...},
                "binary_2": {"tokens": [...], "edges": [...], ...},
                "label": 1,
                "metadata": {...}
            }
    
    Returns:
        Dictionary with:
            - binary_1_tokens: [batch_size, max_len] padded tokens
            - binary_1_attention_mask: [batch_size, max_len] attention mask
            - binary_1_edges: List of edge lists (one per sample)
            - binary_1_node_counts: [batch_size] number of nodes per graph
            - binary_2_tokens: [batch_size, max_len] padded tokens
            - binary_2_attention_mask: [batch_size, max_len] attention mask  
            - binary_2_edges: List of edge lists (one per sample)
            - binary_2_node_counts: [batch_size] number of nodes per graph
            - labels: [batch_size] all 1s (positive pairs)
            - metadata: List of pair metadata dicts
    
    Example:
        >>> from torch.utils.data import DataLoader
        >>> dataloader = DataLoader(
        ...     dataset,
        ...     batch_size=16,
        ...     collate_fn=collate_heterogeneous
        ... )
        >>> for batch in dataloader:
        ...     print(batch["binary_1_tokens"].shape)  # [16, max_len]
        ...     break
    """
    batch_size = len(batch)
    
    # Extract binary_1 and binary_2 data
    binary_1_list = [sample["binary_1"] for sample in batch]
    binary_2_list = [sample["binary_2"] for sample in batch]
    labels = [sample["label"] for sample in batch]
    metadata = [sample["metadata"] for sample in batch]
    
    # Collate binary_1
    binary_1_collated = _collate_binary_list(binary_1_list)
    
    # Collate binary_2
    binary_2_collated = _collate_binary_list(binary_2_list)
    
    # Combine into batch dictionary (T033)
    batch_dict = {
        # Binary 1 data
        "binary_1_tokens": binary_1_collated["tokens"],
        "binary_1_attention_mask": binary_1_collated["attention_mask"],
        "binary_1_edges": binary_1_collated["edges"],
        "binary_1_node_counts": binary_1_collated["node_counts"],
        
        # Binary 2 data
        "binary_2_tokens": binary_2_collated["tokens"],
        "binary_2_attention_mask": binary_2_collated["attention_mask"],
        "binary_2_edges": binary_2_collated["edges"],
        "binary_2_node_counts": binary_2_collated["node_counts"],
        
        # Labels and metadata
        "labels": torch.tensor(labels, dtype=torch.long),
        "metadata": metadata
    }
    
    return batch_dict


def _collate_binary_list(binary_list: List[Dict]) -> Dict[str, Any]:
    """
    Collate list of binaries (helper for binary_1 and binary_2).
    
    Args:
        binary_list: List of binary dictionaries
        
    Returns:
        Dictionary with padded tokens, attention masks, edges, node counts
    """
    # Check if we have pre-tokenized data
    has_token_ids = any("token_ids" in b and b["token_ids"] is not None for b in binary_list)
    
    if has_token_ids:
        # Use pre-tokenized IDs
        sequences = [b.get("token_ids", []) or [] for b in binary_list]
        pad_id = 1  # CLAP-ASM pad_token_id
    else:
        # Use raw tokens strings (legacy)
        sequences = [b.get("tokens", []) or ["<PAD>"] for b in binary_list]
        pad_id = "<PAD>"
        
    # Find max length
    max_len = max((len(seq) for seq in sequences), default=0)
    
    # Pad sequences
    padded_sequences = []
    attention_masks = []
    
    for seq in sequences:
        actual_len = len(seq)
        padding_len = max_len - actual_len
        
        # Pad
        padded = seq + [pad_id] * padding_len
        padded_sequences.append(padded)
        
        # Create attention mask
        mask = [1] * actual_len + [0] * padding_len
        attention_masks.append(mask)
    
    # Extract graph data (T032)
    edges = [binary.get("edges", []) for binary in binary_list]
    node_counts = torch.tensor([binary.get("node_count", 0) for binary in binary_list], dtype=torch.long)
    
    # Convert to tensor if using IDs
    if has_token_ids:
        padded_sequences = torch.tensor(padded_sequences, dtype=torch.long)
        attention_masks = torch.tensor(attention_masks, dtype=torch.long)
    
    return {
        "tokens": padded_sequences,  # Tensor [batch, max_len] or List[List[str]]
        "attention_mask": attention_masks,  # Tensor [batch, max_len]
        "edges": edges,  # List of edge lists
        "node_counts": node_counts  # [batch_size]
    }


def collate_with_tokenizer(tokenizer):
    """
    Factory function to create collate_fn with tokenizer.
    
    This will be used in training when we have the actual tokenizer
    to convert token strings to IDs.
    
    Args:
        tokenizer: ClapASMTokenizer instance
        
    Returns:
        Collate function with tokenizer closure
    """
    def collate_fn(batch: List[Dict]) -> Dict[str, Any]:
        """Collate with actual tokenization."""
        batch_size = len(batch)
        
        # Extract data
        binary_1_list = [sample["binary_1"] for sample in batch]
        binary_2_list = [sample["binary_2"] for sample in batch]
        labels = [sample["label"] for sample in batch]
        metadata = [sample["metadata"] for sample in batch]
        
        # Collate with tokenization
        binary_1_collated = _collate_with_tokenization(binary_1_list, tokenizer)
        binary_2_collated = _collate_with_tokenization(binary_2_list, tokenizer)
        
        batch_dict = {
            "binary_1_tokens": binary_1_collated["token_ids"],
            "binary_1_attention_mask": binary_1_collated["attention_mask"],
            "binary_1_edges": binary_1_collated["edges"],
            "binary_1_node_counts": binary_1_collated["node_counts"],
            
            "binary_2_tokens": binary_2_collated["token_ids"],
            "binary_2_attention_mask": binary_2_collated["attention_mask"],
            "binary_2_edges": binary_2_collated["edges"],
            "binary_2_node_counts": binary_2_collated["node_counts"],
            
            "labels": torch.tensor(labels, dtype=torch.long),
            "metadata": metadata
        }
        
        return batch_dict
    
    return collate_fn


def _collate_with_tokenization(binary_list: List[Dict], tokenizer) -> Dict[str, Any]:
    """
    Collate with actual tokenization (for use during training).
    
    Args:
        binary_list: List of binary dictionaries
        tokenizer: AssemblyTokenizer instance
        
    Returns:
        Dictionary with token IDs, attention masks, edges, node counts
    """
    # Tokenize each sequence
    tokenized_list = []
    for binary in binary_list:
        tokens = binary["tokens"]
        if not tokens:
            tokens = []
        
        # Use tokenizer (assumes tokens are instruction strings)
        # In reality, they're already parsed tokens, so we'd need to reconstruct instructions
        # For now, treat each token as an instruction
        result = tokenizer.tokenize(tokens, add_special_tokens=True)
        tokenized_list.append(result)
    
    # Find max length
    max_len = max(result["length"] for result in tokenized_list)
    
    # Pad to max length
    token_ids = []
    attention_masks = []
    
    for result in tokenized_list:
        token_ids.append(result["token_ids"][:max_len])
        attention_masks.append(result["attention_mask"][:max_len])
    
    # Extract graph data
    edges = [binary["edges"] for binary in binary_list]
    node_counts = torch.tensor([binary["node_count"] for binary in binary_list], dtype=torch.long)
    
    return {
        "token_ids": torch.tensor(token_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_masks, dtype=torch.long),
        "edges": edges,
        "node_counts": node_counts
    }

