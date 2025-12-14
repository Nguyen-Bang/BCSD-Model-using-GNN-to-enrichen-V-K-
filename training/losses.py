"""
Loss Functions for BCSD Training

Implements three loss functions for the BCSD model:
1. MLMLoss: Masked Language Model loss for token prediction
2. InfoNCELoss: Contrastive loss with in-batch negatives
3. JointLoss: Combines MLM + λ*contrastive loss

Module: training.losses
Owner: User Story 7 (US7) - Training Infrastructure
Tasks: T074-T076
"""

import logging
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger("bcsd.training")


class MLMLoss(nn.Module):
    """
    Masked Language Model Loss (T074).
    
    Computes cross-entropy loss for masked token prediction. Used during
    Stage 1 pretraining to learn assembly instruction semantics.
    
    Architecture:
    1. Model outputs logits for all tokens: [batch, seq_len, vocab_size]
    2. Select only masked positions using boolean indexing
    3. Compute cross-entropy loss against true token IDs
    
    Example:
        >>> mlm_loss = MLMLoss()
        >>> logits = model(input_ids, attention_mask)  # [batch, seq_len, vocab_size]
        >>> labels = input_ids.clone()  # Ground truth (before masking)
        >>> masked_positions = (input_ids == MASK_TOKEN_ID)
        >>> loss = mlm_loss(logits, labels, masked_positions)
    """
    
    def __init__(self, ignore_index: int = -100):
        """
        Initialize MLM loss.
        
        Args:
            ignore_index: Token ID to ignore in loss computation (padding tokens)
        """
        super().__init__()
        self.ignore_index = ignore_index
        self.criterion = nn.CrossEntropyLoss(ignore_index=ignore_index)
        
        logger.info(f"MLMLoss initialized with ignore_index={ignore_index}")
    
    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        masked_positions: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute MLM loss.
        
        Args:
            logits: Model output [batch, seq_len, vocab_size]
            labels: Ground truth token IDs [batch, seq_len]
            masked_positions: Boolean mask [batch, seq_len] indicating which tokens were masked
                             If None, compute loss on all positions
        
        Returns:
            Scalar loss tensor
        """
        batch_size, seq_len, vocab_size = logits.shape
        
        if masked_positions is not None:
            # Only compute loss on masked tokens
            # Flatten and select masked positions
            logits_flat = logits.view(-1, vocab_size)  # [batch*seq_len, vocab_size]
            labels_flat = labels.view(-1)  # [batch*seq_len]
            masked_flat = masked_positions.view(-1)  # [batch*seq_len]
            
            # Select only masked positions
            logits_masked = logits_flat[masked_flat]  # [num_masked, vocab_size]
            labels_masked = labels_flat[masked_flat]  # [num_masked]
            
            if logits_masked.size(0) == 0:
                # No masked tokens in this batch
                return torch.tensor(0.0, device=logits.device)
            
            loss = self.criterion(logits_masked, labels_masked)
        else:
            # Compute loss on all positions (reshape for CrossEntropyLoss)
            logits_flat = logits.view(-1, vocab_size)
            labels_flat = labels.view(-1)
            loss = self.criterion(logits_flat, labels_flat)
        
        return loss


class InfoNCELoss(nn.Module):
    """
    InfoNCE Contrastive Loss (T075).
    
    Also known as NT-Xent (Normalized Temperature-scaled Cross Entropy).
    Used for contrastive learning in Siamese architecture.
    
    Architecture:
    1. Given batch of positive pairs: [(emb_A1, emb_B1), (emb_A2, emb_B2), ...]
    2. All embeddings: [emb_A1, emb_B1, emb_A2, emb_B2, ...] → shape [2*batch, dim]
    3. Compute similarity matrix: cosine_sim(all, all) → [2*batch, 2*batch]
    4. For each positive pair (i, j):
        - Positive: sim(i, j)
        - Negatives: all other embeddings in batch (in-batch negatives)
    5. Loss = -log(exp(sim(i,j)/τ) / Σ exp(sim(i,k)/τ))
    
    Temperature (τ):
    - τ=0.07 (typical): Sharpens similarity distribution
    - Lower τ → harder negative mining
    
    Example:
        >>> loss_fn = InfoNCELoss(temperature=0.07)
        >>> emb_A = model(sample_A)  # [batch, 768]
        >>> emb_B = model(sample_B)  # [batch, 768]
        >>> loss = loss_fn(emb_A, emb_B)
    """
    
    def __init__(self, temperature: float = 0.07):
        """
        Initialize InfoNCE loss.
        
        Args:
            temperature: Scaling factor for similarities (default: 0.07)
        """
        super().__init__()
        self.temperature = temperature
        
        logger.info(f"InfoNCELoss initialized with temperature={temperature}")
    
    def forward(
        self,
        embeddings_a: torch.Tensor,
        embeddings_b: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute InfoNCE loss with in-batch negatives.
        
        Args:
            embeddings_a: Embeddings from variant A [batch, embedding_dim]
            embeddings_b: Embeddings from variant B [batch, embedding_dim]
                         (positive pairs with embeddings_a)
        
        Returns:
            Scalar loss tensor
        """
        batch_size = embeddings_a.size(0)
        device = embeddings_a.device
        
        # Normalize embeddings to unit sphere
        embeddings_a = F.normalize(embeddings_a, p=2, dim=1)
        embeddings_b = F.normalize(embeddings_b, p=2, dim=1)
        
        # Concatenate all embeddings: [2*batch, dim]
        embeddings_all = torch.cat([embeddings_a, embeddings_b], dim=0)
        
        # Compute similarity matrix: [2*batch, 2*batch]
        # sim[i,j] = cosine_similarity(emb_i, emb_j)
        similarity_matrix = torch.mm(embeddings_all, embeddings_all.t()) / self.temperature
        
        # Create labels for positive pairs
        # For embeddings_a[i] at index i, its positive is embeddings_b[i] at index batch_size+i
        # For embeddings_b[i] at index batch_size+i, its positive is embeddings_a[i] at index i
        labels = torch.cat([
            torch.arange(batch_size, 2 * batch_size),  # Positives for embeddings_a
            torch.arange(0, batch_size)                # Positives for embeddings_b
        ]).to(device)
        
        # Mask out self-similarities (diagonal) by setting to large negative value
        mask = torch.eye(2 * batch_size, dtype=torch.bool, device=device)
        similarity_matrix = similarity_matrix.masked_fill(mask, -1e9)
        
        # Compute cross-entropy loss
        # For each embedding, the positive is the label, all others are negatives
        loss = F.cross_entropy(similarity_matrix, labels)
        
        return loss


class JointLoss(nn.Module):
    """
    Joint Loss: MLM + λ*Contrastive (T076).
    
    Combines masked language modeling with contrastive learning.
    
    Architecture:
    - Stage 1 (MLM only): λ=0, train on individual samples
    - Stage 2 (Contrastive): λ ∈ {0.3, 0.5, 0.7}, train on positive pairs
    
    Total Loss = MLM Loss + λ * Contrastive Loss
    
    Example:
        >>> joint_loss = JointLoss(lambda_contrastive=0.5)
        >>> 
        >>> # Forward pass through model
        >>> logits_a, emb_a = model(sample_a, return_logits=True)
        >>> logits_b, emb_b = model(sample_b, return_logits=True)
        >>> 
        >>> # Compute joint loss
        >>> loss = joint_loss(
        ...     mlm_logits=logits_a,
        ...     mlm_labels=labels_a,
        ...     mlm_masked_positions=masked_a,
        ...     embeddings_a=emb_a,
        ...     embeddings_b=emb_b
        ... )
    """
    
    def __init__(
        self,
        lambda_contrastive: float = 0.5,
        temperature: float = 0.07,
        ignore_index: int = -100
    ):
        """
        Initialize joint loss.
        
        Args:
            lambda_contrastive: Weight for contrastive loss (default: 0.5)
            temperature: Temperature for InfoNCE (default: 0.07)
            ignore_index: Ignore index for MLM loss (default: -100)
        """
        super().__init__()
        
        self.lambda_contrastive = lambda_contrastive
        
        # Initialize component losses
        self.mlm_loss = MLMLoss(ignore_index=ignore_index)
        self.contrastive_loss = InfoNCELoss(temperature=temperature)
        
        logger.info(f"JointLoss initialized: λ={lambda_contrastive}, τ={temperature}")
    
    def forward(
        self,
        mlm_logits: Optional[torch.Tensor] = None,
        mlm_labels: Optional[torch.Tensor] = None,
        mlm_masked_positions: Optional[torch.Tensor] = None,
        embeddings_a: Optional[torch.Tensor] = None,
        embeddings_b: Optional[torch.Tensor] = None,
        return_components: bool = False
    ) -> torch.Tensor:
        """
        Compute joint loss.
        
        Args:
            mlm_logits: MLM logits [batch, seq_len, vocab_size] (optional)
            mlm_labels: MLM labels [batch, seq_len] (optional)
            mlm_masked_positions: Masked positions [batch, seq_len] (optional)
            embeddings_a: Embeddings from variant A [batch, dim] (optional)
            embeddings_b: Embeddings from variant B [batch, dim] (optional)
            return_components: If True, return (total_loss, mlm_loss, contrastive_loss)
        
        Returns:
            If return_components=False: Scalar loss tensor
            If return_components=True: Tuple (total_loss, mlm_loss, contrastive_loss)
        """
        total_loss = 0.0
        mlm_loss_value = 0.0
        contrastive_loss_value = 0.0
        
        # Compute MLM loss if inputs provided
        if mlm_logits is not None and mlm_labels is not None:
            mlm_loss_value = self.mlm_loss(mlm_logits, mlm_labels, mlm_masked_positions)
            total_loss = total_loss + mlm_loss_value
        
        # Compute contrastive loss if embeddings provided
        if embeddings_a is not None and embeddings_b is not None and self.lambda_contrastive > 0:
            contrastive_loss_value = self.contrastive_loss(embeddings_a, embeddings_b)
            total_loss = total_loss + self.lambda_contrastive * contrastive_loss_value
        
        if return_components:
            return total_loss, mlm_loss_value, contrastive_loss_value
        else:
            return total_loss
    
    def set_lambda(self, lambda_contrastive: float):
        """Update lambda_contrastive weight (useful for staged training)."""
        self.lambda_contrastive = lambda_contrastive
        logger.info(f"Updated λ_contrastive to {lambda_contrastive}")


def test_losses():
    """Test loss functions with dummy data (T074-T076)."""
    print("Testing Loss Functions...")
    
    batch_size = 4
    seq_len = 20
    vocab_size = 5000
    embedding_dim = 768
    
    # Test MLMLoss (T074)
    print("\n1. Testing MLMLoss...")
    mlm_loss_fn = MLMLoss()
    
    logits = torch.randn(batch_size, seq_len, vocab_size)
    labels = torch.randint(0, vocab_size, (batch_size, seq_len))
    masked_positions = torch.rand(batch_size, seq_len) > 0.85  # 15% masked
    
    loss = mlm_loss_fn(logits, labels, masked_positions)
    print(f"  MLM Loss: {loss.item():.4f}")
    assert loss.item() > 0, "MLM loss should be positive"
    print("  ✓ MLMLoss test passed")
    
    # Test InfoNCELoss (T075)
    print("\n2. Testing InfoNCELoss...")
    contrastive_loss_fn = InfoNCELoss(temperature=0.07)
    
    embeddings_a = torch.randn(batch_size, embedding_dim)
    embeddings_b = torch.randn(batch_size, embedding_dim)
    
    loss = contrastive_loss_fn(embeddings_a, embeddings_b)
    print(f"  InfoNCE Loss: {loss.item():.4f}")
    assert loss.item() > 0, "InfoNCE loss should be positive"
    print("  ✓ InfoNCELoss test passed")
    
    # Test JointLoss (T076)
    print("\n3. Testing JointLoss...")
    joint_loss_fn = JointLoss(lambda_contrastive=0.5, temperature=0.07)
    
    total_loss, mlm, contrastive = joint_loss_fn(
        mlm_logits=logits,
        mlm_labels=labels,
        mlm_masked_positions=masked_positions,
        embeddings_a=embeddings_a,
        embeddings_b=embeddings_b,
        return_components=True
    )
    
    print(f"  Total Loss: {total_loss.item():.4f}")
    print(f"  MLM Component: {mlm.item():.4f}")
    print(f"  Contrastive Component: {contrastive.item():.4f}")
    print(f"  Weighted Contrastive: {0.5 * contrastive.item():.4f}")
    
    expected_total = mlm.item() + 0.5 * contrastive.item()
    assert abs(total_loss.item() - expected_total) < 1e-5, "Joint loss composition mismatch"
    print("  ✓ JointLoss test passed")
    
    # Test lambda adjustment
    print("\n4. Testing lambda adjustment...")
    joint_loss_fn.set_lambda(0.7)
    total_loss_new, mlm_new, contrastive_new = joint_loss_fn(
        mlm_logits=logits,
        mlm_labels=labels,
        mlm_masked_positions=masked_positions,
        embeddings_a=embeddings_a,
        embeddings_b=embeddings_b,
        return_components=True
    )
    
    expected_total_new = mlm_new.item() + 0.7 * contrastive_new.item()
    assert abs(total_loss_new.item() - expected_total_new) < 1e-5, "Lambda adjustment failed"
    print(f"  New Total Loss (λ=0.7): {total_loss_new.item():.4f}")
    print("  ✓ Lambda adjustment test passed")
    
    print("\n✓ All loss function tests passed!")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_losses()
