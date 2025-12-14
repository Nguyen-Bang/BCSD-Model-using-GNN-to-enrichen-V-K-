"""
Validation Metrics for BCSD Training

Implements validation functions to evaluate model performance during training.

Module: training.metrics
Owner: User Story 7 (US7) - Training Infrastructure
Tasks: T077
"""

import logging
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

logger = logging.getLogger("bcsd.training")


def compute_similarity_accuracy(
    embeddings_a: torch.Tensor,
    embeddings_b: torch.Tensor,
    threshold: float = 0.5
) -> Dict[str, float]:
    """
    Compute similarity accuracy for positive pairs (T077).
    
    Measures how well the model identifies positive pairs based on cosine similarity.
    
    Metrics:
    - mean_positive_similarity: Average cosine similarity of positive pairs
    - std_positive_similarity: Standard deviation of positive similarities
    - accuracy_at_threshold: % of pairs with similarity > threshold
    
    Args:
        embeddings_a: Embeddings from variant A [batch, dim]
        embeddings_b: Embeddings from variant B [batch, dim] (positive pairs)
        threshold: Similarity threshold for accuracy (default: 0.5)
    
    Returns:
        Dictionary with accuracy metrics
    
    Example:
        >>> metrics = compute_similarity_accuracy(emb_a, emb_b, threshold=0.5)
        >>> print(f"Mean similarity: {metrics['mean_positive_similarity']:.4f}")
        >>> print(f"Accuracy@0.5: {metrics['accuracy_at_threshold']:.2%}")
    """
    # Normalize embeddings
    embeddings_a = F.normalize(embeddings_a, p=2, dim=1)
    embeddings_b = F.normalize(embeddings_b, p=2, dim=1)
    
    # Compute cosine similarities for positive pairs
    # Element-wise dot product: sum(emb_a * emb_b, dim=1)
    positive_similarities = (embeddings_a * embeddings_b).sum(dim=1)
    
    # Convert to numpy for statistics
    similarities_np = positive_similarities.cpu().detach().numpy()
    
    # Compute metrics
    mean_sim = float(np.mean(similarities_np))
    std_sim = float(np.std(similarities_np))
    min_sim = float(np.min(similarities_np))
    max_sim = float(np.max(similarities_np))
    
    # Accuracy: % of pairs above threshold
    accuracy = float((similarities_np > threshold).mean())
    
    return {
        "mean_positive_similarity": mean_sim,
        "std_positive_similarity": std_sim,
        "min_positive_similarity": min_sim,
        "max_positive_similarity": max_sim,
        "accuracy_at_threshold": accuracy,
        "threshold": threshold
    }


def compute_ranking_metrics(
    query_embeddings: torch.Tensor,
    candidate_embeddings: torch.Tensor,
    positive_indices: torch.Tensor
) -> Dict[str, float]:
    """
    Compute ranking metrics for retrieval evaluation (T077).
    
    Measures how well the model ranks true positives among candidates.
    
    Metrics:
    - mean_reciprocal_rank (MRR): Average 1/rank of first positive
    - recall_at_k: % of queries where positive is in top-k
    
    Args:
        query_embeddings: Query embeddings [num_queries, dim]
        candidate_embeddings: Candidate embeddings [num_candidates, dim]
        positive_indices: Index of positive candidate for each query [num_queries]
    
    Returns:
        Dictionary with ranking metrics
    
    Example:
        >>> metrics = compute_ranking_metrics(queries, candidates, positives)
        >>> print(f"MRR: {metrics['mean_reciprocal_rank']:.4f}")
        >>> print(f"Recall@10: {metrics['recall_at_10']:.2%}")
    """
    # Normalize embeddings
    query_embeddings = F.normalize(query_embeddings, p=2, dim=1)
    candidate_embeddings = F.normalize(candidate_embeddings, p=2, dim=1)
    
    # Compute similarity matrix: [num_queries, num_candidates]
    similarity_matrix = torch.mm(query_embeddings, candidate_embeddings.t())
    
    # Get rankings (descending similarity)
    # argsort gives indices in ascending order, so negate similarities
    rankings = torch.argsort(-similarity_matrix, dim=1)  # [num_queries, num_candidates]
    
    # Find rank of positive for each query
    num_queries = query_embeddings.size(0)
    ranks = []
    
    for i in range(num_queries):
        positive_idx = positive_indices[i].item()
        # Find where positive_idx appears in rankings[i]
        rank = (rankings[i] == positive_idx).nonzero(as_tuple=True)[0].item() + 1  # 1-indexed
        ranks.append(rank)
    
    ranks = np.array(ranks)
    
    # Compute metrics
    mrr = float(np.mean(1.0 / ranks))
    
    # Recall@k for different k values
    recall_metrics = {}
    for k in [1, 5, 10, 20]:
        recall_at_k = float((ranks <= k).mean())
        recall_metrics[f"recall_at_{k}"] = recall_at_k
    
    return {
        "mean_reciprocal_rank": mrr,
        "mean_rank": float(np.mean(ranks)),
        "median_rank": float(np.median(ranks)),
        **recall_metrics
    }


def log_embedding_statistics(
    embeddings: torch.Tensor,
    prefix: str = "train"
) -> Dict[str, float]:
    """
    Compute and log embedding statistics (T077).
    
    Tracks embedding quality during training:
    - Norm statistics (should stabilize during training)
    - Value distribution (mean, std, sparsity)
    
    Args:
        embeddings: Embeddings [batch, dim]
        prefix: Prefix for metric keys (e.g., "train", "val")
    
    Returns:
        Dictionary with embedding statistics
    
    Example:
        >>> stats = log_embedding_statistics(embeddings, prefix="val")
        >>> print(f"Mean L2 norm: {stats['val_mean_l2_norm']:.4f}")
    """
    # Convert to numpy
    embeddings_np = embeddings.cpu().detach().numpy()
    
    # Compute L2 norms
    l2_norms = np.linalg.norm(embeddings_np, axis=1)
    
    # Compute statistics
    stats = {
        f"{prefix}_mean_l2_norm": float(np.mean(l2_norms)),
        f"{prefix}_std_l2_norm": float(np.std(l2_norms)),
        f"{prefix}_min_l2_norm": float(np.min(l2_norms)),
        f"{prefix}_max_l2_norm": float(np.max(l2_norms)),
        f"{prefix}_mean_embedding_value": float(np.mean(embeddings_np)),
        f"{prefix}_std_embedding_value": float(np.std(embeddings_np)),
        f"{prefix}_sparsity": float((np.abs(embeddings_np) < 0.01).mean())
    }
    
    return stats


def compute_validation_metrics(
    model,
    val_loader,
    loss_fn,
    device: str = "cuda"
) -> Dict[str, float]:
    """
    Compute comprehensive validation metrics (T077).
    
    Runs validation loop and computes:
    - Validation loss
    - Similarity accuracy
    - Embedding statistics
    
    Args:
        model: BCSD model
        val_loader: Validation DataLoader
        loss_fn: Loss function (JointLoss)
        device: Device for computation
    
    Returns:
        Dictionary with all validation metrics
    
    Example:
        >>> metrics = compute_validation_metrics(model, val_loader, loss_fn)
        >>> print(f"Val Loss: {metrics['val_loss']:.4f}")
        >>> print(f"Val Accuracy: {metrics['val_accuracy_at_threshold']:.2%}")
    """
    model.eval()
    
    total_loss = 0.0
    all_similarities = []
    all_embeddings = []
    num_batches = 0
    
    with torch.no_grad():
        for batch in val_loader:
            # Move batch to device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            graph_x = batch["graph_batch"].x.to(device)
            graph_edge_index = batch["graph_batch"].edge_index.to(device)
            graph_batch_idx = batch["graph_batch"].batch.to(device)
            
            # Assume batch has pairs: first half is variant A, second half is variant B
            batch_size = input_ids.size(0) // 2
            
            # Get embeddings for both variants
            embeddings = model.get_embeddings(
                input_ids=input_ids,
                attention_mask=attention_mask,
                graph_x=graph_x,
                graph_edge_index=graph_edge_index,
                graph_batch=graph_batch_idx
            )
            
            embeddings_a = embeddings[:batch_size]
            embeddings_b = embeddings[batch_size:]
            
            # Compute loss
            loss = loss_fn(
                embeddings_a=embeddings_a,
                embeddings_b=embeddings_b
            )
            
            total_loss += loss.item()
            
            # Collect embeddings and compute similarities
            all_embeddings.append(embeddings)
            
            # Compute similarities for this batch
            sim_metrics = compute_similarity_accuracy(embeddings_a, embeddings_b)
            all_similarities.append(sim_metrics["mean_positive_similarity"])
            
            num_batches += 1
    
    # Aggregate metrics
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    avg_similarity = np.mean(all_similarities) if all_similarities else 0.0
    
    # Embedding statistics
    all_embeddings_tensor = torch.cat(all_embeddings, dim=0)
    embedding_stats = log_embedding_statistics(all_embeddings_tensor, prefix="val")
    
    metrics = {
        "val_loss": avg_loss,
        "val_mean_similarity": avg_similarity,
        **embedding_stats
    }
    
    return metrics


def test_metrics():
    """Test validation metrics with dummy data (T077)."""
    print("Testing Validation Metrics...")
    
    batch_size = 8
    embedding_dim = 768
    
    # Test similarity accuracy (T077)
    print("\n1. Testing compute_similarity_accuracy...")
    embeddings_a = torch.randn(batch_size, embedding_dim)
    embeddings_b = embeddings_a + torch.randn(batch_size, embedding_dim) * 0.1  # Similar but not identical
    
    metrics = compute_similarity_accuracy(embeddings_a, embeddings_b, threshold=0.5)
    print(f"  Mean positive similarity: {metrics['mean_positive_similarity']:.4f}")
    print(f"  Std positive similarity: {metrics['std_positive_similarity']:.4f}")
    print(f"  Accuracy@0.5: {metrics['accuracy_at_threshold']:.2%}")
    assert 0 <= metrics['mean_positive_similarity'] <= 1, "Similarity should be in [0, 1]"
    print("  ✓ Similarity accuracy test passed")
    
    # Test ranking metrics
    print("\n2. Testing compute_ranking_metrics...")
    num_queries = 5
    num_candidates = 20
    query_embeddings = torch.randn(num_queries, embedding_dim)
    candidate_embeddings = torch.randn(num_candidates, embedding_dim)
    positive_indices = torch.randint(0, num_candidates, (num_queries,))
    
    metrics = compute_ranking_metrics(query_embeddings, candidate_embeddings, positive_indices)
    print(f"  Mean Reciprocal Rank: {metrics['mean_reciprocal_rank']:.4f}")
    print(f"  Mean Rank: {metrics['mean_rank']:.2f}")
    print(f"  Recall@10: {metrics['recall_at_10']:.2%}")
    assert 0 <= metrics['mean_reciprocal_rank'] <= 1, "MRR should be in [0, 1]"
    print("  ✓ Ranking metrics test passed")
    
    # Test embedding statistics
    print("\n3. Testing log_embedding_statistics...")
    embeddings = torch.randn(batch_size, embedding_dim)
    stats = log_embedding_statistics(embeddings, prefix="test")
    
    print(f"  Mean L2 norm: {stats['test_mean_l2_norm']:.4f}")
    print(f"  Std L2 norm: {stats['test_std_l2_norm']:.4f}")
    print(f"  Sparsity: {stats['test_sparsity']:.2%}")
    assert stats['test_mean_l2_norm'] > 0, "L2 norm should be positive"
    print("  ✓ Embedding statistics test passed")
    
    print("\n✓ All validation metrics tests passed!")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_metrics()
