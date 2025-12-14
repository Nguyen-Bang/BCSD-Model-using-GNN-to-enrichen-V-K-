"""
Similarity Search for BCSD Embeddings

Mathematical similarity search using numpy/scipy - NO GPU required.
This demonstrates that once embeddings are generated, similarity detection
is purely mathematical and extremely fast.

Module: inference.similarity
Owner: User Story 8 (US8) - Vectorization & Similarity Search
Tasks: T094-T096
"""

import logging
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

logger = logging.getLogger("bcsd.inference")


def cosine_similarity(
    embedding_a: np.ndarray,
    embedding_b: np.ndarray
) -> float:
    """
    Compute cosine similarity between two embeddings (T094).
    
    Uses numpy for CPU-only computation - no GPU needed.
    
    Cosine similarity = (A · B) / (||A|| × ||B||)
    Range: [-1, 1] where 1 = identical, 0 = orthogonal, -1 = opposite
    
    Args:
        embedding_a: First embedding vector [dim]
        embedding_b: Second embedding vector [dim]
    
    Returns:
        Cosine similarity score in [-1, 1]
    
    Example:
        >>> emb_a = np.array([1, 2, 3])
        >>> emb_b = np.array([2, 4, 6])  # Same direction, scaled
        >>> similarity = cosine_similarity(emb_a, emb_b)
        >>> print(similarity)  # ~1.0 (very similar)
    """
    # Normalize vectors
    norm_a = np.linalg.norm(embedding_a)
    norm_b = np.linalg.norm(embedding_b)
    
    if norm_a == 0 or norm_b == 0:
        return 0.0
    
    # Compute cosine similarity
    dot_product = np.dot(embedding_a, embedding_b)
    similarity = dot_product / (norm_a * norm_b)
    
    return float(similarity)


def cosine_similarity_batch(
    query_embedding: np.ndarray,
    database_embeddings: np.ndarray
) -> np.ndarray:
    """
    Compute cosine similarities between query and all database embeddings.
    
    Vectorized implementation for speed.
    
    Args:
        query_embedding: Query vector [dim]
        database_embeddings: Database vectors [num_embeddings, dim]
    
    Returns:
        Similarity scores [num_embeddings]
    """
    # Normalize query
    query_norm = query_embedding / np.linalg.norm(query_embedding)
    
    # Normalize database
    database_norms = database_embeddings / np.linalg.norm(database_embeddings, axis=1, keepdims=True)
    
    # Compute dot products (cosine similarities)
    similarities = np.dot(database_norms, query_norm)
    
    return similarities


def top_k_similar(
    query_embedding: np.ndarray,
    database_embeddings: Dict[str, np.ndarray],
    k: int = 10
) -> List[Tuple[str, float]]:
    """
    Find top-K most similar embeddings (T095).
    
    Returns K closest matches with their similarity scores.
    Uses pure numpy - runs on CPU in <1 second for 10K vectors.
    
    Args:
        query_embedding: Query vector [dim]
        database_embeddings: Dict mapping IDs to embedding vectors
        k: Number of top results to return (default: 10)
    
    Returns:
        List of (id, similarity_score) tuples, sorted by similarity (descending)
    
    Example:
        >>> database = {
        ...     "func_1": np.array([1, 2, 3]),
        ...     "func_2": np.array([2, 4, 6]),
        ...     "func_3": np.array([-1, -2, -3])
        ... }
        >>> query = np.array([1.5, 3, 4.5])
        >>> results = top_k_similar(query, database, k=2)
        >>> for func_id, score in results:
        ...     print(f"{func_id}: {score:.4f}")
    """
    # Compute similarities for all database entries
    similarities = []
    
    for func_id, embedding in database_embeddings.items():
        similarity = cosine_similarity(query_embedding, embedding)
        similarities.append((func_id, similarity))
    
    # Sort by similarity (descending)
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    # Return top-k
    return similarities[:k]


def build_embedding_database(
    embedding_dir: str,
    pattern: str = "*.npy"
) -> Dict[str, np.ndarray]:
    """
    Build searchable embedding database from .npy files (T096).
    
    Loads all embedding files into memory for fast similarity search.
    
    Args:
        embedding_dir: Directory containing .npy embedding files
        pattern: Glob pattern for embedding files (default: "*.npy")
    
    Returns:
        Dictionary mapping binary names to embedding vectors
    
    Example:
        >>> database = build_embedding_database("embeddings/")
        >>> print(f"Loaded {len(database)} embeddings")
        >>> print(f"Database keys: {list(database.keys())[:5]}")
    """
    embedding_path = Path(embedding_dir)
    
    if not embedding_path.exists():
        logger.error(f"Embedding directory not found: {embedding_dir}")
        return {}
    
    # Find all embedding files
    embedding_files = list(embedding_path.glob(pattern))
    logger.info(f"Loading embeddings from {embedding_dir}")
    logger.info(f"Found {len(embedding_files)} embedding files")
    
    database = {}
    
    for embedding_file in embedding_files:
        try:
            # Load embedding
            embedding = np.load(embedding_file)
            
            # Use filename (without .npy) as ID
            func_id = embedding_file.stem
            
            database[func_id] = embedding
            
        except Exception as e:
            logger.warning(f"Failed to load {embedding_file}: {e}")
    
    logger.info(f"Loaded {len(database)} embeddings into database")
    
    # Log statistics
    if database:
        embedding_dims = [emb.shape[0] for emb in database.values()]
        logger.info(f"Embedding dimensions: {embedding_dims[0]}")
        logger.info(f"Total database size: {sum(emb.nbytes for emb in database.values()) / 1024 / 1024:.2f} MB")
    
    return database


def search_similar_functions(
    query_binary: str,
    database: Dict[str, np.ndarray],
    query_embedding: np.ndarray,
    k: int = 10
) -> list:
    """
    Search for similar functions and format results with metadata.
    
    Args:
        query_binary: Name of the query binary
        database: Embedding database (dict: binary_name → embedding)
        query_embedding: Query embedding vector
        k: Number of top results to return
    
    Returns:
        list: Results with rank, binary name, similarity, and metadata
    """
    # Get top-K similar functions
    top_results = top_k_similar(query_embedding, database, k=k)
    
    # Format results with metadata
    formatted_results = []
    for rank, (binary_name, similarity) in enumerate(top_results, start=1):
        result = {
            "rank": rank,
            "binary": binary_name,
            "similarity": similarity,
            "is_query": binary_name == query_binary
        }
        formatted_results.append(result)
    
    return formatted_results


def search_similar_functions(
    query_binary: str,
    database: Dict[str, np.ndarray],
    query_embedding: np.ndarray,
    k: int = 10
) -> List[Dict[str, any]]:
    """
    Search for similar functions with detailed results.
    
    Args:
        query_binary: Query binary name (for display)
        database: Embedding database
        query_embedding: Query embedding vector
        k: Number of results to return
    
    Returns:
        List of result dictionaries with binary name and similarity score
    """
    start_time = time.time()
    
    # Find top-k similar
    results = top_k_similar(query_embedding, database, k=k)
    
    elapsed = time.time() - start_time
    
    # Format results
    formatted_results = []
    for i, (func_id, similarity) in enumerate(results):
        formatted_results.append({
            "rank": i + 1,
            "binary": func_id,
            "similarity": similarity,
            "is_query": func_id == query_binary
        })
    
    logger.info(f"Search completed in {elapsed*1000:.2f}ms")
    logger.info(f"Searched {len(database)} embeddings")
    
    return formatted_results


def test_similarity_functions():
    """Test similarity functions with synthetic data (T099-T100)."""
    print("Testing Similarity Functions...")
    
    # Test cosine_similarity (T099)
    print("\n1. Testing cosine_similarity...")
    emb_a = np.array([1.0, 2.0, 3.0])
    emb_b = np.array([2.0, 4.0, 6.0])  # Same direction
    emb_c = np.array([-1.0, -2.0, -3.0])  # Opposite direction
    
    sim_ab = cosine_similarity(emb_a, emb_b)
    sim_ac = cosine_similarity(emb_a, emb_c)
    
    print(f"  Similarity(A, B): {sim_ab:.4f} (expected ~1.0)")
    print(f"  Similarity(A, C): {sim_ac:.4f} (expected ~-1.0)")
    
    assert abs(sim_ab - 1.0) < 0.01, "Same direction should have similarity ~1.0"
    assert abs(sim_ac - (-1.0)) < 0.01, "Opposite direction should have similarity ~-1.0"
    print("  ✓ Cosine similarity test passed")
    
    # Test top_k_similar
    print("\n2. Testing top_k_similar...")
    database = {
        f"func_{i}": np.random.randn(768) for i in range(100)
    }
    query = np.random.randn(768)
    
    results = top_k_similar(query, database, k=10)
    
    print(f"  Found {len(results)} results")
    print(f"  Top result: {results[0][0]}, similarity: {results[0][1]:.4f}")
    
    # Verify results are sorted
    similarities = [score for _, score in results]
    assert similarities == sorted(similarities, reverse=True), "Results should be sorted by similarity"
    print("  ✓ Top-K retrieval test passed")
    
    # Test performance (T100)
    print("\n3. Testing performance with 10,000 embeddings...")
    large_database = {
        f"func_{i}": np.random.randn(768) for i in range(10000)
    }
    
    start_time = time.time()
    results = top_k_similar(query, large_database, k=10)
    elapsed = time.time() - start_time
    
    print(f"  Search time: {elapsed*1000:.2f}ms")
    print(f"  Throughput: {len(large_database)/elapsed:.0f} embeddings/second")
    
    assert elapsed < 1.0, f"Search should complete in <1 second, took {elapsed:.2f}s"
    print("  ✓ Performance test passed (<1 second for 10K vectors)")
    
    # Test build_embedding_database
    print("\n4. Testing build_embedding_database...")
    print("  (Skipped - requires actual .npy files)")
    
    print("\n✓ All similarity function tests passed!")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_similarity_functions()
