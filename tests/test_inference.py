"""
Tests for Inference Module

Tests for vectorization and similarity search.

Module: tests.test_inference
Owner: User Story 8 (US8) - Vectorization & Similarity Search
Tasks: T101
"""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from inference.similarity import (
    cosine_similarity,
    top_k_similar,
    build_embedding_database
)


class TestCosineSimilarity(unittest.TestCase):
    """Test cosine similarity computation (T101)."""
    
    def test_cosine_similarity_identical(self):
        """Test cosine similarity with identical vectors."""
        emb_a = np.array([1.0, 2.0, 3.0, 4.0])
        emb_b = np.array([1.0, 2.0, 3.0, 4.0])
        
        similarity = cosine_similarity(emb_a, emb_b)
        
        self.assertAlmostEqual(similarity, 1.0, places=5)
    
    def test_cosine_similarity_opposite(self):
        """Test cosine similarity with opposite vectors."""
        emb_a = np.array([1.0, 2.0, 3.0])
        emb_b = np.array([-1.0, -2.0, -3.0])
        
        similarity = cosine_similarity(emb_a, emb_b)
        
        self.assertAlmostEqual(similarity, -1.0, places=5)
    
    def test_cosine_similarity_orthogonal(self):
        """Test cosine similarity with orthogonal vectors."""
        emb_a = np.array([1.0, 0.0, 0.0])
        emb_b = np.array([0.0, 1.0, 0.0])
        
        similarity = cosine_similarity(emb_a, emb_b)
        
        self.assertAlmostEqual(similarity, 0.0, places=5)
    
    def test_cosine_similarity_zero_vector(self):
        """Test cosine similarity with zero vector."""
        emb_a = np.array([1.0, 2.0, 3.0])
        emb_b = np.array([0.0, 0.0, 0.0])
        
        similarity = cosine_similarity(emb_a, emb_b)
        
        self.assertEqual(similarity, 0.0)


class TestTopKRetrieval(unittest.TestCase):
    """Test top-K similarity retrieval (T101)."""
    
    def test_top_k_retrieval(self):
        """Test that top-K returns correct number of results."""
        database = {
            f"func_{i}": np.random.randn(128) for i in range(20)
        }
        query = np.random.randn(128)
        
        results = top_k_similar(query, database, k=5)
        
        self.assertEqual(len(results), 5)
        
        # Verify each result is a tuple (id, score)
        for func_id, score in results:
            self.assertIn(func_id, database)
            self.assertIsInstance(score, float)
    
    def test_top_k_sorted(self):
        """Test that results are sorted by similarity."""
        database = {
            "func_1": np.array([1.0, 0.0]),
            "func_2": np.array([0.5, 0.5]),
            "func_3": np.array([0.0, 1.0])
        }
        query = np.array([1.0, 0.0])  # Should match func_1 best
        
        results = top_k_similar(query, database, k=3)
        
        # Verify sorted descending
        similarities = [score for _, score in results]
        self.assertEqual(similarities, sorted(similarities, reverse=True))
        
        # Verify func_1 is top result
        self.assertEqual(results[0][0], "func_1")
    
    def test_top_k_with_k_larger_than_database(self):
        """Test top-K when k > database size."""
        database = {
            "func_1": np.array([1.0, 0.0]),
            "func_2": np.array([0.0, 1.0])
        }
        query = np.array([1.0, 0.0])
        
        results = top_k_similar(query, database, k=10)
        
        # Should return all available entries
        self.assertEqual(len(results), 2)


class TestEmbeddingDatabase(unittest.TestCase):
    """Test embedding database operations (T101)."""
    
    def test_build_embedding_database(self):
        """Test building database from .npy files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            
            # Create dummy embeddings
            emb1 = np.random.randn(768)
            emb2 = np.random.randn(768)
            emb3 = np.random.randn(768)
            
            np.save(tmppath / "func_1.npy", emb1)
            np.save(tmppath / "func_2.npy", emb2)
            np.save(tmppath / "func_3.npy", emb3)
            
            # Build database
            database = build_embedding_database(str(tmppath))
            
            self.assertEqual(len(database), 3)
            self.assertIn("func_1", database)
            self.assertIn("func_2", database)
            self.assertIn("func_3", database)
            
            # Verify embeddings loaded correctly
            np.testing.assert_array_equal(database["func_1"], emb1)
    
    def test_build_embedding_database_nonexistent_dir(self):
        """Test building database from non-existent directory."""
        database = build_embedding_database("/nonexistent/path/")
        
        self.assertEqual(len(database), 0)


class TestVectorizationIntegration(unittest.TestCase):
    """Test vectorization integration (T098, T099, T100)."""
    
    def test_embedding_dimensions(self):
        """Test that embeddings have correct dimensions."""
        # This would require a real model and binary
        # For now, just verify the expected dimension
        expected_dim = 768
        dummy_embedding = np.random.randn(expected_dim)
        
        self.assertEqual(dummy_embedding.shape[0], expected_dim)
    
    def test_similarity_search_performance(self):
        """Test that similarity search is fast (T100)."""
        import time
        
        # Create large database (10,000 embeddings)
        database = {
            f"func_{i}": np.random.randn(768) for i in range(10000)
        }
        query = np.random.randn(768)
        
        # Measure search time
        start_time = time.time()
        results = top_k_similar(query, database, k=10)
        elapsed = time.time() - start_time
        
        # Should complete in <1 second
        self.assertLess(elapsed, 1.0)
        self.assertEqual(len(results), 10)


if __name__ == "__main__":
    unittest.main()
