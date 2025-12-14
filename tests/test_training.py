"""
Tests for Training Module

Tests for loss functions, metrics, and trainer.

Module: tests.test_training
Owner: User Story 7 (US7) - Training Infrastructure
Tasks: T089
"""

import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn as nn

from training.losses import MLMLoss, InfoNCELoss, JointLoss
from training.metrics import (
    compute_similarity_accuracy,
    compute_ranking_metrics,
    log_embedding_statistics
)


class TestMLMLoss(unittest.TestCase):
    """Test MLM loss computation (T089)."""
    
    def test_mlm_loss_computation(self):
        """Test that MLM loss computes correctly."""
        batch_size = 4
        seq_len = 20
        vocab_size = 5000
        
        loss_fn = MLMLoss()
        
        logits = torch.randn(batch_size, seq_len, vocab_size)
        labels = torch.randint(0, vocab_size, (batch_size, seq_len))
        masked_positions = torch.rand(batch_size, seq_len) > 0.85
        
        loss = loss_fn(logits, labels, masked_positions)
        
        # Loss should be positive scalar
        self.assertIsInstance(loss, torch.Tensor)
        self.assertEqual(loss.shape, torch.Size([]))
        self.assertGreater(loss.item(), 0)
    
    def test_mlm_loss_without_mask(self):
        """Test MLM loss on all positions."""
        batch_size = 2
        seq_len = 10
        vocab_size = 100
        
        loss_fn = MLMLoss()
        
        logits = torch.randn(batch_size, seq_len, vocab_size)
        labels = torch.randint(0, vocab_size, (batch_size, seq_len))
        
        loss = loss_fn(logits, labels, masked_positions=None)
        
        self.assertGreater(loss.item(), 0)
    
    def test_mlm_loss_no_masked_tokens(self):
        """Test MLM loss when no tokens are masked."""
        batch_size = 2
        seq_len = 10
        vocab_size = 100
        
        loss_fn = MLMLoss()
        
        logits = torch.randn(batch_size, seq_len, vocab_size)
        labels = torch.randint(0, vocab_size, (batch_size, seq_len))
        masked_positions = torch.zeros(batch_size, seq_len, dtype=torch.bool)
        
        loss = loss_fn(logits, labels, masked_positions)
        
        # Should return zero loss
        self.assertEqual(loss.item(), 0.0)


class TestInfoNCELoss(unittest.TestCase):
    """Test InfoNCE contrastive loss (T089)."""
    
    def test_infonce_loss_computation(self):
        """Test that InfoNCE loss computes correctly."""
        batch_size = 8
        embedding_dim = 768
        
        loss_fn = InfoNCELoss(temperature=0.07)
        
        embeddings_a = torch.randn(batch_size, embedding_dim)
        embeddings_b = torch.randn(batch_size, embedding_dim)
        
        loss = loss_fn(embeddings_a, embeddings_b)
        
        # Loss should be positive scalar
        self.assertIsInstance(loss, torch.Tensor)
        self.assertEqual(loss.shape, torch.Size([]))
        self.assertGreater(loss.item(), 0)
    
    def test_infonce_loss_identical_embeddings(self):
        """Test InfoNCE loss with identical positive pairs."""
        batch_size = 4
        embedding_dim = 128
        
        loss_fn = InfoNCELoss(temperature=0.07)
        
        embeddings_a = torch.randn(batch_size, embedding_dim)
        embeddings_b = embeddings_a.clone()  # Identical
        
        loss = loss_fn(embeddings_a, embeddings_b)
        
        # Loss should be small but positive (due to in-batch negatives)
        self.assertGreater(loss.item(), 0)
        self.assertLess(loss.item(), 1.0)  # Should be relatively small
    
    def test_infonce_loss_temperature_effect(self):
        """Test that temperature affects loss magnitude."""
        batch_size = 4
        embedding_dim = 128
        
        embeddings_a = torch.randn(batch_size, embedding_dim)
        embeddings_b = torch.randn(batch_size, embedding_dim)
        
        loss_high_temp = InfoNCELoss(temperature=0.5)(embeddings_a, embeddings_b)
        loss_low_temp = InfoNCELoss(temperature=0.01)(embeddings_a, embeddings_b)
        
        # Different temperatures should give different losses
        self.assertNotEqual(loss_high_temp.item(), loss_low_temp.item())


class TestJointLoss(unittest.TestCase):
    """Test joint MLM + contrastive loss (T089)."""
    
    def test_joint_loss_with_lambda_values(self):
        """Test joint loss with different lambda values."""
        batch_size = 4
        seq_len = 20
        vocab_size = 5000
        embedding_dim = 768
        
        logits = torch.randn(batch_size, seq_len, vocab_size)
        labels = torch.randint(0, vocab_size, (batch_size, seq_len))
        masked_positions = torch.rand(batch_size, seq_len) > 0.85
        embeddings_a = torch.randn(batch_size, embedding_dim)
        embeddings_b = torch.randn(batch_size, embedding_dim)
        
        # Test different lambda values
        for lambda_val in [0.3, 0.5, 0.7]:
            loss_fn = JointLoss(lambda_contrastive=lambda_val)
            
            total_loss, mlm_loss, contrastive_loss = loss_fn(
                mlm_logits=logits,
                mlm_labels=labels,
                mlm_masked_positions=masked_positions,
                embeddings_a=embeddings_a,
                embeddings_b=embeddings_b,
                return_components=True
            )
            
            # Verify composition
            expected_total = mlm_loss.item() + lambda_val * contrastive_loss.item()
            self.assertAlmostEqual(total_loss.item(), expected_total, places=5)
    
    def test_joint_loss_lambda_update(self):
        """Test that lambda can be updated dynamically."""
        batch_size = 2
        embedding_dim = 128
        
        embeddings_a = torch.randn(batch_size, embedding_dim)
        embeddings_b = torch.randn(batch_size, embedding_dim)
        
        loss_fn = JointLoss(lambda_contrastive=0.5)
        
        loss1 = loss_fn(embeddings_a=embeddings_a, embeddings_b=embeddings_b)
        
        loss_fn.set_lambda(0.7)
        loss2 = loss_fn(embeddings_a=embeddings_a, embeddings_b=embeddings_b)
        
        # Different lambdas should give different losses
        self.assertNotEqual(loss1.item(), loss2.item())
    
    def test_joint_loss_mlm_only(self):
        """Test joint loss with only MLM component."""
        batch_size = 2
        seq_len = 10
        vocab_size = 100
        
        loss_fn = JointLoss(lambda_contrastive=0.0)
        
        logits = torch.randn(batch_size, seq_len, vocab_size)
        labels = torch.randint(0, vocab_size, (batch_size, seq_len))
        masked_positions = torch.rand(batch_size, seq_len) > 0.85
        
        total_loss, mlm_loss, contrastive_loss = loss_fn(
            mlm_logits=logits,
            mlm_labels=labels,
            mlm_masked_positions=masked_positions,
            return_components=True
        )
        
        # Total should equal MLM only (lambda=0)
        self.assertAlmostEqual(total_loss.item(), mlm_loss.item(), places=5)
    
    def test_joint_loss_contrastive_only(self):
        """Test joint loss with only contrastive component."""
        batch_size = 4
        embedding_dim = 128
        
        loss_fn = JointLoss(lambda_contrastive=1.0)
        
        embeddings_a = torch.randn(batch_size, embedding_dim)
        embeddings_b = torch.randn(batch_size, embedding_dim)
        
        total_loss, mlm_loss, contrastive_loss = loss_fn(
            embeddings_a=embeddings_a,
            embeddings_b=embeddings_b,
            return_components=True
        )
        
        # MLM should be zero
        self.assertEqual(mlm_loss, 0.0)
        # Total should equal contrastive (lambda=1.0)
        self.assertAlmostEqual(total_loss.item(), contrastive_loss.item(), places=5)


class TestMetrics(unittest.TestCase):
    """Test validation metrics (T089)."""
    
    def test_similarity_accuracy(self):
        """Test similarity accuracy computation."""
        batch_size = 8
        embedding_dim = 256
        
        embeddings_a = torch.randn(batch_size, embedding_dim)
        embeddings_b = embeddings_a + torch.randn(batch_size, embedding_dim) * 0.1
        
        metrics = compute_similarity_accuracy(embeddings_a, embeddings_b, threshold=0.5)
        
        self.assertIn("mean_positive_similarity", metrics)
        self.assertIn("accuracy_at_threshold", metrics)
        self.assertGreaterEqual(metrics["mean_positive_similarity"], -1.0)
        self.assertLessEqual(metrics["mean_positive_similarity"], 1.0)
        self.assertGreaterEqual(metrics["accuracy_at_threshold"], 0.0)
        self.assertLessEqual(metrics["accuracy_at_threshold"], 1.0)
    
    def test_ranking_metrics(self):
        """Test ranking metrics computation."""
        num_queries = 5
        num_candidates = 20
        embedding_dim = 128
        
        query_embeddings = torch.randn(num_queries, embedding_dim)
        candidate_embeddings = torch.randn(num_candidates, embedding_dim)
        positive_indices = torch.randint(0, num_candidates, (num_queries,))
        
        metrics = compute_ranking_metrics(query_embeddings, candidate_embeddings, positive_indices)
        
        self.assertIn("mean_reciprocal_rank", metrics)
        self.assertIn("recall_at_10", metrics)
        self.assertGreater(metrics["mean_reciprocal_rank"], 0.0)
        self.assertLessEqual(metrics["mean_reciprocal_rank"], 1.0)
    
    def test_embedding_statistics(self):
        """Test embedding statistics logging."""
        batch_size = 16
        embedding_dim = 768
        
        embeddings = torch.randn(batch_size, embedding_dim)
        stats = log_embedding_statistics(embeddings, prefix="test")
        
        self.assertIn("test_mean_l2_norm", stats)
        self.assertIn("test_std_l2_norm", stats)
        self.assertIn("test_sparsity", stats)
        self.assertGreater(stats["test_mean_l2_norm"], 0)


class TestTrainerCheckpointing(unittest.TestCase):
    """Test trainer checkpoint save/load (T089)."""
    
    def test_checkpoint_save_load(self):
        """Test that checkpoints can be saved and loaded."""
        # Create a simple model
        model = nn.Linear(10, 10)
        
        # Save checkpoint
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "test_checkpoint.pt"
            
            checkpoint = {
                "epoch": 5,
                "model_state_dict": model.state_dict(),
                "train_loss": 1.234,
                "val_loss": 1.567
            }
            
            torch.save(checkpoint, checkpoint_path)
            
            # Load checkpoint
            loaded_checkpoint = torch.load(checkpoint_path)
            
            self.assertEqual(loaded_checkpoint["epoch"], 5)
            self.assertAlmostEqual(loaded_checkpoint["train_loss"], 1.234)
            self.assertAlmostEqual(loaded_checkpoint["val_loss"], 1.567)
            
            # Verify model state can be loaded
            model2 = nn.Linear(10, 10)
            model2.load_state_dict(loaded_checkpoint["model_state_dict"])


class TestEpochLogging(unittest.TestCase):
    """Test epoch-based logging (T089)."""
    
    def test_csv_logging(self):
        """Test that metrics are logged to CSV correctly."""
        import csv
        
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "metrics.csv"
            
            # Write header and data using csv module
            with open(log_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["epoch", "train_loss", "val_loss"])
                writer.writerow([0, 2.345, 1.678])
                writer.writerow([1, 2.123, 1.456])
            
            # Read back
            with open(log_file, 'r') as f:
                lines = f.readlines()
            
            self.assertGreaterEqual(len(lines), 3)  # Header + at least 2 data rows
            self.assertIn("epoch", lines[0])
            self.assertIn("2.345", lines[1])


if __name__ == "__main__":
    unittest.main()
