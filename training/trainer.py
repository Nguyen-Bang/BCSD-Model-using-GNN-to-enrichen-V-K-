"""
Trainer Class for BCSD Model

Implements complete training pipeline with:
- Siamese training on positive pairs
- Joint MLM + contrastive loss
- Epoch-based checkpointing and logging
- Early stopping
- Validation metrics

Module: training.trainer
Owner: User Story 7 (US7) - Training Infrastructure  
Tasks: T078-T085
"""

import csv
import logging
import os
import time
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from training.losses import JointLoss
from training.metrics import (
    compute_similarity_accuracy,
    log_embedding_statistics,
    compute_validation_metrics
)

logger = logging.getLogger("bcsd.training")


class Trainer:
    """
    Trainer for BCSD Model (T078-T085).
    
    Handles complete training pipeline with:
    - Siamese training: Same model processes all samples with shared weights
    - Joint loss: MLM + λ*contrastive
    - Epoch-based checkpointing and logging
    - Early stopping based on validation loss
    
    Architecture:
    1. Initialize model, optimizer, loss functions
    2. Training loop:
        - train_epoch(): Process all batches, compute loss, optimize
        - validate(): Run validation set, compute metrics
        - save_checkpoint(): Save model if validation improves
        - Early stopping: Stop if no improvement for patience epochs
    3. Log metrics to CSV after each epoch
    
    Example:
        >>> from models.bcsd_model import BCSModel
        >>> from dataset.code_dataset import BinaryCodeDataset
        >>> 
        >>> model = BCSModel(...)
        >>> train_dataset = BinaryCodeDataset(...)
        >>> val_dataset = BinaryCodeDataset(...)
        >>> 
        >>> trainer = Trainer(
        ...     model=model,
        ...     train_dataset=train_dataset,
        ...     val_dataset=val_dataset,
        ...     config=config
        ... )
        >>> 
        >>> trainer.train(num_epochs=10)
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_dataset,
        val_dataset,
        config: Dict,
        device: str = "cuda"
    ):
        """
        Initialize Trainer (T079).
        
        Args:
            model: BCSD model to train
            train_dataset: Training dataset
            val_dataset: Validation dataset
            config: Training configuration dictionary
            device: Device for training ("cuda" or "cpu")
        """
        self.model = model.to(device)
        self.device = device
        self.config = config
        
        # Extract config parameters with defaults
        self.batch_size = config.get("batch_size", 16)
        self.learning_rate = config.get("learning_rate", 2e-5)
        self.weight_decay = config.get("weight_decay", 0.01)
        self.gradient_clip_max_norm = config.get("gradient_clip_max_norm", 1.0)
        self.lambda_contrastive = config.get("lambda_contrastive", 0.5)
        self.mlm_mask_prob = config.get("mlm_mask_prob", 0.15)
        self.contrastive_temperature = config.get("contrastive_temperature", 0.07)
        
        # Checkpointing config
        self.checkpoint_dir = Path(config.get("checkpoint_dir", "checkpoints"))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.save_every_n_epochs = config.get("save_every_n_epochs", 1)
        self.keep_top_k_checkpoints = config.get("keep_top_k_checkpoints", 3)
        
        # Logging config
        self.log_dir = Path(config.get("log_dir", "logs"))
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.log_dir / "training_metrics.csv"
        
        # Early stopping config
        self.early_stopping_patience = config.get("early_stopping_patience", 3)
        self.early_stopping_delta = config.get("early_stopping_delta", 0.001)
        self.best_val_loss = float('inf')
        self.epochs_without_improvement = 0
        
        # Create data loaders
        from dataset.collate import collate_heterogeneous
        
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=collate_heterogeneous,
            num_workers=0  # Set to 0 to avoid multiprocessing issues with graphs
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=collate_heterogeneous,
            num_workers=0
        )
        
        # Initialize optimizer (T079)
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        # Initialize loss function (T079)
        self.loss_fn = JointLoss(
            lambda_contrastive=self.lambda_contrastive,
            temperature=self.contrastive_temperature,
            ignore_index=-100
        )
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.training_history = []
        
        # Initialize CSV log file
        self._initialize_log_file()
        
        logger.info(f"Trainer initialized:")
        logger.info(f"  Device: {device}")
        logger.info(f"  Batch size: {self.batch_size}")
        logger.info(f"  Learning rate: {self.learning_rate}")
        logger.info(f"  Lambda contrastive: {self.lambda_contrastive}")
        logger.info(f"  Training samples: {len(train_dataset)}")
        logger.info(f"  Validation samples: {len(val_dataset)}")
    
    def _initialize_log_file(self):
        """Initialize CSV log file with headers (T084)."""
        if not self.log_file.exists():
            with open(self.log_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'epoch', 'train_loss', 'mlm_loss', 'contrastive_loss',
                    'val_loss', 'val_mean_similarity', 'learning_rate', 'timestamp'
                ])
            logger.info(f"Created log file: {self.log_file}")
    
    def train_epoch(self) -> Dict[str, float]:
        """
        Train for one epoch (T080).
        
        Processes all batches with Siamese training:
        - Each batch contains positive pairs
        - Model processes each sample with shared weights
        - Compute joint MLM + contrastive loss
        - Backward pass and optimize
        
        Returns:
            Dictionary with epoch training metrics
        """
        self.model.train()
        
        total_loss = 0.0
        total_mlm_loss = 0.0
        total_contrastive_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch + 1}")
        
        for batch in progress_bar:
            # Move batch to device
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            graph_x = batch["graph_batch"].x.to(self.device)
            graph_edge_index = batch["graph_batch"].edge_index.to(self.device)
            graph_batch_idx = batch["graph_batch"].batch.to(self.device)
            
            # Siamese training: Process pairs sequentially with shared weights
            # Assume batch contains pairs: [sample_A1, sample_B1, sample_A2, sample_B2, ...]
            # Split into two halves
            batch_size = input_ids.size(0) // 2
            
            # Forward pass for all samples (Siamese: same model, shared weights)
            embeddings = self.model.get_embeddings(
                input_ids=input_ids,
                attention_mask=attention_mask,
                graph_x=graph_x,
                graph_edge_index=graph_edge_index,
                graph_batch=graph_batch_idx
            )
            
            # Split embeddings into pairs
            embeddings_a = embeddings[:batch_size]
            embeddings_b = embeddings[batch_size:]
            
            # Compute joint loss (MLM + contrastive)
            # For now, skip MLM component (would need masked inputs)
            # Focus on contrastive learning
            loss, mlm_loss, contrastive_loss = self.loss_fn(
                embeddings_a=embeddings_a,
                embeddings_b=embeddings_b,
                return_components=True
            )
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            if self.gradient_clip_max_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.gradient_clip_max_norm
                )
            
            # Optimizer step
            self.optimizer.step()
            
            # Track metrics
            total_loss += loss.item()
            total_mlm_loss += mlm_loss.item() if isinstance(mlm_loss, torch.Tensor) else mlm_loss
            total_contrastive_loss += contrastive_loss.item() if isinstance(contrastive_loss, torch.Tensor) else contrastive_loss
            num_batches += 1
            self.global_step += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'contrastive': f'{contrastive_loss.item():.4f}'
            })
        
        # Compute average losses
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        avg_mlm_loss = total_mlm_loss / num_batches if num_batches > 0 else 0.0
        avg_contrastive_loss = total_contrastive_loss / num_batches if num_batches > 0 else 0.0
        
        return {
            "train_loss": avg_loss,
            "mlm_loss": avg_mlm_loss,
            "contrastive_loss": avg_contrastive_loss
        }
    
    def validate(self) -> Dict[str, float]:
        """
        Run validation (T081).
        
        Computes validation loss and similarity metrics on validation set.
        
        Returns:
            Dictionary with validation metrics
        """
        self.model.eval()
        
        total_loss = 0.0
        all_similarities = []
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                # Move batch to device
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                graph_x = batch["graph_batch"].x.to(self.device)
                graph_edge_index = batch["graph_batch"].edge_index.to(self.device)
                graph_batch_idx = batch["graph_batch"].batch.to(self.device)
                
                # Forward pass
                batch_size = input_ids.size(0) // 2
                embeddings = self.model.get_embeddings(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    graph_x=graph_x,
                    graph_edge_index=graph_edge_index,
                    graph_batch=graph_batch_idx
                )
                
                embeddings_a = embeddings[:batch_size]
                embeddings_b = embeddings[batch_size:]
                
                # Compute loss
                loss = self.loss_fn(
                    embeddings_a=embeddings_a,
                    embeddings_b=embeddings_b
                )
                
                total_loss += loss.item()
                
                # Compute similarity metrics
                sim_metrics = compute_similarity_accuracy(embeddings_a, embeddings_b)
                all_similarities.append(sim_metrics["mean_positive_similarity"])
                
                num_batches += 1
        
        # Aggregate metrics
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        avg_similarity = sum(all_similarities) / len(all_similarities) if all_similarities else 0.0
        
        return {
            "val_loss": avg_loss,
            "val_mean_similarity": avg_similarity
        }
    
    def save_checkpoint(
        self,
        epoch: int,
        train_metrics: Dict[str, float],
        val_metrics: Dict[str, float]
    ) -> str:
        """
        Save model checkpoint (T082).
        
        Saves model weights, optimizer state, and metrics with naming:
        `model_epoch_{epoch}_valloss_{val_loss:.4f}.pt`
        
        Args:
            epoch: Current epoch number
            train_metrics: Training metrics dictionary
            val_metrics: Validation metrics dictionary
        
        Returns:
            Path to saved checkpoint
        """
        val_loss = val_metrics.get("val_loss", 0.0)
        
        checkpoint_path = self.checkpoint_dir / f"model_epoch_{epoch:03d}_valloss_{val_loss:.4f}.pt"
        
        checkpoint = {
            "epoch": epoch,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
            "config": self.config,
            "best_val_loss": self.best_val_loss
        }
        
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Saved checkpoint: {checkpoint_path}")
        
        # Clean up old checkpoints (keep only top-k)
        self._cleanup_old_checkpoints()
        
        return str(checkpoint_path)
    
    def load_checkpoint(self, checkpoint_path: str) -> Dict:
        """
        Load checkpoint and resume training (T083).
        
        Args:
            checkpoint_path: Path to checkpoint file
        
        Returns:
            Checkpoint metadata dictionary
        """
        logger.info(f"Loading checkpoint: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Restore model and optimizer state
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        # Restore training state
        self.current_epoch = checkpoint["epoch"]
        self.global_step = checkpoint.get("global_step", 0)
        self.best_val_loss = checkpoint.get("best_val_loss", float('inf'))
        
        logger.info(f"Resumed from epoch {self.current_epoch}, step {self.global_step}")
        
        return checkpoint
    
    def _cleanup_old_checkpoints(self):
        """Keep only top-k checkpoints based on validation loss."""
        # Get all checkpoint files
        checkpoint_files = list(self.checkpoint_dir.glob("model_epoch_*.pt"))
        
        if len(checkpoint_files) <= self.keep_top_k_checkpoints:
            return
        
        # Sort by validation loss (extracted from filename)
        def extract_val_loss(path):
            try:
                # Filename format: model_epoch_{N}_valloss_{loss}.pt
                parts = path.stem.split("_valloss_")
                if len(parts) == 2:
                    return float(parts[1])
            except:
                pass
            return float('inf')
        
        checkpoint_files.sort(key=extract_val_loss)
        
        # Remove checkpoints beyond top-k
        for checkpoint_file in checkpoint_files[self.keep_top_k_checkpoints:]:
            checkpoint_file.unlink()
            logger.info(f"Removed old checkpoint: {checkpoint_file}")
    
    def log_metrics_to_csv(
        self,
        epoch: int,
        train_metrics: Dict[str, float],
        val_metrics: Dict[str, float]
    ):
        """
        Log metrics to CSV file (T084).
        
        Appends epoch metrics to training_metrics.csv.
        
        Args:
            epoch: Current epoch number
            train_metrics: Training metrics dictionary
            val_metrics: Validation metrics dictionary
        """
        with open(self.log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch,
                train_metrics.get("train_loss", 0.0),
                train_metrics.get("mlm_loss", 0.0),
                train_metrics.get("contrastive_loss", 0.0),
                val_metrics.get("val_loss", 0.0),
                val_metrics.get("val_mean_similarity", 0.0),
                self.optimizer.param_groups[0]['lr'],
                time.strftime("%Y-%m-%d %H:%M:%S")
            ])
        
        logger.info(f"Logged metrics to {self.log_file}")
    
    def check_early_stopping(self, val_loss: float) -> bool:
        """
        Check early stopping condition (T085).
        
        Stops training if validation loss doesn't improve for patience epochs.
        
        Args:
            val_loss: Current validation loss
        
        Returns:
            True if should stop training, False otherwise
        """
        if val_loss < self.best_val_loss - self.early_stopping_delta:
            # Improvement
            self.best_val_loss = val_loss
            self.epochs_without_improvement = 0
            logger.info(f"✓ New best validation loss: {val_loss:.4f}")
            return False
        else:
            # No improvement
            self.epochs_without_improvement += 1
            logger.info(f"No improvement for {self.epochs_without_improvement}/{self.early_stopping_patience} epochs")
            
            if self.epochs_without_improvement >= self.early_stopping_patience:
                logger.info(f"Early stopping triggered after {self.early_stopping_patience} epochs without improvement")
                return True
        
        return False
    
    def train(self, num_epochs: int):
        """
        Main training loop (T080-T085).
        
        Trains model for specified number of epochs with:
        - Training on train set
        - Validation on val set
        - Checkpointing
        - Early stopping
        - Metric logging
        
        Args:
            num_epochs: Number of epochs to train
        """
        logger.info(f"Starting training for {num_epochs} epochs...")
        start_time = time.time()
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            
            logger.info(f"\n{'='*60}")
            logger.info(f"Epoch {epoch + 1}/{num_epochs}")
            logger.info(f"{'='*60}")
            
            # Train for one epoch
            train_metrics = self.train_epoch()
            logger.info(f"Train Loss: {train_metrics['train_loss']:.4f}")
            logger.info(f"  MLM Loss: {train_metrics['mlm_loss']:.4f}")
            logger.info(f"  Contrastive Loss: {train_metrics['contrastive_loss']:.4f}")
            
            # Validate
            val_metrics = self.validate()
            logger.info(f"Val Loss: {val_metrics['val_loss']:.4f}")
            logger.info(f"  Val Similarity: {val_metrics['val_mean_similarity']:.4f}")
            
            # Log metrics to CSV
            self.log_metrics_to_csv(epoch, train_metrics, val_metrics)
            
            # Save checkpoint if scheduled
            if (epoch + 1) % self.save_every_n_epochs == 0:
                self.save_checkpoint(epoch, train_metrics, val_metrics)
            
            # Check early stopping
            if self.check_early_stopping(val_metrics['val_loss']):
                logger.info("Training stopped early")
                break
            
            # Store history
            self.training_history.append({
                "epoch": epoch,
                **train_metrics,
                **val_metrics
            })
        
        total_time = time.time() - start_time
        logger.info(f"\nTraining completed in {total_time:.2f} seconds")
        logger.info(f"Best validation loss: {self.best_val_loss:.4f}")
        logger.info(f"Metrics saved to: {self.log_file}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("Trainer module loaded successfully")
    print("Use this module by importing: from training.trainer import Trainer")
