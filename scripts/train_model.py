"""
Training Entry Point for BCSD Model

Command-line script to train the BCSD model with configuration.

Usage:
    python scripts/train_model.py --config configs/train_config.yaml

Module: scripts.train_model
Owner: User Story 7 (US7) - Training Infrastructure
Tasks: T086
"""

import argparse
import logging
import sys
import yaml
from pathlib import Path

import torch

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from models.bcsd_model import BCSModel
from dataset.code_dataset import BinaryCodeDataset
from training.trainer import Trainer
from utils.reproducibility import set_seed

logger = logging.getLogger("bcsd.scripts")


def load_config(config_path: str) -> dict:
    """
    Load training configuration from YAML file.
    
    Args:
        config_path: Path to YAML config file
    
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    logger.info(f"Loaded config from {config_path}")
    return config


def create_model(config: dict, device: str) -> BCSModel:
    """
    Create BCSD model from configuration.
    
    Args:
        config: Configuration dictionary
        device: Device to place model on
    
    Returns:
        Initialized BCSD model
    """
    model_config = config.get("model", {})
    
    model = BCSModel(
        node_feature_dim=model_config.get("node_feature_dim", 128),
        gnn_hidden_dim=model_config.get("gnn_hidden_dim", 256),
        gnn_output_dim=model_config.get("gnn_output_dim", 256),
        gnn_num_layers=model_config.get("gnn_num_layers", 3),
        gnn_num_heads=model_config.get("gnn_num_heads", 4),
        bert_model_name=model_config.get("bert_model_name", "bert-base-uncased"),
        dropout=model_config.get("dropout", 0.1)
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logger.info(f"Model created:")
    logger.info(f"  Total parameters: {total_params:,}")
    logger.info(f"  Trainable parameters: {trainable_params:,}")
    
    return model


def create_datasets(config: dict):
    """
    Create training and validation datasets.
    
    Args:
        config: Configuration dictionary
    
    Returns:
        Tuple of (train_dataset, val_dataset)
    """
    data_config = config.get("data", {})
    
    metadata_path = data_config.get("metadata_path", "data/metadata.csv")
    vocab_path = data_config.get("vocab_path", "data/vocab.json")
    
    # Create training dataset
    train_dataset = BinaryCodeDataset(
        metadata_path=metadata_path,
        vocab_path=vocab_path,
        split_set="train",
        max_seq_length=data_config.get("max_seq_length", 512)
    )
    
    # Create validation dataset
    val_dataset = BinaryCodeDataset(
        metadata_path=metadata_path,
        vocab_path=vocab_path,
        split_set="val",
        max_seq_length=data_config.get("max_seq_length", 512)
    )
    
    logger.info(f"Datasets created:")
    logger.info(f"  Training samples: {len(train_dataset)}")
    logger.info(f"  Validation samples: {len(val_dataset)}")
    
    return train_dataset, val_dataset


def main():
    """Main training function."""
    # Parse arguments
    parser = argparse.ArgumentParser(description="Train BCSD model")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/train_config.yaml",
        help="Path to training config YAML file"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint to resume from"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cuda or cpu). Auto-detect if not specified."
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger.info("="*60)
    logger.info("BCSD Model Training")
    logger.info("="*60)
    
    # Load configuration
    config = load_config(args.config)
    
    # Set device
    if args.device:
        device = args.device
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    logger.info(f"Using device: {device}")
    
    # Set random seed for reproducibility
    seed = config.get("seed", 42)
    set_seed(seed)
    logger.info(f"Random seed set to {seed}")
    
    # Create model
    model = create_model(config, device)
    
    # Create datasets
    train_dataset, val_dataset = create_datasets(config)
    
    # Create trainer
    training_config = config.get("training", {})
    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        config=training_config,
        device=device
    )
    
    # Load checkpoint if provided
    if args.checkpoint:
        trainer.load_checkpoint(args.checkpoint)
    
    # Train model
    num_epochs = training_config.get("epochs", 10)
    logger.info(f"\nStarting training for {num_epochs} epochs...")
    
    try:
        trainer.train(num_epochs=num_epochs)
        logger.info("\n✓ Training completed successfully!")
    except KeyboardInterrupt:
        logger.info("\nTraining interrupted by user")
        logger.info("Saving checkpoint...")
        trainer.save_checkpoint(
            epoch=trainer.current_epoch,
            train_metrics={},
            val_metrics={}
        )
    except Exception as e:
        logger.error(f"\nTraining failed with error: {e}")
        raise
    
    logger.info(f"\nMetrics log: {trainer.log_file}")
    logger.info(f"Checkpoints saved to: {trainer.checkpoint_dir}")


if __name__ == "__main__":
    main()
