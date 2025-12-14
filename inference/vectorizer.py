"""
Vectorizer for BCSD Model

Converts binaries into fixed-size embedding vectors using trained model.
This is the inference/feature extraction module.

Module: inference.vectorizer
Owner: User Story 8 (US8) - Vectorization & Similarity Search
Tasks: T090-T093
"""

import logging
import time
from pathlib import Path
from typing import Dict, Optional, Union

import numpy as np
import torch

from preprocessing.extract_features import extract_single_cfg
from preprocessing.tokenizer import AssemblyTokenizer

logger = logging.getLogger("bcsd.inference")


class Vectorizer:
    """
    Vectorizer: Converts binaries to embeddings (T090-T093).
    
    Uses trained BCSD model as feature extractor to generate fixed-size
    embedding vectors (768-dim) from binary executables.
    
    Architecture:
    1. Load trained model checkpoint
    2. Extract CFG and tokenize instructions (preprocessing)
    3. Forward pass through model (GNN + BERT)
    4. Extract [CLS] embedding as function fingerprint
    
    This is INFERENCE ONLY - no gradient computation, no training.
    
    Example:
        >>> vectorizer = Vectorizer(
        ...     checkpoint_path="checkpoints/model_best.pt",
        ...     vocab_path="data/vocab.json"
        ... )
        >>> 
        >>> # Vectorize single binary
        >>> embedding = vectorizer.vectorize_binary("path/to/binary")
        >>> print(embedding.shape)  # (768,)
        >>> 
        >>> # Vectorize entire directory
        >>> embeddings = vectorizer.vectorize_directory("Dataset-1/clamav/")
    """
    
    def __init__(
        self,
        checkpoint_path: str,
        vocab_path: str,
        device: str = "cuda",
        max_seq_length: int = 512
    ):
        """
        Initialize Vectorizer (T091).
        
        Args:
            checkpoint_path: Path to trained model checkpoint
            vocab_path: Path to tokenizer vocabulary
            device: Device for inference ("cuda" or "cpu")
            max_seq_length: Maximum sequence length for tokenization
        """
        self.device = device if torch.cuda.is_available() else "cpu"
        self.max_seq_length = max_seq_length
        
        # Load tokenizer
        self.tokenizer = AssemblyTokenizer(vocab_size=5000)
        self.tokenizer.load_vocab(vocab_path)
        logger.info(f"Loaded tokenizer from {vocab_path}")
        
        # Load model
        self.model = self._load_model(checkpoint_path)
        self.model.eval()  # Set to evaluation mode
        logger.info(f"Loaded model from {checkpoint_path}")
        logger.info(f"Using device: {self.device}")
    
    def _load_model(self, checkpoint_path: str):
        """
        Load trained model from checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
        
        Returns:
            Loaded model in eval mode
        """
        from models.bcsd_model import BCSModel
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Extract model config from checkpoint
        config = checkpoint.get("config", {})
        model_config = config.get("model", {})
        
        # Create model with same architecture
        model = BCSModel(
            node_feature_dim=model_config.get("node_feature_dim", 128),
            gnn_hidden_dim=model_config.get("gnn_hidden_dim", 256),
            gnn_output_dim=model_config.get("gnn_output_dim", 256),
            gnn_num_layers=model_config.get("gnn_num_layers", 3),
            gnn_num_heads=model_config.get("gnn_num_heads", 4),
            bert_model_name=model_config.get("bert_model_name", "bert-base-uncased"),
            dropout=model_config.get("dropout", 0.1)
        )
        
        # Load weights
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(self.device)
        
        logger.info(f"Model loaded from epoch {checkpoint.get('epoch', 'unknown')}")
        logger.info(f"Checkpoint val loss: {checkpoint.get('val_metrics', {}).get('val_loss', 'unknown')}")
        
        return model
    
    def vectorize_binary(
        self,
        binary_path: str,
        timeout: int = 300
    ) -> Optional[np.ndarray]:
        """
        Vectorize single binary to embedding vector (T092).
        
        Pipeline:
        1. Extract CFG with angr
        2. Tokenize assembly instructions
        3. Create graph data
        4. Forward pass through model
        5. Extract [CLS] embedding
        
        Args:
            binary_path: Path to binary executable
            timeout: Timeout for angr CFG extraction (seconds)
        
        Returns:
            Embedding vector (768,) as numpy array, or None if extraction fails
        """
        try:
            logger.info(f"Vectorizing: {binary_path}")
            start_time = time.time()
            
            # Step 1: Extract CFG
            cfg_data = extract_single_cfg(binary_path, timeout=timeout)
            
            if not cfg_data or not cfg_data.get("nodes"):
                logger.warning(f"Failed to extract CFG from {binary_path}")
                return None
            
            # Step 2: Tokenize instructions
            # Collect all instructions from nodes
            all_instructions = []
            for node in cfg_data["nodes"]:
                all_instructions.extend(node["instructions"])
            
            # Tokenize
            tokens, attention_mask = self.tokenizer.tokenize(
                all_instructions,
                max_length=self.max_seq_length
            )
            
            # Convert to tensors
            input_ids = torch.tensor([tokens], dtype=torch.long).to(self.device)
            attention_mask = torch.tensor([attention_mask], dtype=torch.long).to(self.device)
            
            # Step 3: Create graph data
            from torch_geometric.data import Data
            
            num_nodes = len(cfg_data["nodes"])
            
            # Create dummy node features (in real case, would use instruction embeddings)
            node_features = torch.randn(num_nodes, 128).to(self.device)
            
            # Build edge index
            edge_list = []
            node_id_to_idx = {node["id"]: idx for idx, node in enumerate(cfg_data["nodes"])}
            
            for edge in cfg_data["edges"]:
                src_idx = node_id_to_idx.get(edge["source"])
                tgt_idx = node_id_to_idx.get(edge["target"])
                
                if src_idx is not None and tgt_idx is not None:
                    edge_list.append([src_idx, tgt_idx])
            
            if edge_list:
                edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous().to(self.device)
            else:
                edge_index = torch.zeros((2, 0), dtype=torch.long).to(self.device)
            
            graph_batch = torch.zeros(num_nodes, dtype=torch.long).to(self.device)
            
            # Step 4: Forward pass (no gradients)
            with torch.no_grad():
                embedding = self.model.get_embeddings(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    graph_x=node_features,
                    graph_edge_index=edge_index,
                    graph_batch=graph_batch
                )
            
            # Step 5: Extract embedding as numpy array
            embedding_np = embedding.cpu().numpy().squeeze()
            
            elapsed = time.time() - start_time
            logger.info(f"Vectorized {binary_path} in {elapsed:.2f}s")
            logger.info(f"  Embedding shape: {embedding_np.shape}")
            logger.info(f"  L2 norm: {np.linalg.norm(embedding_np):.4f}")
            
            return embedding_np
        
        except Exception as e:
            logger.error(f"Failed to vectorize {binary_path}: {e}")
            return None
    
    def vectorize_directory(
        self,
        directory_path: str,
        output_dir: str,
        pattern: str = "*"
    ) -> Dict[str, str]:
        """
        Vectorize all binaries in directory (T093).
        
        Processes all matching files and saves embeddings to .npy files.
        
        Args:
            directory_path: Directory containing binaries
            output_dir: Directory to save embedding .npy files
            pattern: Glob pattern for binary files (default: "*")
        
        Returns:
            Dictionary mapping binary paths to embedding file paths
        """
        directory = Path(directory_path)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Find all matching files
        binary_files = list(directory.glob(pattern))
        logger.info(f"Found {len(binary_files)} files in {directory_path}")
        
        results = {}
        successful = 0
        failed = 0
        
        for binary_file in binary_files:
            # Skip directories
            if binary_file.is_dir():
                continue
            
            # Vectorize
            embedding = self.vectorize_binary(str(binary_file))
            
            if embedding is not None:
                # Save embedding
                output_file = output_path / f"{binary_file.name}.npy"
                np.save(output_file, embedding)
                
                results[str(binary_file)] = str(output_file)
                successful += 1
                logger.info(f"✓ Saved embedding: {output_file}")
            else:
                failed += 1
                logger.warning(f"✗ Failed: {binary_file}")
        
        logger.info(f"\nVectorization complete:")
        logger.info(f"  Successful: {successful}/{len(binary_files)}")
        logger.info(f"  Failed: {failed}/{len(binary_files)}")
        logger.info(f"  Output directory: {output_dir}")
        
        return results


def test_vectorizer():
    """Test vectorizer with dummy checkpoint (T098)."""
    print("Testing Vectorizer...")
    print("Note: Requires a trained model checkpoint to run")
    print("This is a placeholder test - actual testing should be done with real checkpoint")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_vectorizer()
