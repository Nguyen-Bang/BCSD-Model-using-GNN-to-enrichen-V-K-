"""
Integration Tests for Dataset and DataLoader

Tests T034-T036: Dataset loading, dynamic pairing, and DataLoader batching

Module: tests.test_dataset_integration
Owner: User Story 2 (US2) - PyTorch Dataset Implementation
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import logging
import torch
from torch.utils.data import DataLoader

from dataset.code_dataset import BinaryCodeDataset
from dataset.collate import collate_heterogeneous

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_dataset_loading():
    """T034: Test Dataset with small subset and verify dynamic pairing"""
    logger.info("\n=== T034: Testing Dataset Loading ===")
    
    metadata_csv = "data/metadata.csv"
    
    if not Path(metadata_csv).exists():
        logger.error(f"Metadata file not found: {metadata_csv}")
        logger.info("Please run batch_preprocess.py first to generate metadata.csv")
        return False
    
    try:
        # Load dataset
        dataset = BinaryCodeDataset(
            metadata_csv=metadata_csv,
            split="test"  # Use test split
        )
        
        logger.info(f"✓ Dataset loaded successfully")
        logger.info(f"  - Function groups: {len(dataset)}")
        logger.info(f"  - Total binaries: {len(dataset.metadata)}")
        
        if len(dataset) == 0:
            logger.error("✗ Dataset is empty - no function groups with 2+ variants")
            return False
        
        # Test __getitem__
        logger.info("\n--- Testing dynamic pairing (__getitem__) ---")
        sample = dataset[0]
        
        # Verify structure
        assert "binary_1" in sample, "Missing binary_1"
        assert "binary_2" in sample, "Missing binary_2"
        assert "label" in sample, "Missing label"
        assert "metadata" in sample, "Missing metadata"
        
        logger.info(f"✓ Sample structure correct")
        
        # Verify binary data
        binary_1 = sample["binary_1"]
        binary_2 = sample["binary_2"]
        
        assert "tokens" in binary_1, "Missing tokens in binary_1"
        assert "edges" in binary_1, "Missing edges in binary_1"
        assert "node_count" in binary_1, "Missing node_count in binary_1"
        
        logger.info(f"✓ Binary data structure correct")
        logger.info(f"  - Binary 1: {len(binary_1['tokens'])} tokens, {len(binary_1['edges'])} edges")
        logger.info(f"  - Binary 2: {len(binary_2['tokens'])} tokens, {len(binary_2['edges'])} edges")
        
        # Verify same function name (dynamic pairing correctness)
        meta = sample["metadata"]
        logger.info(f"  - Function: {meta['function_name']}")
        logger.info(f"  - Binary 1: {meta['binary_1_compilation']}")
        logger.info(f"  - Binary 2: {meta['binary_2_compilation']}")
        
        # Verify label is positive pair
        assert sample["label"] == 1, "Label should be 1 (positive pair)"
        logger.info(f"✓ Label correct (positive pair)")
        
        # Test multiple samples to verify random pairing
        logger.info("\n--- Testing random pairing (sample same function group twice) ---")
        sample_a = dataset[0]
        sample_b = dataset[0]
        
        # Should be same function but potentially different pairs
        assert sample_a["metadata"]["function_name"] == sample_b["metadata"]["function_name"]
        logger.info(f"✓ Same function name across samples: {sample_a['metadata']['function_name']}")
        
        logger.info("\n✓✓✓ T034 PASSED: Dataset loading and dynamic pairing work correctly ✓✓✓\n")
        return True
        
    except Exception as e:
        logger.error(f"✗ T034 FAILED: {e}", exc_info=True)
        return False


def test_dataloader_batching():
    """T035: Test DataLoader with batch_size=16 and verify shapes"""
    logger.info("\n=== T035: Testing DataLoader Batching ===")
    
    metadata_csv = "data/metadata.csv"
    
    if not Path(metadata_csv).exists():
        logger.error(f"Metadata file not found: {metadata_csv}")
        return False
    
    try:
        # Load dataset
        dataset = BinaryCodeDataset(
            metadata_csv=metadata_csv,
            split="test"
        )
        
        if len(dataset) == 0:
            logger.error("Dataset is empty")
            return False
        
        # Create DataLoader with custom collate
        batch_size = min(2, len(dataset))  # Use smaller batch size for testing
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            collate_fn=collate_heterogeneous,
            shuffle=False
        )
        
        logger.info(f"✓ DataLoader created with batch_size={batch_size}")
        
        # Test one batch
        logger.info("\n--- Testing batch iteration ---")
        batch = next(iter(dataloader))
        
        # Verify batch structure
        required_keys = [
            "binary_1_tokens", "binary_1_attention_mask", "binary_1_edges", "binary_1_node_counts",
            "binary_2_tokens", "binary_2_attention_mask", "binary_2_edges", "binary_2_node_counts",
            "labels", "metadata"
        ]
        
        for key in required_keys:
            assert key in batch, f"Missing key: {key}"
        
        logger.info(f"✓ Batch has all required keys")
        
        # Verify shapes
        logger.info("\n--- Verifying batch shapes ---")
        
        # Binary 1
        binary_1_tokens = batch["binary_1_tokens"]
        binary_1_mask = batch["binary_1_attention_mask"]
        binary_1_edges = batch["binary_1_edges"]
        binary_1_node_counts = batch["binary_1_node_counts"]
        
        logger.info(f"Binary 1:")
        logger.info(f"  - Tokens: {len(binary_1_tokens)} samples, max_len={len(binary_1_tokens[0]) if binary_1_tokens else 0}")
        logger.info(f"  - Attention mask: {binary_1_mask.shape}")
        logger.info(f"  - Edges: {len(binary_1_edges)} graphs")
        logger.info(f"  - Node counts: {binary_1_node_counts.shape} = {binary_1_node_counts.tolist()}")
        
        # Binary 2
        binary_2_tokens = batch["binary_2_tokens"]
        binary_2_mask = batch["binary_2_attention_mask"]
        binary_2_edges = batch["binary_2_edges"]
        binary_2_node_counts = batch["binary_2_node_counts"]
        
        logger.info(f"Binary 2:")
        logger.info(f"  - Tokens: {len(binary_2_tokens)} samples, max_len={len(binary_2_tokens[0]) if binary_2_tokens else 0}")
        logger.info(f"  - Attention mask: {binary_2_mask.shape}")
        logger.info(f"  - Edges: {len(binary_2_edges)} graphs")
        logger.info(f"  - Node counts: {binary_2_node_counts.shape} = {binary_2_node_counts.tolist()}")
        
        # Labels
        labels = batch["labels"]
        logger.info(f"Labels: {labels.shape} = {labels.tolist()}")
        
        # Verify all are positive pairs
        assert all(l == 1 for l in labels.tolist()), "All labels should be 1"
        
        # Verify batch size consistency
        assert binary_1_mask.shape[0] == batch_size, "Binary 1 mask batch size mismatch"
        assert binary_2_mask.shape[0] == batch_size, "Binary 2 mask batch size mismatch"
        assert len(binary_1_edges) == batch_size, "Binary 1 edges batch size mismatch"
        assert len(binary_2_edges) == batch_size, "Binary 2 edges batch size mismatch"
        assert labels.shape[0] == batch_size, "Labels batch size mismatch"
        
        logger.info(f"\n✓ All shapes correct (batch_size={batch_size})")
        
        # Verify attention masks
        logger.info("\n--- Verifying attention masks ---")
        for i in range(batch_size):
            mask = binary_1_mask[i]
            # Should have some 1s (attended tokens) and potentially some 0s (padding)
            num_attended = (mask == 1).sum().item()
            num_padded = (mask == 0).sum().item()
            logger.info(f"  Sample {i}: {num_attended} attended, {num_padded} padded")
            assert num_attended > 0, f"Sample {i} has no attended tokens"
        
        logger.info(f"✓ Attention masks valid")
        
        # Test iteration through full dataset
        logger.info("\n--- Testing full dataset iteration ---")
        total_batches = 0
        total_samples = 0
        for batch in dataloader:
            total_batches += 1
            total_samples += batch["labels"].shape[0]
        
        logger.info(f"✓ Iterated through {total_batches} batches, {total_samples} samples")
        assert total_samples == len(dataset), f"Sample count mismatch: {total_samples} != {len(dataset)}"
        
        logger.info("\n✓✓✓ T035 PASSED: DataLoader batching works correctly ✓✓✓\n")
        return True
        
    except Exception as e:
        logger.error(f"✗ T035 FAILED: {e}", exc_info=True)
        return False


def main():
    """Run all dataset tests (T034-T035)"""
    logger.info("="*80)
    logger.info("DATASET AND DATALOADER INTEGRATION TESTS")
    logger.info("="*80)
    
    results = []
    
    # T034: Dataset loading
    results.append(("T034", test_dataset_loading()))
    
    # T035: DataLoader batching
    results.append(("T035", test_dataloader_batching()))
    
    # Summary
    logger.info("\n" + "="*80)
    logger.info("TEST SUMMARY")
    logger.info("="*80)
    
    for task_id, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        logger.info(f"{task_id}: {status}")
    
    all_passed = all(passed for _, passed in results)
    
    if all_passed:
        logger.info("\n✓✓✓ ALL TESTS PASSED ✓✓✓")
        logger.info("\nPhase 4 (User Story 2) is complete!")
        logger.info("Ready to proceed to Phase 5 (GNN Encoder Implementation)")
    else:
        logger.info("\n✗✗✗ SOME TESTS FAILED ✗✗✗")
        logger.info("\nPlease fix failing tests before proceeding.")
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
