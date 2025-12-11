"""
BCSD Model Pipeline

Main pipeline for Binary Code Similarity Detection using:
1. Angr for disassembly
2. BERT for encoding
3. GNN for graph-based enrichment

This module orchestrates the complete pipeline from binary input to similarity detection.
"""

import logging
import argparse
from pathlib import Path
from typing import Dict, List, Any, Tuple
import json
import numpy as np

from angr_disassembly import disassemble_binary, disassemble_binaries_folder
from bert_encoder import BertEncoder
from gnn_model import GNNProcessor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def convert_to_serializable(obj):
    """
    Convert numpy arrays and other non-serializable objects to JSON-serializable format.
    
    Args:
        obj: Object to convert
        
    Returns:
        JSON-serializable version of the object
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    return obj


class BCSDPipeline:
    """
    Complete pipeline for Binary Code Similarity Detection.
    """
    
    def __init__(self, 
                 bert_model: str = 'bert-base-uncased',
                 gnn_config: Dict[str, Any] = None,
                 device: str = None):
        """
        Initialize the BCSD pipeline.
        
        Args:
            bert_model: Name of the BERT model to use
            gnn_config: Configuration for the GNN model
            device: Device to run models on ('cuda' or 'cpu')
        """
        logger.info("Initializing BCSD Pipeline...")
        
        self.bert_encoder = BertEncoder(model_name=bert_model, device=device)
        self.gnn_processor = GNNProcessor(model_config=gnn_config, device=device)
        
        logger.info("Pipeline initialized successfully")
    
    def process_single_binary(self, binary_path: str) -> Dict[str, Any]:
        """
        Process a single binary through the complete pipeline.
        
        Args:
            binary_path: Path to the binary file
            
        Returns:
            Dictionary containing all pipeline results
        """
        logger.info(f"Processing binary: {binary_path}")
        
        # Step 1: Disassemble with angr
        logger.info("Step 1/3: Disassembling binary...")
        disassembly_data = disassemble_binary(binary_path)
        
        if not disassembly_data:
            logger.error(f"Failed to disassemble {binary_path}")
            return {}
        
        # Step 2: Encode with BERT
        logger.info("Step 2/3: Encoding with BERT...")
        encoded_data = self.bert_encoder.encode_binary(disassembly_data)
        
        # Step 3: Process with GNN
        logger.info("Step 3/3: Processing with GNN...")
        enriched_embeddings = self.gnn_processor.process_binary(encoded_data)
        
        result = {
            'binary_path': binary_path,
            'disassembly': disassembly_data,
            'encoded': encoded_data,
            'enriched_embeddings': enriched_embeddings
        }
        
        logger.info(f"Successfully processed {binary_path}")
        return result
    
    def process_binary_folder(self, folder_path: str) -> Dict[str, Dict[str, Any]]:
        """
        Process all binaries in a folder through the pipeline.
        
        Args:
            folder_path: Path to folder containing binaries
            
        Returns:
            Dictionary mapping binary names to their results
        """
        logger.info(f"Processing binaries in folder: {folder_path}")
        
        folder = Path(folder_path)
        if not folder.exists() or not folder.is_dir():
            logger.error(f"Invalid folder path: {folder_path}")
            return {}
        
        results = {}
        binary_files = [f for f in folder.iterdir() if f.is_file()]
        
        logger.info(f"Found {len(binary_files)} files to process")
        
        for i, binary_file in enumerate(binary_files, 1):
            logger.info(f"[{i}/{len(binary_files)}] Processing: {binary_file.name}")
            try:
                result = self.process_single_binary(str(binary_file))
                if result:
                    results[binary_file.name] = result
            except Exception as e:
                logger.error(f"Error processing {binary_file.name}: {e}", exc_info=True)
        
        logger.info(f"Successfully processed {len(results)}/{len(binary_files)} binaries")
        return results
    
    def compare_binaries(self, 
                        binary1_result: Dict[str, Any],
                        binary2_result: Dict[str, Any]) -> Dict[str, float]:
        """
        Compare two processed binaries and compute similarity scores.
        
        Args:
            binary1_result: Pipeline result for first binary
            binary2_result: Pipeline result for second binary
            
        Returns:
            Dictionary of similarity metrics
        """
        logger.info("Comparing binaries...")
        
        emb1 = binary1_result.get('enriched_embeddings', {})
        emb2 = binary2_result.get('enriched_embeddings', {})
        
        if not emb1 or not emb2:
            logger.warning("One or both binaries have no embeddings")
            return {'overall_similarity': 0.0}
        
        # Compute pairwise similarities between functions
        similarities = []
        
        for func_addr1, func_emb1 in emb1.items():
            for func_addr2, func_emb2 in emb2.items():
                sim = self.gnn_processor.compute_similarity(func_emb1, func_emb2)
                similarities.append(sim)
        
        if not similarities:
            return {'overall_similarity': 0.0}
        
        # Aggregate similarity metrics
        result = {
            'overall_similarity': float(np.mean(similarities)),
            'max_similarity': float(np.max(similarities)),
            'min_similarity': float(np.min(similarities)),
            'num_comparisons': len(similarities)
        }
        
        logger.info(f"Similarity: {result['overall_similarity']:.4f}")
        return result
    
    def find_similar_binaries(self, 
                             results: Dict[str, Dict[str, Any]],
                             threshold: float = 0.7) -> List[Tuple[str, str, float]]:
        """
        Find pairs of similar binaries from a set of results.
        
        Args:
            results: Dictionary of binary processing results
            threshold: Minimum similarity threshold
            
        Returns:
            List of tuples (binary1, binary2, similarity)
        """
        logger.info(f"Finding similar binaries (threshold: {threshold})...")
        
        binary_names = list(results.keys())
        similar_pairs = []
        
        for i in range(len(binary_names)):
            for j in range(i + 1, len(binary_names)):
                bin1 = binary_names[i]
                bin2 = binary_names[j]
                
                comparison = self.compare_binaries(results[bin1], results[bin2])
                similarity = comparison.get('overall_similarity', 0.0)
                
                if similarity >= threshold:
                    similar_pairs.append((bin1, bin2, similarity))
        
        # Sort by similarity (descending)
        similar_pairs.sort(key=lambda x: x[2], reverse=True)
        
        logger.info(f"Found {len(similar_pairs)} similar pairs")
        return similar_pairs


def main():
    """Main entry point for the BCSD pipeline."""
    parser = argparse.ArgumentParser(description='Binary Code Similarity Detection Pipeline')
    parser.add_argument('input', help='Path to binary file or folder containing binaries')
    parser.add_argument('--output', '-o', help='Output file for results (JSON)', default=None)
    parser.add_argument('--compare', help='Path to second binary for comparison', default=None)
    parser.add_argument('--threshold', '-t', type=float, default=0.7,
                       help='Similarity threshold for finding similar binaries (default: 0.7)')
    parser.add_argument('--bert-model', default='bert-base-uncased',
                       help='BERT model to use (default: bert-base-uncased)')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default=None,
                       help='Device to run on (default: auto-detect)')
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = BCSDPipeline(
        bert_model=args.bert_model,
        device=args.device
    )
    
    # Process input
    input_path = Path(args.input)
    
    if input_path.is_file():
        # Single binary
        result = pipeline.process_single_binary(str(input_path))
        
        if args.compare:
            # Compare with another binary
            compare_result = pipeline.process_single_binary(args.compare)
            similarity = pipeline.compare_binaries(result, compare_result)
            print(f"\nSimilarity between {input_path.name} and {Path(args.compare).name}:")
            print(f"  Overall: {similarity['overall_similarity']:.4f}")
            print(f"  Max: {similarity['max_similarity']:.4f}")
            print(f"  Min: {similarity['min_similarity']:.4f}")
        
        # Save results
        if args.output:
            serializable_result = convert_to_serializable(result)
            with open(args.output, 'w') as f:
                json.dump(serializable_result, f, indent=2)
            logger.info(f"Results saved to {args.output}")
    
    elif input_path.is_dir():
        # Multiple binaries
        results = pipeline.process_binary_folder(str(input_path))
        
        # Find similar pairs
        similar_pairs = pipeline.find_similar_binaries(results, threshold=args.threshold)
        
        print(f"\nProcessed {len(results)} binaries")
        print(f"Found {len(similar_pairs)} similar pairs (threshold: {args.threshold}):")
        
        for bin1, bin2, similarity in similar_pairs[:10]:  # Show top 10
            print(f"  {bin1} <-> {bin2}: {similarity:.4f}")
        
        if len(similar_pairs) > 10:
            print(f"  ... and {len(similar_pairs) - 10} more")
        
        # Save results
        if args.output:
            output_data = {
                'results': results,
                'similar_pairs': [
                    {'binary1': p[0], 'binary2': p[1], 'similarity': p[2]}
                    for p in similar_pairs
                ]
            }
            
            serializable_output = convert_to_serializable(output_data)
            with open(args.output, 'w') as f:
                json.dump(serializable_output, f, indent=2)
            logger.info(f"Results saved to {args.output}")
    
    else:
        logger.error(f"Invalid input path: {args.input}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
