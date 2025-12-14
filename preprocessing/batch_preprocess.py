"""
Batch Preprocessing Script

Processes multiple binaries: extracts CFG + tokenizes instructions → unified JSON format.
Generates metadata.csv for dynamic pair sampling.

Module: preprocessing.batch_preprocess
Owner: User Story 1 (US1) - Binary Preprocessing
Task: T022-T025
"""

import argparse
import csv
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

from tqdm import tqdm

from preprocessing.extract_features import extract_single_cfg, compute_hash
from preprocessing.tokenizer import AssemblyTokenizer

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)-8s | %(asctime)s | %(name)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("bcsd.batch_preprocess")


# Removed: using compute_hash from extract_features module instead


def extract_metadata_from_path(binary_path: Path) -> Dict[str, str]:
    """
    Extract metadata from binary filename.
    
    Expected format: <project>_<function>_<compiler>_<opt>.bin
    Example: openssl_ssl_read_gcc_O0.bin
    
    Args:
        binary_path: Path to binary file
        
    Returns:
        Dictionary with project, function_name, compiler, optimization
    """
    filename = binary_path.stem  # Remove extension
    parts = filename.split('_')
    
    # Try to parse structured filename
    if len(parts) >= 4:
        return {
            "project": parts[0],
            "function_name": '_'.join(parts[1:-2]),  # Handle multi-word function names
            "compiler": parts[-2],
            "optimization": parts[-1]
        }
    else:
        # Fallback for simple filenames
        return {
            "project": "unknown",
            "function_name": filename,
            "compiler": "unknown",
            "optimization": "unknown"
        }


# Removed: using create_unified_json_v2 instead (handles new CFG format)


def process_single_binary(binary_path: Path, output_dir: Path, tokenizer: AssemblyTokenizer,
                          split_set: str = "train") -> Optional[Dict]:
    """
    Process single binary: extract CFG + tokenize + create unified JSON.
    
    Args:
        binary_path: Path to binary file
        output_dir: Directory to save preprocessed output
        tokenizer: AssemblyTokenizer instance
        split_set: Dataset split ("train", "validation", "test")
        
    Returns:
        Metadata dictionary or None if failed
    """
    try:
        # Compute file hash
        file_hash = compute_hash(str(binary_path))
        
        # Extract metadata from filename
        metadata = extract_metadata_from_path(binary_path)
        
        # Extract CFG
        cfg_result = extract_single_cfg(str(binary_path), str(output_dir))
        
        if cfg_result["status"] != "success":
            logger.warning(f"CFG extraction failed for {binary_path.name}: {cfg_result.get('error', 'Unknown error')}")
            return None
        
        # Use cfg_data instead of cfg (fixed for new format)
        cfg_data = cfg_result.get("cfg_data", {})
        
        # Collect all instructions for tokenization
        all_instructions = []
        for node in cfg_data.get("nodes", []):
            for inst in node.get("instructions", []):
                all_instructions.append(inst.get("full", ""))
        
        # Tokenize (get token strings, not IDs for storage)
        tokens = []
        for inst in all_instructions:
            if inst:  # Skip empty instructions
                parsed = tokenizer._parse_instruction(inst)
                tokens.extend(parsed)
        
        # Create unified JSON
        output_file = output_dir / f"{file_hash}.json"
        create_unified_json_v2(cfg_data, tokens, file_hash, metadata["function_name"], output_file)
        
        # Build metadata entry
        metadata_entry = {
            "file_hash": file_hash,
            "project": metadata["project"],
            "function_name": metadata["function_name"],
            "optimization": metadata["optimization"],
            "compiler": metadata["compiler"],
            "split_set": split_set,
            "file_path": str(binary_path.absolute()),
            "file_size_bytes": binary_path.stat().st_size,
            "node_count": cfg_data.get("node_count", 0),
            "edge_count": cfg_data.get("edge_count", 0),
            "token_count": len(tokens),
            "preprocessing_status": "success"
        }
        
        return metadata_entry
        
    except Exception as e:
        logger.error(f"Error processing {binary_path.name}: {str(e)}", exc_info=True)
        return None


def create_unified_json_v2(cfg_data: Dict, tokens: List[str], file_hash: str, 
                           function_name: str, output_file: Path) -> None:
    """
    Create unified JSON file with CFG edges + tokenized sequence (updated format).
    
    Format: {hash}.json containing:
        - id: file hash
        - function_name: extracted function name
        - tokens: tokenized instruction sequence
        - token_count: number of tokens
        - edges: CFG edge list
        - node_count: number of CFG nodes
        - edge_count: number of CFG edges
    
    Args:
        cfg_data: CFG extraction result (cfg_data dict from extract_single_cfg)
        tokens: Tokenized instruction sequence
        file_hash: SHA256 hash of binary
        function_name: Function name
        output_file: Path to save JSON
    """
    unified = {
        "id": file_hash,
        "function_name": function_name,
        "tokens": tokens,
        "token_count": len(tokens),
        "edges": cfg_data.get("edges", []),
        "node_count": cfg_data.get("node_count", 0),
        "edge_count": cfg_data.get("edge_count", 0)
    }
    
    with open(output_file, 'w') as f:
        json.dump(unified, f, indent=2)



def batch_preprocess(binary_dir: str, output_dir: str, vocab_file: Optional[str] = None,
                    split_set: str = "train") -> Dict:
    """
    Batch process all binaries in directory.
    
    Args:
        binary_dir: Directory containing binary files
        output_dir: Directory to save outputs
        vocab_file: Path to existing vocabulary JSON (if None, builds new vocab)
        split_set: Dataset split ("train", "validation", "test")
        
    Returns:
        Statistics dictionary
    """
    binary_dir = Path(binary_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all binary files (improved filtering)
    binary_files = []
    for f in binary_dir.glob("*"):
        # Skip directories
        if not f.is_file():
            continue
        # Skip files with known non-binary extensions
        if f.suffix.lower() in ['.txt', '.md', '.sh', '.py', '.json', '.csv', '.c', '.cpp', '.h', '.hpp']:
            continue
        # Skip hidden files
        if f.name.startswith('.'):
            continue
        binary_files.append(f)
    
    logger.info(f"Found {len(binary_files)} binary files in {binary_dir}")
    
    if len(binary_files) == 0:
        logger.warning("No binary files found!")
        return {"total_binaries": 0, "successful": 0, "failed": 0}
    
    # Initialize tokenizer
    tokenizer = AssemblyTokenizer(vocab_size=5000, max_seq_length=512)
    
    if vocab_file and Path(vocab_file).exists():
        logger.info(f"Loading existing vocabulary from {vocab_file}")
        tokenizer.load_vocab(vocab_file)
    else:
        logger.info("Building new vocabulary from all binaries...")
        # First pass: collect all instructions for vocabulary building
        all_instruction_sequences = []
        for binary_path in tqdm(binary_files, desc="Collecting instructions"):
            try:
                cfg_result = extract_single_cfg(str(binary_path), str(output_dir))
                if cfg_result["status"] == "success":
                    cfg_data = cfg_result.get("cfg_data", {})
                    instructions = []
                    for node in cfg_data.get("nodes", []):
                        for inst in node.get("instructions", []):
                            inst_text = inst.get("full", "")
                            if inst_text:
                                instructions.append(inst_text)
                    if instructions:
                        all_instruction_sequences.append(instructions)
            except Exception as e:
                logger.warning(f"Error collecting from {binary_path.name}: {e}")
                continue
        
        tokenizer.build_vocab(all_instruction_sequences)
        
        # Save vocabulary
        if vocab_file:
            tokenizer.save_vocab(vocab_file)
        else:
            default_vocab = output_dir.parent / "vocab.json"
            tokenizer.save_vocab(str(default_vocab))
            logger.info(f"Vocabulary saved to {default_vocab}")
    
    # Process all binaries
    metadata_entries = []
    successful = 0
    failed = 0
    
    logger.info("Processing binaries and creating unified JSON files...")
    for binary_path in tqdm(binary_files, desc="Processing"):
        metadata = process_single_binary(binary_path, output_dir, tokenizer, split_set)
        
        if metadata:
            metadata_entries.append(metadata)
            successful += 1
        else:
            failed += 1
    
    # Write metadata.csv
    if metadata_entries:
        metadata_csv = output_dir.parent / "metadata.csv"
        fieldnames = ["file_hash", "project", "function_name", "optimization", "compiler", 
                     "split_set", "file_path", "file_size_bytes", "node_count", "edge_count",
                     "token_count", "preprocessing_status"]
        
        with open(metadata_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(metadata_entries)
        
        logger.info(f"Metadata written to {metadata_csv}")
    
    stats = {
        "total_binaries": len(binary_files),
        "successful": successful,
        "failed": failed,
        "output_dir": str(output_dir),
        "vocab_size": len(tokenizer.vocab)
    }
    
    logger.info(f"Batch preprocessing complete: {successful}/{len(binary_files)} successful")
    
    return stats


def main():
    """Command-line interface for batch preprocessing."""
    parser = argparse.ArgumentParser(description="Batch preprocess binaries for BCSD pipeline")
    parser.add_argument("binary_dir", help="Directory containing binary files")
    parser.add_argument("--output-dir", default="data/preprocessed", help="Output directory for preprocessed files")
    parser.add_argument("--vocab-file", default=None, help="Path to vocabulary file (builds new if not provided)")
    parser.add_argument("--split", default="train", choices=["train", "validation", "test"], help="Dataset split")
    
    args = parser.parse_args()
    
    # Run batch preprocessing
    stats = batch_preprocess(
        binary_dir=args.binary_dir,
        output_dir=args.output_dir,
        vocab_file=args.vocab_file,
        split_set=args.split
    )
    
    print("\n" + "="*60)
    print("BATCH PREPROCESSING COMPLETE")
    print("="*60)
    print(f"Total binaries:  {stats['total_binaries']}")
    print(f"Successful:      {stats['successful']}")
    print(f"Failed:          {stats['failed']}")
    print(f"Output dir:      {stats['output_dir']}")
    print(f"Vocabulary size: {stats['vocab_size']}")
    print("="*60)


if __name__ == "__main__":
    main()
