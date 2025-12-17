"""
Batch Preprocessing Script (Updated for ClapASM + New Graph Format)

Processes multiple binaries: 
1. Extracts CFG with edge types (jumpkinds).
2. Tokenizes instructions using pre-trained ClapASM tokenizer.
3. Generates unified JSON files with 'token_ids'.
4. Creates metadata.csv for dataset loading.

Module: preprocessing.batch_preprocess
Owner: User Story 1 (US1) - Binary Preprocessing
Task: T022-T025
"""

import argparse
import csv
import logging
from pathlib import Path
from typing import Dict, List, Optional

from tqdm import tqdm

from preprocessing.extract_features import extract_cfg, compute_hash
from preprocessing.tokenizer import ClapASMTokenizer

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)-8s | %(asctime)s | %(name)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("bcsd.batch_preprocess")


def extract_metadata_from_path(binary_path: Path) -> Dict[str, str]:
    """
    Extract metadata from binary filename/path.
    
    Expected format often found in research datasets:
    Dataset-1/<project>/<optimization>/<compiler>/.../<binary>
    or flat: <project>_<function>_<compiler>_<opt>.bin
    
    We'll try to be robust.
    """
    # Simple heuristic: split filename by underscores
    filename = binary_path.stem
    parts = filename.split('_')
    
    metadata = {
        "project": "unknown",
        "function_name": filename,
        "compiler": "unknown",
        "optimization": "unknown"
    }
    
    # Try to parse standard "project_func_compiler_opt" format
    if len(parts) >= 4:
        metadata["project"] = parts[0]
        metadata["compiler"] = parts[-2]
        metadata["optimization"] = parts[-1]
        metadata["function_name"] = '_'.join(parts[1:-2])
    
    # Override with path info if available (e.g. Dataset-1/openssl/...)
    # This depends on specific folder structure of Dataset-1
    # For now, filename parsing is safer if filenames are descriptive
    
    return metadata


def batch_preprocess(binary_dir: str, output_dir: str, split_set: str = "train") -> Dict:
    """
    Batch process all binaries in directory using ClapASM tokenizer.
    
    Args:
        binary_dir: Directory containing binary files (searched recursively)
        output_dir: Directory to save outputs
        split_set: Dataset split ("train", "validation", "test")
        
    Returns:
        Statistics dictionary
    """
    binary_dir = Path(binary_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize ClapASM Tokenizer
    try:
        tokenizer = ClapASMTokenizer()
        logger.info(f"Loaded ClapASM Tokenizer (vocab size: {len(tokenizer.tokenizer.vocab)})")
    except Exception as e:
        logger.error(f"Failed to load tokenizer: {e}")
        return {"error": str(e)}
    
    # Find binary files recursively
    binary_files = []
    # extensions to ignore
    ignore_exts = {'.json', '.txt', '.md', '.csv', '.py', '.sh', '.c', '.cpp', '.h', '.o', '.obj'}
    
    for f in binary_dir.rglob("*"):
        if f.is_file() and f.suffix.lower() not in ignore_exts and not f.name.startswith('.'):
            # Check if it's executable (optional, might be slow)
            # For now, assume anything not ignored is a binary target
            binary_files.append(f)
            
    logger.info(f"Found {len(binary_files)} potential binaries in {binary_dir}")
    
    metadata_entries = []
    successful = 0
    failed = 0
    
    # Process binaries
    for binary_path in tqdm(binary_files, desc="Processing"):
        try:
            # Extract Features (CFG + Tokenization)
            result = extract_cfg(
                binary_path=str(binary_path),
                output_dir=str(output_dir),
                tokenizer=tokenizer,
                timeout=300 # 5 min timeout per binary
            )
            
            if result["status"] == "success":
                # Extract metadata for CSV
                meta = extract_metadata_from_path(binary_path)
                
                entry = {
                    "file_hash": result["binary_hash"],
                    "project": meta["project"],
                    "function_name": meta["function_name"],
                    "optimization": meta["optimization"],
                    "compiler": meta["compiler"],
                    "split_set": split_set,
                    "file_path": str(binary_path.absolute()),
                    "output_file": result["output_file"],
                    "node_count": result["total_nodes"],
                    "edge_count": result["total_edges"],
                    "function_count": result["total_functions"],
                    "preprocessing_status": "success"
                }
                metadata_entries.append(entry)
                successful += 1
            else:
                failed += 1
                # logger.warning(f"Failed {binary_path.name}: {result.get('error')}")
                
        except Exception as e:
            logger.error(f"Crash processing {binary_path.name}: {e}")
            failed += 1
            
    # Write metadata.csv
    if metadata_entries:
        metadata_csv = output_dir.parent / "metadata.csv"
        # Ensure parent exists
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        
        fieldnames = ["file_hash", "project", "function_name", "optimization", "compiler", 
                     "split_set", "file_path", "output_file", "node_count", "edge_count",
                     "function_count", "preprocessing_status"]
        
        # Check if file exists to append or write header
        file_exists = metadata_csv.exists()
        mode = 'a' if file_exists else 'w'
        
        with open(metadata_csv, mode, newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerows(metadata_entries)
            
        logger.info(f"Metadata written to {metadata_csv} (entries: {len(metadata_entries)})")
        
    return {
        "total": len(binary_files),
        "successful": successful,
        "failed": failed
    }

def main():
    parser = argparse.ArgumentParser(description="Batch preprocess binaries with ClapASM")
    parser.add_argument("binary_dir", help="Input directory containing binaries")
    parser.add_argument("--output-dir", default="data/preprocessed", help="Output directory")
    parser.add_argument("--split", default="train", help="Split set name (train/val/test)")
    
    args = parser.parse_args()
    
    batch_preprocess(args.binary_dir, args.output_dir, args.split)

if __name__ == "__main__":
    main()