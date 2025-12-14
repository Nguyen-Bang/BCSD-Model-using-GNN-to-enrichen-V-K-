"""
Binary Code Dataset for PyTorch

Implements PyTorch Dataset for loading preprocessed CFG data with dynamic pairing
for Siamese contrastive learning.

Module: dataset.code_dataset
Owner: User Story 2 (US2) - PyTorch Dataset Implementation
Tasks: T027-T029
"""

import csv
import json
import logging
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

import torch
from torch.utils.data import Dataset

logger = logging.getLogger("bcsd.dataset")


class BinaryCodeDataset(Dataset):
    """
    PyTorch Dataset for Binary Code with dynamic pair sampling (T027-T029).
    
    Loads preprocessed data from metadata.csv and dynamically pairs binaries
    from the same function for Siamese training (US2).
    
    Key Features:
    - Reads metadata.csv for file paths and function information
    - Groups binaries by (project, function_name) for positive pairing
    - Dynamically samples 2 binaries from same function per __getitem__
    - Filters by split_set (train/validation/test)
    
    Example:
        >>> dataset = BinaryCodeDataset(
        ...     metadata_csv="data/metadata.csv",
        ...     split="train"
        ... )
        >>> sample = dataset[0]  # Returns pair of binaries from same function
        >>> print(sample.keys())  # ['binary_1', 'binary_2', 'label', 'metadata']
    """
    
    def __init__(
        self,
        metadata_csv: str,
        split: str = "train",
        vocab_file: Optional[str] = None
    ):
        """
        Initialize dataset (T028).
        
        Args:
            metadata_csv: Path to metadata CSV file
            split: Dataset split ("train", "validation", "test")
            vocab_file: Path to vocabulary JSON (for future tokenization)
        """
        self.metadata_csv = metadata_csv
        self.split = split
        self.vocab_file = vocab_file
        
        # Load metadata
        self.metadata = self._load_metadata(metadata_csv, split)
        
        # Group by (project, function_name) for dynamic pairing
        self.function_groups = self._build_function_groups()
        
        # Create list of function groups (for indexing)
        self.function_group_list = list(self.function_groups.keys())
        
        logger.info(f"BinaryCodeDataset initialized: {len(self.function_group_list)} function groups, {len(self.metadata)} binaries in split '{split}'")
    
    def _load_metadata(self, csv_path: str, split: str) -> List[Dict]:
        """
        Load metadata from CSV and filter by split.
        
        Args:
            csv_path: Path to metadata CSV
            split: Split to filter by
            
        Returns:
            List of metadata dictionaries
        """
        metadata = []
        
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("split_set") == split:
                    metadata.append(row)
        
        if not metadata:
            raise ValueError(f"No samples found for split '{split}' in {csv_path}")
        
        logger.info(f"Loaded {len(metadata)} samples from {csv_path} (split: {split})")
        return metadata
    
    def _build_function_groups(self) -> Dict[Tuple[str, str], List[Dict]]:
        """
        Group binaries by (project, function_name) for pairing (T028).
        
        Returns:
            Dictionary mapping (project, function_name) to list of binary metadata
        """
        groups = defaultdict(list)
        
        for sample in self.metadata:
            key = (sample["project"], sample["function_name"])
            groups[key].append(sample)
        
        # Filter out groups with only 1 sample (can't pair)
        groups = {k: v for k, v in groups.items() if len(v) >= 2}
        
        logger.info(f"Built {len(groups)} function groups with 2+ variants")
        return groups
    
    def __len__(self) -> int:
        """
        Return number of function groups (not individual binaries).
        
        Each group can generate multiple pairs through random sampling.
        """
        return len(self.function_group_list)
    
    def __getitem__(self, idx: int) -> Dict:
        """
        Dynamically sample pair from same function group (T029).
        
        Args:
            idx: Index of function group
            
        Returns:
            Dictionary with:
                - binary_1: First binary data (tokens, edges, metadata)
                - binary_2: Second binary data (from same function, different compilation)
                - label: Always 1 (positive pair)
                - metadata: Pair metadata (function name, compilations, etc.)
        """
        # Get function group
        function_key = self.function_group_list[idx]
        group_samples = self.function_groups[function_key]
        
        # Randomly sample 2 different binaries from this group
        if len(group_samples) >= 2:
            sample_1, sample_2 = random.sample(group_samples, 2)
        else:
            # Fallback: same binary twice (shouldn't happen after filtering)
            sample_1 = sample_2 = group_samples[0]
        
        # Load binary data
        binary_1 = self._load_binary(sample_1)
        binary_2 = self._load_binary(sample_2)
        
        # Create pair metadata
        pair_metadata = {
            "function_name": function_key[1],
            "project": function_key[0],
            "binary_1_hash": sample_1["file_hash"],
            "binary_2_hash": sample_2["file_hash"],
            "binary_1_compilation": f"{sample_1['compiler']}_{sample_1['optimization']}",
            "binary_2_compilation": f"{sample_2['compiler']}_{sample_2['optimization']}",
        }
        
        return {
            "binary_1": binary_1,
            "binary_2": binary_2,
            "label": 1,  # Positive pair
            "metadata": pair_metadata
        }
    
    def _load_binary(self, metadata: Dict) -> Dict:
        """
        Load binary data from JSON file.
        
        Args:
            metadata: Metadata dictionary with file_hash
            
        Returns:
            Dictionary with tokens, edges, and metadata
        """
        # Construct output file path from file_hash
        # Try both formats: {hash}_cfg.json (old format) and {hash}.json (new format)
        file_hash = metadata["file_hash"]
        base_path = Path(metadata.get("file_path", "")).parent / "data" / "preprocessed"
        
        output_file = base_path / f"{file_hash}_cfg.json"
        if not output_file.exists():
            output_file = base_path / f"{file_hash}.json"
        
        # If still not found, try relative to current directory
        if not output_file.exists():
            output_file = Path("data/preprocessed") / f"{file_hash}_cfg.json"
        if not output_file.exists():
            output_file = Path("data/preprocessed") / f"{file_hash}.json"
        
        try:
            with open(output_file, 'r') as f:
                data = json.load(f)
            
            # Handle both old format (_cfg.json) and new format (.json)
            if "cfg_data" in data:
                # Old format from extract_single_cfg()
                cfg_data = data["cfg_data"]
                
                # Extract instructions from all nodes as tokens
                tokens = []
                for node in cfg_data.get("nodes", []):
                    for inst in node.get("instructions", []):
                        tokens.append(inst.get("full", ""))
                
                return {
                    "file_hash": file_hash,
                    "tokens": tokens,
                    "edges": cfg_data.get("edges", []),
                    "node_count": cfg_data.get("node_count", 0),
                    "edge_count": cfg_data.get("edge_count", 0),
                    "function_name": metadata.get("function_name", "unknown")
                }
            
            elif "functions" in data:
                # New format from extract_cfg() - merge all functions
                all_tokens = []
                all_edges = []
                total_nodes = 0
                
                for func in data.get("functions", []):
                    for node in func.get("nodes", []):
                        all_tokens.extend(node.get("instructions", []))
                    all_edges.extend(func.get("edges", []))
                    total_nodes += len(func.get("nodes", []))
                
                return {
                    "file_hash": file_hash,
                    "tokens": all_tokens,
                    "edges": all_edges,
                    "node_count": total_nodes,
                    "edge_count": len(all_edges),
                    "function_name": metadata.get("function_name", data.get("functions", [{}])[0].get("function_name", "unknown"))
                }
            
            else:
                raise ValueError(f"Unknown JSON format in {output_file}")
        
        except Exception as e:
            logger.error(f"Error loading {output_file}: {e}")
            # Return empty data as fallback
            return {
                "file_hash": file_hash,
                "tokens": [],
                "edges": [],
                "node_count": 0,
                "edge_count": 0,
                "function_name": metadata.get("function_name", "unknown")
            }

