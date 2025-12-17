"""
Unit Tests for Preprocessing Module

Tests for CFG extraction, tokenization (ClapASM), and batch preprocessing pipeline.

Module: tests.test_preprocessing
Owner: User Story 1 (US1) - Binary Preprocessing
"""

import unittest
import json
import tempfile
import shutil
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from preprocessing.extract_features import extract_single_cfg, extract_cfg, compute_hash
from preprocessing.tokenizer import ClapASMTokenizer
from preprocessing.batch_preprocess import batch_preprocess


class TestCFGExtraction(unittest.TestCase):
    """Test CFG extraction functionality"""
    
    def setUp(self):
        """Setup test fixtures"""
        self.test_binary = Path(__file__).parent.parent / "test_binaries" / "test_gnn_gcc_O0"
        self.temp_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        """Clean up temporary files"""
        if Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)
    
    def test_extract_cfg_success(self):
        """Test successful CFG extraction from test binary"""
        if not self.test_binary.exists():
            self.skipTest("Test binary not found")
        
        result = extract_single_cfg(str(self.test_binary), self.temp_dir)
        
        # Verify result structure
        self.assertEqual(result["status"], "success")
        self.assertIn("binary_hash", result)
        self.assertIn("cfg_data", result)
        
        # Verify CFG data
        cfg_data = result["cfg_data"]
        self.assertGreater(cfg_data["node_count"], 0, "Should extract at least one node")
        self.assertGreater(cfg_data["edge_count"], 0, "Should extract at least one edge")
        
        # Check new edge format [src, dst, type]
        if cfg_data["edges"]:
            edge = cfg_data["edges"][0]
            self.assertEqual(len(edge), 3, "Edge should have 3 elements [src, dst, type]")
            self.assertIsInstance(edge[2], str, "Edge type should be string")


class TestClapASMTokenizer(unittest.TestCase):
    """Test ClapASM tokenization functionality"""
    
    def setUp(self):
        """Setup test fixtures"""
        self.tokenizer = ClapASMTokenizer()
    
    def test_loading(self):
        """Test tokenizer loads correctly"""
        self.assertIsNotNone(self.tokenizer.tokenizer)
        self.assertGreater(len(self.tokenizer.tokenizer.vocab), 0)
    
    def test_tokenization(self):
        """Test instruction tokenization"""
        instructions = ["mov rax, rdi", "add rax, 0x10"]
        
        # Tokenize
        result = self.tokenizer.tokenize(instructions, add_special_tokens=True)
        
        # Verify structure
        self.assertIn("token_ids", result)
        self.assertIn("attention_mask", result)
        self.assertEqual(len(result["token_ids"]), len(result["attention_mask"]))
        
        # Check values
        token_ids = result["token_ids"]
        self.assertGreater(len(token_ids), 0)
    
    def test_comma_filtering(self):
        """Test that commas are filtered out (normalized)"""
        # "mov rax, rdi" -> should not have [UNK] if comma is handled
        # But wait, CLAP-ASM maps comma to [UNK] (ID 4) or we filter it.
        # My updated code filters it: text.replace(",", " ")
        
        instructions = ["mov rax, rdi"]
        result = self.tokenizer.tokenize(instructions)
        ids = result["token_ids"]
        
        # Check if UNK (4) is present. It SHOULD NOT be if we filter commas.
        # Note: Depending on vocab, 4 might be UNK.
        unk_id = self.tokenizer.tokenizer.unk_token_id
        if unk_id is None:
            unk_id = 4 # Fallback assumption
            
        self.assertNotIn(unk_id, ids, "Comma should be filtered, so no UNK tokens expected")


class TestBatchProcessing(unittest.TestCase):
    """Test batch preprocessing functionality"""
    
    def setUp(self):
        """Setup test fixtures"""
        self.test_binaries_dir = Path(__file__).parent.parent / "test_binaries"
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up temporary files"""
        if Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)
    
    def test_batch_processing(self):
        """Test batch processing of test binaries"""
        if not self.test_binaries_dir.exists():
            self.skipTest("Test binaries directory not found")
        
        output_dir = Path(self.temp_dir) / "preprocessed"
        
        # Process test binaries
        stats = batch_preprocess(
            binary_dir=str(self.test_binaries_dir),
            output_dir=str(output_dir),
            split_set="train"
        )
        
        # Verify stats
        self.assertGreater(stats["total"], 0)
        self.assertGreater(stats["successful"], 0)
        
        # Verify metadata file created
        metadata_path = output_dir.parent / "metadata.csv"
        self.assertTrue(metadata_path.exists())
        
        # Verify preprocessed files created
        self.assertTrue(output_dir.exists())
        json_files = list(output_dir.glob("*.json"))
        self.assertGreater(len(json_files), 0, "Should create at least one preprocessed file")


class TestIntegration(unittest.TestCase):
    """Integration tests for complete preprocessing pipeline"""
    
    def setUp(self):
        """Setup test fixtures"""
        self.test_binaries_dir = Path(__file__).parent.parent / "test_binaries"
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up temporary files"""
        if Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)
    
    def test_end_to_end_preprocessing(self):
        """Test complete preprocessing pipeline from binary to preprocessed JSON"""
        if not self.test_binaries_dir.exists():
            self.skipTest("Test binaries directory not found")
        
        # Find test binary
        test_binary = self.test_binaries_dir / "test_gnn_gcc_O0"
        if not test_binary.exists():
            self.skipTest("Test binary not found")
        
        output_dir = Path(self.temp_dir) / "preprocessed"
        output_dir.mkdir(parents=True)
        
        # Initialize Tokenizer
        tokenizer = ClapASMTokenizer()
        
        # Step 3: Full preprocessing with tokenization
        result = extract_cfg(
            str(test_binary),
            str(output_dir),
            tokenizer=tokenizer
        )
        
        self.assertEqual(result["status"], "success")
        self.assertIn("output_file", result)
        
        # Verify output file exists and is valid
        output_file = Path(result["output_file"])
        self.assertTrue(output_file.exists())
        
        with open(output_file, 'r') as f:
            data = json.load(f)
            self.assertIn("functions", data)
            self.assertGreater(len(data["functions"]), 0)
            
            # Check for token_ids in nodes
            first_func = data["functions"][0]
            if first_func["nodes"]:
                first_node = first_func["nodes"][0]
                self.assertIn("token_ids", first_node)
                self.assertIsInstance(first_node["token_ids"], list)


if __name__ == "__main__":
    unittest.main()