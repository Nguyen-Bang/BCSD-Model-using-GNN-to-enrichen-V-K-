"""
Unit Tests for Preprocessing Module

Tests for CFG extraction, tokenization, and batch preprocessing pipeline.

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
from preprocessing.tokenizer import AssemblyTokenizer
from preprocessing.batch_preprocess import process_binary_directory


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
        self.assertIn("functions", result)
        
        # Verify CFG data
        cfg_data = result["cfg_data"]
        self.assertGreater(cfg_data["node_count"], 0, "Should extract at least one node")
        self.assertGreater(cfg_data["edge_count"], 0, "Should extract at least one edge")
        self.assertIsInstance(cfg_data["nodes"], list)
        self.assertIsInstance(cfg_data["edges"], list)
        
        # Verify functions extracted
        self.assertGreater(len(result["functions"]), 0, "Should find at least one function")
        
        # Verify output file created
        output_file = Path(self.temp_dir) / f"{result['binary_hash']}_cfg.json"
        self.assertTrue(output_file.exists(), "Output file should be created")
        
        # Verify output file is valid JSON
        with open(output_file, 'r') as f:
            loaded_data = json.load(f)
            self.assertEqual(loaded_data["status"], "success")
    
    def test_extract_cfg_nonexistent_binary(self):
        """Test CFG extraction fails gracefully for nonexistent binary"""
        result = extract_single_cfg("/nonexistent/binary", self.temp_dir)
        
        self.assertEqual(result["status"], "failed")
        self.assertIn("error", result)
        self.assertIn("not found", result["error"].lower())
    
    def test_compute_hash(self):
        """Test binary hash computation"""
        if not self.test_binary.exists():
            self.skipTest("Test binary not found")
        
        hash1 = compute_hash(str(self.test_binary))
        hash2 = compute_hash(str(self.test_binary))
        
        # Hash should be deterministic
        self.assertEqual(hash1, hash2)
        
        # Hash should be 64 characters (SHA256 hex)
        self.assertEqual(len(hash1), 64)
        self.assertTrue(all(c in '0123456789abcdef' for c in hash1))
    
    def test_cfg_node_structure(self):
        """Test that CFG nodes have correct structure"""
        if not self.test_binary.exists():
            self.skipTest("Test binary not found")
        
        result = extract_single_cfg(str(self.test_binary))
        
        if result["status"] == "success":
            nodes = result["cfg_data"]["nodes"]
            
            for node in nodes:
                # Each node should have required fields
                self.assertIn("id", node)
                self.assertIn("addr", node)
                self.assertIn("size", node)
                self.assertIn("instructions", node)
                
                # Node ID should be integer
                self.assertIsInstance(node["id"], int)
                
                # Instructions should be a list
                self.assertIsInstance(node["instructions"], list)
    
    def test_cfg_edge_structure(self):
        """Test that CFG edges have correct structure"""
        if not self.test_binary.exists():
            self.skipTest("Test binary not found")
        
        result = extract_single_cfg(str(self.test_binary))
        
        if result["status"] == "success":
            edges = result["cfg_data"]["edges"]
            node_count = result["cfg_data"]["node_count"]
            
            for edge in edges:
                # Each edge should be [src, dst]
                self.assertIsInstance(edge, list)
                self.assertEqual(len(edge), 2)
                
                src, dst = edge
                # Node IDs should be valid
                self.assertIsInstance(src, int)
                self.assertIsInstance(dst, int)
                self.assertGreaterEqual(src, 0)
                self.assertGreaterEqual(dst, 0)
                self.assertLess(src, node_count)
                self.assertLess(dst, node_count)


class TestAssemblyTokenizer(unittest.TestCase):
    """Test assembly tokenization functionality"""
    
    def setUp(self):
        """Setup test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.tokenizer = AssemblyTokenizer(vocab_size=100)
    
    def tearDown(self):
        """Clean up temporary files"""
        if Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)
    
    def test_vocab_building(self):
        """Test vocabulary building from instructions"""
        instructions = [
            ["mov rax, rdi", "add rax, 0x10", "ret"],
            ["push rbp", "mov rbp, rsp", "pop rbp"],
            ["call 0x1000", "test rax, rax", "jne 0x2000"]
        ]
        
        self.tokenizer.build_vocab(instructions)
        
        # Verify special tokens
        self.assertEqual(self.tokenizer.token_to_id["[PAD]"], 0)
        self.assertEqual(self.tokenizer.token_to_id["[CLS]"], 101)
        self.assertEqual(self.tokenizer.token_to_id["[SEP]"], 102)
        self.assertEqual(self.tokenizer.token_to_id["[MASK]"], 103)
        self.assertEqual(self.tokenizer.token_to_id["[UNK]"], 104)
        
        # Verify opcodes are in vocabulary
        self.assertIn("mov", self.tokenizer.token_to_id)
        self.assertIn("add", self.tokenizer.token_to_id)
        self.assertIn("ret", self.tokenizer.token_to_id)
        self.assertIn("push", self.tokenizer.token_to_id)
        self.assertIn("pop", self.tokenizer.token_to_id)
        
        # Verify vocab size doesn't exceed limit
        self.assertLessEqual(len(self.tokenizer.token_to_id), self.tokenizer.vocab_size)
    
    def test_special_tokens(self):
        """Test special token handling"""
        instructions = [["mov rax, rdi"]]
        self.tokenizer.build_vocab(instructions)
        
        # All special tokens should be present
        special_tokens = ["[PAD]", "[CLS]", "[SEP]", "[MASK]", "[UNK]"]
        for token in special_tokens:
            self.assertIn(token, self.tokenizer.token_to_id)
            self.assertIn(token, self.tokenizer.id_to_token.values())
    
    def test_tokenization(self):
        """Test instruction tokenization"""
        instructions = [
            ["mov rax, rdi", "add rax, 0x10"],
            ["push rbp", "mov rbp, rsp"]
        ]
        self.tokenizer.build_vocab(instructions)
        
        # Tokenize a single instruction list
        result = self.tokenizer.tokenize(instructions[0], add_special_tokens=True)
        
        # Verify result structure
        self.assertIn("token_ids", result)
        self.assertIn("attention_mask", result)
        
        # Verify special tokens added
        token_ids = result["token_ids"]
        self.assertEqual(token_ids[0], self.tokenizer.token_to_id["[CLS]"])
        self.assertEqual(token_ids[-1], self.tokenizer.token_to_id["[SEP]"])
        
        # Attention mask should be all 1s for non-padded tokens
        attention_mask = result["attention_mask"]
        self.assertTrue(all(mask == 1 for mask in attention_mask))
    
    def test_padding(self):
        """Test sequence padding"""
        instructions = [["mov rax, rdi"]]
        self.tokenizer.build_vocab(instructions)
        
        # Tokenize with padding
        result = self.tokenizer.tokenize(
            instructions[0],
            add_special_tokens=True,
            max_length=20,
            padding=True
        )
        
        token_ids = result["token_ids"]
        attention_mask = result["attention_mask"]
        
        # Verify padding length
        self.assertEqual(len(token_ids), 20)
        self.assertEqual(len(attention_mask), 20)
        
        # Verify padding tokens
        pad_id = self.tokenizer.token_to_id["[PAD]"]
        self.assertTrue(any(tid == pad_id for tid in token_ids))
        
        # Verify attention mask for padding
        self.assertTrue(all(mask == 0 for tid, mask in zip(token_ids, attention_mask) if tid == pad_id))
    
    def test_unknown_token_handling(self):
        """Test handling of unknown tokens"""
        # Build vocab with limited instructions
        instructions = [["mov rax, rdi"]]
        self.tokenizer.build_vocab(instructions)
        
        # Try tokenizing with unknown instruction
        new_instructions = ["veryunknowninstruction xyz, abc"]
        result = self.tokenizer.tokenize(new_instructions, add_special_tokens=False)
        
        # Unknown tokens should be mapped to [UNK]
        unk_id = self.tokenizer.token_to_id["[UNK]"]
        token_ids = result["token_ids"]
        self.assertTrue(any(tid == unk_id for tid in token_ids))
    
    def test_save_load_vocab(self):
        """Test vocabulary save and load"""
        instructions = [["mov rax, rdi", "add rax, 0x10"]]
        self.tokenizer.build_vocab(instructions)
        
        # Save vocabulary
        vocab_path = Path(self.temp_dir) / "test_vocab.json"
        self.tokenizer.save_vocab(str(vocab_path))
        
        # Verify file created
        self.assertTrue(vocab_path.exists())
        
        # Load vocabulary into new tokenizer
        new_tokenizer = AssemblyTokenizer(vocab_size=100)
        new_tokenizer.load_vocab(str(vocab_path))
        
        # Verify vocabularies match
        self.assertEqual(
            self.tokenizer.token_to_id,
            new_tokenizer.token_to_id
        )
        self.assertEqual(
            self.tokenizer.id_to_token,
            new_tokenizer.id_to_token
        )


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
        metadata_path = process_binary_directory(
            binary_dir=str(self.test_binaries_dir),
            output_dir=str(output_dir),
            vocab_file=None,  # Build vocab from scratch
            timeout=300
        )
        
        # Verify metadata file created
        self.assertTrue(Path(metadata_path).exists())
        
        # Verify preprocessed files created
        self.assertTrue(output_dir.exists())
        json_files = list(output_dir.glob("*.json"))
        self.assertGreater(len(json_files), 0, "Should create at least one preprocessed file")
    
    def test_merged_json_format(self):
        """Test that merged JSON has correct format"""
        if not self.test_binaries_dir.exists():
            self.skipTest("Test binaries directory not found")
        
        # Find a test binary
        test_binary = self.test_binaries_dir / "test_gnn_gcc_O0"
        if not test_binary.exists():
            self.skipTest("Test binary not found")
        
        output_dir = Path(self.temp_dir) / "preprocessed"
        output_dir.mkdir(parents=True)
        
        # Extract CFG with tokenization
        tokenizer = AssemblyTokenizer(vocab_size=100)
        tokenizer.build_vocab([[]])  # Initialize with empty vocab
        
        result = extract_cfg(
            str(test_binary),
            str(output_dir),
            tokenizer=tokenizer
        )
        
        if result["status"] == "success":
            # Load the output file
            output_file = Path(result["output_file"])
            with open(output_file, 'r') as f:
                data = json.load(f)
            
            # Verify top-level structure
            self.assertIn("binary_hash", data)
            self.assertIn("binary_path", data)
            self.assertIn("binary_name", data)
            self.assertIn("functions", data)
            self.assertIn("metadata", data)
            
            # Verify functions structure
            functions = data["functions"]
            self.assertIsInstance(functions, list)
            
            if len(functions) > 0:
                func = functions[0]
                self.assertIn("function_name", func)
                self.assertIn("function_addr", func)
                self.assertIn("nodes", func)
                self.assertIn("edges", func)
                
                # Verify nodes structure
                nodes = func["nodes"]
                if len(nodes) > 0:
                    node = nodes[0]
                    self.assertIn("id", node)
                    self.assertIn("addr", node)
                    self.assertIn("instructions", node)


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
        
        # Step 1: Extract CFG
        cfg_result = extract_single_cfg(str(test_binary), str(output_dir))
        self.assertEqual(cfg_result["status"], "success")
        
        # Step 2: Build tokenizer
        tokenizer = AssemblyTokenizer(vocab_size=200)
        
        # Collect instructions from CFG
        instructions = []
        for node in cfg_result["cfg_data"]["nodes"]:
            node_instructions = [inst["full"] for inst in node["instructions"]]
            if node_instructions:
                instructions.append(node_instructions)
        
        tokenizer.build_vocab(instructions)
        
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


if __name__ == "__main__":
    unittest.main()
