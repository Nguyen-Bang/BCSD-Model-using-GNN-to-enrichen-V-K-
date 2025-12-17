"""
Assembly Instruction Tokenizer

Custom domain-specific tokenizer for assembly instructions.
NOT using BERT's WordPiece tokenizer - builds vocabulary from opcodes and operands.

Module: preprocessing.tokenizer
Owner: User Story 1 (US1) - Binary Preprocessing
Task: T016-T021
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from collections import Counter

logger = logging.getLogger("bcsd.preprocessing")


class AssemblyTokenizer:
    """
    Domain-specific tokenizer for x86/x64 assembly instructions.
    
    Builds vocabulary from opcodes, registers, and operands rather than
    using generic NLP tokenization. Vocabulary size up to 5000 tokens (configurable).
    
    Special tokens:
        [PAD] = 0   : Padding token
        [CLS] = 101 : Classification token (start of sequence)
        [SEP] = 102 : Separator token (end of sequence)
        [MASK] = 103: Masking token (for MLM training)
        [UNK] = 104 : Unknown token
    """
    
    def __init__(self, vocab_size: int = 5000, max_seq_length: int = 512):
        """
        Initialize tokenizer.
        
        Args:
            vocab_size: Maximum vocabulary size (default: 5000)
            max_seq_length: Maximum sequence length for padding/truncation
        """
        self.vocab_size = vocab_size
        self.max_seq_length = max_seq_length
        
        # Special tokens (T019)
        self.PAD = 0
        self.CLS = 101
        self.SEP = 102
        self.MASK = 103
        self.UNK = 104
        
        # Vocabulary mappings
        self.vocab = {
            "[PAD]": self.PAD,
            "[CLS]": self.CLS,
            "[SEP]": self.SEP,
            "[MASK]": self.MASK,
            "[UNK]": self.UNK
        }
        self.reverse_vocab = {v: k for k, v in self.vocab.items()}
        
        # Token frequency counter
        self.token_counts = Counter()
        
        logger.info(f"AssemblyTokenizer initialized: vocab_size={vocab_size}, max_len={max_seq_length}")
    
    def _parse_instruction(self, instruction: str) -> List[str]:
        """
        Parse assembly instruction into tokens (T017).
        
        Example:
            "mov rax, rdi" → ["mov", "rax", "rdi"]
            "call 0x401000" → ["call", "0x401000"]
            "lea rax, [rip + 0xe1f]" → ["lea", "rax", "[", "rip", "+", "0xe1f", "]"]
        
        Args:
            instruction: Assembly instruction string
            
        Returns:
            List of tokens
        """
        if not instruction or not instruction.strip():
            return []
        
        # Split by common delimiters but preserve them
        tokens = []
        current = ""
        
        for char in instruction:
            if char in [' ', ',', '[', ']', '+', '-', '*', ':', '(', ')']:
                if current:
                    tokens.append(current)
                    current = ""
                if char.strip():  # Skip pure whitespace
                    tokens.append(char)
            else:
                current += char
        
        if current:
            tokens.append(current)
        
        return tokens
    
    def build_vocab(self, instruction_sequences: List[List[str]]) -> None:
        """
        Build vocabulary from collection of instruction sequences (T018).
        
        Collects all unique tokens (opcodes, registers, immediates) and assigns
        IDs based on frequency. Most common tokens get lower IDs.
        
        Args:
            instruction_sequences: List of instruction sequences (each sequence is list of instruction strings)
        """
        logger.info(f"Building vocabulary from {len(instruction_sequences)} sequences")
        
        # Count all tokens
        for sequence in instruction_sequences:
            for instruction in sequence:
                tokens = self._parse_instruction(instruction)
                self.token_counts.update(tokens)
        
        # Get most common tokens (excluding special tokens)
        # Reserve space for special tokens
        available_slots = self.vocab_size - len(self.vocab)
        most_common = self.token_counts.most_common(available_slots)
        
        # Assign IDs starting after special tokens (T020)
        next_id = 105  # After [UNK]=104
        for token, count in most_common:
            if token not in self.vocab:
                self.vocab[token] = next_id
                self.reverse_vocab[next_id] = token
                next_id += 1
        
        actual_vocab_size = len(self.vocab)
        logger.info(f"Vocabulary built: {actual_vocab_size} tokens (target: {self.vocab_size})")
        
        if actual_vocab_size < self.vocab_size:
            logger.info(f"Actual vocab size ({actual_vocab_size}) < target ({self.vocab_size}) - this is normal if training data has fewer unique tokens")
    
    def tokenize(self, instruction_sequence: List[str], add_special_tokens: bool = True) -> Dict[str, List[int]]:
        """
        Tokenize instruction sequence to token IDs (T021).
        
        Args:
            instruction_sequence: List of instruction strings
            add_special_tokens: Whether to add [CLS] and [SEP] tokens
            
        Returns:
            Dictionary with:
                - token_ids: List of token IDs
                - attention_mask: Mask (1 for real tokens, 0 for padding)
                - length: Actual sequence length before padding
        """
        # Parse all instructions
        all_tokens = []
        for instruction in instruction_sequence:
            tokens = self._parse_instruction(instruction)
            all_tokens.extend(tokens)
        
        # Convert tokens to IDs
        token_ids = []
        
        if add_special_tokens:
            token_ids.append(self.CLS)
        
        for token in all_tokens:
            token_ids.append(self.vocab.get(token, self.UNK))
        
        if add_special_tokens:
            token_ids.append(self.SEP)
        
        # Truncate if too long
        if len(token_ids) > self.max_seq_length:
            token_ids = token_ids[:self.max_seq_length]
            if add_special_tokens:
                token_ids[-1] = self.SEP  # Ensure SEP at end
        
        # Create attention mask (1 for real tokens)
        actual_length = len(token_ids)
        attention_mask = [1] * actual_length
        
        # Pad to max length
        padding_length = self.max_seq_length - actual_length
        if padding_length > 0:
            token_ids.extend([self.PAD] * padding_length)
            attention_mask.extend([0] * padding_length)
        
        return {
            "token_ids": token_ids,
            "attention_mask": attention_mask,
            "length": actual_length
        }
    
    def save_vocab(self, vocab_file: str) -> None:
        """
        Save vocabulary to JSON file for reproducibility.
        
        Args:
            vocab_file: Path to save vocabulary JSON
        """
        vocab_data = {
            "vocab_size": len(self.vocab),
            "max_seq_length": self.max_seq_length,
            "vocab": self.vocab,
            "token_counts": dict(self.token_counts.most_common(100))  # Save top 100 for analysis
        }
        
        with open(vocab_file, 'w') as f:
            json.dump(vocab_data, f, indent=2)
        
        logger.info(f"Vocabulary saved to {vocab_file}")
    
    def load_vocab(self, vocab_file: str) -> None:
        """
        Load vocabulary from JSON file.
        
        Args:
            vocab_file: Path to vocabulary JSON
        """
        with open(vocab_file, 'r') as f:
            vocab_data = json.load(f)
        
        self.vocab = {k: int(v) for k, v in vocab_data["vocab"].items()}
        self.reverse_vocab = {v: k for k, v in self.vocab.items()}
        self.max_seq_length = vocab_data["max_seq_length"]
        
        logger.info(f"Vocabulary loaded from {vocab_file}: {len(self.vocab)} tokens")
    
    def decode(self, token_ids: List[int]) -> List[str]:
        """
        Decode token IDs back to tokens.
        
        Args:
            token_ids: List of token IDs
            
        Returns:
            List of token strings
        """
        return [self.reverse_vocab.get(tid, "[UNK]") for tid in token_ids]


class ClapASMTokenizer:
    """
    Wrapper for the CLAP-ASM tokenizer (Hugging Face).
    
    Uses the pre-trained vocabulary from CLAP-ASM project:
    https://huggingface.co/hustcw/clap-asm
    
    This provides a more robust, assembly-specific BPE tokenizer compared
    to the simple custom implementation.
    """
    
    def __init__(self, model_path: str = "preprocessing/clap_asm_tokenizer", max_seq_length: int = 512):
        """
        Initialize CLAP-ASM tokenizer.
        
        Args:
            model_path: Path to directory containing tokenizer.json and vocab.txt
            max_seq_length: Maximum sequence length
        """
        from transformers import PreTrainedTokenizerFast
        
        self.max_seq_length = max_seq_length
        try:
            self.tokenizer = PreTrainedTokenizerFast.from_pretrained(
                model_path,
                model_max_length=max_seq_length,
                padding_side="right",
                truncation_side="right"
            )
            # Ensure pad token is set (CLAP-ASM usually uses [PAD] or similar)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = "[PAD]"
                
            logger.info(f"ClapASMTokenizer loaded from {model_path}")
            
        except Exception as e:
            logger.error(f"Failed to load CLAP-ASM tokenizer from {model_path}: {e}")
            raise

    def tokenize(self, instruction_sequence: List[str], add_special_tokens: bool = True) -> Dict[str, List[int]]:
        """
        Tokenize instruction sequence.
        
        The CLAP-ASM tokenizer expects a single text string. We join the instructions
        (e.g., with newlines or spaces) before tokenizing.
        
        Args:
            instruction_sequence: List of instruction strings
            add_special_tokens: Whether to add special tokens ([CLS], [SEP])
            
        Returns:
            Dictionary with token_ids, attention_mask, length
        """
        # Join instructions into a single block of text
        # CLAP-ASM likely expects raw assembly code
        text_block = "\n".join(instruction_sequence)
        
        # Tokenize using HF tokenizer
        encoding = self.tokenizer(
            text_block,
            add_special_tokens=add_special_tokens,
            max_length=self.max_seq_length,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors=None # Return python lists
        )
        
        # Calculate actual length (sum of attention mask)
        actual_length = sum(encoding["attention_mask"])
        
        return {
            "token_ids": encoding["input_ids"],
            "attention_mask": encoding["attention_mask"],
            "length": actual_length
        }
    
    def decode(self, token_ids: List[int]) -> str:
        """
        Decode token IDs back to text.
        """
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)

