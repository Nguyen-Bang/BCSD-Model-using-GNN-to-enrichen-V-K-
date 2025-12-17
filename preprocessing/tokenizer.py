"""
Assembly Instruction Tokenizer using CLAP-ASM

Wrapper for the pre-trained CLAP-ASM tokenizer from Hugging Face.
This is the only tokenizer used in the BCSD pipeline.

Module: preprocessing.tokenizer
Owner: User Story 1 (US1) - Binary Preprocessing
"""

import logging
from typing import Dict, List

logger = logging.getLogger("bcsd.preprocessing")


class ClapASMTokenizer:
    """
    Wrapper for the CLAP-ASM tokenizer (Hugging Face).
    
    Uses the pre-trained vocabulary from CLAP-ASM project:
    https://huggingface.co/hustcw/clap-asm
    
    This provides a robust, assembly-specific BPE tokenizer with 33k+ vocabulary
    specifically designed for x86/x64 assembly code.
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
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = "[PAD]"
                
            logger.info(f"ClapASMTokenizer loaded from {model_path} (vocab: {len(self.tokenizer.vocab)} tokens)")
            
        except Exception as e:
            logger.error(f"Failed to load CLAP-ASM tokenizer from {model_path}: {e}")
            raise

    def tokenize(self, instruction_sequence: List[str], add_special_tokens: bool = True) -> Dict[str, List[int]]:
        """
        Tokenize instruction sequence.
        
        Args:
            instruction_sequence: List of instruction strings
            add_special_tokens: Whether to add special tokens ([CLS], [SEP])
            
        Returns:
            Dictionary with token_ids, attention_mask, length
        """
        # Join instructions into a single block of text
        # CLAP-ASM likely expects raw assembly code
        text_block = "\n".join(instruction_sequence)
        
        # Normalize: remove commas to match CLAP-ASM's custom logic (prevents [UNK])
        text_block = text_block.replace(",", "")
        
        # Tokenize using HF tokenizer
        encoding = self.tokenizer(
            text_block,
            add_special_tokens=add_special_tokens,
            max_length=self.max_seq_length,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors=None  # Return python lists
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
        
        Args:
            token_ids: List of token IDs
            
        Returns:
            Decoded text string
        """
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)

