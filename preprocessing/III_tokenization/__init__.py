"""
Tokenization subpackage.

Wraps the CLAP-ASM WordPiece tokenizer and provides single-file and batch
tokenization of enriched CFG JSON files (function-level tokenization).

Files:
    tokenizer.py       -- ClapASMTokenizer class
    tokenize_cfg.py    -- Single-file CLI: tokenize_cfg.py <input_file> [output_file]
    batch_tokenize.py  -- Batch CLI: batch_tokenize.py <input_dir> <output_dir>
"""

from .tokenizer import ClapASMTokenizer as ClapASMTokenizer

__all__ = ["ClapASMTokenizer"]
