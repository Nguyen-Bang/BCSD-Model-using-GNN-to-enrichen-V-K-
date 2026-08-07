#!/usr/bin/env python3
"""
Batch tokenization of enriched CFG JSON files.

Recursively processes all JSON files in input_dir, tokenizes each
function's instruction sequence with CLAP-compatible instruction-level
tokenization (token_type_ids + configurable max length), and writes
output to output_dir preserving subdirectory structure.

Usage:
    python -m preprocessing.III_tokenization.batch_tokenize <input_dir> <output_dir> [--max-seq-length 1024]

Example:
    python -m preprocessing.III_tokenization.batch_tokenize data/II_static_enriched data/III_tokenized

Module: preprocessing.III_tokenization.batch_tokenize
"""

import argparse
import sys
from pathlib import Path

from .tokenize_cfg import tokenize_cfg_file
from .tokenizer import ClapASMTokenizer


def batch_tokenize(input_dir: str, output_dir: str, max_seq_length: int = 1024) -> int:
    """
    Tokenize all JSON files in input_dir, writing to output_dir.

    Recurses into subdirectories and preserves folder structure.
    Initializes the tokenizer once and reuses it for all files.
    Output filenames get '_tokenized' appended to the stem.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    json_files = sorted(input_dir.rglob("*.json"))

    if not json_files:
        print(f"No JSON files found in {input_dir}")
        return 0

    print(f"Tokenizing {len(json_files)} files from {input_dir} -> {output_dir}")
    print(f"Max sequence length: {max_seq_length}")

    tokenizer = ClapASMTokenizer(max_seq_length=max_seq_length)
    print(f"Tokenizer loaded (vocab: {tokenizer.vocab_size} tokens)\n")

    successful = 0
    failed = 0

    for i, json_path in enumerate(json_files, 1):
        rel_path = json_path.relative_to(input_dir)
        out_path = output_dir / rel_path.parent / f"{json_path.stem}_tokenized.json"

        print(f"[{i}/{len(json_files)}] {rel_path}")
        try:
            tokenize_cfg_file(str(json_path), str(out_path), tokenizer=tokenizer)
            successful += 1
        except Exception as e:
            print(f"  Error: {e}")
            failed += 1

    print(f"\nDone. {successful} succeeded, {failed} failed.")
    return failed


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Batch tokenize enriched CFG JSON files")
    parser.add_argument(
        "input_dir",
        help="Input directory containing enriched CFG JSONs (searched recursively)",
    )
    parser.add_argument("output_dir", help="Output directory for tokenized JSON files")
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=1024,
        help="Maximum token sequence length per function (default: 1024)",
    )
    args = parser.parse_args(argv)

    return min(
        batch_tokenize(args.input_dir, args.output_dir, args.max_seq_length),
        1,
    )


if __name__ == "__main__":
    sys.exit(main())
