"""
Single-file CFG tokenization.

Loads an enriched CFG JSON and tokenizes each function's rebased_instructions
using CLAP-compatible instruction-level tokenization (with token_type_ids).

Usage:
    python -m preprocessing.III_tokenization.tokenize_cfg <input_file> [output_file]

If output_file is omitted, appends '_tokenized' to the input filename:
    input_enriched.json -> input_enriched_tokenized.json

Module: preprocessing.III_tokenization.tokenize_cfg
"""

import json
import sys
from pathlib import Path
from typing import Any

from .tokenizer import ClapASMTokenizer


def tokenize_cfg_data(data: dict[str, Any], tokenizer: ClapASMTokenizer) -> dict[str, Any]:
    """
    Tokenize all functions in a CFG dict (function-level tokenization).

    Passes each function's rebased_instructions dict directly to the
    CLAP-compatible tokenizer, producing token_ids, attention_mask,
    and token_type_ids per function.

    Args:
        data: Enriched CFG dict with functions[].rebased_instructions.
        tokenizer: Initialized ClapASMTokenizer instance.

    Returns:
        The same dict, mutated in-place with tokenization fields per function.
    """
    for function in data.get("functions", []):
        rebased = function.get("rebased_instructions", {})

        if rebased:
            tokens = tokenizer.tokenize(rebased)
            function["token_ids"] = tokens["token_ids"]
            function["attention_mask"] = tokens["attention_mask"]
            function["token_type_ids"] = tokens["token_type_ids"]
            function["token_length"] = tokens["length"]
        else:
            function["token_ids"] = []
            function["attention_mask"] = []
            function["token_type_ids"] = []
            function["token_length"] = 0

    if "metadata" not in data:
        data["metadata"] = {}
    data["metadata"]["tokenized"] = True
    data["metadata"]["max_seq_length"] = tokenizer.max_seq_length

    return data


def default_output_path(input_path: str) -> str:
    """Derive default output path by appending '_tokenized' before extension."""
    p = Path(input_path)
    return str(p.with_stem(p.stem + "_tokenized"))


def tokenize_cfg_file(
    input_path: str,
    output_path: str | None = None,
    tokenizer: ClapASMTokenizer | None = None,
) -> str:
    """
    Tokenize a single enriched CFG JSON file and write the result.

    Args:
        input_path: Path to enriched CFG JSON.
        output_path: Path to write tokenized JSON. If None, appends '_tokenized'.
        tokenizer: ClapASMTokenizer instance. Created if None.

    Returns:
        The output file path that was written.
    """
    if output_path is None:
        output_path = default_output_path(input_path)

    if tokenizer is None:
        tokenizer = ClapASMTokenizer()

    with open(input_path) as f:
        data = json.load(f)

    tokenize_cfg_data(data, tokenizer)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    func_count = len(data.get("functions", []))
    print(f"  Tokenized {func_count} functions -> {output_path}")
    return output_path


def main():
    if len(sys.argv) < 2:
        print(
            "Usage: python -m preprocessing.III_tokenization.tokenize_cfg "
            "<input_file> [output_file]"
        )
        print("  If output_file is omitted, appends '_tokenized' to input filename.")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None

    if not Path(input_path).exists():
        print(f"Error: File not found: {input_path}")
        sys.exit(1)

    tokenize_cfg_file(input_path, output_path)


if __name__ == "__main__":
    main()
