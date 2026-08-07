"""
Assembly Instruction Tokenizer using CLAP-ASM vocabulary.

Matches CLAP's AsmTokenizer.tokenize_function() behavior:
- Instruction-level tokenization (max 20 tokens per instruction)
- token_type_ids using INSTR{N} vocabulary entries (shared with word embeddings)
- Function-level truncation at max_seq_length (default 1024)
- No special tokens ([CLS]/[SEP]) -- CLAP uses mean pooling, not [CLS]
- Commas stripped from instructions (CLAP convention)

The CLAP vocabulary contains INSTR1-INSTR1024 as special tokens for
instruction-index encoding. token_type_ids use these to give the model
instruction-boundary awareness within the flat token sequence.

Module: preprocessing.III_tokenization.tokenizer
"""

import logging
from pathlib import Path

logger = logging.getLogger("bcsd.preprocessing")

MAX_TOKENS_PER_INSTRUCTION = 20
PAD_TOKEN_ID = 1  # CLAP uses <pad> = 1


class ClapASMTokenizer:
    """
    CLAP-compatible assembly tokenizer with instruction-level token_type_ids.

    Uses the pre-trained WordPiece vocabulary from CLAP-ASM (33,555 tokens)
    but replicates the instruction-level tokenization logic from
    CLAP's AsmTokenizer.tokenize_function().
    """

    def __init__(
        self,
        model_path: str | Path | None = None,
        max_seq_length: int = 1024,
    ):
        from transformers import PreTrainedTokenizerFast

        if model_path is None:
            model_path = Path(__file__).resolve().parents[1] / "clap_asm_tokenizer"

        self.max_seq_length = max_seq_length
        try:
            self._tokenizer = PreTrainedTokenizerFast.from_pretrained(
                model_path,
                model_max_length=max_seq_length,
                padding_side="right",
                truncation_side="right",
            )
            if self._tokenizer.pad_token is None:
                self._tokenizer.pad_token = "<pad>"

            logger.info(
                f"ClapASMTokenizer loaded from {model_path} "
                f"(vocab: {self._tokenizer.vocab_size} tokens, "
                f"max_seq_length: {max_seq_length})"
            )
        except Exception as e:
            logger.error(f"Failed to load CLAP-ASM tokenizer from {model_path}: {e}")
            raise

    def tokenize(self, rebased_instructions: dict[str, str]) -> dict[str, list[int]]:
        """
        Tokenize a function's rebased_instructions dict.

        Matches CLAP AsmTokenizer.tokenize_function() behavior:
        - Iterates instruction dict in insertion order
        - Each instruction tokenized with at most 20 WordPiece tokens
        - Commas stripped (CLAP convention)
        - No special tokens added
        - token_type_ids built from INSTR{key} vocab entries
        - Truncates total sequence at max_seq_length

        Args:
            rebased_instructions: Dict mapping string indices ("1", "2", ...)
                to assembly instruction strings. Must be in program order.

        Returns:
            Dict with token_ids, attention_mask, token_type_ids, length.
            No padding -- padding is deferred to collate.
        """
        if not rebased_instructions:
            return {
                "token_ids": [],
                "attention_mask": [],
                "token_type_ids": [],
                "length": 0,
            }

        all_tokens: list[str] = []
        all_instr_tokens: list[str] = []
        total_len = 0

        for key, instruction in rebased_instructions.items():
            inst_tokens = self._tokenizer.tokenize(
                instruction.replace(",", ""),
                max_length=MAX_TOKENS_PER_INSTRUCTION,
                truncation=True,
                add_special_tokens=False,
            )[:MAX_TOKENS_PER_INSTRUCTION]
            instr_marker = "INSTR" + key
            all_tokens.extend(inst_tokens)
            all_instr_tokens.extend([instr_marker] * len(inst_tokens))
            total_len += len(inst_tokens)

            if total_len >= self.max_seq_length:
                all_tokens = all_tokens[: self.max_seq_length]
                all_instr_tokens = all_instr_tokens[: self.max_seq_length]
                break

        token_ids = self._tokenizer.convert_tokens_to_ids(all_tokens)
        type_ids = self._tokenizer.convert_tokens_to_ids(all_instr_tokens)
        length = len(token_ids)

        return {
            "token_ids": token_ids,
            "attention_mask": [1] * length,
            "token_type_ids": type_ids,
            "length": length,
        }

    def decode(self, token_ids: list[int]) -> str:
        """Decode token IDs back to text."""
        return self._tokenizer.decode(token_ids, skip_special_tokens=True)

    @property
    def vocab_size(self) -> int:
        return self._tokenizer.vocab_size

    @property
    def pad_token_id(self) -> int:
        return PAD_TOKEN_ID
