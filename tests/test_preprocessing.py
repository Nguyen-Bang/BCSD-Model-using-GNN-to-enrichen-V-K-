"""Hermetic tests for the staged preprocessing pipeline."""

import importlib
import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]

from preprocessing.I_ida_extraction import batch_extract, cfg_extractor
from preprocessing.III_tokenization import batch_tokenize, tokenize_cfg
from preprocessing.III_tokenization.tokenizer import ClapASMTokenizer


class _HashValues(list):
    def tolist(self):
        return list(self)


class _MinHash:
    """Small stand-in that keeps enrichment tests independent of datasketch."""

    def __init__(self, num_perm):
        self.hashvalues = _HashValues(range(num_perm))

    def update(self, _value):
        pass


_datasketch = types.ModuleType("datasketch")
_datasketch.MinHash = _MinHash
with patch.dict(sys.modules, {"datasketch": _datasketch}):
    batch_enrich = importlib.import_module("preprocessing.II_static_enrichment.batch_enrich")
    enrich_cfg = importlib.import_module("preprocessing.II_static_enrichment.enrich_cfg")


def _raw_cfg():
    return {
        "binary_name": "sample",
        "binary_path": "/input/sample",
        "functions": [
            {
                "function_name": "add_one",
                "function_addr": "0x1000",
                "nodes": [
                    {
                        "id": 0,
                        "size": 5,
                        "instructions": {"1": "mov rax, rdi", "2": "add rax, 1"},
                    },
                    {
                        "id": 1,
                        "size": 1,
                        "instructions": {"3": "ret"},
                    },
                ],
                "edges": [[0, 1, "fallthrough"]],
                "rebased_instructions": {
                    "1": "mov rax, rdi",
                    "2": "add rax, 1",
                    "3": "ret",
                },
            }
        ],
        "metadata": {
            "total_functions": 1,
            "total_nodes": 2,
            "total_edges": 1,
        },
    }


class TestBatchExtraction(unittest.TestCase):
    def test_job_is_isolated_and_publishes_valid_json(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_dir = root / "input"
            output_dir = root / "output"
            input_dir.mkdir()
            binary = input_dir / "program"
            binary.write_bytes(b"binary")
            source_sidecar = binary.with_suffix(".i64")
            source_sidecar.write_text("keep", encoding="utf-8")
            observed = {}

            def fake_run(command, **kwargs):
                work_dir = Path(kwargs["cwd"])
                observed["command"] = command
                observed["cwd"] = work_dir
                self.assertNotEqual(work_dir, binary.parent)
                database_path = next(a for a in command if a.startswith("-o"))[2:]
                self.assertTrue(database_path.startswith(str(work_dir)))

                script_argument = next(a for a in command if a.startswith("-S"))
                _, output_name = shlex.split(script_argument[2:])
                Path(output_name).write_text(json.dumps(_raw_cfg()), encoding="utf-8")
                return subprocess.CompletedProcess(command, 0, "", "")

            with patch.object(batch_extract.subprocess, "run", side_effect=fake_run):
                failures = batch_extract.batch_extract(
                    input_dir,
                    output_dir,
                    ida_path="fake-ida",
                    timeout=1,
                )

            self.assertEqual(failures, 0)
            self.assertEqual(source_sidecar.read_text(encoding="utf-8"), "keep")
            output = output_dir / "program_cfg.json"
            self.assertEqual(json.loads(output.read_text(encoding="utf-8")), _raw_cfg())
            self.assertFalse(observed["cwd"].exists())

    def test_nonzero_ida_status_does_not_replace_existing_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_dir = root / "input"
            output_dir = root / "output"
            input_dir.mkdir()
            output_dir.mkdir()
            (input_dir / "program").write_bytes(b"binary")
            output = output_dir / "program_cfg.json"
            output.write_text("previous", encoding="utf-8")
            result = subprocess.CompletedProcess([], 7, "", "IDA failed")

            with patch.object(batch_extract.subprocess, "run", return_value=result):
                failures = batch_extract.batch_extract(input_dir, output_dir, "fake-ida")

            self.assertEqual(failures, 1)
            self.assertEqual(output.read_text(encoding="utf-8"), "previous")

    def test_invalid_json_is_reported_as_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_dir = root / "input"
            input_dir.mkdir()
            (input_dir / "program").write_bytes(b"binary")

            def fake_run(command, **_kwargs):
                script_argument = next(a for a in command if a.startswith("-S"))
                _, output_name = shlex.split(script_argument[2:])
                Path(output_name).write_text("not JSON", encoding="utf-8")
                return subprocess.CompletedProcess(command, 0, "", "")

            with patch.object(batch_extract.subprocess, "run", side_effect=fake_run):
                failures = batch_extract.batch_extract(input_dir, root / "output", "fake-ida")

            self.assertEqual(failures, 1)
            self.assertFalse((root / "output" / "program_cfg.json").exists())


class TestCFGExtractionLogic(unittest.TestCase):
    def test_conditional_branch_distinguishes_target_and_fallthrough(self):
        fake_idc = SimpleNamespace(
            BADADDR=-1,
            prev_head=lambda _end: 0x1004,
            print_insn_mnem=lambda _address: "jnz",
            get_operand_value=lambda _address, _operand: 0x2000,
        )
        block = SimpleNamespace(end_ea=0x1005)

        with patch.object(cfg_extractor, "idc", fake_idc, create=True):
            self.assertEqual(
                cfg_extractor.get_edge_type(block, SimpleNamespace(start_ea=0x2000)),
                "conditional_jump",
            )
            self.assertEqual(
                cfg_extractor.get_edge_type(block, SimpleNamespace(start_ea=0x1005)),
                "fallthrough",
            )


class TestStaticEnrichment(unittest.TestCase):
    def test_enrichment_preserves_cfg_and_adds_expected_features(self):
        data = _raw_cfg()
        result = enrich_cfg.enrich_cfg_data(data)
        function = result["functions"][0]

        self.assertIs(result, data)
        self.assertEqual(function["edges"], [[0, 1, "fallthrough"]])
        self.assertEqual(function["node_count"], 2)
        self.assertEqual(function["edge_count"], 1)
        self.assertEqual(function["function_size"], 6)
        self.assertEqual(function["function_cyclomatic_complexity"], 1)
        self.assertEqual(len(function["opcode_minhash"]), 128)
        self.assertEqual(function["nodes"][0]["instruction_count"], 2)
        self.assertEqual(function["nodes"][0]["arithmetic_density"], 0.5)
        self.assertTrue(result["metadata"]["enriched"])

    def test_batch_cli_returns_nonzero_when_an_item_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_dir = root / "input"
            input_dir.mkdir()
            (input_dir / "broken.json").write_text("broken", encoding="utf-8")

            self.assertEqual(
                batch_enrich.main([str(input_dir), str(root / "output")]),
                1,
            )


class _FastTokenizer:
    pad_token = "<pad>"
    vocab_size = 33_555
    unk_token_id = 4

    @classmethod
    def from_pretrained(cls, model_path, **_kwargs):
        cls.loaded_path = Path(model_path)
        return cls()

    def tokenize(self, text, max_length, **_kwargs):
        return text.split()[:max_length]

    def convert_tokens_to_ids(self, tokens):
        return [sum(map(ord, token)) % 1000 for token in tokens]

    def decode(self, token_ids, **_kwargs):
        return " ".join(map(str, token_ids))


class _FunctionTokenizer:
    max_seq_length = 1024
    vocab_size = 33_555

    def tokenize(self, instructions):
        self.instructions = instructions
        return {
            "token_ids": [10, 11],
            "attention_mask": [1, 1],
            "token_type_ids": [101, 102],
            "length": 2,
        }


class TestTokenization(unittest.TestCase):
    def test_bundled_model_is_wordpiece(self):
        tokenizer_json = REPOSITORY_ROOT / "preprocessing/clap_asm_tokenizer/tokenizer.json"
        with tokenizer_json.open(encoding="utf-8") as tokenizer_file:
            data = json.load(tokenizer_file)
        self.assertEqual(data["model"]["type"], "WordPiece")

    def test_default_assets_are_resolved_independently_of_cwd(self):
        transformers = types.ModuleType("transformers")
        transformers.PreTrainedTokenizerFast = _FastTokenizer

        with tempfile.TemporaryDirectory() as directory:
            previous_directory = Path.cwd()
            try:
                os.chdir(directory)
                with patch.dict(sys.modules, {"transformers": transformers}):
                    tokenizer = ClapASMTokenizer(max_seq_length=8)
            finally:
                os.chdir(previous_directory)

        self.assertTrue(_FastTokenizer.loaded_path.is_absolute())
        self.assertEqual(_FastTokenizer.loaded_path.name, "clap_asm_tokenizer")
        tokens = tokenizer.tokenize({"1": "mov rax, rdi", "2": "ret"})
        self.assertEqual(tokens["length"], len(tokens["token_ids"]))
        self.assertEqual(tokens["attention_mask"], [1] * tokens["length"])
        self.assertEqual(len(tokens["token_type_ids"]), tokens["length"])

    def test_each_instruction_is_limited_to_twenty_subwords(self):
        tokenizer = ClapASMTokenizer(max_seq_length=1024)
        result = tokenizer.tokenize({"1": "abcdefghijk " * 100})
        self.assertEqual(result["length"], 20)

    def test_function_level_schema_is_preserved(self):
        data = _raw_cfg()
        tokenizer = _FunctionTokenizer()
        result = tokenize_cfg.tokenize_cfg_data(data, tokenizer)
        function = result["functions"][0]

        self.assertEqual(tokenizer.instructions, function["rebased_instructions"])
        self.assertEqual(function["token_ids"], [10, 11])
        self.assertEqual(function["attention_mask"], [1, 1])
        self.assertEqual(function["token_type_ids"], [101, 102])
        self.assertEqual(function["token_length"], 2)
        self.assertNotIn("token_ids", function["nodes"][0])
        self.assertTrue(result["metadata"]["tokenized"])

    def test_batch_cli_returns_nonzero_when_an_item_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_dir = root / "input"
            input_dir.mkdir()
            (input_dir / "broken.json").write_text("broken", encoding="utf-8")

            with patch.object(
                batch_tokenize, "ClapASMTokenizer", return_value=_FunctionTokenizer()
            ):
                self.assertEqual(
                    batch_tokenize.main([str(input_dir), str(root / "output")]),
                    1,
                )


@unittest.skipUnless(
    os.environ.get("RUN_IDA_TESTS") == "1",
    "set RUN_IDA_TESTS=1 to run the IDA integration test",
)
class TestIDAIntegration(unittest.TestCase):
    def test_compiled_fixture(self):
        ida_path = os.environ.get("IDA_PATH", "ida64")
        if shutil.which(ida_path) is None:
            self.skipTest(f"IDA executable not found: {ida_path}")
        compiler = shutil.which(os.environ.get("CC", "cc"))
        if compiler is None:
            self.skipTest("C compiler not found")

        fixture_source = REPOSITORY_ROOT / "test_binaries/test_gnn.c"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_dir = root / "input"
            input_dir.mkdir()
            subprocess.run(
                [
                    compiler,
                    "-O0",
                    str(fixture_source),
                    "-o",
                    str(input_dir / "test_gnn"),
                ],
                check=True,
                capture_output=True,
            )
            failures = batch_extract.batch_extract(input_dir, root / "output", ida_path)
            self.assertEqual(failures, 0)


if __name__ == "__main__":
    unittest.main()
