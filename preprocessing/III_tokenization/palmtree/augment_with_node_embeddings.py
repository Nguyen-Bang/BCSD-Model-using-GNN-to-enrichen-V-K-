"""Build a self-contained reduced dataset with inline PalmTree node embeddings."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import subprocess
import sys
import types
import uuid
from collections import OrderedDict
from collections.abc import Callable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

PALMTREE_DIM = 128
PALMTREE_SCHEMA_VERSION = 1
PALMTREE_REPOSITORY = "https://github.com/palmtreemodel/PalmTree"


def _checkout_state(checkout: Path) -> tuple[str | None, bool | None]:
    try:
        revision = subprocess.check_output(
            ["git", "-C", str(checkout), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "-C", str(checkout), "status", "--porcelain"],
                stderr=subprocess.DEVNULL,
                text=True,
            ).strip()
        )
    except (OSError, subprocess.CalledProcessError):
        return None, None
    return revision, dirty


class EmbeddingCache:
    """A small LRU cache that bounds retained instruction embeddings."""

    def __init__(self, max_entries: int) -> None:
        if max_entries < 0:
            raise ValueError("cache_entries must be nonnegative")
        self.max_entries = max_entries
        self._values: OrderedDict[str, torch.Tensor] = OrderedDict()

    def get(self, key: str) -> torch.Tensor | None:
        value = self._values.get(key)
        if value is not None:
            self._values.move_to_end(key)
        return value

    def put(self, key: str, value: torch.Tensor) -> None:
        if self.max_entries == 0:
            return
        self._values[key] = value
        self._values.move_to_end(key)
        while len(self._values) > self.max_entries:
            self._values.popitem(last=False)

    def __len__(self) -> int:
        return len(self._values)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as error:
        raise ValueError(f"Cannot read {path}: {error}") from error
    return digest.hexdigest()


def _load_pt(path: Path) -> dict[str, Any]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as error:
        raise ValueError(f"Cannot load {path}: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a mapping in {path}")
    return payload


def _load_json(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"Cannot load {path}: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a mapping in {path}")
    return payload


def _functions(payload: Mapping[str, Any], context: Path) -> list[dict[str, Any]]:
    functions = payload.get("functions")
    if not isinstance(functions, list) or not all(
        isinstance(function, dict) for function in functions
    ):
        raise ValueError(f"Expected a list of function mappings in {context}")
    return functions


def _tokenized_functions(payload: Mapping[str, Any], context: Path) -> list[dict[str, Any]]:
    return [
        function
        for function in _functions(payload, context)
        if function.get("token_ids") is not None and len(function["token_ids"]) > 0
    ]


def _nodes(function: Mapping[str, Any], context: str) -> list[dict[str, Any]]:
    nodes = function.get("nodes", [])
    if not isinstance(nodes, list) or not all(isinstance(node, dict) for node in nodes):
        raise ValueError(f"Expected a list of node mappings in {context}")
    declared = function.get("node_count", len(nodes))
    try:
        declared_count = int(declared)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Invalid node_count in {context}: {declared!r}") from error
    if declared_count != len(nodes):
        raise ValueError(
            f"Node count mismatch in {context}: declared {declared_count}, got {len(nodes)}"
        )
    return nodes


def _validate_alignment(
    base: Mapping[str, Any],
    tokenized: Mapping[str, Any],
    base_path: Path,
    tokenized_path: Path,
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    base_functions = _functions(base, base_path)
    source_functions = _tokenized_functions(tokenized, tokenized_path)
    if len(base_functions) != len(source_functions):
        raise ValueError(
            f"Function count mismatch for {base_path}: base has {len(base_functions)}, "
            f"tokenized source has {len(source_functions)} nonempty functions"
        )

    aligned = []
    for function_index, (base_function, source_function) in enumerate(
        zip(base_functions, source_functions, strict=True)
    ):
        context = f"{base_path}, function {function_index}"
        base_name = base_function.get("function_name")
        source_name = source_function.get("function_name")
        if base_name != source_name:
            raise ValueError(f"Function mismatch in {context}: {base_name!r} != {source_name!r}")
        base_nodes = _nodes(base_function, context)
        source_nodes = _nodes(source_function, f"{tokenized_path}, function {function_index}")
        if len(base_nodes) != len(source_nodes):
            raise ValueError(
                f"Node count mismatch in {context}: base has {len(base_nodes)}, "
                f"tokenized source has {len(source_nodes)}"
            )
        for node_index, (base_node, source_node) in enumerate(
            zip(base_nodes, source_nodes, strict=True)
        ):
            if base_node.get("id") != source_node.get("id"):
                raise ValueError(
                    f"Node mismatch in {context}, node {node_index}: "
                    f"{base_node.get('id')!r} != {source_node.get('id')!r}"
                )
        aligned.append((base_function, source_function))
    return aligned


def _instruction_texts(node: Mapping[str, Any], context: str) -> list[str]:
    instructions = node.get("instructions", {})
    if not isinstance(instructions, dict):
        raise ValueError(f"Expected an instruction mapping in {context}")

    def sort_key(item: tuple[Any, Any]) -> tuple[int, int | str]:
        key = item[0]
        try:
            return (0, int(key))
        except (TypeError, ValueError):
            return (1, str(key))

    texts = []
    for _, instruction in sorted(instructions.items(), key=sort_key):
        if not isinstance(instruction, str):
            raise ValueError(f"Expected instruction text in {context}")
        texts.append(instruction)
    return texts


def _as_embedding_batch(payload: Any, expected_rows: int) -> torch.Tensor:
    try:
        embeddings = torch.as_tensor(payload, dtype=torch.float32, device="cpu")
    except (TypeError, ValueError, RuntimeError) as error:
        raise ValueError(f"PalmTree returned invalid embeddings: {error}") from error
    if tuple(embeddings.shape) != (expected_rows, PALMTREE_DIM):
        raise ValueError(
            f"PalmTree returned shape {tuple(embeddings.shape)}, "
            f"expected ({expected_rows}, {PALMTREE_DIM})"
        )
    if not torch.isfinite(embeddings).all():
        raise ValueError("PalmTree returned non-finite embeddings")
    return embeddings


def _encode_missing(
    texts: Sequence[str],
    encoder: Any,
    cache: EmbeddingCache,
) -> dict[str, torch.Tensor]:
    missing = list(dict.fromkeys(text for text in texts if cache.get(text) is None))
    if not missing:
        return {}
    with torch.inference_mode():
        payload = encoder.encode(missing, output_option="lst")
    encoded = _as_embedding_batch(payload, len(missing))
    result = {}
    for text, embedding in zip(missing, encoded, strict=True):
        vector = embedding.detach().clone()
        cache.put(text, vector)
        result[text] = vector
    return result


def _node_embedding(
    instructions: Sequence[str],
    normalize: Callable[[str], str],
    encoder: Any,
    cache: EmbeddingCache,
    batch_size: int,
) -> torch.Tensor:
    if not instructions:
        return torch.zeros(PALMTREE_DIM, dtype=torch.float32)

    total = torch.zeros(PALMTREE_DIM, dtype=torch.float32)
    count = 0
    for start in range(0, len(instructions), batch_size):
        normalized = []
        for instruction in instructions[start : start + batch_size]:
            text = normalize(instruction)
            if not isinstance(text, str):
                raise ValueError("PalmTree normalizer must return text")
            if text:
                normalized.append(text)
        fresh = _encode_missing(normalized, encoder, cache)
        for text in normalized:
            embedding = cache.get(text)
            if embedding is None:
                embedding = fresh[text]
            total += embedding
            count += 1
    return total / count if count else total


def _payload_equal(left: Any, right: Any) -> bool:
    if isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor):
        return (
            isinstance(left, torch.Tensor)
            and isinstance(right, torch.Tensor)
            and torch.equal(left, right)
        )
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        return left.keys() == right.keys() and all(
            _payload_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return len(left) == len(right) and all(
            _payload_equal(a, b) for a, b in zip(left, right, strict=True)
        )
    return left == right


def validate_output(
    output: Mapping[str, Any],
    base: Mapping[str, Any],
    provenance: Mapping[str, Any],
    path: Path,
) -> None:
    if output.get("palmtree") != provenance:
        raise ValueError(f"PalmTree provenance mismatch in {path}")
    base_keys = set(base)
    if set(output) != base_keys | {"palmtree"}:
        raise ValueError(f"Base payload fields changed in {path}")
    for key in base_keys - {"functions"}:
        if not _payload_equal(output[key], base[key]):
            raise ValueError(f"Base payload field {key!r} changed in {path}")

    output_functions = _functions(output, path)
    base_functions = _functions(base, path)
    if len(output_functions) != len(base_functions):
        raise ValueError(f"Function count changed in {path}")
    for index, (output_function, base_function) in enumerate(
        zip(output_functions, base_functions, strict=True)
    ):
        expected_keys = set(base_function) | {"node_features"}
        if set(output_function) != expected_keys:
            raise ValueError(f"Function fields changed in {path}, function {index}")
        for key in base_function:
            if not _payload_equal(output_function[key], base_function[key]):
                raise ValueError(f"Base function field {key!r} changed in {path}, function {index}")
        expected_rows = max(len(_nodes(base_function, f"{path}, function {index}")), 1)
        features = output_function["node_features"]
        if not isinstance(features, torch.Tensor):
            raise ValueError(f"Missing node feature tensor in {path}, function {index}")
        if features.dtype != torch.float16 or tuple(features.shape) != (
            expected_rows,
            PALMTREE_DIM,
        ):
            raise ValueError(
                f"Invalid node feature tensor in {path}, function {index}: "
                f"dtype={features.dtype}, shape={tuple(features.shape)}"
            )
        if not torch.isfinite(features).all():
            raise ValueError(f"Non-finite node features in {path}, function {index}")


def _atomic_save(
    output: dict[str, Any],
    base: Mapping[str, Any],
    provenance: Mapping[str, Any],
    destination: Path,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    try:
        torch.save(output, temporary)
        validate_output(_load_pt(temporary), base, provenance, temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def enrich_file(
    base_path: Path,
    tokenized_path: Path,
    destination: Path,
    encoder: Any,
    normalize: Callable[[str], str],
    cache: EmbeddingCache,
    batch_size: int,
    provenance: Mapping[str, Any],
) -> tuple[int, int]:
    base = _load_pt(base_path)
    tokenized = _load_json(tokenized_path)
    aligned = _validate_alignment(base, tokenized, base_path, tokenized_path)

    node_count = 0
    instruction_count = 0
    for function_index, (base_function, source_function) in enumerate(aligned):
        source_nodes = _nodes(source_function, f"{tokenized_path}, function {function_index}")
        rows = []
        for node_index, source_node in enumerate(source_nodes):
            instructions = _instruction_texts(
                source_node,
                f"{tokenized_path}, function {function_index}, node {node_index}",
            )
            rows.append(_node_embedding(instructions, normalize, encoder, cache, batch_size))
            node_count += 1
            instruction_count += len(instructions)
        base_function["node_features"] = (
            torch.stack(rows).to(torch.float16)
            if rows
            else torch.zeros((1, PALMTREE_DIM), dtype=torch.float16)
        )

    output = dict(base)
    output["palmtree"] = dict(provenance)
    _atomic_save(output, _load_pt(base_path), provenance, destination)
    return node_count, instruction_count


def _source_path(tokenized_dir: Path, relative_base_path: Path) -> Path:
    if len(relative_base_path.parts) < 3:
        raise ValueError(
            f"Expected base path <split>/<project>/<file>.pt, got {relative_base_path}"
        )
    project = relative_base_path.parts[-2]
    return tokenized_dir / project / relative_base_path.with_suffix(".json").name


def _iter_files(
    base_dir: Path, projects: set[str] | None, max_files: int
) -> Iterator[tuple[Path, Path]]:
    emitted = 0
    for base_path in sorted(base_dir.rglob("*.pt")):
        relative = base_path.relative_to(base_dir)
        if projects and (len(relative.parts) < 2 or relative.parts[-2] not in projects):
            continue
        yield base_path, relative
        emitted += 1
        if max_files and emitted >= max_files:
            return


def _install_legacy_modules(checkout: Path) -> tuple[Any, Callable[[str], str]]:
    pretrained = checkout / "pre-trained_model"
    source = checkout / "src"
    if not pretrained.is_dir() or not source.is_dir():
        raise FileNotFoundError(
            f"PalmTree checkout must contain pre-trained_model/ and src/: {checkout}"
        )
    sys.path[:0] = [str(pretrained), str(source)]

    local_vocab = importlib.import_module("vocab")
    module_names = {
        "bert_pytorch.model.bert": "palmtree.model.bert",
        "bert_pytorch.model.language_model": "palmtree.model.language_model",
        "bert_pytorch.model.transformer": "palmtree.model.transformer",
        "bert_pytorch.model.attention.multi_head": ("palmtree.model.attention.multi_head"),
        "bert_pytorch.model.attention.single": "palmtree.model.attention.single",
        "bert_pytorch.model.utils.layer_norm": "palmtree.model.utils.layer_norm",
        "bert_pytorch.model.utils.feed_forward": "palmtree.model.utils.feed_forward",
        "bert_pytorch.model.utils.gelu": "palmtree.model.utils.gelu",
        "bert_pytorch.model.utils.sublayer": "palmtree.model.utils.sublayer",
        "bert_pytorch.model.embedding.bert": "palmtree.model.embedding.bert",
        "bert_pytorch.model.embedding.token": "palmtree.model.embedding.token",
        "bert_pytorch.model.embedding.position": "palmtree.model.embedding.position",
        "bert_pytorch.model.embedding.segment": "palmtree.model.embedding.segment",
    }
    for package in (
        "bert_pytorch",
        "bert_pytorch.dataset",
        "bert_pytorch.model",
        "bert_pytorch.model.embedding",
        "bert_pytorch.model.attention",
        "bert_pytorch.model.utils",
    ):
        sys.modules.setdefault(package, types.ModuleType(package))
    vocab_module = types.ModuleType("bert_pytorch.dataset.vocab")
    for name in ("TorchVocab", "Vocab", "WordVocab"):
        setattr(vocab_module, name, getattr(local_vocab, name))
    sys.modules["bert_pytorch.dataset.vocab"] = vocab_module
    for legacy_name, current_name in module_names.items():
        sys.modules[legacy_name] = importlib.import_module(current_name)

    eval_utils = importlib.import_module("eval_utils")
    eval_utils.USE_CUDA = False

    # Normalization follows the official PalmTree inference implementation:
    # https://github.com/palmtreemodel/PalmTree/blob/master/pre-trained_model/eval_utils.py
    def normalize(instruction: str) -> str:
        return eval_utils.parse_instruction(instruction, {}, {})

    return eval_utils, normalize


def load_palmtree(
    checkout: Path, model_path: Path, vocab_path: Path
) -> tuple[Any, Callable[[str], str]]:
    for path, label in ((model_path, "model"), (vocab_path, "vocabulary")):
        if not path.is_file():
            raise FileNotFoundError(f"PalmTree {label} not found: {path}")
    eval_utils, normalize = _install_legacy_modules(checkout)
    original_load = torch.load

    def compatible_load(*args: Any, **kwargs: Any) -> Any:
        kwargs.setdefault("weights_only", False)
        return original_load(*args, **kwargs)

    try:
        torch.load = compatible_load
        encoder = eval_utils.UsableTransformer(
            model_path=str(model_path), vocab_path=str(vocab_path)
        )
    finally:
        torch.load = original_load
    return encoder, normalize


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=("Build reduced_data_palmtree_full with inline 128-d node embeddings"),
        epilog=f"PalmTree source and attribution: {PALMTREE_REPOSITORY}",
    )
    parser.add_argument("--base-dir", type=Path, required=True)
    parser.add_argument("--tokenized-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--palmtree-checkout",
        type=Path,
        required=True,
        help="External checkout of the official PalmTree repository",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        help=("Checkpoint path (default: CHECKOUT/pre-trained_model/palmtree/transformer.ep19)"),
    )
    parser.add_argument(
        "--vocab-path",
        type=Path,
        help="Vocabulary path (default: CHECKOUT/pre-trained_model/palmtree/vocab)",
    )
    parser.add_argument("--encode-batch-size", type=int, default=256)
    parser.add_argument("--cache-entries", type=int, default=50_000)
    parser.add_argument("--projects", default="", help="Comma-separated project allowlist")
    parser.add_argument("--max-files", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--progress-every", type=int, default=20)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.encode_batch_size < 1:
        raise ValueError("encode-batch-size must be positive")
    if args.cache_entries < 0 or args.max_files < 0 or args.progress_every < 1:
        raise ValueError(
            "cache-entries and max-files must be nonnegative; progress-every must be positive"
        )

    base_dir = args.base_dir.resolve()
    tokenized_dir = args.tokenized_dir.resolve()
    output_dir = args.output_dir.resolve()
    checkout = args.palmtree_checkout.resolve()
    if not base_dir.is_dir() or not tokenized_dir.is_dir():
        raise FileNotFoundError("base-dir and tokenized-dir must be existing directories")
    if (
        output_dir == base_dir
        or output_dir.is_relative_to(base_dir)
        or base_dir.is_relative_to(output_dir)
    ):
        raise ValueError("output-dir must be separate from base-dir")

    default_weights = checkout / "pre-trained_model" / "palmtree"
    model_path = (args.model_path or default_weights / "transformer.ep19").resolve()
    vocab_path = (args.vocab_path or default_weights / "vocab").resolve()
    for path, label in ((model_path, "model"), (vocab_path, "vocabulary")):
        if not path.is_file():
            raise FileNotFoundError(f"PalmTree {label} not found: {path}")
    model_sha256 = _sha256(model_path)
    vocab_sha256 = _sha256(vocab_path)
    checkout_revision, checkout_dirty = _checkout_state(checkout)

    projects = {project.strip() for project in args.projects.split(",") if project.strip()} or None
    candidates = list(_iter_files(base_dir, projects, args.max_files))
    if not candidates:
        raise FileNotFoundError(f"No .pt files found under {base_dir}")

    jobs = []
    for base_path, relative in candidates:
        source_path = _source_path(tokenized_dir, relative)
        if not source_path.is_file():
            raise FileNotFoundError(f"Tokenized source not found for {base_path}: {source_path}")
        provenance = {
            "schema_version": PALMTREE_SCHEMA_VERSION,
            "embedding_dim": PALMTREE_DIM,
            "base_sha256": _sha256(base_path),
            "tokenized_sha256": _sha256(source_path),
            "model_sha256": model_sha256,
            "vocab_sha256": vocab_sha256,
            "checkout_revision": checkout_revision,
            "checkout_dirty": checkout_dirty,
        }
        destination = output_dir / relative
        if destination.exists() and not args.overwrite:
            validate_output(_load_pt(destination), _load_pt(base_path), provenance, destination)
            continue
        jobs.append((base_path, source_path, destination, provenance))

    if not jobs:
        print(f"All {len(candidates)} files are complete and current in {output_dir}")
        return 0

    encoder, normalize = load_palmtree(checkout, model_path, vocab_path)
    cache = EmbeddingCache(args.cache_entries)
    total_nodes = 0
    total_instructions = 0
    for index, (base_path, source_path, destination, provenance) in enumerate(jobs, 1):
        nodes, instructions = enrich_file(
            base_path,
            source_path,
            destination,
            encoder,
            normalize,
            cache,
            args.encode_batch_size,
            provenance,
        )
        total_nodes += nodes
        total_instructions += instructions
        if index % args.progress_every == 0 or index == len(jobs):
            print(
                f"[{index}/{len(jobs)}] {destination.relative_to(output_dir)} "
                f"nodes={nodes} instructions={instructions} cache={len(cache)}",
                flush=True,
            )
    print(
        f"Wrote {len(jobs)} files with {total_nodes} nodes and "
        f"{total_instructions} instructions to {output_dir}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
