"""Streaming BinaryCorp-3M embedding and evaluation runner."""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from dataset.code_dataset import ReducedDataset
from dataset.collate import collate_functions
from evaluation.binarycorp.adapters import (
    EmbeddingAdapter,
    collation_options,
    create_adapter,
    load_model_config,
)
from evaluation.provenance import git_commit, git_dirty, sha256_file
from preprocessing.binarycorp.protocol import (
    OPTIMIZATION_LEVELS,
    EmbeddingTable,
    IndexStats,
    evaluate_optimization_pairs,
    iter_split_artifacts,
)
from training.runtime import resolve_device

logger = logging.getLogger("bcsd.binarycorp.runner")


@dataclass(frozen=True)
class EvaluationConfig:
    model: str
    checkpoint: Path | None
    model_config: Path | None
    data_dir: Path
    split: str
    pool_size: int
    seed: int
    batch_size: int
    query_block_size: int
    device: str | None
    pretrained: str
    revision: str
    local_files_only: bool


@dataclass(frozen=True)
class FunctionRecord:
    key: tuple[str, str]
    optimization: str
    function: dict[str, Any]


def _plain_function(
    record: FunctionRecord,
    *,
    include_function_features: bool,
    include_graph: bool,
) -> dict[str, Any]:
    if include_graph:
        return ReducedDataset._to_plain(
            record.function,
            context=f"{record.key[0]}:{record.key[1]}:{record.optimization}",
        )

    fields = ["token_ids", "attention_mask", "token_type_ids"]
    if include_function_features:
        fields.append("function_features")
    return {field: record.function[field] for field in fields if field in record.function}


def evaluation_provenance(args: EvaluationConfig) -> dict[str, str | bool | None]:
    zero_shot = args.model == "clap-zero-shot"
    return {
        "model_config_sha256": sha256_file(args.model_config) if args.model_config else None,
        "checkpoint_sha256": sha256_file(args.checkpoint) if args.checkpoint else None,
        "git_commit": git_commit(),
        "git_dirty": git_dirty(),
        "pretrained": args.pretrained if zero_shot else None,
        "revision": args.revision if zero_shot else None,
    }


def encode_split(
    data_dir: str | Path,
    split: str,
    adapter: EmbeddingAdapter,
    *,
    batch_size: int,
    use_edge_features: bool,
    use_reverse_edges: bool,
) -> tuple[dict[str, EmbeddingTable], IndexStats]:
    """Index a split, then stream embeddings into exactly sized arrays."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")

    keys: defaultdict[str, list[tuple[str, str]]] = defaultdict(list)
    ordinals: defaultdict[str, list[int]] = defaultdict(list)
    optimization_bits = {
        optimization: 1 << index for index, optimization in enumerate(sorted(OPTIMIZATION_LEVELS))
    }
    key_state: dict[tuple[str, str], tuple[int, int]] = {}
    files = functions = 0
    for artifact in iter_split_artifacts(data_dir, split):
        files += 1
        bit = optimization_bits[artifact.optimization]
        for function in artifact.functions:
            key = (artifact.project, function["function_name"])
            ordinal, mask = key_state.get(key, (len(key_state), 0))
            if mask & bit:
                raise ValueError(
                    "Duplicate BinaryCorp identity "
                    f"({key[0]!r}, {key[1]!r}, {artifact.optimization!r})"
                )
            key_state[key] = (ordinal, mask | bit)
            keys[artifact.optimization].append(key)
            ordinals[artifact.optimization].append(ordinal)
            functions += 1

    group_count = len(key_state)
    key_state.clear()

    vectors: dict[str, np.ndarray] = {}
    offsets: defaultdict[str, int] = defaultdict(int)
    indexed_offsets: defaultdict[str, int] = defaultdict(int)
    batch_records: list[FunctionRecord] = []
    batches = encoded_functions = 0
    embedding_width: int | None = None

    def flush() -> None:
        nonlocal batches, encoded_functions, embedding_width
        if not batch_records:
            return
        batch = collate_functions(
            [
                _plain_function(
                    record,
                    include_function_features=adapter.include_function_features,
                    include_graph=adapter.include_graph,
                )
                for record in batch_records
            ],
            include_function_features=adapter.include_function_features,
            include_graph=adapter.include_graph,
            use_edge_features=use_edge_features,
            use_reverse_edges=use_reverse_edges,
        )
        encoded = adapter.encode(batch)
        if encoded.ndim != 2 or encoded.shape[0] != len(batch_records):
            raise ValueError("Adapter must return one embedding per function")
        matrix = encoded.detach().float().cpu().numpy()
        if embedding_width is None:
            embedding_width = matrix.shape[1]
        elif matrix.shape[1] != embedding_width:
            raise ValueError("Adapter returned inconsistent embedding widths")

        by_optimization: defaultdict[str, list[int]] = defaultdict(list)
        for row, record in enumerate(batch_records):
            by_optimization[record.optimization].append(row)
        for optimization, rows in by_optimization.items():
            table = vectors.get(optimization)
            if table is None:
                table = np.empty(
                    (len(keys[optimization]), embedding_width),
                    dtype=np.float32,
                )
                vectors[optimization] = table
            start = offsets[optimization]
            stop = start + len(rows)
            table[start:stop] = matrix[rows]
            offsets[optimization] = stop
        encoded_functions += len(batch_records)
        batch_records.clear()
        batches += 1
        if batches % 50 == 0:
            logger.info("Encoded %d batches (%d functions)", batches, encoded_functions)

    for artifact in iter_split_artifacts(data_dir, split):
        for function in artifact.functions:
            key = (artifact.project, function["function_name"])
            index = indexed_offsets[artifact.optimization]
            expected = keys[artifact.optimization]
            if index >= len(expected) or expected[index] != key:
                raise RuntimeError(
                    f"BinaryCorp split changed while encoding {artifact.optimization}"
                )
            indexed_offsets[artifact.optimization] += 1
            batch_records.append(FunctionRecord(key, artifact.optimization, function))
            if len(batch_records) == batch_size:
                flush()
    flush()

    for optimization, optimization_keys in keys.items():
        if indexed_offsets[optimization] != len(optimization_keys) or offsets[optimization] != len(
            optimization_keys
        ):
            raise RuntimeError(f"BinaryCorp split changed while encoding {optimization}")

    tables = {
        optimization: EmbeddingTable(
            optimization_keys,
            np.asarray(ordinals[optimization], dtype=np.int64),
            vectors[optimization],
        )
        for optimization, optimization_keys in keys.items()
    }
    return tables, IndexStats(files, functions, group_count)


def run_evaluation(args: EvaluationConfig) -> dict[str, Any]:
    """Load an adapter, embed the requested split, and run the official protocol."""
    config = load_model_config(args.model_config)
    device = resolve_device(args.device)
    adapter = create_adapter(
        model_kind=args.model,
        checkpoint=args.checkpoint,
        config=config,
        device=device,
        pretrained=args.pretrained,
        revision=args.revision,
        local_files_only=args.local_files_only,
    )
    use_edge_features, use_reverse_edges = collation_options(args.model, config)
    embeddings, stats = encode_split(
        args.data_dir,
        args.split,
        adapter,
        batch_size=args.batch_size,
        use_edge_features=use_edge_features,
        use_reverse_edges=use_reverse_edges,
    )
    if not embeddings:
        raise RuntimeError("BinaryCorp split contains no functions")
    result = evaluate_optimization_pairs(
        embeddings,
        pool_size=args.pool_size,
        seed=args.seed,
        query_block_size=args.query_block_size,
        require_all_pairs=True,
    )
    result.update(
        {
            "model": args.model,
            "checkpoint": str(args.checkpoint) if args.checkpoint else None,
            "model_config": str(args.model_config) if args.model_config else None,
            **evaluation_provenance(args),
            "data": {
                "directory": str(args.data_dir),
                "split": args.split,
                "files": stats.files,
                "functions": stats.functions,
                "groups": stats.groups,
            },
        }
    )
    return result
