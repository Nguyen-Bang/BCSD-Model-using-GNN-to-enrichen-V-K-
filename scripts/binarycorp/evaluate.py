#!/usr/bin/env python3
"""Evaluate BCSD embeddings with the BinaryCorp-3M retrieval protocol."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from evaluation.binarycorp.adapters import (
    DEFAULT_CLAP_MODEL,
    DEFAULT_CLAP_REVISION,
    MODEL_KINDS,
)
from evaluation.binarycorp.runner import EvaluationConfig, run_evaluation
from preprocessing.binarycorp.protocol import DEFAULT_POOL_SIZE

logger = logging.getLogger("bcsd.binarycorp.evaluate")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODEL_KINDS, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--model-config", type=Path)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--pool-size", type=int, default=DEFAULT_POOL_SIZE)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--query-block-size", type=int, default=512)
    parser.add_argument("--device")
    parser.add_argument("--pretrained", default=DEFAULT_CLAP_MODEL)
    parser.add_argument("--revision", default=DEFAULT_CLAP_REVISION)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.model != "clap-zero-shot":
        if args.checkpoint is None:
            parser.error(f"--checkpoint is required for --model {args.model}")
        if args.model_config is None:
            parser.error(f"--model-config is required for --model {args.model}")
    if args.output and args.output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {args.output}")
    config = EvaluationConfig(
        model=args.model,
        checkpoint=args.checkpoint,
        model_config=args.model_config,
        data_dir=args.data_dir,
        split=args.split,
        pool_size=args.pool_size,
        seed=args.seed,
        batch_size=args.batch_size,
        query_block_size=args.query_block_size,
        device=args.device,
        pretrained=args.pretrained,
        revision=args.revision,
        local_files_only=args.local_files_only,
    )
    result = run_evaluation(config)
    for name, metrics in result["per_pair"].items():
        logger.info(
            "%s n=%d pools=%d dropped=%d MRR=%.4f R@1=%.4f",
            name,
            metrics["n_pairs"],
            metrics["n_pools"],
            metrics["n_dropped"],
            metrics["mrr"],
            metrics["recall_at_1"],
        )
    aggregate = result["aggregate"]
    logger.info("Aggregate MRR=%.4f R@1=%.4f", aggregate["mrr"], aggregate["recall_at_1"])
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        logger.info("Wrote %s", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
