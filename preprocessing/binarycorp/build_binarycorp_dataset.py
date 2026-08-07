#!/usr/bin/env python3
"""Build compact BinaryCorp artifacts from extracted CFG JSON files.

The official BinaryCorp CSV lists provide the stable binary identity (``proj``),
optimization level, and train/test assignment. Each output file contains only
the fields consumed by this repository's dataset and collation code.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
from collections.abc import Iterable
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import torch

from dataset.reduction import reduce_function
from preprocessing.II_static_enrichment.enrich_cfg import enrich_cfg_data
from preprocessing.III_tokenization.tokenize_cfg import tokenize_cfg_data
from preprocessing.III_tokenization.tokenizer import ClapASMTokenizer

logger = logging.getLogger("bcsd.binarycorp.build")

_tokenizer: ClapASMTokenizer | None = None


@dataclass(frozen=True)
class SplitEntry:
    project: str
    optimization: str
    split: str


@dataclass(frozen=True)
class BuildJob:
    cfg_path: Path
    output_path: Path
    project: str
    optimization: str


@dataclass(frozen=True)
class BuildResult:
    input_path: Path
    output_path: Path | None
    functions: int
    error: str | None = None


def _read_manifest(path: str | Path, split: str) -> dict[str, SplitEntry]:
    entries: dict[str, SplitEntry] = {}
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {"proj", "filename", "opt"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(f"{path} must contain CSV columns {sorted(required)}")
        for line_number, row in enumerate(reader, start=2):
            filename = row["filename"].strip()
            project = row["proj"].strip()
            optimization = row["opt"].strip()
            if not filename or not project or not optimization:
                raise ValueError(f"{path}:{line_number} contains an empty required field")
            entry = SplitEntry(project, optimization, split)
            previous = entries.setdefault(filename, entry)
            if previous != entry:
                raise ValueError(f"Conflicting rows for filename {filename!r} in {path}")
    return entries


def load_split_map(
    train_csv: str | Path,
    test_csv: str | Path,
) -> dict[str, SplitEntry]:
    """Return the official filename-to-project/optimization/split mapping."""
    combined = _read_manifest(train_csv, "train")
    for filename, entry in _read_manifest(test_csv, "test").items():
        previous = combined.setdefault(filename, entry)
        if previous != entry:
            raise ValueError(f"Filename {filename!r} occurs in both manifests")
    return combined


def _binary_stem(cfg_path: Path) -> str:
    suffix = "_cfg.json"
    if not cfg_path.name.endswith(suffix):
        raise ValueError(f"Expected a filename ending in {suffix!r}: {cfg_path}")
    return cfg_path.name[: -len(suffix)]


def collect_jobs(
    cfg_dir: str | Path,
    split_map: dict[str, SplitEntry],
    output_dir: str | Path,
    *,
    recursive: bool = False,
) -> tuple[list[BuildJob], int]:
    """Match CFG JSONs to manifest entries without relying on directory layout."""
    root = Path(cfg_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"CFG directory not found: {root}")
    paths = root.rglob("*_cfg.json") if recursive else root.glob("*_cfg.json")
    jobs: list[BuildJob] = []
    unmatched = 0
    outputs: set[Path] = set()
    for cfg_path in sorted(paths):
        stem = _binary_stem(cfg_path)
        entry = split_map.get(stem)
        if entry is None:
            unmatched += 1
            continue
        output_path = Path(output_dir) / entry.split / entry.project / f"{stem}.pt"
        if output_path in outputs:
            raise ValueError(f"Multiple CFG files map to the same output: {output_path}")
        outputs.add(output_path)
        jobs.append(BuildJob(cfg_path, output_path, entry.project, entry.optimization))
    return jobs, unmatched


def _initialize_worker(max_seq_length: int) -> None:
    global _tokenizer
    _tokenizer = ClapASMTokenizer(max_seq_length=max_seq_length)


def process_job(job: BuildJob) -> BuildResult:
    """Run enrichment, tokenization, and reduction for one compilation."""
    if _tokenizer is None:
        raise RuntimeError("BinaryCorp worker tokenizer was not initialized")
    try:
        with job.cfg_path.open(encoding="utf-8") as handle:
            data = json.load(handle)
        enrich_cfg_data(data)
        tokenize_cfg_data(data, _tokenizer)
        functions = [
            reduced
            for function in data.get("functions", [])
            if (reduced := reduce_function(function)) is not None
        ]
        if not functions:
            return BuildResult(job.cfg_path, None, 0, "no tokenized functions")
        payload = {
            "binary_name": data.get("binary_name", _binary_stem(job.cfg_path)),
            "project": job.project,
            "opt": job.optimization,
            "functions": functions,
        }
        job.output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, job.output_path)
        return BuildResult(job.cfg_path, job.output_path, len(functions))
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as error:
        return BuildResult(job.cfg_path, None, 0, str(error))


def build_dataset(
    jobs: Iterable[BuildJob],
    *,
    workers: int,
    max_seq_length: int,
) -> list[BuildResult]:
    """Build all jobs, preserving input order in the returned results."""
    jobs = list(jobs)
    if workers < 1:
        raise ValueError("workers must be at least 1")
    if max_seq_length < 1:
        raise ValueError("max_seq_length must be positive")
    existing = [job.output_path for job in jobs if job.output_path.exists()]
    if existing:
        preview = ", ".join(str(path) for path in existing[:3])
        suffix = " ..." if len(existing) > 3 else ""
        raise FileExistsError(f"Refusing to overwrite existing artifacts: {preview}{suffix}")
    if workers == 1:
        _initialize_worker(max_seq_length)
        return [process_job(job) for job in jobs]
    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_initialize_worker,
        initargs=(max_seq_length,),
    ) as executor:
        return list(executor.map(process_job, jobs, chunksize=16))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cfg-dir", type=Path, required=True)
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--test-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, min(8, (os.cpu_count() or 2) - 1)),
    )
    parser.add_argument("--max-seq-length", type=int, default=1024)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--recursive", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.limit is not None and args.limit < 1:
        raise ValueError("limit must be positive")
    split_map = load_split_map(args.train_csv, args.test_csv)
    jobs, unmatched = collect_jobs(
        args.cfg_dir,
        split_map,
        args.output_dir,
        recursive=args.recursive,
    )
    if args.limit is not None:
        jobs = jobs[: args.limit]
    if not jobs:
        raise RuntimeError("No CFG files matched the supplied manifests")
    logger.info("Building %d compilations (%d unmatched CFG files)", len(jobs), unmatched)
    results = build_dataset(
        jobs,
        workers=args.workers,
        max_seq_length=args.max_seq_length,
    )
    failures = [result for result in results if result.error is not None]
    if failures:
        for failure in failures:
            logger.error("%s: %s", failure.input_path, failure.error)
        raise RuntimeError(f"Failed to build {len(failures)} of {len(results)} compilations")
    logger.info(
        "Wrote %d compact files containing %d functions to %s",
        len(results),
        sum(result.functions for result in results),
        args.output_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
