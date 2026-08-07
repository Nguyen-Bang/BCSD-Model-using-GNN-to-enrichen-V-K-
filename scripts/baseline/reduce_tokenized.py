#!/usr/bin/env python3
"""
Reduce tokenized JSON data to compact .pt files.

Reads all JSONs from data/III_tokenized/<project>/, strips instruction text
and addresses (the biggest space consumers), keeps all training-relevant
fields for both the baseline model and future GNN pipeline, and saves
one .pt file per source JSON organized into train/validation/test splits
with project subfolders.

Usage:
    python -m scripts.baseline.reduce_tokenized \
        --input_dir data/III_tokenized \
        --output_dir reduced_data \
        --workers 8

Module: scripts.baseline.reduce_tokenized
"""

import argparse
import json
import logging
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path
from typing import Any

import torch
import yaml

from dataset.reduction import reduce_function

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("reduce_tokenized")

SPLIT_NAMES = ("train", "validation", "test")


def process_file(args: tuple[str, str, str]) -> dict[str, Any]:
    """Process a single JSON file -> .pt file. Designed for multiprocessing."""
    json_path, output_path, project = args

    try:
        with open(json_path) as f:
            data = json.load(f)
    except Exception as e:
        return {"path": json_path, "error": str(e), "functions": 0, "skipped": 0}

    binary_name = data.get("binary_name", Path(json_path).stem)
    functions = []
    skipped = 0

    for func in data.get("functions", []):
        reduced = reduce_function(func)
        if reduced is not None:
            functions.append(reduced)
        else:
            skipped += 1

    output = {
        "binary_name": binary_name,
        "project": project,
        "functions": functions,
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, output_path)

    file_size = os.path.getsize(output_path)
    return {
        "path": json_path,
        "output_path": output_path,
        "functions": len(functions),
        "skipped": skipped,
        "size_mb": file_size / (1024 * 1024),
        "error": None,
    }


def load_split_manifest(path: str | Path) -> dict[str, list[str]]:
    """Load and validate a YAML or JSON split-to-project mapping."""
    manifest_path = Path(path)
    with manifest_path.open() as handle:
        if manifest_path.suffix.lower() == ".json":
            manifest = json.load(handle)
        else:
            manifest = yaml.safe_load(handle)
    if not isinstance(manifest, dict):
        raise ValueError("Split manifest must contain a mapping")
    if set(manifest) != set(SPLIT_NAMES):
        raise ValueError(f"Split manifest must contain exactly {list(SPLIT_NAMES)}")

    split_map: dict[str, list[str]] = {}
    seen: set[str] = set()
    for split in SPLIT_NAMES:
        projects = manifest[split]
        if not isinstance(projects, list) or not all(
            isinstance(project, str) and project for project in projects
        ):
            raise ValueError(f"Split {split!r} must be a list of project names")
        if len(set(projects)) != len(projects):
            raise ValueError(f"Split {split!r} contains duplicate project names")
        duplicates = seen.intersection(projects)
        if duplicates:
            raise ValueError(f"Projects occur in more than one split: {sorted(duplicates)}")
        seen.update(projects)
        split_map[split] = projects
    return split_map


def invert_split_map(split_map: dict[str, list[str]]) -> dict[str, str]:
    return {project: split for split, projects in split_map.items() for project in projects}


def collect_jobs(
    input_dir: str,
    output_dir: str,
    split_map: dict[str, list[str]],
) -> list[tuple[str, str, str]]:
    """Build the list of (json_path, output_path, project) jobs."""
    input_path = Path(input_dir)
    jobs = []
    project_to_split = invert_split_map(split_map)

    for project_dir in sorted(input_path.iterdir()):
        if not project_dir.is_dir():
            continue

        project = project_dir.name
        split = project_to_split.get(project)
        if split is None:
            logger.warning(f"Unknown project '{project}', skipping")
            continue

        for json_file in sorted(project_dir.glob("*.json")):
            out_name = json_file.stem + ".pt"
            out_path = Path(output_dir) / split / project / out_name
            jobs.append((str(json_file), str(out_path), project))

    return jobs


def validate_output_layout(output_dir: str | Path, split_map: dict[str, list[str]]) -> None:
    """Reject project directories left under a different physical split."""
    output_path = Path(output_dir)
    if not output_path.exists():
        return

    expected = invert_split_map(split_map)
    stale = []
    for split in SPLIT_NAMES:
        split_dir = output_path / split
        if not split_dir.is_dir():
            continue
        for project_dir in split_dir.iterdir():
            if project_dir.is_dir() and expected.get(project_dir.name) != split:
                stale.append(str(project_dir))

    if stale:
        raise ValueError(
            "Output directory contains projects assigned to another or unknown split: "
            + ", ".join(sorted(stale))
            + ". Use a clean output directory."
        )


def main():
    parser = argparse.ArgumentParser(description="Reduce tokenized JSONs to compact .pt files")
    parser.add_argument(
        "--input_dir",
        default="data/III_tokenized",
        help="Path to III_tokenized directory",
    )
    parser.add_argument(
        "--output_dir",
        default="reduced_data",
        help="Output directory for reduced .pt files",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of parallel workers",
    )
    parser.add_argument(
        "--split-manifest",
        default="configs/data_splits.yaml",
        help="YAML or JSON mapping train/validation/test to project directories",
    )
    args = parser.parse_args()

    split_map = load_split_manifest(args.split_manifest)
    project_to_split = invert_split_map(split_map)
    validate_output_layout(args.output_dir, split_map)

    logger.info(f"Input:   {args.input_dir}")
    logger.info(f"Output:  {args.output_dir}")
    logger.info(f"Workers: {args.workers}")
    logger.info(f"Splits:  {split_map}")

    jobs = collect_jobs(args.input_dir, args.output_dir, split_map)
    logger.info(f"Found {len(jobs)} JSON files to process")

    if not jobs:
        logger.error("No JSON files found. Check --input_dir path.")
        sys.exit(1)

    t0 = time.time()
    split_stats: dict[str, dict[str, Any]] = {
        s: {"files": 0, "functions": 0, "skipped": 0, "size_mb": 0.0, "errors": 0}
        for s in SPLIT_NAMES
    }

    completed = 0
    failed = 0
    total = len(jobs)

    with Pool(processes=args.workers) as pool:
        for result in pool.imap_unordered(process_file, jobs):
            completed += 1

            if result.get("error"):
                logger.error(f"[{completed}/{total}] FAILED {result['path']}: {result['error']}")
                failed += 1
                continue

            json_path = result["path"]
            project = Path(json_path).parent.name
            split = project_to_split[project]

            stats = split_stats[split]
            stats["files"] += 1
            stats["functions"] += result["functions"]
            stats["skipped"] += result["skipped"]
            stats["size_mb"] += result.get("size_mb", 0)

            if completed % 50 == 0 or completed == total:
                elapsed = time.time() - t0
                rate = completed / elapsed if elapsed > 0 else 0
                logger.info(
                    f"[{completed}/{total}] {rate:.1f} files/s | "
                    f"last: {Path(json_path).name} "
                    f"({result['functions']} funcs, "
                    f"{result.get('size_mb', 0):.1f} MB)"
                )

    elapsed = time.time() - t0
    logger.info(f"\nCompleted in {elapsed:.1f}s ({elapsed / 60:.1f} min)")
    logger.info("=" * 60)

    total_funcs = 0
    total_size = 0.0
    for split, stats in split_stats.items():
        total_funcs += stats["functions"]
        total_size += stats["size_mb"]
        logger.info(
            f"  {split:12s}: {stats['files']:4d} files, "
            f"{stats['functions']:7d} functions, "
            f"{stats['skipped']:5d} skipped, "
            f"{stats['size_mb']:.1f} MB"
        )

    logger.info(
        f"  {'TOTAL':12s}: {sum(s['files'] for s in split_stats.values()):4d} files, "
        f"{total_funcs:7d} functions, {total_size:.1f} MB ({total_size / 1024:.2f} GB)"
    )
    if failed:
        raise SystemExit(f"Dataset reduction failed for {failed} of {total} files")


if __name__ == "__main__":
    main()
