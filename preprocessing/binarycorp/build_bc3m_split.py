#!/usr/bin/env python3
"""Materialize the official BinaryCorp-3M compact split.

``small_train.csv`` is divided into project-disjoint train and validation
subsets. ``small_test.csv`` remains unchanged. Copying is the portable default;
hard links and relative symbolic links are available for local space savings.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import logging
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

from .protocol import DEFAULT_POOL_SIZE, matchable_counts

logger = logging.getLogger("bcsd.binarycorp.split")


@dataclass(frozen=True)
class ManifestRow:
    project: str
    filename: str
    optimization: str


def read_manifest(path: str | Path) -> list[ManifestRow]:
    rows: list[ManifestRow] = []
    seen: set[tuple[str, str]] = set()
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {"proj", "filename", "opt"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(f"{path} must contain CSV columns {sorted(required)}")
        for line_number, raw in enumerate(reader, start=2):
            row = ManifestRow(
                raw["proj"].strip(),
                raw["filename"].strip(),
                raw["opt"].strip(),
            )
            if not row.project or not row.filename or not row.optimization:
                raise ValueError(f"{path}:{line_number} contains an empty required field")
            identity = (row.project, row.filename)
            if identity in seen:
                raise ValueError(f"Duplicate compilation {identity!r} in {path}")
            seen.add(identity)
            rows.append(row)
    return rows


def validation_projects(projects: set[str], fraction: float, seed: int = 0) -> set[str]:
    """Select an exact-size, stable project holdout using a seeded hash order."""
    if not 0 <= fraction < 1:
        raise ValueError("validation fraction must be in [0, 1)")
    if not projects or fraction == 0:
        return set()
    count = round(len(projects) * fraction)
    if count == 0 and len(projects) > 1:
        count = 1
    count = min(count, max(0, len(projects) - 1))

    def score(project: str) -> bytes:
        return hashlib.sha256(f"{seed}:{project}".encode()).digest()

    return set(sorted(projects, key=lambda project: (score(project), project))[:count])


def _materialize(source: Path, destination: Path, mode: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"Destination already exists: {destination}")
    if mode == "copy":
        shutil.copy2(source, destination)
    elif mode == "hardlink":
        os.link(source, destination)
    elif mode == "symlink":
        destination.symlink_to(os.path.relpath(source, destination.parent))
    else:
        raise ValueError(f"Unknown materialization mode: {mode}")


def build_split(
    source_dir: str | Path,
    small_train_csv: str | Path,
    small_test_csv: str | Path,
    output_dir: str | Path,
    *,
    validation_fraction: float = 0.05,
    seed: int = 0,
    mode: str = "copy",
) -> dict[str, dict[str, int]]:
    """Create train/validation/test trees and return per-split counts."""
    source_dir = Path(source_dir)
    output_dir = Path(output_dir)
    if any((output_dir / split).exists() for split in ("train", "validation", "test")):
        raise FileExistsError(f"Output split directories already exist under {output_dir}")
    train_rows = read_manifest(small_train_csv)
    test_rows = read_manifest(small_test_csv)
    train_projects = {row.project for row in train_rows}
    test_projects = {row.project for row in test_rows}
    overlap = train_projects & test_projects
    if overlap:
        raise ValueError(f"Projects occur in both small manifests: {sorted(overlap)[:5]}")
    held_out = validation_projects(train_projects, validation_fraction, seed)
    stats = {split: {"files": 0, "projects": 0} for split in ("train", "validation", "test")}
    missing: list[Path] = []
    plan: list[tuple[Path, Path, str]] = []
    for row in train_rows:
        split = "validation" if row.project in held_out else "train"
        source = source_dir / "train" / row.project / f"{row.filename}.pt"
        destination = output_dir / split / row.project / f"{row.filename}.pt"
        plan.append((source, destination, split))
        if not source.is_file():
            missing.append(source)
    for row in test_rows:
        source = source_dir / "test" / row.project / f"{row.filename}.pt"
        destination = output_dir / "test" / row.project / f"{row.filename}.pt"
        plan.append((source, destination, "test"))
        if not source.is_file():
            missing.append(source)
    if missing:
        preview = ", ".join(str(path) for path in missing[:5])
        raise FileNotFoundError(f"{len(missing)} manifest artifacts are missing: {preview}")
    for source, destination, split in plan:
        _materialize(source, destination, mode)
        stats[split]["files"] += 1
    for split in stats:
        root = output_dir / split
        stats[split]["projects"] = (
            sum(path.is_dir() for path in root.iterdir()) if root.is_dir() else 0
        )
    return stats


def _check_test_pools(data_dir: Path, pool_size: int) -> None:
    counts, index_stats = matchable_counts(data_dir, "test")
    logger.info("Indexed %d test functions in %d groups", index_stats.functions, index_stats.groups)
    for name, count in counts.items():
        logger.info("%s: %d matches, %d full pools", name, count, count // pool_size)
    missing = [name for name, count in counts.items() if count < pool_size]
    if missing:
        raise ValueError(
            f"No complete pool of size {pool_size} for optimization pairs: {', '.join(missing)}"
        )


def publish_split(
    source_dir: str | Path,
    small_train_csv: str | Path,
    small_test_csv: str | Path,
    output_dir: str | Path,
    *,
    validation_fraction: float = 0.05,
    seed: int = 0,
    mode: str = "copy",
    check_test_pools: bool = False,
    pool_size: int = DEFAULT_POOL_SIZE,
) -> dict[str, dict[str, int]]:
    """Build and validate a split before atomically publishing it."""
    output_dir = Path(output_dir)
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"Output directory already exists: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)

    prefix = f".{output_dir.name}.tmp-"
    with tempfile.TemporaryDirectory(prefix=prefix, dir=output_dir.parent) as temporary:
        temporary_dir = Path(temporary)
        stats = build_split(
            source_dir,
            small_train_csv,
            small_test_csv,
            temporary_dir,
            validation_fraction=validation_fraction,
            seed=seed,
            mode=mode,
        )
        if check_test_pools:
            _check_test_pools(temporary_dir, pool_size)
        temporary_dir.rename(output_dir)
    return stats


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--small-train-csv", type=Path, required=True)
    parser.add_argument("--small-test-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--validation-fraction", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mode", choices=("copy", "hardlink", "symlink"), default="copy")
    parser.add_argument("--check-test-pools", action="store_true")
    parser.add_argument("--pool-size", type=int, default=DEFAULT_POOL_SIZE)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.pool_size < 1:
        parser.error("--pool-size must be positive")
    stats = publish_split(
        args.source_dir,
        args.small_train_csv,
        args.small_test_csv,
        args.output_dir,
        validation_fraction=args.validation_fraction,
        seed=args.seed,
        mode=args.mode,
        check_test_pools=args.check_test_pools,
        pool_size=args.pool_size,
    )
    for split, values in stats.items():
        logger.info("%s: %d files across %d projects", split, values["files"], values["projects"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
