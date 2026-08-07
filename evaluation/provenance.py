"""Reproducibility metadata for evaluation artifacts."""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path


def sha256_file(path: str | Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def git_dirty() -> bool | None:
    try:
        status = subprocess.check_output(
            ["git", "status", "--porcelain"], stderr=subprocess.DEVNULL, text=True
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return bool(status.strip())


def provenance(config: str | Path, checkpoint: str | Path) -> dict[str, str | bool | None]:
    return {
        "config": str(config),
        "config_sha256": sha256_file(config),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "git_commit": git_commit(),
        "git_dirty": git_dirty(),
    }
