from pathlib import Path
from types import SimpleNamespace

import pytest

from preprocessing.binarycorp import build_bc3m_split as splitter


def _arguments(tmp_path: Path) -> list[str]:
    return [
        "--source-dir",
        str(tmp_path / "source"),
        "--small-train-csv",
        str(tmp_path / "train.csv"),
        "--small-test-csv",
        str(tmp_path / "test.csv"),
        "--output-dir",
        str(tmp_path / "output"),
        "--check-test-pools",
        "--pool-size",
        "2",
    ]


def _fake_build_split(*args, **kwargs) -> dict[str, dict[str, int]]:
    output_dir = Path(args[3])
    (output_dir / "complete.marker").touch()
    return {split: {"files": 0, "projects": 0} for split in ("train", "validation", "test")}


def test_failed_pool_check_leaves_no_output_and_can_be_rerun(tmp_path, monkeypatch):
    monkeypatch.setattr(splitter, "build_split", _fake_build_split)
    pool_counts = iter(({"O0-O3": 1}, {"O0-O3": 2}))
    index_stats = SimpleNamespace(functions=2, groups=1)
    monkeypatch.setattr(
        splitter,
        "matchable_counts",
        lambda *_args: (next(pool_counts), index_stats),
    )

    with pytest.raises(ValueError, match="No complete pool of size 2"):
        splitter.main(_arguments(tmp_path))

    output_dir = tmp_path / "output"
    assert not output_dir.exists()
    assert not list(tmp_path.glob(".output.tmp-*"))

    assert splitter.main(_arguments(tmp_path)) == 0
    assert (output_dir / "complete.marker").is_file()
    assert not list(tmp_path.glob(".output.tmp-*"))


def test_materialization_failure_cleans_temporary_output(tmp_path, monkeypatch):
    def fail_after_writing(*args, **kwargs):
        output_dir = Path(args[3])
        (output_dir / "partial.marker").touch()
        raise RuntimeError("materialization failed")

    monkeypatch.setattr(splitter, "build_split", fail_after_writing)

    with pytest.raises(RuntimeError, match="materialization failed"):
        splitter.main(_arguments(tmp_path))

    assert not (tmp_path / "output").exists()
    assert not list(tmp_path.glob(".output.tmp-*"))
