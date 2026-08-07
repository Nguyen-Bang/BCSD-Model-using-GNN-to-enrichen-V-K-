from pathlib import Path

import pytest

from scripts.baseline.train import build_parser, prepare_output_dir, resolve_output_dir


def _config(checkpoint_dir: Path) -> dict:
    return {"training": {"checkpoint_dir": str(checkpoint_dir)}}


def test_default_output_directories_are_seed_specific(tmp_path):
    config = _config(tmp_path / "runs")

    outputs = {resolve_output_dir(config, seed=seed) for seed in (0, 101, 202)}

    assert outputs == {
        tmp_path / "runs" / "seed_0",
        tmp_path / "runs" / "seed_101",
        tmp_path / "runs" / "seed_202",
    }


def test_explicit_output_directory_is_supported(tmp_path):
    output_dir = tmp_path / "portable-run"
    args = build_parser().parse_args(["--output-dir", str(output_dir)])

    assert args.output_dir == str(output_dir)
    assert (
        resolve_output_dir(
            _config(tmp_path / "runs"),
            seed=101,
            output_dir=output_dir,
        )
        == output_dir
    )


@pytest.mark.parametrize(
    "artifact",
    [
        "training_log.csv",
        "best_mlm.pt",
        "mlm_final.pt",
        "best_model.pt",
        "checkpoint_s2_epoch_5.pt",
    ],
)
def test_fresh_training_refuses_to_overwrite_artifacts(tmp_path, artifact):
    output_dir = tmp_path / "seed_0"
    output_dir.mkdir()
    (output_dir / artifact).touch()

    with pytest.raises(FileExistsError, match="already contains training artifacts"):
        prepare_output_dir(output_dir)


def test_non_training_files_do_not_block_fresh_run(tmp_path):
    output_dir = tmp_path / "seed_0"
    output_dir.mkdir()
    (output_dir / "notes.txt").touch()

    prepare_output_dir(output_dir)
