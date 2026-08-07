from pathlib import Path

import pytest

import scripts.gnn.train as train_cli
from scripts.gnn.train import build_parser, prepare_output_dir, resolve_output_dir
from training.runtime import load_config


def _config(checkpoint_dir: Path) -> dict:
    return {"training": {"checkpoint_dir": str(checkpoint_dir)}}


def test_default_output_directories_are_seed_specific(tmp_path):
    config = _config(tmp_path / "runs")

    outputs = {resolve_output_dir(config, seed=seed) for seed in (101, 202, 303)}

    assert outputs == {
        tmp_path / "runs" / "seed_101",
        tmp_path / "runs" / "seed_202",
        tmp_path / "runs" / "seed_303",
    }


def test_explicit_output_directory_is_supported(tmp_path):
    output_dir = tmp_path / "portable-run"
    args = build_parser().parse_args(["--output-dir", str(output_dir)])

    assert args.output_dir == str(output_dir)
    assert (
        resolve_output_dir(_config(tmp_path / "runs"), seed=101, output_dir=output_dir)
        == output_dir
    )


def test_fresh_training_refuses_to_overwrite_artifacts(tmp_path):
    output_dir = tmp_path / "seed_101"
    output_dir.mkdir()
    (output_dir / "best_mrr.pt").touch()

    with pytest.raises(FileExistsError, match="already contains training artifacts"):
        prepare_output_dir(output_dir, resume=False)


def test_resume_uses_checkpoint_directory(tmp_path):
    checkpoint = tmp_path / "seed_101" / "checkpoint_epoch_5.pt"

    assert (
        resolve_output_dir(_config(tmp_path / "runs"), seed=101, resume=checkpoint)
        == checkpoint.parent
    )
    with pytest.raises(ValueError, match="must match the resume checkpoint directory"):
        resolve_output_dir(
            _config(tmp_path / "runs"),
            seed=101,
            resume=checkpoint,
            output_dir=tmp_path / "different-run",
        )


def test_palmtree_typed_edge_config_uses_embedded_nodes_and_typed_cfg():
    config = load_config("configs/palmtree_typed_edges.yaml")

    assert config["data"]["data_dir"] == "reduced_data_palmtree_full"
    assert config["gnn"]["input_dim"] == 128
    assert config["gnn"]["encoder_type"] == "graph"
    assert config["gnn"]["use_edge_features"] is True
    assert config["gnn"]["use_reverse_edges"] is True
    assert config["gnn"]["num_edge_types"] == 11


def test_training_cli_does_not_own_reusable_model_or_epoch_code():
    assert not hasattr(train_cli, "create_model")
    assert not hasattr(train_cli, "train_epoch")
