"""Tests for the evaluation command-line interfaces."""

import tempfile
from pathlib import Path

import pytest
import torch

from scripts.gnn.clap_frozen.evaluate import main as run_clap_evaluation
from scripts.gnn.evaluate import (
    build_parser as build_ranking_parser,
)
from scripts.gnn.evaluate import (
    main as run_ranking,
)
from scripts.gnn.evaluate_controls import (
    build_parser as build_intervention_parser,
)
from scripts.gnn.evaluate_controls import (
    main as run_interventions,
)
from training.evaluation import load_checkpoint_strict

REQUIRED = [
    "--config",
    "config.yaml",
    "--checkpoint",
    "model.pt",
    "--output-dir",
    "out",
]


def test_fixed_pair_ranking_protocol_defaults():
    args = build_ranking_parser().parse_args(REQUIRED)
    assert args.split == "test"
    assert args.batch_size == 64
    assert args.pair_seed == 0


def test_intervention_protocol_defaults():
    args = build_intervention_parser().parse_args(REQUIRED)
    assert args.split == "test"
    assert args.batch_size == 33
    assert args.pair_seed == 0
    assert args.sampling_seeds == [0, 1, 2, 3, 4]
    assert args.modes == ["noise_global", "noise_perdim", "shuffle"]


def test_fixed_protocol_overrides_are_rejected_before_loading_data():
    for runner, override in (
        (run_ranking, ["--split", "validation"]),
        (run_interventions, ["--batch-size", "64"]),
    ):
        try:
            runner(REQUIRED + override)
        except ValueError:
            pass
        else:
            raise AssertionError("noncanonical evaluation protocol was accepted")


def test_evaluators_refuse_existing_outputs_before_loading_models(tmp_path):
    output_dir = tmp_path / "run"
    output_dir.mkdir()
    for runner, filename in (
        (run_ranking, "fixed_pair_ranking.json"),
        (run_interventions, "content_control_hardened.json"),
    ):
        output = output_dir / filename
        output.touch()
        with pytest.raises(FileExistsError, match="Refusing to overwrite"):
            runner(REQUIRED[:-1] + [str(output_dir)])
        output.unlink()

    output = output_dir / "clap.json"
    output.touch()
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        run_clap_evaluation(
            ["--config", "config.yaml", "--checkpoint", "model.pt", "--output", str(output)]
        )


def test_hardened_rejects_unknown_mode():
    try:
        build_intervention_parser().parse_args(REQUIRED + ["--modes", "shuffle,unknown"])
    except SystemExit as error:
        assert error.code != 0
    else:
        raise AssertionError("unknown intervention mode was accepted")


def test_hardened_rejects_duplicate_sampling_seed():
    try:
        build_intervention_parser().parse_args(REQUIRED + ["--sampling-seeds", "0,0"])
    except SystemExit as error:
        assert error.code != 0
    else:
        raise AssertionError("duplicate sampling seed was accepted")


def test_checkpoint_rejects_same_shaped_config_drift_and_allows_legacy():
    model = torch.nn.Linear(2, 2)
    config = {"model": {"hidden_size": 2}, "gnn": {"prefix_source": "gnn"}}
    with tempfile.TemporaryDirectory() as directory:
        directory = Path(directory)
        mismatched = directory / "mismatched.pt"
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "config": {
                    "model": {"hidden_size": 2},
                    "gnn": {"prefix_source": "setencoder"},
                },
            },
            mismatched,
        )
        try:
            load_checkpoint_strict(model, mismatched, torch.device("cpu"), config)
        except ValueError:
            pass
        else:
            raise AssertionError("same-shaped GNN config drift was accepted")

        legacy = directory / "legacy.pt"
        torch.save({"model_state_dict": model.state_dict()}, legacy)
        load_checkpoint_strict(model, legacy, torch.device("cpu"), config)
