import json

import numpy as np
import pytest
import torch

from scripts.gnn import ablate_gate_cells
from training.diagnostics import (
    ablate,
    benjamini_hochberg,
    gate_shape,
    joint_sign_flip_pvalues,
    magnitude_correlations,
)


class GateLayer(torch.nn.Module):
    def __init__(self, heads, prefixes, value):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.full((heads, prefixes), value))
        self.register_buffer("_ablate_mask", torch.ones(heads, prefixes), persistent=False)


class Model(torch.nn.Module):
    def __init__(self, layers=2, heads=3, prefixes=4):
        super().__init__()
        self.kv_prefix_layers = torch.nn.ModuleList(
            GateLayer(heads, prefixes, 0.1 + index * 0.1) for index in range(layers)
        )


def test_ablation_grid_supports_arbitrary_model_shape():
    model = Model(layers=4, heads=5, prefixes=6)
    layers = model.kv_prefix_layers
    assert gate_shape(layers) == (4, 5, 6)
    with ablate(layers, [(3, 4, 5), (0, 0, 0)]):
        assert layers[3]._ablate_mask[4, 5] == 0
        assert layers[0]._ablate_mask[0, 0] == 0
        assert layers[2]._ablate_mask.sum() == 30
    assert all(torch.all(layer._ablate_mask == 1) for layer in layers)
    with ablate(layers):
        assert all(torch.all(layer._ablate_mask == 0) for layer in layers)
    assert all(torch.all(layer._ablate_mask == 1) for layer in layers)


def test_joint_sign_flip_bh_and_magnitude_correlation():
    deltas = np.array([[0, 0, 0, 0], [1, 1, 1, 1]], dtype=np.float64)
    pvalues = joint_sign_flip_pvalues(deltas, 999, 7, chunk_size=64)
    assert pvalues[0] == 1
    assert pvalues[1] < 0.2
    np.testing.assert_allclose(
        benjamini_hochberg([0.01, 0.04, 0.03, 0.8]),
        [0.04, 0.05333333333333334, 0.05333333333333334, 0.8],
    )
    correlations = magnitude_correlations([0.1, -0.2, 0.3], [-0.1, -0.2, -0.3])
    assert correlations["gate_magnitude_vs_mrr_loss"]["rho"] == 1
    assert correlations["gate_magnitude_vs_delta_mrr"]["rho"] == -1


class Dataset:
    _group_keys = (("unrar", "a"), ("zlib", "b"))

    def __len__(self):
        return 4


class Loader:
    dataset = Dataset()


def test_cli_writes_portable_resumable_results(tmp_path, monkeypatch):
    model = Model()
    config_path = tmp_path / "config.yaml"
    checkpoint_path = tmp_path / "checkpoint.pt"
    config_path.write_text("model: {}\ngnn: {}\ndata: {}\ntraining: {}\n")
    checkpoint_path.write_bytes(b"checkpoint")
    monkeypatch.setattr(ablate_gate_cells, "load_config", lambda path: {"training": {}})
    monkeypatch.setattr(
        ablate_gate_cells,
        "_build_and_load",
        lambda config, backbone, checkpoint, device: (model, {"epoch": 3}),
    )
    monkeypatch.setattr(
        ablate_gate_cells,
        "build_split_loader",
        lambda config, split, batch_size: (Loader(), 4),
    )

    calls = []

    def evaluate(model, loader, device, use_amp, amp_dtype, pair_seed):
        calls.append(pair_seed)
        active = sum(layer._ablate_mask.sum().item() for layer in model.kv_prefix_layers)
        ranks = np.array([1, 2, 3, 4] if active == 24 else [2, 2, 3, 4], dtype=np.int32)
        return ranks, 4

    monkeypatch.setattr(ablate_gate_cells, "evaluate_ranks", evaluate)
    output_dir = tmp_path / "out"
    arguments = [
        "--config",
        str(config_path),
        "--checkpoint",
        str(checkpoint_path),
        "--device",
        "cpu",
        "--output-dir",
        str(output_dir),
        "--batch-size",
        "2",
        "--n-boot",
        "20",
        "--n-sign-flips",
        "31",
        "--sign-flip-chunk",
        "8",
        "--max-cells",
        "3",
    ]
    result = ablate_gate_cells.main(arguments)
    assert result["metadata"]["gate_shape"] == [2, 3, 4]
    assert result["final_layer_group_removal"]["index"] == 1
    assert len(result["per_cell"]) == 3
    assert all("sign_flip_p" in row and "bh_q" in row for row in result["per_cell"])
    assert (output_dir / "gate_ablation_ranks.npz").is_file()
    assert json.loads((output_dir / "gate_ablation.json").read_text())["pool_size"] == 4
    first_call_count = len(calls)
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        ablate_gate_cells.main(arguments)
    ablate_gate_cells.main([*arguments, "--resume"])
    assert len(calls) == first_call_count
