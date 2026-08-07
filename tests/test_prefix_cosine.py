import json

import numpy as np
import pytest
import torch

from scripts.gnn import prefix_cosine
from training.diagnostics import deranged_permutation, prefix_cosines


def test_prefix_cosines_use_within_batch_derangement_and_side_centering():
    prefixes_a = torch.tensor(
        [
            [[1.0, 0.0]],
            [[0.0, 1.0]],
            [[1.0, 1.0]],
            [[-1.0, 1.0]],
        ]
    )
    prefixes_b = prefixes_a + torch.tensor([[[4.0, -2.0]]])
    values = prefix_cosines(prefixes_a, prefixes_b, [2, 2], negative_seed=17)
    permutation = values["permutation"]
    assert not np.any(permutation == np.arange(4))
    assert set(permutation[:2]) == {0, 1}
    assert set(permutation[2:]) == {2, 3}
    np.testing.assert_allclose(values["centered_matched"], np.ones(4), atol=1e-6)
    np.testing.assert_allclose(values["centered_deranged"], -np.ones(4), atol=1e-6)


def test_derangement_handles_singleton_remainder_without_self_pairs():
    permutation = deranged_permutation([3, 1], seed=0)

    assert torch.equal(permutation, deranged_permutation([3, 1], seed=0))
    assert torch.equal(permutation.sort().values, torch.arange(4))
    assert not torch.any(permutation == torch.arange(4))


def test_prefix_cosines_handle_singleton_remainder():
    prefixes_a = torch.arange(10, dtype=torch.float32).reshape(5, 1, 2)
    prefixes_b = prefixes_a + torch.tensor([[[0.5, -0.5]]])

    values = prefix_cosines(prefixes_a, prefixes_b, [2, 2, 1], negative_seed=17)

    assert not np.any(values["permutation"] == np.arange(5))
    assert sorted(values["permutation"].tolist()) == list(range(5))
    for name in ("raw_matched", "raw_deranged", "centered_matched", "centered_deranged"):
        assert values[name].shape == (5,)
        assert np.isfinite(values[name]).all()
    assert values["centered_matched"][-1] != 0


class PrefixModel(torch.nn.Module):
    def _make_prefixes(
        self, node_features, edge_index, graph_batch, edge_types, batch_size, device
    ):
        assert batch_size == node_features.shape[0]
        return node_features


class ClapPrefixModel(torch.nn.Module):
    def _prefixes(self, batch_size, node_features, edge_index, edge_types, graph_batch):
        assert batch_size == node_features.shape[0]
        return node_features


def test_clap_prefix_factory_surface_is_supported():
    batch = {
        "input_ids_a": torch.ones(2, 2, dtype=torch.long),
        "node_features_a": torch.ones(2, 1, 3),
    }
    prefixes = prefix_cosine._make_side_prefixes(ClapPrefixModel(), batch, "a")
    assert prefixes.shape == (2, 1, 3)


class Dataset:
    _group_keys = (("unrar", "a"), ("zlib", "b"))

    def __len__(self):
        return 4


class Loader:
    num_workers = 0
    dataset = Dataset()

    def __init__(self, batch):
        self.batch = batch

    def __iter__(self):
        yield self.batch


def test_cli_uses_fixed_pair_seed_without_model_downloads(tmp_path, monkeypatch):
    prefixes_a = torch.tensor(
        [
            [[1.0, 0.0]],
            [[0.0, 1.0]],
            [[1.0, 1.0]],
            [[-1.0, 1.0]],
        ]
    )
    prefixes_b = prefixes_a + 0.1
    batch = {
        "input_ids_a": torch.ones(4, 2, dtype=torch.long),
        "input_ids_b": torch.ones(4, 2, dtype=torch.long),
        "node_features_a": prefixes_a,
        "node_features_b": prefixes_b,
    }
    loader = Loader(batch)
    config_path = tmp_path / "config.yaml"
    checkpoint_path = tmp_path / "checkpoint.pt"
    config_path.write_text("model: {}\ngnn: {}\ndata: {}\ntraining: {}\n")
    checkpoint_path.write_bytes(b"checkpoint")
    monkeypatch.setattr(prefix_cosine, "load_config", lambda path: {})
    monkeypatch.setattr(
        prefix_cosine,
        "_build_and_load",
        lambda config, backbone, checkpoint, device: (PrefixModel(), {"epoch": 9}),
    )
    monkeypatch.setattr(
        prefix_cosine,
        "build_split_loader",
        lambda config, split, batch_size: (loader, 4),
    )
    seen_seeds = []
    monkeypatch.setattr(prefix_cosine, "reset_rng", seen_seeds.append)
    output_dir = tmp_path / "out"
    result = prefix_cosine.main(
        [
            "--config",
            str(config_path),
            "--checkpoint",
            str(checkpoint_path),
            "--device",
            "cpu",
            "--batch-size",
            "4",
            "--pair-seed",
            "23",
            "--output-dir",
            str(output_dir),
        ]
    )
    assert seen_seeds == [23]
    assert result["raw"]["matched"]["mean"] > result["raw"]["deranged"]["mean"]
    assert result["batch_mean_centered"]["definition"].startswith("subtract each side")
    assert json.loads((output_dir / "prefix_cosine.json").read_text())["pool_size"] == 4
    arrays = np.load(output_dir / "prefix_cosine_values.npz")
    assert not np.any(arrays["permutation"] == np.arange(4))
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        prefix_cosine.main(
            [
                "--config",
                str(config_path),
                "--checkpoint",
                str(checkpoint_path),
                "--device",
                "cpu",
                "--batch-size",
                "4",
                "--output-dir",
                str(output_dir),
            ]
        )
