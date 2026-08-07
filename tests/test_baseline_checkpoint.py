import tempfile
import unittest
from pathlib import Path

import torch

from models.baseline import load_backbone_checkpoint


class CheckpointModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = torch.nn.Linear(2, 2)
        self.gnn_head = torch.nn.Linear(2, 2)
        self.kv_prefix_layers = torch.nn.ModuleList([torch.nn.Linear(2, 2)])
        self.register_prefix = torch.nn.Parameter(torch.ones(2))
        self.register_buffer("_rand_node_primes", torch.tensor([2, 3]))


class TestBaselineCheckpoint(unittest.TestCase):
    def test_loader_preserves_structural_prefix_state(self) -> None:
        model = CheckpointModel()
        source = {
            key: torch.full_like(value, 2)
            for key, value in model.state_dict().items()
            if key.startswith("backbone.")
        }
        graph_state = {
            key: value.clone()
            for key, value in model.state_dict().items()
            if key.startswith(("gnn_head.", "kv_prefix_layers."))
            or key in {"register_prefix", "_rand_node_primes"}
        }

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "backbone.pt"
            torch.save({"model_state_dict": source}, path)
            loaded = load_backbone_checkpoint(model, path)

        self.assertEqual(loaded, len(source))
        for key, value in source.items():
            torch.testing.assert_close(model.state_dict()[key], value)
        for key, value in graph_state.items():
            torch.testing.assert_close(model.state_dict()[key], value)


if __name__ == "__main__":
    unittest.main()
