import random
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from models.baseline import load_backbone_checkpoint
from training.runtime import (
    accumulation_window_size,
    capture_rng_state,
    checkpoint_state,
    evaluate_ranking,
    load_training_checkpoint,
    optimizer_steps_per_epoch,
    ranks_from_embeddings,
    ranks_from_similarity,
    restore_rng_state,
    seed_everything,
)


class PairModel(torch.nn.Module):
    def get_embedding_pairs(self, batch):
        return batch["a"], batch["b"]


class RandomPairLoader:
    def __iter__(self):
        value = random.random()
        yield {
            "a": torch.tensor([[1.0, value], [0.0, 1.0]]),
            "b": torch.tensor([[1.0, value], [0.0, 1.0]]),
        }


class WarmStartModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.base = torch.nn.Linear(2, 2)
        self.gnn_head = torch.nn.Linear(2, 2)


class TestTrainingRuntime(unittest.TestCase):
    def test_partial_accumulation_window_and_step_count(self):
        self.assertEqual(optimizer_steps_per_epoch(5, 2), 3)
        self.assertEqual(
            [accumulation_window_size(step, 5, 2) for step in range(5)],
            [2, 2, 2, 2, 1],
        )

    def test_pessimistic_ties_use_worst_tied_rank(self):
        similarity = torch.tensor(
            [
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 1.0],
                [0.0, 0.5, 0.5],
            ]
        )
        torch.testing.assert_close(
            ranks_from_similarity(similarity),
            torch.tensor([2, 2, 2]),
        )

    def test_blockwise_ranks_match_full_similarity_matrix(self):
        devices = [torch.device("cpu")]
        if torch.cuda.is_available():
            devices.append(torch.device("cuda"))

        for device in devices:
            with self.subTest(device=device):
                queries = torch.tensor(
                    [
                        [1.0, 0.0],
                        [0.0, 1.0],
                        [1.0, 1.0],
                        [2.0, -1.0],
                        [0.0, 0.0],
                    ],
                    device=device,
                )
                candidates = torch.tensor(
                    [
                        [1.0, 0.0],
                        [0.0, 1.0],
                        [1.0, 1.0],
                        [2.0, -1.0],
                        [3.0, 4.0],
                    ],
                    device=device,
                )
                expected = ranks_from_similarity(queries @ candidates.T)
                actual = ranks_from_embeddings(queries, candidates, block_size=2)
                torch.testing.assert_close(actual, expected)

    def test_blockwise_ranking_bounds_each_similarity_multiplication(self):
        queries = torch.randn(7, 3)
        candidates = torch.randn(7, 3)
        original_mm = torch.mm

        with patch("training.runtime.torch.mm", wraps=original_mm) as mm:
            ranks_from_embeddings(queries, candidates, block_size=2)

        self.assertGreater(len(mm.call_args_list), 1)
        for call in mm.call_args_list:
            left, right = call.args
            self.assertLessEqual(left.shape[0], 2)
            self.assertLessEqual(right.shape[1], 2)

    def test_pair_seed_is_repeatable_and_does_not_change_training_rng(self):
        seed_everything(91)
        before = capture_rng_state()
        first = evaluate_ranking(
            PairModel(),
            RandomPairLoader(),
            torch.device("cpu"),
            split="test",
            pair_seed=7,
            ranking_block_size=1,
        )
        after = capture_rng_state()
        second = evaluate_ranking(
            PairModel(),
            RandomPairLoader(),
            torch.device("cpu"),
            split="test",
            pair_seed=7,
            ranking_block_size=1,
        )
        self.assertEqual(first, second)
        self.assertEqual(first["split"], "test")
        self.assertEqual(first["pool_size"], 2)
        self.assertEqual(first["tie_policy"], "pessimistic")

        restore_rng_state(before)
        expected = (random.random(), np.random.random(), torch.rand(1))
        restore_rng_state(after)
        actual = (random.random(), np.random.random(), torch.rand(1))
        self.assertEqual(expected[0], actual[0])
        self.assertEqual(expected[1], actual[1])
        torch.testing.assert_close(expected[2], actual[2])

    def test_checkpoint_requires_complete_resume_state_and_strict_model(self):
        model = torch.nn.Linear(2, 2)
        optimizer = torch.optim.AdamW(model.parameters())
        state = checkpoint_state(
            model=model,
            optimizer=optimizer,
            scheduler=None,
            scaler=None,
            epoch=3,
            config={"training": {}},
            seed=11,
        )
        self.assertEqual(state["seed"], 11)
        self.assertIn("rng_state", state)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            broken = dict(state)
            broken["model_state_dict"] = {}
            torch.save(broken, path)
            with self.assertRaises(RuntimeError):
                load_training_checkpoint(
                    path,
                    model=model,
                    optimizer=optimizer,
                    scheduler=None,
                    scaler=None,
                    device=torch.device("cpu"),
                    config={"training": {}},
                )

    def test_backbone_warm_start_requires_every_baseline_key(self):
        model = WarmStartModel()
        baseline = {
            key: value for key, value in model.state_dict().items() if key.startswith("base.")
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "backbone.pt"
            torch.save({"model_state_dict": baseline}, path)
            self.assertEqual(
                load_backbone_checkpoint(model, path),
                len(baseline),
            )
            torch.save(
                {"model_state_dict": {"base.weight": baseline["base.weight"]}},
                path,
            )
            with self.assertRaises(RuntimeError):
                load_backbone_checkpoint(model, path)


if __name__ == "__main__":
    unittest.main()
