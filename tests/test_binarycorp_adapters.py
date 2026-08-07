import pytest
import torch

from evaluation.binarycorp.adapters import (
    ModelAdapter,
    ZeroShotClapAdapter,
    _validate_kind_config,
    collation_options,
)


def test_model_kind_validation_matches_graph_architecture():
    typed = {
        "model": {"variant": "gnn"},
        "gnn": {
            "encoder_type": "graph",
            "use_edge_features": True,
            "use_reverse_edges": True,
        },
    }
    _validate_kind_config("typed-gnn", typed)
    assert collation_options("typed-gnn", typed) == (True, True)

    with pytest.raises(ValueError, match="encoder_type"):
        _validate_kind_config(
            "set-encoder",
            {"model": {"variant": "gnn"}, "gnn": {"encoder_type": "graph"}},
        )


def test_projection_adapter_calls_direct_forward_once():
    class Projection(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def forward(
            self,
            input_ids,
            attention_mask,
            token_type_ids=None,
            function_features=None,
        ):
            self.calls += 1
            values = input_ids[:, :2].float()
            return values

    model = Projection()
    adapter = ModelAdapter(model, "projection-fusion", torch.device("cpu"))
    output = adapter.encode(
        {
            "input_ids": torch.tensor([[3, 4]]),
            "attention_mask": torch.ones(1, 2),
            "token_type_ids": torch.zeros(1, 2, dtype=torch.long),
            "function_features": torch.zeros(1, 132),
        }
    )

    assert model.calls == 1
    torch.testing.assert_close(output.norm(dim=1), torch.ones(1))


def test_set_adapter_does_not_require_edges():
    class SetModel(torch.nn.Module):
        def forward(
            self,
            input_ids,
            attention_mask,
            token_type_ids,
            function_features,
            node_features,
            graph_batch,
        ):
            del attention_mask, token_type_ids, function_features, node_features, graph_batch
            return input_ids[:, :2].float()

    adapter = ModelAdapter(SetModel(), "set-encoder", torch.device("cpu"))
    output = adapter.encode(
        {
            "input_ids": torch.tensor([[3, 4]]),
            "attention_mask": torch.ones(1, 2),
            "token_type_ids": torch.zeros(1, 2, dtype=torch.long),
            "function_features": torch.zeros(1, 132),
            "node_features": torch.zeros(2, 20),
            "batch": torch.zeros(2, dtype=torch.long),
        }
    )

    torch.testing.assert_close(output, torch.tensor([[0.6, 0.8]]))


def test_zero_shot_adapter_uses_complete_clap_encoder_output():
    class Encoder(torch.nn.Module):
        def forward(self, input_ids, attention_mask, token_type_ids=None):
            del attention_mask, token_type_ids
            return torch.stack((input_ids[:, 0].float(), input_ids[:, 1].float()), dim=1)

    adapter = ZeroShotClapAdapter(Encoder(), torch.device("cpu"))
    output = adapter.encode(
        {
            "input_ids": torch.tensor([[3, 4]]),
            "attention_mask": torch.ones(1, 2),
            "token_type_ids": torch.zeros(1, 2, dtype=torch.long),
        }
    )

    torch.testing.assert_close(output, torch.tensor([[0.6, 0.8]]))
