import math
from types import SimpleNamespace

import torch
from torch import nn

from experiments.continual_chemistry.objectives import (
    feedback_statistics, initialize_feedback_predictor, scaled_feedback_loss,
    training_parameter_groups,
)


def test_initial_cost_has_recorded_units_and_remains_trainable():
    records = [{"rust_before": 3, "rust_after": rust, "native_actions": cost}
               for rust, cost in [(3, 8), (2, 12), (0, 18)]]
    statistics = feedback_statistics(records)
    model = SimpleNamespace(predictor=nn.Linear(32, 2))
    initialize_feedback_predictor(model, statistics)
    x = torch.nn.functional.normalize(torch.randn(3, 32), dim=-1)
    initial = model.predictor(x)
    assert (initial[:, 1].expm1() > 1).all()
    expected = sum(math.log1p(row["native_actions"]) for row in records) / 3
    torch.testing.assert_close(initial[:, 1], torch.full((3,), expected))
    returned = torch.tensor([[(3-r["rust_after"])/3, math.log1p(r["native_actions"])]
                             for r in records])
    optimizer = torch.optim.AdamW(model.predictor.parameters(), lr=1e-3)
    initial_loss = float(scaled_feedback_loss(initial, returned, statistics).detach())
    for _ in range(100):
        optimizer.zero_grad()
        loss = scaled_feedback_loss(model.predictor(x), returned, statistics)
        loss.backward()
        optimizer.step()
    assert float(loss.detach()) < initial_loss / 2
    assert model.predictor(x)[:, 1].std() > 0


def test_optimizer_covers_each_trainable_parameter_once():
    model = nn.Module()
    model.core = nn.Linear(2, 2)
    model.head = nn.Linear(2, 2)
    model.core.bias.requires_grad_(False)
    groups = training_parameter_groups(model)
    assigned = [id(p) for group in groups for p in group["params"]]
    assert len(assigned) == len(set(assigned))
    assert set(assigned) == {id(p) for p in model.parameters() if p.requires_grad}
    assert [g["lr"] for g in groups] == [3e-5, 1e-3]
