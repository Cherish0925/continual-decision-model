"""The learned physical distribution must inform acquisition before a test."""
from types import SimpleNamespace

import torch

from cdm.active_experiment import ActiveExperimentConfig, ActiveExperimentPolicy
from cdm.core import ContinualCore, CoreConfig
from experiments.physical_belief.observations import add_physical_context
from experiments.physical_belief.train_policy import prediction_loss


def test_unmeasured_relation_stays_in_predictive_training_target():
    means = torch.tensor([[0., 100.]], requires_grad=True)
    belief = {"means": means, "std": torch.tensor([[2., 20.]]),
              "weights": torch.tensor([[.5, .5]]), "eligible": torch.tensor([[True, False]])}
    loss = prediction_loss(belief, torch.tensor([100.]))
    expected = .9189385332046727 + torch.log(torch.tensor(20.)) - torch.log(torch.tensor(.5))
    torch.testing.assert_close(loss, expected)
    loss.backward()
    assert torch.isfinite(means.grad).all()


def test_shared_physical_context_can_train_measurement_choice():
    torch.manual_seed(31)
    torch.set_num_threads(1)
    core = ContinualCore(CoreConfig(event_dim=3, record_dim=3, actions=2, width=32,
        slots=2, heads=4, hidden=32, action_width=16), torch.tensor([-1., 1.]))
    policy = ActiveExperimentPolicy(ActiveExperimentConfig(36, 20, 30, action_width=16), core)
    means = torch.randn(2, 9, requires_grad=True)
    belief = {"means": means, "std": torch.ones(2, 9), "weights": torch.ones(2, 9) / 9,
              "support_counts": torch.zeros(2, 9), "eligible": torch.zeros(2, 9, dtype=torch.bool)}
    obs = {"event": torch.zeros(2, 36), "evidence": torch.randn(2, 36, 20),
           "evidence_mask": torch.ones(2, 36, dtype=torch.bool),
           "candidates": torch.randn(2, 29, 30), "action_mask": torch.ones(2, 29, dtype=torch.bool)}
    obs["action_mask"][:, 20:] = False
    updated = add_physical_context(obs, belief, SimpleNamespace(y_mean=0., y_scale=1.))
    out = policy(**updated, state=policy.initial(2, "cpu"))
    out["logits"][:, :20].sum().backward()
    assert torch.isfinite(means.grad).all() and means.grad.abs().sum() > 1e-8
