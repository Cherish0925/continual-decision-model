"""Prediction influences decisions without reading future observations."""
import unittest
from unittest.mock import patch

import torch

from cdm.core import ContinualCore, CoreConfig
from cdm.digital import DigitalConfig, DigitalPolicy
from experiments.discoveryworld.belief_actions import BELIEF_DIM, BeliefActionHead
from experiments.discoveryworld.predictive_actions import PredictiveModel
from experiments.discoveryworld.train_predictive_actions import future_targets, forecast_loss


class PredictiveActionTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(19)
        core = ContinualCore(CoreConfig(event_dim=7, record_dim=6, actions=3, width=24,
                                       slots=3, heads=4, hidden=24, action_width=12), torch.tensor([-1., 0., 1.]))
        digital = DigitalPolicy(DigitalConfig(width=24, text_dim=12, image_dim=10), core)
        self.model = PredictiveModel(digital, BeliefActionHead(48, 12, width=16)).eval()
        self.obs = {"numeric": torch.randn(2, 32), "text": torch.randn(2, 4, 12)}
        self.actions = {"text": torch.randn(2, 5, 12), "numeric": torch.randn(2, 5, 8),
                        "present": torch.tensor([[1, 1, 1, 1, 0], [1, 1, 1, 1, 1]], dtype=torch.bool)}
        self.belief = torch.randn(2, 5, BELIEF_DIM)

    def test_forecasts_affect_persistent_state_without_reassimilating_observation(self):
        initial = self.model.digital.initial(2, "cpu")
        with patch.object(self.model.digital.core, "fast", wraps=self.model.digital.core.fast) as fast:
            scores, _, states = self.model.proposals(self.obs, self.actions, self.belief, initial)
        self.assertEqual(fast.call_count, 1)
        _, _, without = self.model.proposals(self.obs, self.actions, self.belief, initial, use_forecasts=False)
        torch.testing.assert_close(states[0]["z"], without[0]["z"])
        self.assertFalse(torch.allclose(states[-1]["z"], without[-1]["z"]))
        self.assertTrue(torch.isneginf(scores[0, :, 4]).all())

    def test_observed_forecast_loss_trains_core_and_feedback(self):
        initial = self.model.digital.initial(2, "cpu")
        scores, predictions, _ = self.model.proposals(self.obs, self.actions, self.belief, initial)
        outcome = torch.zeros(2, 3, 14)
        outcome[..., 0] = 1
        outcome[..., 9:] = 1
        received, future = forecast_loss(predictions[:, :, 0], outcome, torch.ones(2, 3, 4) * .1,
                                        torch.ones(2, 3, 4))
        loss = received.mean() + future.mean() - scores.log_softmax(-1)[..., 0].mean()
        loss.backward()
        for component in (self.model.digital.core.gru, self.model.digital.core.block,
                          self.model.forecaster, self.model.write):
            self.assertGreater(sum(float(p.grad.abs().sum()) for p in component.parameters() if p.grad is not None), 0)

    def test_time_limit_tail_does_not_teach_zero_future_return(self):
        rows = [{"training_reward": .1}, {"training_reward": .2, "truncated": True}]
        targets, masks = future_targets(rows, gamma=1, action_cost=0)
        self.assertEqual(masks, [[1., 0., 0., 0.], [1., 0., 0., 0.]])
        self.assertAlmostEqual(targets[0][1], .3)
        rows[-1]["terminated"] = True
        self.assertTrue(all(all(row) for row in future_targets(rows)[1]))


if __name__ == "__main__":
    unittest.main()
