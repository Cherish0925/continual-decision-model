"""Executed actions, predicted feedback and candidate-specific state access."""
from copy import deepcopy
import unittest

import torch
import torch.nn.functional as F

from cdm.core import ContinualCore, CoreConfig
from cdm.digital import DigitalConfig, DigitalPolicy


class ActionStateTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(7)
        config = CoreConfig(event_dim=16, record_dim=8, actions=1, width=16,
                            slots=3, heads=2, hidden=16, action_width=16)
        self.base = DigitalPolicy(DigitalConfig(width=16, text_dim=8, image_dim=8,
            outcome_dim=14), ContinualCore(config, torch.zeros(1, 1)))
        self.obs = dict(numeric=torch.randn(1, 32).expand(2, -1),
                        text=torch.randn(1, 4, 8).expand(2, -1, -1),
                        text_present=torch.ones(2, 4, dtype=torch.bool))
        self.actions = dict(text=torch.randn(1, 2, 8).expand(2, -1, -1),
                            numeric=torch.randn(1, 2, 8).expand(2, -1, -1),
                            present=torch.ones(2, 2, dtype=torch.bool))

    def test_extension_preserves_loaded_decisions_until_training_and_return_options(self):
        model = DigitalPolicy.with_action_connections(self.base, candidate_read=True, action_conditioned=True)
        old, new = self.base.initial(2, 'cpu'), model.initial(2, 'cpu')
        with torch.no_grad():
            for _ in range(3):
                first, old, outcome = self.base(self.obs, self.actions, old)
                second, new, predicted = model(self.obs, self.actions, new)
                torch.testing.assert_close(first, second)
                torch.testing.assert_close(old['h'], new['h'])
                torch.testing.assert_close(old['z'], new['z'])
                torch.testing.assert_close(outcome, predicted)
                _, _, returned = model(self.obs, self.actions, new, predict_outcomes=False)
                self.assertIsNone(returned)
                new, _ = model.commit_action(self.obs, self.actions, new, torch.tensor([0, 1]))
        self.assertFalse(self.base.config.action_conditioned)

    def test_same_observation_can_require_different_actions_after_different_commands(self):
        model = DigitalPolicy.with_action_connections(self.base, candidate_read=True, action_conditioned=True)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.006)
        executed, target = torch.tensor([0, 1]), torch.tensor([1, 0])
        for _ in range(120):
            optimizer.zero_grad()
            _, state, _ = model(self.obs, self.actions, model.initial(2, 'cpu'), predict_outcomes=False)
            state, _ = model.commit_action(self.obs, self.actions, state, executed)
            logits, _, _ = model(self.obs, self.actions, state, predict_outcomes=False)
            loss = F.cross_entropy(logits, target)
            loss.backward()
            optimizer.step()
        self.assertEqual(logits.argmax(-1).tolist(), target.tolist())
        # A command does not rewrite the observation or reveal the target.
        with torch.no_grad():
            _, state, _ = model(self.obs, self.actions, model.initial(2, 'cpu'))
            control, _ = model.commit_action(self.obs, self.actions, state, torch.tensor([0, 0]))
            logits, _, _ = model(self.obs, self.actions, control)
            torch.testing.assert_close(logits[0], logits[1])
            invalid, _ = model.commit_action(self.obs, self.actions, state, torch.tensor([-1, -1]))
            self.assertFalse(invalid['pending_action'].any())

    def test_prediction_changes_decisions_and_action_loss_reaches_predictor(self):
        model = DigitalPolicy.with_action_connections(self.base, candidate_read=True, action_conditioned=True)
        torch.nn.init.normal_(model.prediction_read.weight, std=.1)
        logits, _, _ = model(self.obs, self.actions, model.initial(2, 'cpu'), predict_outcomes=False)
        loss = F.cross_entropy(logits, torch.zeros(2, dtype=torch.long))
        loss.backward()
        self.assertGreater(model.transition_numeric.weight.grad.abs().sum().item(), 0.)
        self.assertGreater(model.outcome[-1].weight.grad.abs().sum().item(), 0.)
        altered = deepcopy(model)
        with torch.no_grad():
            altered.transition_numeric.weight.add_(torch.randn_like(altered.transition_numeric.weight))
            changed, _, _ = altered(self.obs, self.actions, altered.initial(2, 'cpu'), predict_outcomes=False)
        self.assertGreater((changed - logits).abs().max().item(), 1e-4)


if __name__ == '__main__':
    unittest.main()
