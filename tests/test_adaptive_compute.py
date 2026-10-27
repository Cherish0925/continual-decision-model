"""Observable parity and gradient paths of conditional internal computation."""
import unittest
from unittest.mock import patch

import torch

from cdm.core import ContinualCore, CoreConfig
from cdm.digital import DigitalConfig, DigitalPolicy
from experiments.discoveryworld.adaptive_compute import JointPolicy, DEPTHS
from experiments.discoveryworld.belief_actions import BELIEF_DIM, BeliefActionHead


class AdaptiveComputeTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(9)
        core = ContinualCore(CoreConfig(event_dim=7, record_dim=6, actions=3, width=24,
                                       slots=3, heads=4, hidden=24, action_width=12), torch.tensor([-1., 0., 1.]))
        digital = DigitalPolicy(DigitalConfig(width=24, text_dim=12, image_dim=10), core)
        self.model = JointPolicy(digital, BeliefActionHead(48, 12, width=16)).eval()
        self.obs = {"numeric": torch.randn(2, 32), "text": torch.randn(2, 4, 12)}
        self.actions = {"text": torch.randn(2, 5, 12), "numeric": torch.randn(2, 5, 8),
                        "present": torch.ones(2, 5, dtype=torch.bool)}
        self.belief = torch.randn(2, 5, BELIEF_DIM)

    def test_branch_matches_direct_execution_and_ingests_only_once(self):
        state = self.model.digital.initial(2, "cpu")
        with patch.object(self.model.digital.core, "fast", wraps=self.model.digital.core.fast) as fast:
            _, predictions, states = self.model.proposals(self.obs, self.actions, self.belief, state)
        self.assertEqual(fast.call_count, 1)
        for index, depth in enumerate(DEPTHS):
            direct, continued, counts, _ = self.model.act(self.obs, self.actions, self.belief, state, refinements=depth)
            torch.testing.assert_close(direct, predictions[:, index])
            torch.testing.assert_close(continued["z"], states[index]["z"])
            self.assertEqual(counts.tolist(), [depth, depth])
        legacy, continued, _ = self.model.digital(self.obs, self.actions, state, predict_outcomes=False)
        old_reader = self.model.reader(self.model.digital.core.state_features(continued["h"], continued["z"]),
                                      self.actions["text"], self.belief, legacy)
        torch.testing.assert_close(old_reader, predictions[:, 1])

    def test_joint_loss_reaches_gru_workspace_reader_and_allocator(self):
        state = self.model.digital.initial(2, "cpu")
        allocation, logits, _ = self.model.proposals(self.obs, self.actions, self.belief, state)
        ce = -logits.log_softmax(-1)[..., 0]
        loss = (allocation.softmax(-1) * (ce + .01 * ce.new_tensor(DEPTHS))).mean()
        loss.backward()
        for component in (self.model.digital.core.gru, self.model.digital.core.block,
                          self.model.reader, self.model.allocator):
            total = sum(float(p.grad.abs().sum()) for p in component.parameters() if p.grad is not None)
            self.assertGreater(total, 0)


if __name__ == "__main__":
    unittest.main()
