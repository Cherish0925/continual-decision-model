import copy
import json
from pathlib import Path
import unittest

import torch

from cdm.digital import PublicObjectMemory, public_candidates
from experiments.discoveryworld.belief_actions import BELIEF_DIM, BeliefFeatures, BeliefActionHead
from experiments.discoveryworld.calibration import CalibrationLog
from experiments.discoveryworld.calibration_mixture import AffineMixture


class BeliefActionTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.model = AffineMixture({"x_mean": [0.] * 9, "x_scale": [1.] * 9,
                                   "y_mean_hz": 3000., "y_scale_hz": 1500.}, prior_mode="shared")

    def test_empty_history_and_future_fields_do_not_supply_evidence(self):
        observation = {"ui": {"inventoryObjects": [{"uuid": 4, "name": "thermometer"}],
                               "accessibleEnvironmentObjects": [{"uuid": 8, "name": "quantum crystal"}]},
                       "actions": {"USE": {"args": ["a", "b"]}}}
        log = CalibrationLog()
        log.observe_initial(observation)
        view = BeliefFeatures(log, self.model)
        candidates = public_candidates(observation)
        before = view.encode(observation, candidates)
        self.assertEqual(before.shape, (2, BELIEF_DIM))
        changed = copy.deepcopy(observation)
        changed.update(hidden_rule="temperature", evaluation_target=999., future_reward=1.)
        self.assertTrue(torch.equal(before, view.encode(changed, candidates)))
        transition = {"action": {"action": "USE", "arg1": 4, "arg2": 8},
                      "action_result": {"success": True}}
        view.observe(transition)
        after = view.encode(observation, candidates)
        self.assertFalse(torch.equal(before, after))

    def test_head_preserves_parent_initially_and_learns_action_difference(self):
        torch.manual_seed(9)
        actor = BeliefActionHead(16, 8, width=8)
        state, text, evidence = torch.randn(2, 16), torch.randn(2, 3, 8), torch.randn(2, 3, BELIEF_DIM)
        parent = torch.randn(2, 3)
        self.assertTrue(torch.equal(actor(state, text, evidence, parent), parent))
        optimizer = torch.optim.Adam(actor.parameters(), lr=.01)
        initial = torch.nn.functional.cross_entropy(parent, torch.tensor([0, 2]))
        for _ in range(8):
            optimizer.zero_grad()
            loss = torch.nn.functional.cross_entropy(actor(state, text, evidence, parent), torch.tensor([0, 2]))
            loss.backward()
            optimizer.step()
        self.assertLess(float(loss.detach()), float(initial))


if __name__ == "__main__":
    unittest.main()
