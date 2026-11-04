"""Changing one observed response must change predictions about untried related actions."""
import unittest

import torch

from cdm.conditional_measurement import ConditionalMeasurement


class ConditionalMeasurementTest(unittest.TestCase):
    def test_related_action_changes_more_and_renumbering_preserves_the_result(self):
        head = ConditionalMeasurement(2, 4)
        z = torch.tensor([[[1., 0.], [.8, .6], [-1., 0.]]])
        prior = torch.full((1, 3), .5)
        idx = torch.tensor([[0]])
        low, _ = head((z, prior), idx, torch.tensor([[.1]]))
        high, sd = head((z, prior), idx, torch.tensor([[.9]]))
        self.assertGreater((high - low)[0, 1], (high - low)[0, 2])
        self.assertGreater((high - low)[0, 2], 0.)
        order = [2, 0, 1]
        moved, moved_sd = head((z[:, order], prior[:, order]), torch.tensor([[1]]), torch.tensor([[.9]]))
        torch.testing.assert_close(moved, high[:, order])
        torch.testing.assert_close(moved_sd, sd[:, order])
        masked, masked_sd = head((z, prior), idx, torch.tensor([[.9]]), use_values=False)
        torch.testing.assert_close(masked, prior)
        torch.testing.assert_close(masked_sd, sd)


if __name__ == '__main__':
    unittest.main()
