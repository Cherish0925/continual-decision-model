"""The option-value objective must actually train both activity factors."""
import unittest

import torch

from experiments.lifetime.train import expected_advantage, factor_probs


class ActivityValueGradientTests(unittest.TestCase):
    def test_matches_expected_value_gradient_with_masked_options(self):
        logits = torch.tensor([[0.3, -0.2, 0.1, 0.4, -0.4, 0.2, 0.5]], requires_grad=True)
        estimates = torch.tensor([[0., 3., 2., -1., 4., 1., 3.]], requires_grad=True)
        allowed = torch.tensor([[True, True, True, False, True, True, True]])
        probs = factor_probs(logits, allowed)
        loss = -expected_advantage(probs, estimates.detach())
        gradient = torch.autograd.grad(loss, logits, retain_graph=True)[0]
        expected = torch.autograd.grad(-(probs * estimates.detach()).sum(1).mean(), logits)[0]
        torch.testing.assert_close(gradient, expected)
        self.assertGreater(gradient[:, :2].abs().sum().item(), 0.1)
        self.assertGreater(gradient[:, 2:].abs().sum().item(), 0.1)
        self.assertEqual(gradient[0, 3].item(), 0)
        self.assertLess(gradient[0, 1].item(), 0)  # descent raises the better measurement logit
        self.assertIsNone(estimates.grad)

if __name__ == "__main__":
    unittest.main()
