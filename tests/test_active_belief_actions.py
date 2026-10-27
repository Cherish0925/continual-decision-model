import unittest

import torch

from experiments.discoveryworld.train_active_belief_actions import reader_objective


class ActiveBeliefTests(unittest.TestCase):
    def test_unexecuted_neural_proposal_has_no_policy_gradient(self):
        logits = torch.tensor([[0., 0., -torch.inf], [0., 0., 0.]], requires_grad=True)
        target = torch.tensor([1, 2])
        old = torch.distributions.Categorical(logits=logits.detach()).log_prob(target)
        policy, entropy = reader_objective(logits, target, old, torch.tensor([1., 100.]),
                                           torch.tensor([True, False]), .1)
        (policy - .001 * entropy).backward()
        self.assertLess(float(logits.grad[0, 1]), 0.)
        self.assertGreater(float(logits.grad[0, 0]), 0.)
        self.assertEqual(float(logits.grad[0, 2]), 0.)
        self.assertTrue(torch.equal(logits.grad[1], torch.zeros(3)))
        self.assertTrue(torch.isfinite(logits.grad).all())


if __name__ == '__main__':
    unittest.main()
