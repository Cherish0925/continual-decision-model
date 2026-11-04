"""Opaque reagent labels and hidden table cells must not determine decisions."""
import unittest

import torch

from cdm.categorical_measurement import CategoricalMeasurement, relation_features
from experiments.measured_reactions.study import add_measurement, make_policy, rollout


class CategoricalMeasurementTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(7)
        self.codes = torch.randint(0, 3, (2, 12, 8))
        self.present = torch.ones(2, 12, dtype=torch.bool)
        self.indices = torch.tensor([[0, 3], [0, 4]])
        self.values = torch.tensor([[.2, .9], [.6, .1]])

    def test_relabeling_and_candidate_order(self):
        renamed = self.codes * 17 + torch.arange(8)[None, None] * 101
        x = relation_features(self.codes, self.present, self.indices, self.values)
        torch.testing.assert_close(x, relation_features(renamed, self.present, self.indices, self.values))
        head = CategoricalMeasurement(64, 16)
        mean, sd = head(head.prepare(x, self.codes), self.indices, self.values)
        other, other_sd = head(head.prepare(torch.randn_like(x), renamed), self.indices, self.values)
        torch.testing.assert_close(mean, other)
        torch.testing.assert_close(sd, other_sd)
        order = torch.randperm(12)
        inverse = torch.argsort(order)
        moved = relation_features(renamed[:, order], self.present[:, order], inverse[self.indices], self.values)
        torch.testing.assert_close(moved, x[:, order])
        moved_mean, _ = head(head.prepare(moved, renamed[:, order]), inverse[self.indices], self.values)
        torch.testing.assert_close(moved_mean, mean[:, order])

    def test_reading_changes_related_predictions(self):
        head = CategoricalMeasurement(64, 16)
        codes = torch.zeros(1, 3, 8, dtype=torch.long)
        codes[0, 1, 0] = 1
        codes[0, 2] = 2
        encoded = head.prepare(torch.zeros(1, 3, 64), codes)
        low, _ = head(encoded, torch.tensor([[0]]), torch.tensor([[.1]]))
        high, _ = head(encoded, torch.tensor([[0]]), torch.tensor([[.9]]))
        self.assertGreater((high - low)[0, 1], (high - low)[0, 2])

    def test_all_new_policies_train_and_relational_actions_ignore_names(self):
        y = torch.rand(2, 12)
        for mode in ('categorical', 'relational', 'relational_gru', 'relational_blind', 'relational_kernel'):
            with self.subTest(mode=mode):
                model = make_policy(16, 4)
                add_measurement(model, mode)
                batch = (torch.randn(2, 12, 64), y, self.codes, torch.zeros_like(y), self.present)
                loss = rollout(batch, 4, mode, model, learning=True)
                loss.backward()
                self.assertTrue(torch.isfinite(model.measurement.log_weights.grad).all())
                if mode.startswith('relational'):
                    with torch.no_grad():
                        first = rollout(batch, 4, mode, model)
                        renamed = (torch.randn_like(batch[0]), y, self.codes * 3 + 100,
                                   torch.rand_like(y), self.present)
                        torch.testing.assert_close(rollout(renamed, 4, mode, model), first)


if __name__ == '__main__':
    unittest.main()
