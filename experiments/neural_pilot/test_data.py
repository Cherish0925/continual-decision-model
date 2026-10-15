"""Verify the observed-information contract of the supervised pilot."""
import unittest

import torch

from train import episodes


class ObservedCueTaskTest(unittest.TestCase):
    def test_targets_follow_observed_history_and_queries_hide_cues(self):
        data = episodes(64, torch.Generator().manual_seed(91), torch.device("cpu"), 8)
        self.assertTrue(torch.equal(data["observations"][:, 4:],
                                    torch.zeros_like(data["observations"][:, 4:])))
        for b in range(64):
            remembered = {}
            for t in range(12):
                context = int(data["environments"][b, t].argmax())
                goal = int(data["goals"][b, t].argmax())
                if t < 4:
                    self.assertNotIn(context, remembered)
                    remembered[context] = int(data["observations"][b, t, :4].argmax())
                self.assertIn(context, remembered)
                self.assertEqual(int(data["labels"][b, t]), (remembered[context] + goal) % 4)
            self.assertEqual(len(remembered), 4)

    def test_random_stream_reproduction(self):
        first = episodes(8, torch.Generator().manual_seed(92), torch.device("cpu"), 8)
        second = episodes(8, torch.Generator().manual_seed(92), torch.device("cpu"), 8)
        for name in first:
            self.assertTrue(torch.equal(first[name], second[name]))


if __name__ == "__main__":
    unittest.main()
