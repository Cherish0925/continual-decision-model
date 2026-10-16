"""Replay minibatches must reach later streams and preserve paired fields."""
import unittest

import numpy as np

from experiments.continual_agent.replay import sample_rows


class ReplayTests(unittest.TestCase):
    def test_all_recorded_streams_can_enter_training(self):
        rng = np.random.default_rng(17)
        seen = set()
        for _ in range(256):
            for _, rows in sample_rows([512] * 4, 32, rng):
                seen.update(rows.tolist())
        self.assertEqual(seen, set(range(512)))

    def test_one_minibatch_draws_from_multiple_records_and_later_rows(self):
        selected = sample_rows([512] * 4, 32, np.random.default_rng(0))
        self.assertEqual({i for i, _ in selected}, set(range(4)))
        self.assertEqual(sum(len(rows) for _, rows in selected), 32)
        self.assertTrue(any(np.any(rows >= 32) for _, rows in selected))
        self.assertEqual(len({(i, int(row)) for i, rows in selected for row in rows}), 32)

    def test_unequal_survey_batches_keep_fields_paired(self):
        records = []
        for i, size in enumerate([1, 3, 8]):
            identity = 100 * i + np.arange(size)
            records.append((identity, 10 * identity, identity + 1, -identity))
        selected = sample_rows([len(r[0]) for r in records], 50, np.random.default_rng(1))
        identity, age, goal, target = (
            np.concatenate([records[i][j][rows] for i, rows in selected]) for j in range(4)
        )
        self.assertEqual(set(identity), {0, 100, 101, 102, *range(200, 208)})
        np.testing.assert_array_equal(age, 10 * identity)
        np.testing.assert_array_equal(goal, identity + 1)
        np.testing.assert_array_equal(target, -identity)

    def test_sampling_repeats_for_the_same_seed(self):
        a = sample_rows([2, 9, 13], 8, np.random.default_rng(3))
        b = sample_rows([2, 9, 13], 8, np.random.default_rng(3))
        self.assertEqual([i for i, _ in a], [i for i, _ in b])
        for (_, rows_a), (_, rows_b) in zip(a, b):
            np.testing.assert_array_equal(rows_a, rows_b)


if __name__ == "__main__":
    unittest.main()
