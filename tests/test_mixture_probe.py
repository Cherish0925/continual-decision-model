import unittest

import torch

from experiments.discoveryworld.mixture_probe import MixtureProbe
from test_scalar_probe import dialog


class MixtureProbeTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_failed_dominant_relation_releases_a_distant_alternative(self):
        probe = MixtureProbe([.8, .2], [20., 80.], [.2, 1.], tolerance_hz=2.)
        probe.observe_attempt(20., activated=False, point_attempt=True)
        public = dialog(0, 100, (1, 10))
        current = 20.
        for _ in range(16):
            target = probe.next_target(current, 20., .2, public)
            self.assertIsNotNone(target)
            self.assertIn(abs(target-current), [1., 10.])
            activated = abs(target-80.) < 2
            probe.observe_attempt(target, activated=activated)
            current = target
            if activated:
                break
        self.assertTrue(probe.activated)
        self.assertLess(probe.last_choice['remaining_hypothesis_weights'][0], 1e-6)
        self.assertLessEqual(probe.adjustment_actions, 16)

    def test_overlapping_failed_intervals_are_not_counted_twice(self):
        probe = MixtureProbe([1.], [10.], [4.], tolerance_hz=2.)
        probe.observe_attempt(10., activated=False, point_attempt=True)
        probe.observe_attempt(11., activated=False)
        expected = probe._mass(0., 20.) - probe._mass(8., 13.)
        self.assertTrue(torch.allclose(probe.remaining_mass(0.,20.),expected))
        self.assertEqual(float(probe.remaining_mass(9.,12.)),0.)

    def test_received_activation_ends_experiment(self):
        probe = MixtureProbe([1.], [10.], [4.], tolerance_hz=2.)
        probe.observe_attempt(10., activated=True, point_attempt=True)
        self.assertIsNone(probe.next_target(10.,10.,4.,dialog()))


if __name__ == '__main__':
    unittest.main()
