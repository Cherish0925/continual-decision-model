"""Public-grid probe choices and actual-feedback action budgeting."""
import json
from pathlib import Path
import unittest

from experiments.discoveryworld.scalar_probe import ScalarProbe, next_target, public_scalar_control


def dialog(lower=0, upper=10000, steps=(1, 10, 100, 1000)):
    options = {}
    for index, step in enumerate(steps):
        options[2 * index + 20] = f"Increase frequency by {step} Hz"
        options[2 * index + 21] = f"Decrease frequency by {step} Hz"
    options[99] = "Exit"
    return {"is_in_dialog": True, "dialogIn": f"The allowable range is {lower} to {upper} Hz.",
            "dialogOptions": options}


class ScalarProbeTests(unittest.TestCase):
    def test_native_public_dialog_can_probe_beyond_failed_point(self):
        fixture = json.loads((Path(__file__).parent / "fixtures/discoveryworld_measurements.json").read_text())
        native = next(row["transition"]["next_observation"]["ui"]["dialog_box"]
                      for row in fixture["cases"] if row["label"] == "reactor3")
        self.assertEqual(public_scalar_control(native).minimum_step_hz, 1.)
        probe = ScalarProbe()
        self.assertIsNone(probe.next_target(7115., 7115.47, 2., native))
        probe.observe_attempt(7115., activated=False, point_attempt=True)
        current = 7115.
        for expected in (7116., 7117., 7118., 7119.):
            target = probe.next_target(current, 7115.47, 2., native)
            self.assertEqual(target, expected)
            self.assertNotIn(target, probe.failed_values)
            # Synthetic feedback sequence tests the interface; the component
            # receives no future correct-frequency label or task state.
            probe.observe_attempt(target, activated=target == 7119.)
            current = target
        self.assertEqual(probe.adjustment_actions, 4)
        self.assertIsNone(probe.next_target(current, 7115.47, 2., native))

    def test_shorter_action_path_precedes_closeness_to_mean(self):
        public = dialog()
        self.assertEqual(next_target(5000., 5010., 10., public, {5000., 5010.}), 5001.)
        failed = {5000., 4990., 4999., 5001., 5010.}
        self.assertIsNone(next_target(5000., 5010., 10., public, failed, max_actions=1))
        self.assertEqual(next_target(5000., 5010., 10., public, failed, max_actions=2), 5009.)

    def test_public_fractional_steps_unreachable_values_and_clipped_limits(self):
        fractional = dialog(0, 2, (.25,))
        self.assertEqual(public_scalar_control(fractional).minimum_step_hz, .25)
        self.assertEqual(next_target(1., 1.1, .3, fractional, {1.}), 1.25)
        self.assertIsNone(next_target(0., 3., .1, dialog(0, 20, (2, 4)), {0.}))
        self.assertEqual(next_target(9., 10., .1, dialog(0, 10, (2,)), {9.}), 10.)
        self.assertIsNone(next_target(5., 100., 1., dialog(0, 10), {5.}))

    def test_unoffered_units_and_missing_range_cannot_supply_actions(self):
        public = dialog()
        public["dialogIn"] = "The allowable range is 0 to 10000 Celsius."
        self.assertIsNone(next_target(100., 100., 2., public, {100.}))
        self.assertIsNone(next_target(100., 100., 2., {"is_in_dialog": False}, {100.}))
        self.assertIsNone(next_target(100., 100., 2., dialog(), set()))
        self.assertIsNone(next_target(100., 100., 0., dialog(), {100.}))

    def test_only_completed_point_failure_starts_probing_and_proposals_do_not_count(self):
        probe = ScalarProbe()
        probe.observe_attempt(99., activated=False)
        self.assertFalse(probe.started)
        self.assertIsNone(probe.next_target(99., 100., 2., dialog()))
        probe.observe_attempt(100., activated=None, point_attempt=True)
        self.assertFalse(probe.started)
        self.assertNotIn(100., probe.failed_values)
        probe.observe_attempt(100., activated=False, point_attempt=True, adjustment_actions=0)
        target = probe.next_target(100., 100., 2., dialog())
        self.assertEqual(target, 101.)
        self.assertNotIn(target, probe.failed_values)
        self.assertEqual(probe.adjustment_actions, 0)
        self.assertEqual(probe.next_target(100., 100., 2., dialog()), target)
        probe.observe_attempt(target, activated=None)
        self.assertEqual(probe.adjustment_actions, 1)
        self.assertIsNone(probe.next_target(target, 100., 2., dialog()))
        probe.observe_attempt(target, activated=False, adjustment_actions=0)
        self.assertEqual(probe.adjustment_actions, 1)
        self.assertIn(target, probe.failed_values)

    def test_sixteen_extra_actual_adjustments_are_a_hard_budget(self):
        probe = ScalarProbe()
        public = dialog(0, 1000, (1,))
        probe.observe_attempt(100., activated=False, point_attempt=True)
        current = 100.
        for _ in range(16):
            target = probe.next_target(current, 100., 100., public)
            self.assertIsNotNone(target)
            self.assertNotIn(target, probe.failed_values)
            probe.observe_attempt(target, activated=False, point_attempt=True)
            current = target
        self.assertEqual(probe.adjustment_actions, 16)
        self.assertEqual(probe.remaining_actions, 0)
        self.assertIsNone(probe.next_target(current, 100., 100., public))
        with self.assertRaises(ValueError):
            ScalarProbe(max_adjustment_actions=17)

    def test_exhausted_band_and_already_active_point_stop(self):
        public = dialog(0, 4, (1,))
        self.assertIsNone(next_target(2., 2., 1., public, {0., 1., 2., 3., 4.}))
        probe = ScalarProbe()
        probe.observe_attempt(2., activated=True, point_attempt=True)
        self.assertIsNone(probe.next_target(2., 2., 1., public))


if __name__ == "__main__":
    unittest.main()
