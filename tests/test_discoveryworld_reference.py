"""Scientific reference behavior, numerical parsing, and real task completion."""
from importlib.util import find_spec
from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from experiments.discoveryworld.reference import (
    collect_episode, create, fit_hypotheses, parse_measurement,
    split_for_seed, tuning_choice,
)


class ReferenceReasoningTests(unittest.TestCase):
    def test_measurements_preserve_coordinates_and_ignore_crystal_numbers(self):
        self.assertEqual(parse_measurement(
            "thermometer", "quantum crystal 142: temperature of -4.25 degrees Celsius."),
            {"temperature_c": -4.25})
        self.assertEqual(parse_measurement(
            "spectrometer", "crystal 42\n- Channel 1: 39.01\n- Channel 5: 49.09"),
            {"spectrum_channel_1": 39.01, "spectrum_channel_5": 49.09})
        with self.assertRaises(ValueError):
            parse_measurement("thermometer", "Object with UUID 12345 is not accessible.")

    def test_competing_hypotheses_depend_on_observed_data(self):
        readings = [{"density": 2.5, "temperature": 7.8},
                    {"density": 4.5, "temperature": 9.3}]
        first = fit_hypotheses([(readings[0], 29.5), (readings[1], 51.5)])
        self.assertEqual(first[0].feature, "density")
        self.assertAlmostEqual(first[0].predict({"density": 8}), 90)
        # Both candidates interpolate two points; neither is silently removed.
        self.assertEqual({h.feature for h in first}, {"density", "temperature"})
        second = fit_hypotheses([(readings[0], 19.6), (readings[1], 22.6)])
        self.assertEqual(second[0].feature, "temperature")
        self.assertAlmostEqual(second[0].predict({"temperature": 8}), 20)

    def test_tuning_reads_dialog_options_instead_of_assuming_option_numbers(self):
        dialog = {"dialogIn": "The current resonance frequence is: 910 Hertz.\n"
                              "The allowable range is 0 to 10,000 Hz.",
                  "dialogOptions": {"7": "Increase frequency by 100 Hz",
                                    "2": "Decrease frequency by 10 Hz",
                                    "1": "Increase frequency by 1 Hz", "9": "Exit"}}
        self.assertEqual(tuning_choice(dialog, 1008), 7)
        self.assertEqual(tuning_choice(dialog, 900), 2)
        self.assertIsNone(tuning_choice(dialog, 910.1))

    def test_collector_split_is_not_a_policy_input(self):
        self.assertEqual(split_for_seed(100), "train")
        self.assertEqual(split_for_seed(200), "validation")
        self.assertEqual(split_for_seed(104), "heldout_rule")
        with self.assertRaises(ValueError):
            split_for_seed(0)
        with self.assertRaises(ValueError):
            create({"seed": 100})


@unittest.skipUnless(find_spec("discoveryworld"), "Install the pinned DiscoveryWorld simulator")
class ReferenceNativeTaskTests(unittest.TestCase):
    def test_measure_infer_and_activate_actual_reactors(self):
        # A shadow reference sees poison fields beside the public fields. Any
        # attempt to consult their contents fails; both references must choose
        # identical actions throughout the real experiment, not just initially.
        class Forbidden:
            def __getitem__(self, key):
                raise AssertionError("The policy consulted hidden simulator state")

            def __iter__(self):
                raise AssertionError("The policy iterated hidden simulator state")

            def __bool__(self):
                raise AssertionError("The policy used a hidden-state condition")

        class ShadowPolicy:
            def __init__(self):
                self.clean = create()
                self.poisoned = create()

            def __getattr__(self, name):
                return getattr(self.clean, name)

            def __call__(self, observation):
                poisoned = deepcopy(observation)
                poisoned.update({key: Forbidden() for key in
                                 ("seed", "rule", "task_metrics", "scoreCard", "world")})
                for group in poisoned["ui"]["nearbyObjects"]["objects"].values():
                    for obj in group:
                        obj["attributes"] = Forbidden()
                action = self.clean(observation)
                other = self.poisoned(poisoned)
                if action != other:
                    raise AssertionError("Hidden-state fields changed the policy action")
                return action

        shadow = ShadowPolicy()
        with TemporaryDirectory() as directory:
            output = Path(directory)
            with patch("experiments.discoveryworld.reference.create", return_value=shadow):
                # This instance's nearby radiation sources and closely spaced
                # calibration readings defeated the single-reading procedure.
                result = collect_episode(output, seed=103)
            self.assertIsNone(result["error"])
            self.assertTrue(result["task_metrics"][0]["completed_successfully"])
            summary = json.loads((output / "summary.json").read_text())
            measured = [e for e in summary["reference_evidence"] if e["kind"] == "measurement"]
            self.assertEqual(len(measured), 20)
            self.assertEqual(len({e["crystal"] for e in measured}), 4)
            tested = [e for e in summary["reference_evidence"] if e["kind"] == "hypothesis_test"]
            self.assertEqual(sum(e["activated"] for e in tested), 2)
            self.assertGreater(result["transitions"], 20)
            rows = [json.loads(line) for line in (output / "trajectory.jsonl").read_text().splitlines()]
            self.assertEqual(len(rows), result["transitions"])
            self.assertTrue(all(row["action_result"]["success"] for row in rows))
            self.assertTrue(any(row["action"].get("action") == "PUT" for row in rows))
            self.assertTrue(any("chosen_dialog_option_int" in row["action"] for row in rows))
            self.assertFalse(summary["is_learned_policy"])


if __name__ == "__main__":
    unittest.main()
