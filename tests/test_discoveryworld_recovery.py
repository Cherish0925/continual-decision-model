"""Recovery after real student deviations, with no simulator-state oracle."""
import contextlib
from copy import deepcopy
from importlib.util import find_spec
import io
from tempfile import TemporaryDirectory
import unittest

from experiments.discoveryworld.adapter import DiscoveryWorldEnv
from experiments.discoveryworld.recovery import RecoveryExpert
from experiments.discoveryworld.reference import AffineHypothesis, INSTRUMENTS


class RecoveryReasoningTests(unittest.TestCase):
    def _tuning_expert(self, prediction):
        objects = []
        for reactor, crystal in zip(range(1, 5), range(11, 15)):
            state = "activated" if reactor < 3 else "uncalibrated"
            objects.extend([{"uuid": reactor, "name": f"crystal reactor ({state})"},
                            {"uuid": crystal, "name": f"quantum crystal {crystal}",
                             "description": f"in reactor [uuid: {reactor}]"}])
        observation = {"ui": {"world_steps": 1,
                              "inventoryObjects": [{"uuid": 20 + i, "name": name}
                                                   for i, name in enumerate(INSTRUMENTS)],
                              "accessibleEnvironmentObjects": objects,
                              "nearbyObjects": {"objects": {}},
                              "taskProgress": [{"completed": False}],
                              "dialog_box": {"is_in_dialog": True,
                                             "dialogIn": "current resonance frequency is: 1250 Hertz",
                                             "dialogOptions": {"1": "Increase frequency by 1 Hz",
                                                               "2": "Decrease frequency by 1 Hz",
                                                               "3": "Exit"}}}}
        expert = RecoveryExpert(observation)
        expert.calibration = {11: 100, 12: 200}
        expert.readings = {c: {name: {"x": prediction} for name in INSTRUMENTS}
                           for c in range(11, 15)}
        expert.hypotheses = [AffineHypothesis("x", slope, 0, 0) for slope in (1, 2)]
        expert.dialog_owner = 3
        expert._advance()
        return expert

    def _tuning_result(self, expert, *, activated, completed=False):
        observation = deepcopy(expert.observation)
        observation["ui"]["world_steps"] += 1
        observation["ui"]["taskProgress"][0]["completed"] = completed
        observation["ui"]["dialog_box"]["dialogIn"] = "current resonance frequency is: 1251 Hertz"
        for obj in observation["ui"]["accessibleEnvironmentObjects"]:
            if obj["uuid"] == 3:
                obj["name"] = f"crystal reactor ({'activated' if activated else 'uncalibrated'})"
        expert.observe({"chosen_dialog_option_int": 1}, observation, {"success": True})

    def test_native_activation_before_prediction_and_terminal_action_are_recorded(self):
        for completed in (False, True):
            with self.subTest(completed=completed):
                expert = self._tuning_expert(1252.75)
                self._tuning_result(expert, activated=True, completed=completed)
                trial, = expert.trials
                self.assertEqual((trial["prediction_hz"], trial["applied_hz"]), (1252.75, 1251))
                self.assertTrue(trial["activation_changed"])
                self.assertFalse(trial["prediction_reached"])
                self.assertTrue(trial["supports_hypothesis"])
                self.assertEqual(expert.selected, expert.hypotheses[0])
                self.assertEqual(expert.completed, completed)
                if completed:
                    self.assertEqual(expert.phase, "complete")
                else:
                    self._tuning_result(expert, activated=True)
                    self.assertEqual(len(expert.trials), 1)

    def test_incidental_activation_does_not_support_a_different_prediction(self):
        for prediction in (1253, 2000):
            with self.subTest(prediction=prediction):
                expert = self._tuning_expert(prediction)
                self._tuning_result(expert, activated=True, completed=True)
                trial, = expert.trials
                self.assertTrue(trial["activated"])
                self.assertFalse(trial["prediction_consistent"])
                self.assertFalse(trial["supports_hypothesis"])
                self.assertIsNone(expert.selected)
                self.assertEqual(expert.trial_indices, {})

    def test_exhausted_prediction_cannot_block_own_or_other_terminal_outcome(self):
        for exhausted in (13, 14):
            with self.subTest(exhausted_crystal=exhausted):
                expert = self._tuning_expert(1252.75)
                expert.trial_indices[exhausted] = len(expert.hypotheses)
                self._tuning_result(expert, activated=True, completed=True)
                trial, = expert.trials
                self.assertTrue(expert.completed)
                self.assertTrue(trial["activated"])
                self.assertEqual(trial["applied_hz"], 1251)
                self.assertEqual(trial["crystal"], 13)
                if exhausted == 13:
                    self.assertIsNone(trial["hypothesis"])
                    self.assertIsNone(trial["prediction_hz"])
                    self.assertFalse(trial["supports_hypothesis"])
                    self.assertIsNone(expert.selected)
                else:
                    self.assertTrue(trial["supports_hypothesis"])
                    self.assertEqual(expert.selected, expert.hypotheses[0])

    def test_an_old_frequency_is_not_relabelled_as_the_current_applied_value(self):
        expert = self._tuning_expert(1252.75)
        expert.reactor_frequencies[3] = (1251, 0)
        observation = deepcopy(expert.observation)
        observation["ui"]["world_steps"] += 1
        observation["ui"]["dialog_box"] = {}
        for obj in observation["ui"]["accessibleEnvironmentObjects"]:
            if obj["uuid"] == 3:
                obj["name"] = "crystal reactor (activated)"
        expert.observe({"action": "PUT", "arg1": 13, "arg2": 3}, observation, {"success": True})
        trial, = expert.trials
        self.assertIsNone(trial["applied_hz"])
        self.assertEqual(trial["last_observed_frequency_hz"], 1251)
        self.assertEqual(trial["frequency_observed_event"], 0)
        self.assertFalse(trial["supports_hypothesis"])
        self.assertIsNone(expert.selected)

    def test_failed_prediction_and_later_deactivation_remain_distinct(self):
        expert = self._tuning_expert(1251.25)
        self._tuning_result(expert, activated=False)
        trial, = expert.trials
        self.assertTrue(trial["prediction_reached"])
        self.assertFalse(trial["activated"])
        self.assertFalse(trial["activation_changed"])
        self.assertEqual(expert.trial_indices[13], 1)
        expert = self._tuning_expert(1252.75)
        self._tuning_result(expert, activated=True)
        self._tuning_result(expert, activated=False)
        self.assertEqual(len(expert.trials), 2)
        self.assertTrue(expert.trials[-1]["previously_activated"])
        self.assertFalse(expert.trials[-1]["activated"])
        self.assertFalse(expert.trials[-1]["supports_hypothesis"])

    def test_newly_supported_hypothesis_leads_recovery_after_prior_failed_trials(self):
        observation = {"ui": {"world_steps": 1, "inventoryObjects": [],
                              "accessibleEnvironmentObjects": [],
                              "nearbyObjects": {"objects": {}}, "taskProgress": []}}
        expert = RecoveryExpert(observation)
        expert.hypotheses = [AffineHypothesis("x", slope, 0, 0) for slope in (1, 2, 3)]
        expert.readings[17] = {"densitometer": {"x": 10}}
        expert.readings[29] = {"densitometer": {"x": 20}}
        # Two crystals already reached different points in the old trial order.
        # New experimental support changes that order; positional cursors from
        # the old order must not make either crystal skip the supported relation.
        expert.trial_indices = {17: 2, 29: 1}
        expert._accept_hypothesis(expert.hypotheses[2])
        self.assertEqual(expert._target(17)[0], 30)
        self.assertEqual(expert._target(29)[0], 60)


@unittest.skipUnless(find_spec("discoveryworld"), "Install the pinned DiscoveryWorld simulator")
class RecoveryTests(unittest.TestCase):
    def test_actual_deviations_preserve_measurements_and_resume_science(self):
        with TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            env = DiscoveryWorldEnv(directory, max_steps=150)
            observation = env.reset("Reactor Lab", "Normal", 110)
            expert = RecoveryExpert(observation)

            def step(action):
                nonlocal observation
                transition = env.step(action)
                observation = env.policy_input()
                expert.observe(action, observation, transition["action_result"])
                return transition

            # Querying an unexecuted suggestion cannot increment evidence,
            # consume a measurement, or change a pending scientific objective.
            initial = deepcopy(expert.summary())
            first = expert.recommend(observation)
            for _ in range(3):
                self.assertEqual(expert.recommend(observation), first)
                self.assertEqual(expert.summary(), initial)

            while expert._instrument("densitometer") not in expert._inventory():
                step(expert.recommend(observation))
            density = expert._instrument("densitometer")
            step({"action": "TELEPORT_TO_LOCATION", "arg1": "reactor lab"})
            reactor = expert._ids("crystal reactor")[0]
            step({"action": "TELEPORT_TO_OBJECT", "arg1": reactor})
            failed = step({"action": "PICKUP", "arg1": reactor})
            self.assertFalse(failed["action_result"]["success"])
            self.assertNotEqual(expert.recommend(observation), {"action": "PICKUP", "arg1": reactor})

            # An off-route but valid density observation is retained and will
            # satisfy that coordinate later. Repetition does not advance phases.
            step({"action": "TELEPORT_TO_LOCATION", "arg1": "science lab (crystal bench)"})
            crystal = next(c for c in expert._ids("quantum crystal") if c not in expert.anchor_pairs.values())
            step({"action": "TELEPORT_TO_OBJECT", "arg1": crystal})
            measurement = {"action": "USE", "arg1": density, "arg2": crystal}
            step(measurement)
            values = deepcopy(expert.readings[crystal]["densitometer"])
            step(measurement)
            self.assertEqual(expert.readings[crystal]["densitometer"], values)
            self.assertEqual(len(expert.raw_measurements), 2)
            self.assertNotEqual(expert.recommend(observation), measurement)

            # The native USE route can enter a device dialogue; correction must
            # exit that actual conversation before attempting ordinary actions.
            step({"action": "TELEPORT_TO_OBJECT", "arg1": reactor})
            entered = step({"action": "USE", "arg1": reactor, "arg2": density})
            self.assertTrue(entered["action_result"]["success"])
            self.assertTrue(observation["ui"]["dialog_box"]["dialogOptions"])
            correction = expert.recommend(observation)
            self.assertIn("chosen_dialog_option_int", correction)
            step(correction)

            # Resume to a full controlled radiation batch, not just one valid
            # navigation action. Repeated radiation USE is intentional science.
            moved_during_sampling = False
            while not expert.radiation_complete and not env.ended:
                qualified = [r for r in expert.raw_measurements
                             if r["instrument"] == "radiation meter" and r["qualified"]]
                if len(qualified) == 10 and not moved_during_sampling:
                    location = observation["ui"]["agentLocation"]
                    for direction, (dx, dy) in {"north": (0, -1), "east": (1, 0),
                                               "south": (0, 1), "west": (-1, 0)}.items():
                        if direction not in location["directions_you_can_move"]:
                            continue
                        hypothetical = deepcopy(observation)
                        hypothetical["ui"]["agentLocation"].update(x=location["x"] + dx, y=location["y"] + dy)
                        if expert._separation(hypothetical) >= 9:
                            step({"action": "MOVE_DIRECTION", "arg1": direction})
                            moved_during_sampling = True
                            break
                step(expert.recommend(observation))
            self.assertTrue(expert.radiation_complete)
            self.assertTrue(moved_during_sampling)
            samples = next(iter(expert.radiation_complete.values()))
            self.assertEqual(len(samples), 32)
            self.assertEqual(len(set(samples)), 32)
            self.assertGreater(len(expert.calibration), 1)
            self.assertGreater(expert.summary()["measured_coordinates"], 1)
            self.assertEqual(expert.readings[crystal]["densitometer"], values)
            # A qualified observation is tied to a real event, not to a query
            # count; all those event numbers refer to the executed prefix.
            self.assertTrue(all(0 < event <= env.steps for event in samples))


if __name__ == "__main__":
    unittest.main()
