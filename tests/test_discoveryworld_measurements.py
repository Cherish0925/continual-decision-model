"""Recorded native messages, binding and statistics; no simulator/model needed."""
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import re
import unittest

from experiments.discoveryworld.measurements import MeasurementLog, MeasurementSummary


FIXTURE = Path(__file__).parent / "fixtures" / "discoveryworld_measurements.json"


class MeasurementTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = {case["label"]: case["transition"] for case in json.loads(FIXTURE.read_text())["cases"]}

    def test_actual_five_instruments_preserve_values_units_and_exact_text(self):
        expected = {"density": ("density_g_cm3", 56.630, "g/cm^3"),
                    "temperature": ("temperature_c", 19.11, "degC"),
                    "gap": ("quantum_gap_nm", 31.77, "nm"),
                    "radiation_a1": ("radiation_usv_h", 17.15, "uSv/h")}
        log = MeasurementLog()
        for label, (quantity, value, unit) in expected.items():
            row = self.rows[label]
            records = log.append_transition(row)
            self.assertEqual(len(records), 1)
            record = records[0]
            self.assertEqual((record.quantity, record.value, record.unit), (quantity, value, unit))
            self.assertEqual(record.instrument_handle, row["action"]["arg1"])
            self.assertEqual(record.target_handle, row["action"]["arg2"])
            self.assertEqual(record.text, row["next_observation"]["ui"]["lastActionMessage"])
            self.assertEqual(record.sequence, row["index"])
        spectrum = log.append_transition(self.rows["spectrum"])
        self.assertEqual([record.value for record in spectrum], [23.81, 45.10, 38.96, 39.41, 16.46])
        self.assertEqual([record.quantity for record in spectrum], [f"spectrum_channel_{i}" for i in range(1, 6)])
        self.assertTrue(all(record.unit is None for record in spectrum))  # Native message specifies no unit.

    def test_repeated_samples_all_remain_and_targets_and_conditions_are_separate(self):
        log = MeasurementLog()
        for label in ("radiation_a1", "radiation_a2", "radiation_a3", "radiation_b"):
            log.append_transition(self.rows[label])
        self.assertEqual(len(log.records), 4)
        summaries = log.summaries()
        self.assertEqual(len(summaries), 2)
        a = next(summary for summary in summaries if summary.count == 3)
        self.assertEqual((a.minimum, a.maximum), (17.15, 17.19))
        self.assertAlmostEqual(a.mean, (17.15 + 17.16 + 17.19) / 3)
        self.assertEqual(a.latest, 17.19)
        self.assertEqual(len(a.numeric_row()), 8)
        self.assertIn("radiation meter measuring quantum crystal 4", a.text())
        self.assertAlmostEqual(a.numeric_features()["last_change"], .03)
        self.assertAlmostEqual(a.numeric_features()["total_change"], .04)
        # Another actual action with the same payload is retained, not deduplicated.
        log.observe(self.rows["radiation_a3"]["action"], self.rows["radiation_a3"]["action_result"],
                    self.rows["radiation_a3"]["next_observation"], sequence=1000)
        self.assertEqual(len(log.records), 5)
        self.assertEqual(max(summary.count for summary in log.summaries()), 4)
        moved = deepcopy(self.rows["radiation_a3"])
        moved["next_observation"]["ui"]["agentLocation"]["x"] += 1
        log.append_transition(moved)
        self.assertEqual(len(log.summaries()), 3)

    def test_failed_or_nonmeasurement_actions_cannot_reuse_a_popup_as_a_new_measurement(self):
        log = MeasurementLog()
        self.assertEqual(log.append_transition(self.rows["failed_action"]), [])
        stale = deepcopy(self.rows["radiation_a1"])
        stale["next_observation"]["ui"]["lastActionMessage"] = "The device did not return a measurement."
        self.assertTrue(stale["next_observation"]["ui"]["extended_action_message"])
        self.assertEqual(log.append_transition(stale), [])
        stale["action"] = {"action": "ROTATE_DIRECTION", "arg1": "north"}
        stale["next_observation"]["ui"]["lastActionMessage"] = self.rows["radiation_a1"]["next_observation"]["ui"]["lastActionMessage"]
        self.assertEqual(log.append_transition(stale), [])
        self.assertEqual(log.records, [])

    def test_dialog_frequency_tracks_the_interlocutor_and_its_public_contents(self):
        log = MeasurementLog()
        first = log.append_transition(self.rows["reactor1"])[0]
        self.assertEqual((first.target_handle, first.value, first.unit), (50759, 1657.6, "Hz"))
        self.assertTrue(first.contained_objects)
        self.assertEqual(log.append_transition(self.rows["exit1"]), [])
        self.assertIsNone(log.dialog_owner)
        for label, value in (("reactor3", 5000.), ("adjust3a", 4000.), ("adjust3b", 3000.)):
            record = log.append_transition(self.rows[label])[0]
            self.assertEqual((record.instrument_handle, record.target_handle, record.value), (10581, 10581, value))
        self.assertNotEqual(first.target_handle, record.target_handle)
        swapped = MeasurementLog().append_transition(self.rows["swapped_device"])[0]
        self.assertEqual(swapped.target_handle, 50759)
        self.assertEqual(swapped.value, 2171.12)
        # A partial history contains a reading but not enough evidence to name its owner.
        unknown = MeasurementLog()
        unbound = unknown.append_transition(self.rows["adjust3a"])[0]
        self.assertFalse(unbound.bound)
        self.assertEqual(unbound.value, 4000.)
        self.assertEqual(unknown.summaries(), [])

    def test_actual_reactor_contents_remain_bound_in_separate_historical_summaries(self):
        # Actual TALK returns from correction seed111: the same reactor reads
        # 5000 Hz with crystal 3, then with crystal 4, then again with crystal 4.
        log = MeasurementLog()
        for label in ("reactor_contents_before", "reactor_contents_after", "reactor_contents_after_again"):
            log.append_transition(self.rows[label])
        first, second = log.summaries()
        self.assertEqual(first.target_handle, second.target_handle)
        self.assertEqual(first.target_handle, 34810)
        self.assertEqual((first.contained_objects, second.contained_objects), ((16434,), (6122,)))
        self.assertEqual((first.count, second.count), (1, 2))
        self.assertEqual((first.first_sequence, second.first_sequence, second.last_sequence), (70, 266, 268))
        self.assertEqual((first.mean, second.mean), (5000., 5000.))
        # A consumer can resolve each historical handle from its own snapshot,
        # without consulting today's object registry or interpreting UUID values.
        for summary, name in ((first, "quantum crystal 3"), (second, "quantum crystal 4")):
            conditions = summary.conditions
            objects = {obj["uuid"]: obj for group in conditions["nearby"].values() for obj in group}
            objects.update((obj["uuid"], obj) for obj in conditions["accessible"] + conditions["inventory"])
            contained = objects[summary.contained_objects[0]]
            self.assertEqual(contained["name"], name)
            self.assertIn("[uuid: 34810]", contained["description"])
        self.assertEqual([r.contained_objects for r in log.records], [(16434,), (6122,), (6122,)])
        legacy = asdict(first)
        legacy.pop("contained_objects")
        self.assertEqual(MeasurementSummary(**legacy).contained_objects, ())

    def test_object_renaming_changes_routing_only_not_numeric_features(self):
        normal, renamed = MeasurementLog(), MeasurementLog()

        def rename(value, key=None):
            if isinstance(value, dict):
                return {k: rename(v, k) for k, v in value.items()}
            if isinstance(value, list):
                return [rename(v) for v in value]
            if key in ("uuid", "arg1", "arg2") and isinstance(value, int):
                return value + 700000
            if isinstance(value, str):
                return re.sub(r"\[uuid: (\d+)\]", lambda match: f"[uuid: {int(match.group(1)) + 700000}]", value)
            return value

        for label in ("radiation_a1", "radiation_a2", "radiation_a3"):
            normal.append_transition(self.rows[label])
            renamed.append_transition(rename(self.rows[label]))
        self.assertNotEqual(normal.records[0].target_handle, renamed.records[0].target_handle)
        self.assertEqual(normal.summaries()[0].numeric_features(), renamed.summaries()[0].numeric_features())


if __name__ == "__main__":
    unittest.main()
