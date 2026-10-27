"""Recorded public calibration history, causal joins, and counterfactual boundaries."""
from copy import deepcopy
from dataclasses import asdict
import gzip
import json
from pathlib import Path
import unittest

from experiments.discoveryworld.calibration import CalibrationLog, FEATURE_QUANTITIES


def fixture():
    path = Path(__file__).parent / "fixtures/discoveryworld_calibration.json.gz"
    return json.loads(gzip.decompress(path.read_bytes()))["transitions"]


class CalibrationTests(unittest.TestCase):
    def test_actual_anchor_join_and_preintervention_queries(self):
        log, first_support, snapshots = CalibrationLog(), {}, {}
        for row in fixture():
            log.append_transition(row)
            for support in log.support():
                first_support.setdefault(support.label.object_handle, support.first_available_sequence)
            if row["index"] in (103, 114, 150, 161, 204, 218):
                snapshots[row["index"]] = log.snapshot()
        self.assertEqual(first_support, {63031: 114, 17557: 161, 6122: 213, 16434: 224})
        self.assertEqual(len(snapshots[103]["labels"]), 1)
        self.assertEqual(snapshots[103]["support"], ())  # Hz preceded that anchor's properties.
        self.assertEqual(len(snapshots[114]["support"]), 1)
        self.assertEqual(len(snapshots[150]["support"]), 1)
        self.assertEqual(len(snapshots[161]["support"]), 2)
        examples = log.query_examples
        self.assertEqual([(e.opportunity.decision_sequence, e.label.received_sequence,
                           e.label.activation_sequence, e.label.frequency_hz) for e in examples],
                         [(205, 213, 213, 6592.), (219, 224, 224, 7119.)])
        self.assertEqual([len(e.opportunity.support) for e in examples], [2, 3])
        for example in examples:
            opportunity = example.opportunity
            self.assertTrue(opportunity.clean)
            self.assertEqual(tuple(opportunity.query.presence), FEATURE_QUANTITIES)
            self.assertTrue(all(opportunity.query.presence.values()))
            self.assertNotIn("resonance_frequency_hz", opportunity.query.features)
            self.assertTrue(all(r.sequence < opportunity.decision_sequence for r in opportunity.query.records))
            self.assertTrue(all(s.label.received_sequence < opportunity.decision_sequence for s in opportunity.support))
            self.assertTrue(all(s.available_after_sequence < opportunity.decision_sequence for s in opportunity.support))
            self.assertGreater(example.label.received_sequence, opportunity.decision_sequence)
        # The first target's later, slightly different working setting remains
        # a raw fact, without becoming an independent support crystal.
        self.assertEqual([r.frequency_hz for r in log.labels if r.object_handle == 6122], [6592., 6593.44])
        self.assertEqual(len(log.support()), 4)
        events = log.snapshot()["activations"]
        self.assertEqual([(e.sequence, e.object_handle) for e in events if e.kind == "restoration"],
                         [(148, 63031), (195, 17557)])
        self.assertEqual([(e.sequence, e.object_handle) for e in events if e.kind == "new_activation"],
                         [(213, 6122), (224, 16434)])
        # Previously returned snapshots cannot acquire later properties/labels.
        self.assertEqual(snapshots[103]["support"], ())
        self.assertEqual(len(snapshots[204]["labels"]), 2)

    def test_native_values_units_conditions_and_uncalibrated_settings(self):
        log = CalibrationLog()
        for row in fixture():
            log.append_transition(row)
            if row["index"] == 204:
                break
        query = log.query(6122)
        self.assertEqual(len(query.records), 11)  # Four scalar tools, five channels, three radiation returns.
        radiation = query.features["radiation_usv_h"]
        self.assertEqual(len(radiation), 1)
        self.assertEqual((radiation[0].count, radiation[0].unit), (3, "uSv/h"))
        readings = [r.value for r in query.records if r.quantity == "radiation_usv_h"]
        self.assertAlmostEqual(radiation[0].mean, sum(readings) / 3)
        self.assertTrue(radiation[0].conditions["nearby"])
        self.assertTrue(all(r.unit is None for r in query.records if r.quantity.startswith("spectrum_channel")))
        self.assertFalse(any(label.object_handle == 6122 for label in log.labels))
        self.assertEqual(log.current_dialog(), {"reactor_handle": 10581, "object_handle": 6122,
                                               "frequency_hz": 5000., "unit": "Hz", "activated": False,
                                               "available_after_sequence": 204})
        # Re-querying is read-only; a correct-frequency label needs a new real return.
        before = log.snapshot()
        log.observe_initial(deepcopy(row["next_observation"]))
        self.assertEqual(before, log.snapshot())
        log.query(6122)
        self.assertEqual(before, log.snapshot())

    def test_future_return_changes_target_only_and_never_rebinds_old_hz(self):
        normal, changed = CalibrationLog(), CalibrationLog()
        rows = fixture()
        for original in rows:
            altered = deepcopy(original)
            if altered["index"] == 213:
                # A counterfactual public return changes only this later label.
                ui = altered["next_observation"]["ui"]
                for key in ("lastActionMessage", "extended_action_message"):
                    ui[key] = ui.get(key, "").replace("6592", "6501")
                ui["dialog_box"]["dialogIn"] = ui["dialog_box"]["dialogIn"].replace("6592", "6501")
            normal.append_transition(original)
            changed.append_transition(altered)
        self.assertEqual(normal.query_examples[0].opportunity, changed.query_examples[0].opportunity)
        self.assertEqual(changed.query_examples[0].label.frequency_hz, 6501.)
        self.assertNotEqual(normal.query_examples[0].label.frequency_hz, changed.query_examples[0].label.frequency_hz)
        # At the actual activation, remove its fresh frequency return. An old
        # 6590-Hz setting must not become a correct label from the active flag.
        log = CalibrationLog()
        for original in rows:
            if original["index"] >= 213:
                row = deepcopy(original)
                row["action"] = {"action": "ROTATE_DIRECTION", "arg1": "north"}
                row["next_observation"]["ui"]["dialog_box"] = {"is_in_dialog": False}
                row["next_observation"]["ui"]["lastActionMessage"] = "Turned north."
                log.append_transition(row)
                break
            log.append_transition(original)
        self.assertFalse(any(label.object_handle == 6122 for label in log.labels))
        self.assertEqual(log.query_examples, ())

    def test_prior_trial_frequency_excludes_late_queries_and_hidden_metadata_is_ignored(self):
        normal, manipulated = CalibrationLog(), CalibrationLog()
        for original in fixture():
            row = deepcopy(original)
            # Extra teacher/oracle keys cannot contribute evidence.
            row["expert_action"] = {"action": "TALK", "arg1": 999999}
            row["hypotheses"] = [{"frequency": -987654}]
            row["next_observation"]["scorecard"] = {"correct_frequency": -987654}
            ui = row["next_observation"]["ui"]
            objects = ui.get("accessibleEnvironmentObjects", []) + ui.get("inventoryObjects", [])
            objects += [o for group in ui.get("nearbyObjects", {}).get("objects", {}).values() for o in group]
            for obj in objects:
                obj["attributes"] = {"isActivated": True, "resonanceFreq": -987654}
            normal.append_transition(original)
            manipulated.append_transition(row)
        self.assertEqual(normal.snapshot(), manipulated.snapshot())
        self.assertNotIn("-987654", json.dumps(asdict(manipulated.query_examples[0])))
        # Counterfactual arrival order: a property arrives only after a trial.
        # Reuse native payloads, with explicit synthetic sequence indices.
        rows = {r["index"]: r for r in fixture()}
        late_log = CalibrationLog(required_quantities=("density_g_cm3",))
        for index, source_index in enumerate((204, 205, 13, 204, 213)):
            row = deepcopy(rows[source_index])
            row["index"] = index
            if source_index == 205:
                options = row["observation"]["ui"]["dialog_box"]["dialogOptions"]
                row["observation"]["ui"]["dialog_box"]["dialogOptions"] = {int(k): v for k, v in options.items()}
            late_log.append_transition(row)
        late = late_log.query_examples[0]
        self.assertFalse(late.opportunity.clean)
        self.assertEqual(late.opportunity.exclusion_reason, "prior_frequency_intervention")
        self.assertEqual(late.opportunity.decision_sequence, 4)

    def test_changed_crystal_and_missing_current_contents_cannot_inherit_a_label(self):
        path = Path(__file__).parent / "fixtures/discoveryworld_measurements.json"
        rows = {c["label"]: c["transition"] for c in json.loads(path.read_text())["cases"]}
        log = CalibrationLog()
        log.append_transition(rows["reactor_contents_before"])
        self.assertEqual(log.current_dialog()["object_handle"], 16434)
        log.append_transition(rows["reactor_contents_after"])
        self.assertEqual(log.current_dialog()["object_handle"], 6122)
        self.assertEqual(log.labels, ())  # Both real returns are uncalibrated 5000 Hz.
        hidden = deepcopy(rows["reactor_contents_after"]["next_observation"])
        ui = hidden["ui"]
        ui["nearbyObjects"], ui["inventoryObjects"], ui["accessibleEnvironmentObjects"] = {}, [], []
        log.observe_initial(hidden)
        self.assertIsNone(log.current_dialog())


if __name__ == "__main__":
    unittest.main()
