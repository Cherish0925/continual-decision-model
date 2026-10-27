"""CPU protocol tests; controlled model outputs are not pretrained model results."""
from copy import deepcopy
import contextlib
from importlib.util import find_spec
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import torch
from torch import nn

from cdm.semantic import SemanticExpert
from experiments.discoveryworld.semantic_policy import SemanticPolicy, SemanticPolicyError, evaluate_episode


class ProtocolProcessor:
    """Small reversible output codec, with a token-counted chat input."""
    def __init__(self):
        self.thinking_modes = []

    def apply_chat_template(self, messages, **kwargs):
        self.thinking_modes.append(kwargs["enable_thinking"])
        assert all(isinstance(message["content"], list) for message in messages)
        length = max(1, len(json.dumps(messages)) // 8)
        return {"input_ids": torch.ones(1, length, dtype=torch.long),
                "attention_mask": torch.ones(1, length, dtype=torch.long)}

    def decode(self, tokens, **kwargs):
        return "".join(chr(token) for token in tokens)


class ControlledBackbone(nn.Module):
    def __init__(self, outputs):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1))
        self.config = SimpleNamespace(text_config=SimpleNamespace(hidden_size=1))
        self.outputs, self.calls = list(outputs), []

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        response = torch.tensor([[ord(char) for char in self.outputs.pop(0)]], dtype=torch.long)
        return torch.cat((kwargs["input_ids"], response), dim=1)


def policy(outputs, **kwargs):
    expert = SemanticExpert(ControlledBackbone(outputs))
    kwargs.setdefault("vision", False)
    return SemanticPolicy(expert=expert, processor=ProtocolProcessor(), **kwargs)


def fixture_row(label="radiation_a1"):
    data = json.loads((Path(__file__).parent / "fixtures/discoveryworld_measurements.json").read_text())
    return deepcopy(next(case["transition"] for case in data["cases"] if case["label"] == label))


def model_payload(agent):
    payload = json.loads(agent.decisions[-1]["messages"][1]["content"][0]["text"])
    records = []
    for row in payload["measurements"]:
        group = dict(zip(payload["measurement_columns"], row))
        samples = group.pop("samples")
        records.extend({**group, **dict(zip(payload["sample_columns"], sample))} for sample in samples)
    payload["measurements"] = records
    return payload


class SemanticPolicyTests(unittest.TestCase):
    def test_single_fenced_json_from_recorded_thinking_run_keeps_raw_output(self):
        # Exact final response from semantic_thinking/35b/seed310, decision 1.
        final = '\n\n```json\n{"action": "MOVE_DIRECTION", "arg1": "east"}\n```'
        text = 'Reasoning stays separate.</think>' + final
        agent = policy([text], thinking=True)
        self.assertEqual(agent(fixture_row()["observation"]), {"action": "MOVE_DIRECTION", "arg1": "east"})
        record = agent.decisions[-1]
        self.assertEqual(record["final_output"], final)
        self.assertEqual(record["raw_output"], text)
        self.assertEqual(record["output_ids"], [ord(char) for char in text])
        self.assertEqual(len(agent.expert.backbone.calls), 1)
        unlabelled = ' \n```\n{"chosen_dialog_option_int":2}\n```\n '
        agent = policy([unlabelled])
        self.assertEqual(agent(fixture_row()["observation"]), {"chosen_dialog_option_int": 2})
        self.assertEqual(agent.decisions[-1]["final_output"], unlabelled)

    def test_fenced_response_rejects_prose_multiple_blocks_and_invalid_json(self):
        valid = '```json\n{"action":"READ","arg1":12}\n```'
        for final in ('Here is the action:\n' + valid, valid + '\nExplanation follows.',
                      valid + '\n' + valid, '```json\n{"action":"READ",}\n```',
                      '```python\n{"action":"READ"}\n```', '```json\n{"action":"READ"}'):
            with self.subTest(final=final):
                # A valid action inside reasoning must not rescue the final response.
                text = valid + '\n</think>\n' + final
                agent = policy([text], thinking=True)
                with self.assertRaises(SemanticPolicyError):
                    agent(fixture_row()["observation"])
                self.assertNotIn("action", agent.decisions[-1])
                self.assertEqual(agent.decisions[-1]["raw_output"], text)
                self.assertEqual(len(agent.expert.backbone.calls), 1)

    def test_notebook_waits_for_actual_feedback_and_never_becomes_measurement_evidence(self):
        row = fixture_row()
        generated = {**row["action"], "memory": "My guess is a reading of 999999.",
                     "running_hypotheses": ["An untested relation"], "explanation": "Take a reading."}
        with TemporaryDirectory() as folder:
            trace = Path(folder) / "decisions.jsonl"
            agent = policy([json.dumps(generated), '{}'], research_notebook=True, trace_path=trace)
            action = agent(row["observation"])
            self.assertEqual(action, generated)
            self.assertIsNone(agent.model_authored_notes)
            self.assertIsNone(model_payload(agent)["model_authored_notes"])
            self.assertFalse(trace.with_suffix(".notes.jsonl").exists())
            row["action"] = action
            agent.observe(row)
            self.assertEqual(len(agent.expert.backbone.calls), 1)  # No knowledge-update generation.
            saved = json.loads(trace.with_suffix(".notes.jsonl").read_text())
            self.assertEqual(saved["content"]["memory"], generated["memory"])
            self.assertEqual(saved["received_feedback"]["sequence"], row["index"])
            agent(row["next_observation"])
            payload = model_payload(agent)
            self.assertEqual(payload["model_authored_notes"], saved)
            self.assertEqual(payload["model_authored_notes"]["source"], "model_generated_before_action")
            self.assertEqual(len(payload["measurements"]), 1)
            self.assertEqual(payload["measurements"][0]["value"], agent.measurements.records[0].value)
            self.assertNotEqual(payload["measurements"][0]["value"], 999999.)
            self.assertNotIn("memory", payload["actual_history"][0]["action"])
            self.assertEqual(saved["content"]["running_hypotheses"], generated["running_hypotheses"])
            agent.reset()
            self.assertIsNone(agent.model_authored_notes)
            self.assertEqual(agent.notebook_updates, [])

    def test_unexecuted_proposals_and_replayed_annotations_cannot_write_notebook(self):
        agent = policy(['{"action":"READ","arg1":42,"memory":"Never executed"}'], research_notebook=True)
        row = fixture_row()
        agent(row["observation"])
        row["action"].update(memory="Another agent's note", running_hypotheses="Teacher hypothesis")
        row["expert_action"] = {"memory": "Teacher annotation"}
        agent.observe(row)
        self.assertIsNone(agent.model_authored_notes)
        self.assertEqual(agent.notebook_updates, [])
        self.assertEqual(len(agent.expert.backbone.calls), 1)
        agent.reset()
        agent.observe(row)  # Prefix replay supplies actual feedback but no current-model proposal.
        self.assertIsNone(agent.model_authored_notes)

    @unittest.skipUnless(find_spec("discoveryworld"), "Install the pinned DiscoveryWorld simulator")
    def test_native_failed_action_keeps_authored_notes_separate_from_its_real_result(self):
        first = {"action": "ROTATE_DIRECTION", "arg1": "up", "memory": "I expect to face up.",
                 "running_hypotheses": [], "explanation": "Check this direction."}
        second = {"action": "ROTATE_DIRECTION", "arg1": "north", "memory": "The up action failed.",
                  "running_hypotheses": [], "explanation": "Use a native direction."}
        with TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()):
            agent = policy([json.dumps(first), json.dumps(second)], research_notebook=True)
            summary = evaluate_episode(agent, Path(folder), "Reactor Lab", "Normal", 110, 2)
            rows = [json.loads(line) for line in (Path(folder) / "trajectory.jsonl").read_text().splitlines()]
            self.assertEqual([row["action"] for row in rows], [first, second])
            self.assertEqual((summary["steps"], summary["model_calls"]), (2, 2))
            notes = model_payload(agent)["model_authored_notes"]
            self.assertEqual(notes["content"]["memory"], first["memory"])
            self.assertFalse(notes["received_feedback"]["action_result"]["success"])
            self.assertTrue(rows[-1]["action_result"]["success"])
            self.assertEqual(agent.model_authored_notes["content"]["memory"], second["memory"])
            self.assertEqual(len(agent.measurements.records), 0)

    def test_thinking_uses_only_final_json_and_retains_the_full_generation(self):
        text = 'Consider {"action":"USE","arg1":999}.\n</think>\n{"action":"READ","arg1":12}'
        agent = policy([text], thinking=True, max_new_tokens=2048, temperature=.6, top_p=.95, top_k=20)
        self.assertEqual(agent(fixture_row()["observation"]), {"action": "READ", "arg1": 12})
        record = agent.decisions[-1]
        self.assertEqual(record["raw_output_with_special_tokens"], text)
        self.assertEqual(record["output_ids"], [ord(char) for char in text])
        self.assertTrue(record["thinking_complete"])
        self.assertFalse(record["output_limit_reached"])
        self.assertIn('"arg1":999', record["reasoning_output"])
        self.assertEqual(json.loads(record["final_output"]), record["action"])
        self.assertEqual(agent.processor.thinking_modes, [True])
        self.assertEqual(agent.expert.backbone.calls[0]["max_new_tokens"], 2048)
        self.assertEqual({key: agent.expert.backbone.calls[0][key] for key in ("temperature", "top_p", "top_k")},
                         {"temperature": .6, "top_p": .95, "top_k": 20})
        self.assertTrue(agent.expert.backbone.calls[0]["do_sample"])

    def test_unfinished_thinking_never_executes_json_inside_reasoning(self):
        text = 'Consider {"action":"READ","arg1":12}'
        agent = policy([text], thinking=True, max_new_tokens=len(text))
        with self.assertRaises(SemanticPolicyError):
            agent(fixture_row()["observation"])
        record = agent.decisions[-1]
        self.assertFalse(record["thinking_complete"])
        self.assertTrue(record["output_limit_reached"])
        self.assertEqual(record["stop_reason"], "length")
        self.assertEqual(record["raw_output"], text)
        self.assertIsNone(record["final_output"])
        self.assertNotIn("action", record)
        self.assertEqual(len(agent.expert.backbone.calls), 1)

        partial = 'Done.</think>{"action":'
        agent = policy([partial], thinking=True, max_new_tokens=len(partial))
        with self.assertRaises(SemanticPolicyError):
            agent(fixture_row()["observation"])
        record = agent.decisions[-1]
        self.assertTrue(record["thinking_complete"])
        self.assertTrue(record["output_limit_reached"])
        self.assertEqual(record["error"]["type"], "JSONDecodeError")
        self.assertNotIn("action", record)

    def test_default_plain_json_and_eos_at_token_limit_are_unchanged(self):
        text = '{"action":"READ","arg1":12}'
        agent = policy([text], max_new_tokens=len(text))
        # A controlled EOS equal to the final codec token tests stop accounting;
        # no real tokenizer or model result is implied by this protocol fixture.
        agent.expert.backbone.generation_config = SimpleNamespace(eos_token_id=[ord('}')])
        self.assertEqual(agent(fixture_row()["observation"]), json.loads(text))
        record = agent.decisions[-1]
        self.assertEqual(agent.processor.thinking_modes, [False])
        self.assertIsNone(record["thinking_complete"])
        self.assertEqual(record["stop_reason"], "eos")
        self.assertFalse(record["output_limit_reached"])
        self.assertNotIn("top_p", agent.expert.backbone.calls[0])
        self.assertNotIn("top_k", agent.expert.backbone.calls[0])
        self.assertNotIn("model_authored_notes", model_payload(agent))

    def test_prompt_protocol_follows_current_options_after_dialogue_exit(self):
        # Actual native UI snapshots include the exiting-dialog display cache:
        # it can still say is_in_dialog while its selectable options are empty.
        entry, exit_row = fixture_row("reactor1"), fixture_row("exit1")
        agent = policy(['{}', '{}', '{}'])
        agent(entry["observation"])
        ordinary = model_payload(agent)["output_protocol"]
        self.assertEqual(ordinary["schema"]["required"], ["action"])
        self.assertEqual(ordinary["syntax_examples"][0], {"action": "USE", "arg1": 5, "arg2": 12})
        self.assertNotIn("chosen_dialog_option_int", json.dumps(ordinary))
        self.assertNotIn("chosen_dialog_option_int", agent.decisions[-1]["messages"][0]["content"][0]["text"])
        agent.observe(entry)
        agent(entry["next_observation"])
        active = model_payload(agent)["output_protocol"]
        self.assertEqual(active["schema"]["required"], ["chosen_dialog_option_int"])
        self.assertEqual(active["schema"]["properties"]["chosen_dialog_option_int"]["enum"],
                         [int(key) for key in entry["next_observation"]["ui"]["dialog_box"]["dialogOptions"]])
        agent.observe(exit_row)
        agent(exit_row["next_observation"])
        self.assertFalse(exit_row["next_observation"]["ui"]["dialog_box"].get("dialogOptions"))
        self.assertEqual(model_payload(agent)["output_protocol"], ordinary)

    def test_only_actual_feedback_populates_raw_measurements_and_native_dialog_is_unchanged(self):
        row = fixture_row()
        observed = {**row["observation"], "actions": {"USE": {"args": ["arg1", "arg2"]}},
                    "seed": "PRIVATE_SEED", "gold_rule": "PRIVATE_RULE", "task_metrics": "PRIVATE_SCORE"}
        observed["ui"]["taskProgress"] = [{"taskName": "Public task", "description": "Measure it",
                                             "score": "PRIVATE_SCORE", "solution": "PRIVATE_RULE"}]
        agent = policy(['{"action":"PICKUP","arg1":999}', '{"chosen_dialog_option_int":2}', '{}'])
        self.assertEqual(agent(observed), {"action": "PICKUP", "arg1": 999})
        self.assertEqual(model_payload(agent)["measurements"], [])
        self.assertEqual(agent(observed), {"chosen_dialog_option_int": 2})
        self.assertEqual(model_payload(agent)["actual_history"], [])
        row.update(expert_action={"action": "ROTATE_DIRECTION", "arg1": "south"},
                   gold_rule="PRIVATE_RULE", task_metrics="PRIVATE_SCORE")
        agent.observe(row)
        agent(row["next_observation"])
        payload = model_payload(agent)
        self.assertEqual(payload["actual_history"][0]["action"], row["action"])
        self.assertEqual(len(payload["measurements"]), 1)
        record = payload["measurements"][0]
        exact = agent.measurements.records[0]
        self.assertEqual((record["instrument_handle"], record["target_handle"], record["value"], record["unit"]),
                         (row["action"]["arg1"], row["action"]["arg2"], exact.value, exact.unit))
        self.assertIn("contained_objects", record)
        condition = payload["measurement_conditions"][record["conditions_id"]]
        self.assertEqual(condition["agent_location"], exact.conditions["agent_location"])
        self.assertEqual(condition["inventory"], [obj["uuid"] for obj in exact.conditions["inventory"]])
        self.assertEqual(condition["nearby"], {direction: [[obj["uuid"], obj.get("distance")] for obj in group]
                                               for direction, group in exact.conditions["nearby"].items()})
        self.assertNotIn("PRIVATE_", json.dumps(agent.decisions))
        self.assertEqual(len(agent.expert.backbone.calls), 3)
        self.assertTrue(all(not call["do_sample"] for call in agent.expert.backbone.calls))
        self.assertTrue(all(not parameter.requires_grad for parameter in agent.expert.parameters()))

    def test_received_calibration_keeps_its_public_crystal_reactor_binding(self):
        row = fixture_row("reactor1")
        agent = policy(['{}', '{}'])
        agent(row["observation"])
        self.assertEqual(model_payload(agent)["measurements"], [])
        agent.observe(row)
        agent(row["next_observation"])
        record = model_payload(agent)["measurements"][0]
        self.assertEqual((record["instrument_handle"], record["target_handle"]), (50759, 50759))
        self.assertEqual(record["contained_objects"], [6122])
        self.assertEqual((record["value"], record["unit"]), (1657.6, "Hz"))
        self.assertEqual(record["sequence"], row["index"])

    def test_one_invalid_generation_is_saved_without_retry_or_action_replacement(self):
        with TemporaryDirectory() as folder:
            path = Path(folder) / "decisions.jsonl"
            agent = policy(["not JSON"], trace_path=path)
            with self.assertRaises(SemanticPolicyError):
                agent(fixture_row()["observation"])
            self.assertEqual(len(agent.expert.backbone.calls), 1)
            recorded = json.loads(path.read_text())
            self.assertEqual(recorded["raw_output"], "not JSON")
            self.assertNotIn("action", recorded)
            self.assertEqual(recorded["error"]["type"], "JSONDecodeError")
            self.assertTrue(recorded["input_ids"])

    def test_budget_omits_whole_old_entries_and_keeps_exact_ledger(self):
        agent = policy(['{}', '{}'])
        observation = fixture_row()["observation"]
        agent(observation)
        mandatory_budget = agent.decisions[-1]["input_tokens"]
        initial_ui = model_payload(agent)["ui"]
        for index in range(20):
            row = fixture_row()
            row["index"] = index
            agent.observe(row)
        agent.max_input_tokens = mandatory_budget + 800
        agent(observation)
        decision = agent.decisions[-1]
        self.assertLessEqual(decision["input_tokens"], agent.max_input_tokens)
        self.assertGreater(sum(decision["omitted"].values()), 0)
        self.assertEqual(len(agent.measurements.records), 20)
        payload = model_payload(agent)
        self.assertEqual(payload["ui"], initial_ui)
        self.assertTrue(all(record["value"] == agent.measurements.records[0].value for record in payload["measurements"]))
        agent.reset()
        self.assertEqual((agent.measurements.records, agent.history, agent.decisions), ([], [], []))

    @unittest.skipUnless(find_spec("discoveryworld"), "Install the pinned DiscoveryWorld simulator")
    def test_native_error_is_observed_and_later_parse_error_preserves_actual_prefix(self):
        with TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()):
            agent = policy(['{"action":"ROTATE_DIRECTION","arg1":"up"}',
                            '{"action":"ROTATE_DIRECTION","arg1":"north"}', 'invalid'], vision=True)
            output = Path(folder)
            summary = evaluate_episode(agent, output, "Reactor Lab", "Normal", 110, 4)
            rows = [json.loads(line) for line in (output / "trajectory.jsonl").read_text().splitlines()]
            self.assertEqual((summary["steps"], summary["model_calls"]), (2, 3))
            self.assertFalse(rows[0]["action_result"]["success"])
            self.assertEqual(rows[0]["action"]["arg1"], "up")
            self.assertTrue(rows[1]["action_result"]["success"])
            self.assertEqual(agent.history[0]["result"], rows[0]["action_result"])
            self.assertEqual(summary["error"]["type"], "SemanticPolicyError")
            self.assertFalse(summary["completed_successfully"])
            self.assertTrue(list((output / "images").glob("*.png")))
            self.assertTrue(list((output / "semantic_images").glob("*.png")))
            self.assertEqual(len((output / "decisions.jsonl").read_text().splitlines()), 3)

if __name__ == "__main__":
    unittest.main()
