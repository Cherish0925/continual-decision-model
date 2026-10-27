"""Dynamic native actions, identity invariance and continued neural gradients."""
from copy import deepcopy
from dataclasses import asdict
import unittest
from unittest.mock import patch

import torch

from cdm.core import ContinualCore, CoreConfig
from cdm.digital import (DigitalConfig, DigitalPolicy, PublicObjectMemory, public_candidates,
                         NativeCandidate, observation_numeric, observation_text, observed_numbers)


def observation():
    tool = {"uuid": 900, "name": "thermometer", "description": "a thermometer"}
    sample = {"uuid": 42, "name": "crystal", "description": "a crystal (in reactor [uuid: 77])"}
    distant = {"uuid": 77, "name": "reactor", "description": "a reactor", "distance": 2}
    return {"ui": {"agentLocation": {"x": 2, "y": 3, "faceDirection": "north",
                                    "directions_you_can_move": ["north", "east"]},
                   "inventoryObjects": [tool], "accessibleEnvironmentObjects": [sample],
                   "nearbyObjects": {"objects": {"east": [distant]}}, "world_steps": 5,
                   "lastActionMessage": "Measured crystal [uuid: 42] at 32.22 degrees Celsius.",
                   "dialog_box": {"is_in_dialog": False}},
            "actions": {"PICKUP": {"args": ["arg1"]}, "USE": {"args": ["arg1", "arg2"]},
                        "PUT": {"args": ["arg1", "arg2"]}, "TELEPORT_TO_OBJECT": {"args": ["arg1"]},
                        "MOVE_DIRECTION": {"args": ["arg1"]}, "ROTATE_DIRECTION": {"args": ["arg1"]}},
            "teleport_locations": {}}


class PublicCandidateTests(unittest.TestCase):
    def test_object_argument_handles_are_routing_metadata_only(self):
        candidates = public_candidates(observation())
        use = next(c for c in candidates if c.action == {"action": "USE", "arg1": 900, "arg2": 42})
        self.assertEqual(use.object_arguments, (900, 42))
        teleport = next(c for c in candidates if c.action == {"action": "TELEPORT_TO_OBJECT", "arg1": 77})
        self.assertEqual(teleport.object_arguments, (77, None))
        move = next(c for c in candidates if c.action.get("action") == "MOVE_DIRECTION")
        self.assertEqual(move.object_arguments, (None, None))
        self.assertEqual(NativeCandidate({}, "", ()).object_arguments, (None, None))

    def test_readings_keep_sign_precision_exponent_and_grouped_digits(self):
        self.assertEqual(observed_numbers("UUID '9999': -32.22 C; range 0 to 10,000 Hz; 1.25e-3 A"),
                         [-32.22, 0., 10000., .00125])

    def test_public_object_memory_allows_navigation_without_inventing_access(self):
        memory = PublicObjectMemory()
        first = observation()
        memory.observe(first)
        later = deepcopy(first)
        later["ui"]["nearbyObjects"] = {"objects": {}}
        later["ui"]["accessibleEnvironmentObjects"] = []
        enriched = memory.observe(later)
        actions = [candidate.action for candidate in public_candidates(enriched)]
        self.assertIn({"action": "TELEPORT_TO_OBJECT", "arg1": 77}, actions)
        self.assertIn({"action": "TELEPORT_TO_OBJECT", "arg1": 42}, actions)
        self.assertNotIn({"action": "PICKUP", "arg1": 42}, actions)
        self.assertNotIn("remembered_objects", later)
        self.assertIn("previously observed", observation_text(enriched))

    def test_public_argument_domains_and_dialog(self):
        obs = observation()
        actions = [candidate.action for candidate in public_candidates(obs)]
        self.assertIn({"action": "USE", "arg1": 900, "arg2": 42}, actions)
        self.assertIn({"action": "TELEPORT_TO_OBJECT", "arg1": 77}, actions)
        self.assertNotIn({"action": "USE", "arg1": 900, "arg2": 77}, actions)
        self.assertNotIn({"action": "PICKUP", "arg1": 900}, actions)
        self.assertNotIn({"action": "PUT", "arg1": 42, "arg2": 900}, actions)
        self.assertNotIn({"action": "MOVE_DIRECTION", "arg1": "west"}, actions)
        obs["gold_action"] = {"action": "PICKUP", "arg1": 123456}
        self.assertEqual(actions, [candidate.action for candidate in public_candidates(obs)])
        obs["ui"]["dialog_box"] = {"is_in_dialog": True,
                                   "dialogOptions": {"1": "Increase frequency by 1000 Hz", "2": "Exit"}}
        candidates = public_candidates(obs)
        self.assertEqual({c.action["chosen_dialog_option_int"] for c in candidates}, {1, 2})
        self.assertTrue(all("action" not in c.action for c in candidates))
        obs["ui"]["dialog_box"]["dialogOptions"] = {}
        self.assertIn({"action": "USE", "arg1": 900, "arg2": 42},
                      [c.action for c in public_candidates(obs)])

    def test_uuid_renaming_does_not_change_learned_inputs(self):
        obs = observation()
        renamed = deepcopy(obs)
        mapping = {900: 2, 42: 999999, 77: 5}

        def rename(value):
            if isinstance(value, dict):
                return {key: mapping[item] if key == "uuid" else rename(item) for key, item in value.items()}
            if isinstance(value, list):
                return [rename(item) for item in reversed(value)]
            if isinstance(value, str):
                for old, new in mapping.items():
                    value = value.replace(f"uuid: {old}]", f"uuid: {new}]")
            return value

        renamed = rename(renamed)
        # Direction ordering is immaterial to the candidate set as well.
        self.assertEqual(observation_text(obs), observation_text(renamed))
        self.assertEqual(observation_numeric(obs), observation_numeric(renamed))
        self.assertEqual([(c.text, c.numeric) for c in public_candidates(obs)],
                         [(c.text, c.numeric) for c in public_candidates(renamed)])
        self.assertIn("32.22 degrees Celsius", observation_text(obs))
        self.assertEqual(len(observation_numeric(obs)), 32)
        self.assertEqual(observation_numeric(obs)[15], 1.0)
        self.assertEqual(observation_numeric(obs)[19], 0.0)


class DigitalPolicyTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(4)
        torch.set_num_threads(1)
        self.core_config = CoreConfig(event_dim=7, record_dim=6, actions=3, width=24,
                                      slots=3, heads=4, hidden=24, action_width=12)
        self.source = ContinualCore(self.core_config, torch.tensor([-1., 0., 1.]))
        self.config = DigitalConfig(width=24, text_dim=12, image_dim=10)
        self.model = DigitalPolicy(self.config, self.source)
        self.obs = {"numeric": torch.randn(2, 32), "text": torch.randn(2, 4, 12),
                    "image": torch.randn(2, 5, 10), "timing": torch.zeros(2, 3)}
        self.candidates = {"text": torch.randn(2, 7, 12), "numeric": torch.randn(2, 7, 8),
                           "present": torch.tensor([[True] * 6 + [False], [True] * 7])}

    def test_action_permutation_state_carry_and_core_gradients(self):
        state = self.model.initial(2, "cpu")
        logits, continued, outcome = self.model(self.obs, self.candidates, state)
        order = torch.tensor([4, 2, 6, 0, 1, 5, 3])
        reordered = {key: value[:, order] for key, value in self.candidates.items()}
        shuffled, shuffled_state, shuffled_outcome = self.model(self.obs, reordered, state)
        torch.testing.assert_close(shuffled, logits[:, order])
        torch.testing.assert_close(shuffled_outcome, outcome[:, order])
        torch.testing.assert_close(continued["h"], shuffled_state["h"])
        self.assertTrue(torch.isneginf(logits[0, -1]))
        second, final, _ = self.model(self.obs, self.candidates, continued)
        reset, _, _ = self.model(self.obs, self.candidates, continued, persistent=False)
        self.assertGreater((second[:, :6] - reset[:, :6]).abs().max().item(), 1e-6)
        continued["h"].retain_grad()
        (second[:, :6].square().mean() + outcome.square().mean()).backward()
        for parameter in (self.source.gru.weight_hh_l0, self.source.block.cross_attn.in_proj_weight,
                          self.model.text.weight, self.model.candidate_text.weight):
            self.assertGreater(parameter.grad.abs().sum().item(), 0)
        self.assertGreater(continued["h"].grad.abs().sum().item(), 0)
        self.assertIs(final["A"], state["A"])
        self.assertIs(final["B"], state["B"])

    def test_source_load_preserves_every_original_tensor_and_ignores_process_actions(self):
        saved = {"config": asdict(self.core_config), "model": self.source.state_dict()}
        loaded = DigitalPolicy.from_lifetime_checkpoint(saved, self.config).eval()
        for name, value in self.source.state_dict().items():
            torch.testing.assert_close(loaded.core.state_dict()[name], value)
        state = loaded.initial(2, "cpu")
        before = loaded(self.obs, self.candidates, state)[0]
        with torch.no_grad():
            loaded.core.action_values.fill_(999)
            loaded.core.control_out[-1].weight.fill_(100)
        torch.testing.assert_close(before, loaded(self.obs, self.candidates, state)[0])

    def test_workspace_restart_keeps_recurrent_state_and_instance_adapters(self):
        state = self.model.initial(2, "cpu")
        _, state, _ = self.model(self.obs, self.candidates, state)
        regular_logits, regular, _ = self.model(self.obs, self.candidates, state)
        reset_logits, reset, _ = self.model(self.obs, self.candidates, state, restart_workspace=True)
        torch.testing.assert_close(reset["h"], regular["h"])
        self.assertIs(reset["A"], state["A"])
        self.assertIs(reset["B"], state["B"])
        self.assertGreater((reset["z"] - regular["z"]).abs().max().item(), 1e-6)
        self.assertGreater((reset_logits[:, :6] - regular_logits[:, :6]).abs().max().item(), 1e-6)

    def test_frozen_features_padding_and_gru_mode(self):
        self.obs["text"].requires_grad_()
        self.obs["image_present"] = torch.tensor([False, True])
        state = self.model.initial(2, "cpu")
        logits, continued, outcome = self.model(self.obs, self.candidates, state, mode="gru")
        altered = {**self.obs, "image": self.obs["image"].clone()}
        altered["image"][0] += 1000
        torch.testing.assert_close(logits[0], self.model(altered, self.candidates, state, mode="gru")[0][0])
        torch.testing.assert_close(continued["z"], state["z"])
        outcome.square().mean().backward()
        self.assertIsNone(self.obs["text"].grad)
        self.assertIsNone(self.source.block.cross_attn.in_proj_weight.grad)


class ObjectRelationTests(unittest.TestCase):
    def setUp(self):
        DigitalPolicyTests.setUp(self)
        self.obs.update(measurement_numeric=torch.randn(2, 3, self.core_config.record_dim),
                        measurement_text=torch.randn(2, 3, self.config.text_dim),
                        measurement_age=torch.rand(2, 3),
                        measurement_present=torch.tensor([[True, True, False], [True, True, True]]))
        self.candidates["measurement_relations"] = torch.randint(2, (2, 7, 3, 8)).float()

    def assert_outputs_equal(self, actual, expected):
        torch.testing.assert_close(actual[0], expected[0], rtol=0, atol=0)
        torch.testing.assert_close(actual[2], expected[2], rtol=0, atol=0)
        for key in expected[1]:
            torch.testing.assert_close(actual[1][key], expected[1][key], rtol=0, atol=0)

    def test_explicit_extension_preserves_old_weights_and_initial_outputs_exactly(self):
        self.model.eval()
        extended = DigitalPolicy.with_object_relations(self.model)
        old = self.model.state_dict()
        added = {"measurement_key.weight", "measurement_value.weight",
                 "measurement_relation_bias.weight", "measurement_output.weight"}
        self.assertEqual(set(extended.state_dict()) - set(old), added)
        self.assertFalse(self.model.config.object_relations)
        self.assertTrue(extended.config.object_relations)
        self.assertFalse(extended.training)
        for key, value in old.items():
            torch.testing.assert_close(extended.state_dict()[key], value, rtol=0, atol=0)
        self.assertNotEqual(extended.candidate_text.weight.data_ptr(), self.model.candidate_text.weight.data_ptr())
        state = self.model.initial(2, "cpu")
        self.assert_outputs_equal(extended(self.obs, self.candidates, state),
                                  self.model(self.obs, self.candidates, state))
        rebuilt = DigitalPolicy(extended.config, deepcopy(self.source))
        rebuilt.load_state_dict(extended.state_dict(), strict=True)
        self.assert_outputs_equal(rebuilt(self.obs, self.candidates, state),
                                  self.model(self.obs, self.candidates, state))
        disabled = DigitalPolicy.with_object_relations(extended, enabled=False)
        self.model.load_state_dict(disabled.state_dict(), strict=True)
        self.assert_outputs_equal(disabled(self.obs, self.candidates, state),
                                  self.model(self.obs, self.candidates, state))

    def test_training_distinguishes_identical_candidates_by_their_bound_records(self):
        self.model.requires_grad_(False)
        model = DigitalPolicy.with_object_relations(self.model)
        # Candidates have identical language/numbers. Only their record bindings
        # differ, and reversing those bindings reverses the correct candidate.
        candidates = {"text": self.candidates["text"][:1, :1].expand(2, 2, -1),
                      "numeric": self.candidates["numeric"][:1, :1].expand(2, 2, -1),
                      "measurement_relations": torch.zeros(2, 2, 3, 8)}
        candidates["measurement_relations"][0, 0, 0, 4] = 1
        candidates["measurement_relations"][0, 1, 1, 4] = 1
        candidates["measurement_relations"][1] = candidates["measurement_relations"][0, [1, 0]]
        obs = {key: value[:1].expand(2, *value.shape[1:]) for key, value in self.obs.items()}
        state = model.initial(2, "cpu")
        initial = model(obs, candidates, state)[0]
        torch.testing.assert_close(initial[:, 0], initial[:, 1], rtol=0, atol=0)
        optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=.01)
        targets = torch.tensor([0, 1])
        for _ in range(40):
            optimizer.zero_grad(set_to_none=True)
            logits, _, outcomes = model(obs, candidates, state)
            torch.nn.functional.cross_entropy(logits, targets).backward()
            optimizer.step()
        logits, _, outcomes = model(obs, candidates, state)
        self.assertEqual(logits.argmax(-1).tolist(), [0, 1])
        self.assertGreater((logits[:, 0] - logits[:, 1]).abs().min().item(), .1)
        self.assertGreater((outcomes[:, 0] - outcomes[:, 1]).abs().max().item(), 1e-4)
        for name in model._relation_modules:
            self.assertGreater(getattr(model, name).weight.grad.abs().sum().item(), 0)
        swapped = {**candidates, "measurement_relations": candidates["measurement_relations"][:, [1, 0]]}
        torch.testing.assert_close(model(obs, swapped, state)[0], logits[:, [1, 0]])
        cloned = DigitalPolicy.with_object_relations(model)
        self.assert_outputs_equal(cloned(obs, candidates, state), model(obs, candidates, state))

    def test_record_and_candidate_permutations_padding_and_empty_memory(self):
        model = DigitalPolicy.with_object_relations(self.model)
        # Exercise an active readout, not only its initially zero residual.
        with torch.no_grad():
            model.measurement_output.weight.copy_(torch.eye(self.config.width))
        state = model.initial(2, "cpu")
        expected = model(self.obs, self.candidates, state)
        order = torch.tensor([2, 0, 1])
        obs = {key: value[:, order] if key.startswith("measurement_") else value
               for key, value in self.obs.items()}
        candidates = {**self.candidates, "measurement_relations": self.candidates["measurement_relations"][:, :, order]}
        reordered = model(obs, candidates, state)
        torch.testing.assert_close(reordered[0], expected[0])
        torch.testing.assert_close(reordered[2], expected[2])
        order = torch.tensor([6, 3, 1, 0, 2, 4, 5])
        candidates = {key: value[:, order] for key, value in self.candidates.items()}
        torch.testing.assert_close(model(self.obs, candidates, state)[0], expected[0][:, order])
        obs, candidates = deepcopy(self.obs), deepcopy(self.candidates)
        for key in ("measurement_numeric", "measurement_text", "measurement_age"):
            obs[key][0, 2] += 10000
        candidates["measurement_relations"][0, :, 2] += 10000
        self.assert_outputs_equal(model(obs, candidates, state), expected)
        disabled = DigitalPolicy.with_object_relations(model, enabled=False)
        obs["measurement_present"] = torch.zeros(2, 3, dtype=torch.bool)
        self.assert_outputs_equal(model(obs, candidates, state), disabled(obs, candidates, state))
        obs = {key: value for key, value in self.obs.items() if not key.startswith("measurement_")}
        candidates["measurement_relations"] = torch.zeros(2, 7, 0, 8)
        self.assert_outputs_equal(model(obs, candidates, state), disabled(obs, candidates, state))
        encoded = model.encode(self.obs)
        self.assert_outputs_equal(model(encoded, self.candidates, state), expected)
        invalid = {**self.candidates, "measurement_relations": torch.zeros(2, 7, 2, 8)}
        with self.assertRaisesRegex(ValueError, "measurement_relations must be"):
            model(self.obs, invalid, state)


class OutcomeSelectionTests(unittest.TestCase):
    def setUp(self):
        ObjectRelationTests.setUp(self)

    def test_selected_outcomes_match_full_readout_and_loss_gradients(self):
        indices = torch.tensor([2, 6])
        targets = torch.randn(2, self.config.outcome_dim)
        state = self.model.initial(2, "cpu")
        for relations in (False, True):
            with self.subTest(object_relations=relations):
                full = DigitalPolicy.with_object_relations(self.model, enabled=relations)
                if relations:
                    with torch.no_grad():
                        full.measurement_output.weight.copy_(torch.eye(self.config.width))
                selected = deepcopy(full)
                full_logits, full_state, predictions = full(self.obs, self.candidates, state)
                mlp_shapes = []
                hook = selected.outcome.register_forward_pre_hook(lambda module, args: mlp_shapes.append(args[0].shape))
                chosen_logits, chosen_state, chosen = selected(self.obs, self.candidates, state, outcome_indices=indices)
                hook.remove()
                self.assertEqual(mlp_shapes, [torch.Size([2, 3 * self.config.width])])
                torch.testing.assert_close(chosen, predictions[torch.arange(2), indices])
                torch.testing.assert_close(chosen_logits, full_logits, rtol=0, atol=0)
                for key in state:
                    torch.testing.assert_close(chosen_state[key], full_state[key], rtol=0, atol=0)
                full_loss = (torch.nn.functional.cross_entropy(full_logits, indices)
                             + torch.nn.functional.mse_loss(predictions[torch.arange(2), indices], targets))
                chosen_loss = (torch.nn.functional.cross_entropy(chosen_logits, indices)
                               + torch.nn.functional.mse_loss(chosen, targets))
                torch.testing.assert_close(chosen_loss, full_loss)
                full_loss.backward()
                chosen_loss.backward()
                for name, parameter in full.named_parameters():
                    other = dict(selected.named_parameters())[name]
                    self.assertEqual(parameter.grad is None, other.grad is None, name)
                    if parameter.grad is not None:
                        torch.testing.assert_close(other.grad, parameter.grad, rtol=2e-5, atol=2e-7, msg=name)

    def test_skipping_outcomes_never_calls_mlp_and_preserves_logits_and_state(self):
        state = self.model.initial(2, "cpu")
        for relations in (False, True):
            with self.subTest(object_relations=relations):
                model = DigitalPolicy.with_object_relations(self.model, enabled=relations)
                if relations:
                    with torch.no_grad():
                        model.measurement_output.weight.copy_(torch.eye(self.config.width))
                full_logits, full_state, _ = model(self.obs, self.candidates, state)
                with patch.object(model.outcome, "forward", side_effect=AssertionError("Outcome MLP must be skipped")):
                    logits, next_state, outcome = model(self.obs, self.candidates, state,
                                                       predict_outcomes=False, outcome_indices=torch.tensor([2, 6]))
                self.assertIsNone(outcome)
                torch.testing.assert_close(logits, full_logits, rtol=0, atol=0)
                for key in state:
                    torch.testing.assert_close(next_state[key], full_state[key], rtol=0, atol=0)
                logits[:, :6].square().mean().backward()
                self.assertTrue(all(parameter.grad is None for parameter in model.outcome.parameters()))
                self.assertGreater(model.score[-1].weight.grad.abs().sum().item(), 0)


if __name__ == "__main__":
    unittest.main()
