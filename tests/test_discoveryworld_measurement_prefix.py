"""One recorded native prefix has the same causal input in all three consumers.

These are CPU input-pipeline replays, not new simulator task-completion runs.
The fixture retains consecutive radiation and frequency returns with public UI.
"""
from copy import deepcopy
from dataclasses import asdict, is_dataclass
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import torch

from cdm.core import ContinualCore, CoreConfig
from cdm.digital import DigitalConfig, DigitalPolicy
from experiments.discoveryworld.evaluate import LearnedPolicy
from experiments.discoveryworld.features import FeatureCache, FeatureConfig, batch_features, canonical_action, load_features
from experiments.discoveryworld.finetune import StateCritic, run_episode, save_checkpoint
from experiments.discoveryworld.train import (Episode, prepare_episodes, public_observation,
                                              initialize_training, load_checkpoint, parse_args, run_epoch)


def recorded_prefix(folder, *, dialog=False):
    fixture = json.loads((Path(__file__).parent / "fixtures/discoveryworld_measurements.json").read_text())
    cases = {case["label"]: case["transition"] for case in fixture["cases"]}
    names = ("reactor3", "adjust3a", "adjust3b") if dialog else ("radiation_a1", "radiation_a2", "radiation_a3")
    rows = [deepcopy(cases[name]) for name in names]
    for row in rows:
        # Images were excluded from this public-text fixture. The raw readings,
        # native observation clock and original event indices remain unchanged.
        row["observation"]["vision"] = {}
        row["next_observation"]["vision"] = {}
        row.update(training_reward=0., terminated=False, truncated=False,
                   expert_action={"chosen_dialog_option_int": 1} if "chosen_dialog_option_int" in row["action"]
                   else {"action": "ROTATE_DIRECTION", "arg1": "north"})
    metadata = {"scenario": "Reactor Lab", "difficulty": "Normal", "seed": 108,
                "action_descriptions": {"USE": {"args": ["arg1", "arg2"]},
                                        "ROTATE_DIRECTION": {"args": ["arg1"]}},
                "teleport_locations": {}, "fixture_source": fixture["sources"][0]}
    if dialog:
        metadata["action_descriptions"]["TALK"] = {"args": ["arg1"]}
    return Episode(folder / "trajectory.jsonl", metadata, rows)


class CapturingCache(FeatureCache):
    def __init__(self, folder, *, object_relations=False):
        super().__init__(folder, FeatureConfig(mode="lexical", measurement_memory=True,
                                               object_relations=object_relations))
        self.snapshots = []
        self.owners = []
        self.paths = []
        self.last_candidates = []

    def encode(self, observation, candidates, root=None):
        records = observation.get("measurement_records", [])
        self.snapshots.append(deepcopy([asdict(r) if is_dataclass(r) else r for r in records]))
        self.owners.append(observation.get("measurement_dialog_owner", "absent"))
        self.last_candidates = candidates
        path = super().encode(observation, candidates, root)
        self.paths.append(path)
        return path


class RecordedPrefixEnvironment:
    """Replay only the recorded actual actions; a mismatch is a test failure."""
    def __init__(self, episode, output):
        self.episode, self.output = episode, output
        self.actions = episode.metadata["action_descriptions"]
        self.teleports = episode.metadata["teleport_locations"]
        self.configuration = {**episode.metadata, "episode_id": "recorded-public-prefix"}
        self.metrics = [{"task": "recorded-prefix", "progress": 0., "completed": False,
                         "completed_successfully": False}]

    def reset(self, scenario, difficulty, seed):
        assert (scenario, difficulty, seed) == self.episode.identity
        self.steps, self.ended = 0, False
        self.current = self.episode.transitions[0]["observation"]
        return self.policy_input()

    def policy_input(self):
        return {**deepcopy(self.current), "actions": self.actions, "teleport_locations": self.teleports}

    def step(self, action):
        row = deepcopy(self.episode.transitions[self.steps])
        assert action == row["action"]
        self.steps += 1
        self.ended = self.steps == len(self.episode.transitions)
        self.current = row["next_observation"]
        row["truncated"] = self.ended  # The excerpt ends; the scientific task did not terminate.
        return row


class MeasurementPrefixTests(unittest.TestCase):
    def test_train_eval_and_finetune_receive_the_same_actual_prefix(self):
        self.check_actual_prefix(dialog=False)

    def test_actual_dialog_owner_and_records_follow_feedback_in_all_three_consumers(self):
        self.check_actual_prefix(dialog=True)

    def check_actual_prefix(self, *, dialog):
        torch.set_num_threads(1)
        torch.manual_seed(7)
        with TemporaryDirectory() as directory:
            root = Path(directory)
            episode = recorded_prefix(root, dialog=dialog)
            training_cache = CapturingCache(root / "training-cache", object_relations=dialog)
            prepare_episodes([episode], training_cache)
            self.assertEqual([sum(r["count"] for r in records) for records in training_cache.snapshots], [0, 1, 2])
            for row, records in zip(episode.transitions, training_cache.snapshots):
                self.assertTrue(all(r["last_sequence"] < row["index"] for r in records))

            config = CoreConfig(event_dim=16, record_dim=8, actions=1, width=16,
                                slots=2, heads=2, records=2, hidden=16, action_width=16)
            model = DigitalPolicy(DigitalConfig(width=16, text_dim=256, image_dim=256, outcome_dim=14),
                                  ContinualCore(config, torch.zeros(1, 1)))
            checkpoint = root / "policy.pt"
            if dialog:
                # Upgrade an actual legacy checkpoint through the training
                # CLI, then compare the same nonempty measurement input.
                old_digital, old_features = asdict(model.config), asdict(training_cache.config)
                del old_digital["object_relations"], old_features["object_relations"]
                torch.save({"model": model.state_dict(), "core_config": asdict(config),
                            "digital_config": old_digital, "features": old_features,
                            "training_args": {"mode": "cdm"}, "epoch": 0}, checkpoint)
                args = parse_args(["--resume-checkpoint", str(checkpoint), "--object-relations",
                                   "--train-trajectories", str(episode.path), "--validation-trajectories", "unused",
                                   "--output", str(root / "training"), "--feature-cache", str(root / "cache")])
                upgraded, _, effective = initialize_training(args)
                self.assertTrue(effective.object_relations and effective.measurement_memory)
                for key, value in model.state_dict().items():
                    self.assertTrue(torch.equal(value, upgraded.state_dict()[key]), key)
                obs, actions = batch_features([load_features(str(training_cache.paths[1]))], "cpu")
                state = model.initial(1, "cpu")
                with torch.no_grad():
                    before = model(obs, actions, state)
                    after = upgraded(obs, actions, state)
                for index in (0, 2):
                    self.assertTrue(torch.equal(before[index], after[index]))
                self.assertTrue(all(torch.equal(value, after[1][key]) for key, value in before[1].items()))
                model = upgraded
            torch.save({"model": model.state_dict(), "core_config": asdict(config),
                        "digital_config": asdict(model.config), "features": asdict(training_cache.config),
                        "training_args": {"mode": "cdm"}, "epoch": 0}, checkpoint)
            policy = LearnedPolicy(checkpoint, root / "evaluation-cache")
            self.assertEqual(policy.features.config.object_relations, dialog)
            self.assertEqual(policy.model.config.object_relations, dialog)
            policy.features.close()
            policy.features = evaluation_cache = CapturingCache(root / "evaluation-cache", object_relations=dialog)
            outcome_calls = []
            outcome_hook = policy.model.outcome.register_forward_hook(
                lambda *args: outcome_calls.append(True))
            for index, row in enumerate(episode.transitions):
                observation = public_observation(episode, row)
                policy(observation)
                if index:
                    policy(observation)  # Re-querying cannot credit another measurement.
                policy.observe(row)  # Feedback belongs to the actual action, not the expert label.
            policy({**episode.transitions[-1]["next_observation"],
                    "actions": episode.metadata["action_descriptions"],
                    "teleport_locations": {}})
            outcome_hook.remove()
            self.assertEqual(outcome_calls, [])

            environment = RecordedPrefixEnvironment(episode, root / "replay")
            online_cache = CapturingCache(root / "online-cache", object_relations=dialog)
            critic = StateCritic(16)
            optimizer = torch.optim.AdamW([*model.parameters(), *critic.parameters()], lr=1e-4)

            def recorded_action(distribution, sample_shape=torch.Size()):
                # Sampling is replaced only at this replay boundary so the
                # unmodified trainer executes exactly the recorded native action.
                expected = canonical_action(episode.transitions[environment.steps]["action"])
                indices = [i for i, candidate in enumerate(online_cache.last_candidates)
                           if canonical_action(candidate.action) == expected]
                self.assertEqual(len(indices), 1)
                return torch.tensor(indices[0], device=distribution.probs.device)

            with patch.object(torch.distributions.Categorical, "sample", recorded_action):
                result = run_episode(model, critic, optimizer, online_cache, environment,
                                     scenario="Reactor Lab", difficulty="Normal", seed=108,
                                     sequence_length=1)
            self.assertEqual((result["steps"], result["updates"]), (3, 3))
            if dialog:
                self.assertGreater(float(model.measurement_output.weight.detach().abs().sum()), 0)
                save_checkpoint(model, critic, optimizer, policy.checkpoint, root, [result],
                                {"mode": "cdm", "restart_state": False, "restart_workspace": False}, "fixture")
                reloaded, saved = load_checkpoint(root / "last.pt")
                self.assertTrue(saved["features"]["object_relations"])
                self.assertTrue(saved["digital_config"]["object_relations"])
                self.assertTrue(all(torch.equal(value, reloaded.state_dict()[key])
                                    for key, value in model.state_dict().items()))
            self.assertEqual([sum(r["count"] for r in records) for records in online_cache.snapshots],
                             [0, 1, 1, 2, 2, 3])
            # Each intermediate next-state bootstrap and the following action
            # receive the same prefix. The final truncated bootstrap includes
            # all three received measurements exactly once.
            self.assertEqual(evaluation_cache.snapshots, online_cache.snapshots)
            self.assertEqual(training_cache.snapshots, online_cache.snapshots[::2])
            self.assertEqual(evaluation_cache.owners, online_cache.owners)
            self.assertEqual(training_cache.owners, online_cache.owners[::2])
            if dialog:
                # TALK actually opened reactor 10581, despite a different
                # expert label. No owner or reading exists before its return.
                self.assertEqual(online_cache.owners, [None, 10581, 10581, 10581, 10581, 10581])
                self.assertEqual(training_cache.snapshots[0], [])
                self.assertEqual(training_cache.snapshots[1][0]["latest"], 5000.)
                self.assertEqual(training_cache.snapshots[2][0]["latest"], 4000.)
                for path in online_cache.paths[1:]:
                    relations = load_features(str(path))["candidates"]["measurement_relations"]
                    self.assertTrue(relations[..., 6].all())
            else:
                self.assertEqual(online_cache.owners, ["absent"] * 6)
            for cache in (training_cache, evaluation_cache, online_cache):
                cache.close()

    def test_selected_outcomes_preserve_real_prefix_training_gradients(self):
        torch.set_num_threads(1)
        torch.manual_seed(11)
        with TemporaryDirectory() as directory:
            root = Path(directory)
            cache = CapturingCache(root / "cache", object_relations=True)
            prepared = prepare_episodes([recorded_prefix(root)], cache)
            config = CoreConfig(event_dim=16, record_dim=8, actions=1, width=16,
                                slots=2, heads=2, records=2, hidden=16, action_width=16)
            base = DigitalPolicy(DigitalConfig(width=16, text_dim=256, image_dim=256,
                                               outcome_dim=14, object_relations=True),
                                 ContinualCore(config, torch.zeros(1, 1)))
            selective, full = deepcopy(base), deepcopy(base)
            original_forward = full.forward

            def full_readout(observation, candidates, state, **kwargs):
                # Reference computes the legacy [B,C,D] readout even when its
                # immediate result is unused, then selects the actual action.
                indices = kwargs.pop("outcome_indices", None)
                wanted = kwargs.pop("predict_outcomes", True)
                logits, state, predicted = original_forward(observation, candidates, state, **kwargs)
                if indices is not None:
                    predicted = predicted[torch.arange(len(indices)), indices]
                return logits, state, predicted if wanted else None

            full.forward = full_readout
            shapes, metrics = {}, {}
            for name, model in (("selective", selective), ("full", full)):
                shapes[name] = []
                hook = model.outcome.register_forward_pre_hook(
                    lambda _, inputs, name=name: shapes[name].append(tuple(inputs[0].shape)))
                torch.manual_seed(29)
                metrics[name] = run_epoch(model, prepared, batch_size=1, sequence_length=3,
                                          device="cpu", optimizer=torch.optim.SGD(model.parameters(), lr=.001))
                hook.remove()
            self.assertEqual(shapes["selective"], [(1, 48)] * 3)
            self.assertEqual(len(shapes["full"]), 6)
            self.assertTrue(all(len(shape) == 3 and shape[1] > 1 for shape in shapes["full"]))
            for key in ("loss", "outcome_loss", "imitation_loss", "immediate_loss"):
                self.assertAlmostEqual(metrics["selective"][key], metrics["full"][key], places=6)
            for (name, left), (_, right) in zip(selective.named_parameters(), full.named_parameters()):
                torch.testing.assert_close(left, right, rtol=1e-5, atol=1e-7, msg=name)
                if left.grad is None:
                    self.assertIsNone(right.grad, name)
                else:
                    torch.testing.assert_close(left.grad, right.grad, rtol=1e-5, atol=1e-7, msg=name)
            self.assertGreater(float(selective.measurement_output.weight.detach().abs().sum()), 0)
            cache.close()

    def test_future_return_changes_only_the_subsequent_decision_input(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            original = recorded_prefix(root)
            changed = deepcopy(original)
            message = changed.transitions[1]["next_observation"]["ui"]["lastActionMessage"]
            before, value = message.split("reports a level of ", 1)
            changed.transitions[1]["next_observation"]["ui"]["lastActionMessage"] = \
                before + "reports a level of 987654 " + value.split(" ", 1)[1]
            first, second = CapturingCache(root / "first"), CapturingCache(root / "second")
            prepare_episodes([original], first)
            prepare_episodes([changed], second)
            self.assertEqual(first.snapshots[:2], second.snapshots[:2])
            self.assertNotEqual(first.snapshots[2], second.snapshots[2])
            first.close()
            second.close()


if __name__ == "__main__":
    unittest.main()
