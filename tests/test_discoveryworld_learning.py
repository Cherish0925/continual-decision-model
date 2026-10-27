"""Causality, sequence continuity and actual optimizer-path regression tests."""
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import torch

from cdm.core import ContinualCore, CoreConfig
from cdm.digital import DigitalConfig, DigitalPolicy, public_candidates
from experiments.discoveryworld.features import FeatureCache, FeatureConfig, batch_features, load_features
from experiments.discoveryworld.train import (Episode, prepare_episodes, public_observation,
                                             run_epoch, validate_split, outcome_target, received_outcome_loss,
                                             initialize_training, parse_args, main, load_checkpoint, action_run_weights)


def episode(root: Path, seed: int, length: int = 3) -> Episode:
    root.mkdir(parents=True)
    metadata = {"scenario": "test fixture", "difficulty": "Normal", "seed": seed,
                "action_descriptions": {"ROTATE_DIRECTION": {"args": ["arg1"], "desc": "Turn"}},
                "teleport_locations": {}}
    transitions = []
    for i in range(length):
        observation = {"ui": {"agentLocation": {"x": 1, "y": 2, "faceDirection": "north"},
                              "world_steps": i + 1, "taskProgress": [{"taskName": "measure", "description": "Observe a reading"}],
                              "lastActionMessage": "32.22 degrees Celsius" if i else ""},
                       "vision": {}, "capture_seconds": float(i), "arrival_seconds": float(i) + .01}
        transitions.append({"index": i, "observation": observation, "next_observation": observation,
                            "action": {"action": "ROTATE_DIRECTION", "arg1": "south" if i else "east"},
                            "action_result": {"success": True}, "training_reward": .1,
                            "terminated": False, "truncated": i == length - 1})
    path = root / "trajectory.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in transitions))
    (root / "run.json").write_text(json.dumps(metadata))
    return Episode(path, metadata, transitions)


class DiscoveryWorldLearningTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_split_is_by_instance_and_features_exclude_score_supervision(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            first, duplicate = episode(root / "a", 100), episode(root / "b", 100)
            with self.assertRaisesRegex(ValueError, "overlap"):
                validate_split([first], [duplicate])
            cache = FeatureCache(root / "cache", FeatureConfig(mode="lexical"))
            obs = public_observation(first, first.transitions[0])
            candidates = public_candidates(obs)
            path = cache.encode(obs, candidates)
            altered = deepcopy(obs)
            altered["training_reward"] = 99999
            altered["task_metrics"] = [{"progress": 1, "completed": True}]
            altered["ui"]["taskProgress"][0].update(score=999, completed=True)
            altered["capture_seconds"], altered["arrival_seconds"] = 1000., 2000.
            self.assertEqual(path, cache.encode(altered, public_candidates(altered)))
            features = load_features(str(path))
            self.assertNotIn("image", features["observation"])
            self.assertGreater(features["observation"]["text"].abs().sum(), 0)

    def test_sequence_training_updates_core_and_carries_state_across_windows(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            cache = FeatureCache(root / "cache", FeatureConfig(mode="lexical"))
            prepared = prepare_episodes([episode(root / "a", 100), episode(root / "b", 101, 2)], cache)
            torch.manual_seed(11)
            config = CoreConfig(event_dim=16, record_dim=16, actions=1, width=16,
                                heads=2, slots=2, hidden=16, action_width=16)
            model = DigitalPolicy(DigitalConfig(width=16, text_dim=256, image_dim=256, outcome_dim=14),
                                  ContinualCore(config, torch.zeros(1, 1)))
            previous = model.core.gru.weight_hh_l0.detach().clone()
            states = []
            handle = model.core.gru.register_forward_pre_hook(lambda _, inputs: states.append(inputs[1].detach().clone()))
            optimizer = torch.optim.Adam(model.parameters(), lr=.01)
            stats = run_epoch(model, prepared, batch_size=2, sequence_length=1,
                              device="cpu", optimizer=optimizer, immediate_weight=0)
            handle.remove()
            self.assertEqual(stats["transitions"], 5)
            self.assertTrue(torch.isfinite(torch.tensor(stats["loss"])))
            self.assertEqual(float(states[0].abs().sum()), 0)
            self.assertGreater(float(states[1].abs().sum()), 0)
            self.assertGreater(float((model.core.gru.weight_hh_l0.detach() - previous).abs().sum()), 0)
            self.assertIsNotNone(model.score[-1].weight.grad)
            self.assertIsNotNone(model.outcome[-1].weight.grad)
            self.assertIsNotNone(model.core.block.cross_attn.in_proj_weight.grad)

    def test_expert_imitation_and_executed_outcome_have_distinct_gradients(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            data = episode(root / "a", 100, 1)
            row = data.transitions[0]
            row["expert_action"] = {"action": "ROTATE_DIRECTION", "arg1": "west"}
            row["action_result"]["success"] = False
            cache = FeatureCache(root / "cache", FeatureConfig(mode="lexical"))
            prepared = prepare_episodes([data], cache)
            step = prepared[0].steps[0]
            self.assertNotEqual(step.expert_target, step.executed_target)
            config = CoreConfig(event_dim=16, record_dim=16, actions=1, width=16,
                                heads=2, slots=2, hidden=16, action_width=16)
            model = DigitalPolicy(DigitalConfig(width=16, text_dim=256, image_dim=256, outcome_dim=14),
                                  ContinualCore(config, torch.zeros(1, 1)))
            captured = []

            def capture(_, inputs, kwargs, output):
                logits, _, predicted = output
                logits.retain_grad()
                predicted.retain_grad()
                captured.append((logits, predicted, kwargs["outcome_indices"].clone()))

            handle = model.register_forward_hook(capture, with_kwargs=True)
            stats = run_epoch(model, prepared, batch_size=1, sequence_length=1, device="cpu",
                              optimizer=torch.optim.SGD(model.parameters(), lr=0), immediate_weight=0)
            handle.remove()
            logits, predicted, indices = captured[0]
            expected = logits.detach().softmax(-1)
            expected[0, step.expert_target] -= 1
            self.assertTrue(torch.allclose(logits.grad, expected, atol=1e-6))
            self.assertEqual(indices.tolist(), [step.executed_target])
            self.assertEqual(tuple(predicted.shape), (1, 14))
            self.assertGreater(float(predicted.grad.abs().sum()), 0)
            expected_loss = torch.nn.functional.cross_entropy(logits.detach(), torch.tensor([step.expert_target]))
            self.assertAlmostEqual(stats["imitation_loss"], float(expected_loss), places=6)
            # Without an expert label, existing trajectories use one action for both losses.
            del row["expert_action"]
            ordinary = prepare_episodes([data], cache)[0].steps[0]
            self.assertEqual(ordinary.expert_target, ordinary.executed_target)

    def test_action_run_reduction_balances_runs_across_windows_without_changing_outcome_gradients(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            data = episode(root / "a", 100, 6)
            for index, row in enumerate(data.transitions):
                row["expert_action"] = {"action": "ROTATE_DIRECTION", "arg1": "east" if index < 2 else "west"}
                # Actual actions do not determine expert-action run boundaries.
                row["action"] = {"action": "ROTATE_DIRECTION", "arg1": "north" if index % 2 else "south"}
            weights, exits = action_run_weights(data.transitions)
            self.assertEqual(weights, [.5, .5, .25, .25, .25, .25])
            self.assertEqual(exits, [False, False, True, False, False, False])
            self.assertEqual(sum(weights[:2]), sum(weights[2:]))
            prepared = prepare_episodes([data], FeatureCache(root / "cache", FeatureConfig(mode="lexical")))
            config = CoreConfig(event_dim=16, record_dim=16, actions=1, width=16,
                                heads=2, slots=2, hidden=16, action_width=16)
            base = DigitalPolicy(DigitalConfig(width=16, text_dim=256, image_dim=256, outcome_dim=14),
                                 ContinualCore(config, torch.zeros(1, 1)))
            captures, stats = {}, {}
            for reduction in ("event", "action_run"):
                model, captured = deepcopy(base), []

                def capture(_, inputs, output):
                    logits, _, predicted = output
                    logits.retain_grad()
                    if predicted is not None:
                        predicted.retain_grad()
                    captured.append((logits, predicted))

                handle = model.register_forward_hook(capture)
                stats[reduction] = run_epoch(model, prepared, batch_size=1, sequence_length=4,
                    device="cpu", optimizer=torch.optim.SGD(model.parameters(), lr=0),
                    immediate_weight=.25, imitation_reduction=reduction)
                handle.remove()
                captures[reduction] = captured
            self.assertEqual(stats["event"]["outcome_loss"], stats["action_run"]["outcome_loss"])
            self.assertEqual(stats["action_run"]["repeated_run_exits"], 1)
            self.assertEqual(stats["event"]["raw_event_imitation_accuracy"], stats["action_run"]["raw_event_imitation_accuracy"])
            coefficients = []
            for index, step in enumerate(prepared[0].steps):
                # CE normalization is fixed across the 4-event and 2-event
                # windows. Both runs therefore have equal total gradient mass.
                expected_ratio = (1.5 if index < 2 else .75) * (4 if index < 4 else 2) / 3
                for readout in (0, 1):  # refined and immediate CE use the same weights
                    event_logits, event_outcome = captures["event"][2 * index + readout]
                    run_logits, run_outcome = captures["action_run"][2 * index + readout]
                    self.assertTrue(torch.equal(event_logits.detach(), run_logits.detach()))
                    self.assertTrue(torch.allclose(run_logits.grad, event_logits.grad * expected_ratio, atol=1e-6))
                    if readout == 0:
                        self.assertTrue(torch.equal(event_outcome.grad, run_outcome.grad))
                        probability = run_logits.detach().softmax(-1)[0, step.expert_target]
                        coefficients.append(float(run_logits.grad[0, step.expert_target] / (probability - 1)))
                    else:
                        self.assertIsNone(event_outcome)
                        self.assertIsNone(run_outcome)
            self.assertAlmostEqual(sum(coefficients[:2]), sum(coefficients[2:]), places=6)
            # The raw 1/L weighted mean CE has two equally weighted runs.
            manual = []
            for index, step in enumerate(prepared[0].steps):
                logits = captures["action_run"][2 * index][0].detach()
                manual.append(float(torch.nn.functional.cross_entropy(logits, torch.tensor([step.expert_target]))))
            self.assertAlmostEqual(stats["action_run"]["imitation_loss"],
                                   sum(loss * weight for loss, weight in zip(manual, weights)) / sum(weights), places=6)

    def test_resume_preserves_schema_weights_and_records_new_training_history(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            train, validation = episode(root / "train", 102, 1), episode(root / "validation", 101, 2)
            config = CoreConfig(event_dim=16, record_dim=16, actions=1, width=16,
                                heads=2, slots=2, hidden=16, action_width=16)
            original = DigitalPolicy(DigitalConfig(width=16, text_dim=256, image_dim=256, outcome_dim=14),
                                     ContinualCore(config, torch.zeros(1, 1)))
            path = root / "source.pt"
            # A pre-schema checkpoint must keep its original wall-clock representation.
            features = asdict(FeatureConfig(mode="lexical"))
            del features["schema"], features["time_basis"]
            torch.save({"model": original.state_dict(), "core_config": asdict(config),
                        "digital_config": asdict(original.config), "features": features,
                        "training_args": {"mode": "gru", "restart_state": False, "restart_workspace": False},
                        "epoch": 7, "train_instances": [["test fixture", "Normal", 100]],
                        "validation_instances": [["test fixture", "Normal", 101]]}, path)
            cli = ["--resume-checkpoint", str(path), "--train-trajectories", str(train.path),
                   "--validation-trajectories", str(validation.path), "--output", str(root / "resumed"),
                   "--feature-cache", str(root / "cache"), "--epochs", "1", "--threads", "1"]
            args = parse_args(cli)
            restored, source, effective = initialize_training(args)
            self.assertEqual((args.width, args.heads, args.slots, args.mode), (16, 2, 2, "gru"))
            self.assertEqual((effective.mode, effective.schema, effective.time_basis), ("lexical", 1, "wall_clock"))
            self.assertTrue(all(torch.equal(value, restored.state_dict()[key]) for key, value in original.state_dict().items()))
            with self.assertRaisesRegex(ValueError, "--width"):
                initialize_training(parse_args(cli + ["--width", "32"]))
            with self.assertRaisesRegex(ValueError, "--features"):
                initialize_training(parse_args(cli + ["--features", "siglip"]))
            main(cli)
            resumed, saved = load_checkpoint(root / "resumed" / "last.pt")
            self.assertEqual(saved["epoch"], 8)
            self.assertEqual(saved["resume"]["optimizer_state"], "fresh")
            self.assertEqual(saved["core_config"], asdict(config))
            self.assertEqual(saved["features"]["time_basis"], "wall_clock")
            self.assertEqual(saved["train_instances"], [["test fixture", "Normal", 100], ["test fixture", "Normal", 102]])
            self.assertFalse(torch.equal(original.score[-1].weight, resumed.score[-1].weight))

    def test_outcome_target_predicts_received_values_without_uuid_magnitudes(self):
        row = {"action_result": {"success": True}, "training_reward": .2,
               "next_observation": {"ui": {"lastActionMessage": 'Crystal [uuid: 99113] measured 32.22 C and 1,234 Hz'}}}
        target = torch.tensor([outcome_target(row)])
        self.assertEqual(target[0, 9:].tolist(), [1., 1., 0., 0., 0.])
        self.assertAlmostEqual(float(target[0, 4]), float(torch.log1p(torch.tensor(32.22)) / 10), places=6)
        self.assertAlmostEqual(float(target[0, 5]), float(torch.log1p(torch.tensor(1234.)) / 10), places=6)
        prediction = torch.zeros_like(target, requires_grad=True)
        received_outcome_loss(prediction, target).sum().backward()
        self.assertGreater(float(prediction.grad[0, 4:6].abs().sum()), 0)
        self.assertEqual(float(prediction.grad[0, 6:9].abs().sum()), 0)

    def test_object_relation_configuration_survives_training_resume_and_explicit_ablation(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            train, validation = episode(root / "train", 100, 1), episode(root / "validation", 200, 2)
            cli = ["--train-trajectories", str(train.path), "--validation-trajectories", str(validation.path),
                   "--output", str(root / "trained"), "--feature-cache", str(root / "cache"),
                   "--features", "lexical", "--width", "16", "--heads", "2", "--slots", "2",
                   "--epochs", "1", "--threads", "1", "--object-relations"]
            with self.assertRaisesRegex(ValueError, "measurement"):
                initialize_training(parse_args(cli))
            with self.assertRaisesRegex(ValueError, "object_relations"):
                initialize_training(parse_args(cli + ["--measurement-memory", "--mask-object-relations",
                                                       "--no-object-relations"]))
            main(cli + ["--measurement-memory", "--mask-object-relations"])
            path = root / "trained" / "last.pt"
            trained, saved = load_checkpoint(path)
            metadata = json.loads((root / "trained" / "config.json").read_text())
            for payload in (saved, metadata):
                self.assertTrue(payload["features"]["measurement_memory"])
                self.assertTrue(payload["features"]["object_relations"])
                self.assertTrue(payload["features"]["mask_object_relations"])
                self.assertTrue(payload["digital_config"]["object_relations"])
                self.assertEqual(payload["core_config"]["record_dim"], 8)
            resumed_cli = ["--train-trajectories", str(train.path), "--validation-trajectories", str(validation.path),
                           "--output", str(root / "resumed"), "--feature-cache", str(root / "cache"),
                           "--resume-checkpoint", str(path)]
            resumed, _, inherited = initialize_training(parse_args(resumed_cli))
            self.assertTrue(inherited.object_relations)
            self.assertTrue(inherited.mask_object_relations)
            self.assertTrue(all(torch.equal(value, resumed.state_dict()[key])
                                for key, value in trained.state_dict().items()))
            unmasked, _, visible = initialize_training(parse_args(resumed_cli + ["--no-mask-object-relations"]))
            self.assertFalse(visible.mask_object_relations)
            self.assertEqual(unmasked.config, trained.config)
            self.assertEqual(unmasked.state_dict().keys(), trained.state_dict().keys())
            self.assertTrue(all(torch.equal(value, unmasked.state_dict()[key])
                                for key, value in trained.state_dict().items()))
            with self.assertRaisesRegex(ValueError, "measurement"):
                initialize_training(parse_args(resumed_cli + ["--no-measurement-memory"]))
            with self.assertRaisesRegex(ValueError, "object_relations"):
                initialize_training(parse_args(resumed_cli + ["--no-object-relations"]))
            ablated, _, disabled = initialize_training(parse_args(
                resumed_cli + ["--no-object-relations", "--no-measurement-memory", "--no-mask-object-relations"]))
            self.assertFalse(disabled.object_relations or disabled.measurement_memory)
            self.assertFalse(ablated.config.object_relations)
            self.assertTrue(all(torch.equal(value, trained.state_dict()[key])
                                for key, value in ablated.state_dict().items()))
            self.assertLess(len(ablated.state_dict()), len(trained.state_dict()))

    def test_dynamic_padding_never_adds_observation_or_action_evidence(self):
        feature = lambda n, t: {"observation": {"numeric": torch.zeros(32), "timing": torch.zeros(3),
                                                "text": torch.ones(t, 8)},
                                 "candidates": {"text": torch.ones(n, 8), "numeric": torch.zeros(n, 8)}}
        obs, candidates = batch_features([feature(2, 1), feature(5, 3)], "cpu")
        self.assertEqual(obs["text_present"].tolist(), [[True, False, False], [True, True, True]])
        self.assertEqual(candidates["present"].sum(1).tolist(), [2, 5])
        self.assertEqual(float(obs["text"][0, 1:].abs().sum()), 0)

    def test_frozen_text_chunking_keeps_late_tokens_and_uses_cache(self):
        # A deterministic tiny tower tests the cache/chunk boundary; production
        # siglip mode always loads the actual pinned pretrained model.
        from tokenizers import Tokenizer, models, pre_tokenizers, processors
        vocabulary = {str(i): i for i in range(1, 72)}
        vocabulary.update({"<pad>": 0, "<eos>": 72})
        backend = Tokenizer(models.WordLevel(vocabulary, unk_token="<pad>"))
        backend.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
        backend.post_processor = processors.TemplateProcessing(
            single="$A <eos>", special_tokens=[("<eos>", 72)])
        tokenizer = SimpleNamespace(backend_tokenizer=backend, padding_side="right",
                                    pad_token_id=0, pad_token="<pad>",
                                    num_special_tokens_to_add=lambda pair=False: 1)

        class Tower:
            config = SimpleNamespace(text_config=SimpleNamespace(max_position_embeddings=4))
            calls = 0

            def text_model(self, input_ids, **kwargs):
                self.calls += 1
                return SimpleNamespace(last_hidden_state=input_ids.float()[..., None])

        with TemporaryDirectory() as folder:
            cache = FeatureCache(folder, FeatureConfig())
            tower = Tower()
            cache.model, cache.processor = tower, SimpleNamespace(tokenizer=tokenizer)
            text = " ".join(str(i) for i in range(1, 72))
            encoded = cache.texts([text])[0]
            self.assertEqual(encoded[encoded[:, 0] != 72, 0].tolist(), list(range(1, 72)))
            calls = tower.calls
            self.assertTrue(torch.equal(encoded, cache.texts([text])[0]))
            self.assertEqual(tower.calls, calls)


if __name__ == "__main__":
    unittest.main()
