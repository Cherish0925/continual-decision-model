"""Return boundaries and real on-policy updates without model downloads or GPUs."""
import contextlib
from dataclasses import asdict
from importlib.util import find_spec
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import torch

from cdm.core import ContinualCore, CoreConfig
from cdm.digital import DigitalConfig, DigitalPolicy
from experiments.discoveryworld.adapter import DiscoveryWorldEnv
from experiments.discoveryworld.features import FeatureCache, FeatureConfig
from experiments.discoveryworld.finetune import (StateCritic, discounted_returns, parse_args,
                                                run_episode, save_checkpoint, training_schedule, main)
from experiments.discoveryworld.train import load_checkpoint


class FineTuneTests(unittest.TestCase):
    @unittest.skipUnless(find_spec("discoveryworld"), "Install the pinned DiscoveryWorld simulator")
    def test_parallel_native_feedback_trains_one_model_without_losing_episodes(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            with contextlib.redirect_stdout(io.StringIO()):
                main(['--random-native-width','16','--output',str(root/'model'),
                      '--feature-cache',str(root/'features'),'--native-suite','all',
                      '--difficulty','Easy','--seeds','1000','--max-steps','3',
                      '--sequence-length','2','--workers','2','--threads','1'])
            initial,_=load_checkpoint(root/'model/initial.pt','cpu')
            trained,saved=load_checkpoint(root/'model/last.pt','cpu')
            self.assertFalse(torch.equal(initial.score[-1].weight,trained.score[-1].weight))
            episodes=saved['finetune']['episodes']
            self.assertEqual(len(episodes),8)
            self.assertEqual(len({r['scenario'] for r in episodes}),8)
            self.assertEqual(sum(r['steps'] for r in episodes),24)
            updates=[json.loads(row) for row in (root/'model/updates.jsonl').read_text().splitlines()]
            self.assertEqual(updates[-1]['interactions'],24)
            self.assertTrue(all(torch.isfinite(torch.tensor(r['policy_loss'])) for r in updates))

    def test_reward_only_curriculum_starts_with_basic_operations(self):
        with TemporaryDirectory() as folder:
            args = parse_args(['--random-native-width', '32', '--output', folder,
                               '--feature-cache', folder, '--native-suite', 'holdout',
                               '--small-skills-first', '--seeds', '3008', '--difficulty', 'Easy'])
            schedule = training_schedule(args, {})
            self.assertEqual(len(schedule), 16)
            self.assertTrue(all(t.startswith('Small Skills:') for t, _, _, _ in schedule[:10]))
            self.assertFalse({'Plant Nutrients', 'Lost in Translation'} & {t for t, _, _, _ in schedule})

    def test_suite_training_excludes_held_themes_and_preserves_native_limits(self):
        with TemporaryDirectory() as folder:
            common = ["--checkpoint", "source.pt", "--output", folder, "--feature-cache", folder,
                      "--seeds", "3001", "--native-suite", "holdout"]
            args = parse_args(common)
            schedule = training_schedule(args, {})
            self.assertEqual(len(set(schedule)), 18)
            self.assertFalse({"Plant Nutrients", "Lost in Translation"} & {t for t, _, _, _ in schedule})
            self.assertTrue(all(limit == (100 if d == "Easy" else 1000) for _, d, _, limit in schedule))
            with self.assertRaises(ValueError):
                training_schedule(args, {"train_instances": [["Lost in Translation", "Easy", 1000]]})
            t, d, seed, _ = schedule[0]
            with self.assertRaises(ValueError):
                training_schedule(args, {"validation_instances": [[t, d, seed]]})

    def test_terminal_and_time_limit_returns_are_distinct_and_targets_detached(self):
        bootstrap = torch.tensor(4.0, requires_grad=True)
        limited = discounted_returns([1.0, 2.0], [False, False], bootstrap, .9)
        terminal = discounted_returns([1.0, 2.0], [False, True], bootstrap, .9)
        torch.testing.assert_close(limited, torch.tensor([6.04, 5.6]))
        torch.testing.assert_close(terminal, torch.tensor([2.8, 2.0]))
        self.assertFalse(limited.requires_grad)

    def test_official_evaluation_seeds_cannot_enter_training(self):
        with TemporaryDirectory() as folder:
            command = ["--checkpoint", "source.pt", "--output", folder,
                       "--feature-cache", folder, "--seeds", "3"]
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parse_args(command)

    @unittest.skipUnless(find_spec("discoveryworld"), "Install the pinned DiscoveryWorld simulator")
    def test_fresh_native_actions_update_policy_carry_state_and_save_compatible_checkpoint(self):
        torch.manual_seed(17)
        torch.set_num_threads(1)
        core_config = CoreConfig(event_dim=16, record_dim=16, actions=1, width=16,
                                 slots=2, heads=2, hidden=16, action_width=16)
        config = DigitalConfig(width=16, text_dim=256, image_dim=256, outcome_dim=14,
                               candidate_read=True, action_conditioned=True)
        model = DigitalPolicy(config, ContinualCore(core_config, torch.zeros(1, 1)))
        critic = StateCritic(16)
        optimizer = torch.optim.AdamW([*model.parameters(), *critic.parameters()], lr=.001)
        previous = model.score[-1].weight.detach().clone()
        inputs = []
        handle = model.core.gru.register_forward_pre_hook(lambda _, args: inputs.append(args[1].detach().clone()))
        with TemporaryDirectory() as folder:
            root = Path(folder)
            cache = FeatureCache(root / "features", FeatureConfig(mode="lexical", action_history=2))
            env = DiscoveryWorldEnv(root / "episode", max_steps=3)
            with contextlib.redirect_stdout(io.StringIO()):
                result = run_episode(model, critic, optimizer, cache, env, scenario="Reactor Lab",
                                     difficulty="Normal", seed=100, sequence_length=2, outcome_weight=.5)
            handle.remove()
            self.assertEqual(result["steps"], 3)
            self.assertEqual(result["updates"], 2)
            self.assertGreater((model.score[-1].weight.detach() - previous).abs().sum().item(), 0)
            self.assertEqual(inputs[0].abs().sum().item(), 0)
            self.assertTrue(all(state.abs().sum().item() > 0 for state in inputs[1:]))
            # Calls: step0, step1, bootstrap2, step2, bootstrap3. Both reads of
            # observation2 start at the same carried state, not a double update.
            self.assertEqual(len(inputs), 5)
            torch.testing.assert_close(inputs[2], inputs[3])
            rows = [json.loads(line) for line in (env.output / "trajectory.jsonl").read_text().splitlines()]
            self.assertEqual(len(rows), 3)
            self.assertEqual([row["policy_update"] for row in rows], [0, 0, 1])
            self.assertTrue(rows[-1]["truncated"])
            self.assertFalse(rows[-1]["terminated"])
            updates = [json.loads(line) for line in (env.output / "updates.jsonl").read_text().splitlines()]
            self.assertTrue(torch.isfinite(torch.tensor([row["policy_loss"] for row in updates])).all())
            self.assertNotEqual(updates[-1]["bootstrap"], 0.0)
            source = {"core_config": asdict(core_config), "digital_config": asdict(config),
                      "features": asdict(cache.config), "training_args": {"mode": "cdm"},
                      "train_instances": [], "validation_instances": [["Reactor Lab", "Normal", 105]], "epoch": 1}
            save_checkpoint(model, critic, optimizer, source, root, [result],
                            {"mode": "cdm", "restart_state": False, "restart_workspace": False}, "source-hash")
            loaded, saved = load_checkpoint(root / "last.pt")
            torch.testing.assert_close(loaded.score[-1].weight, model.score[-1].weight)
            self.assertNotIn("critic", saved["model"])
            self.assertTrue((root / "critic.pt").exists())
            self.assertEqual(saved["train_instances"], [["Reactor Lab", "Normal", 100]])
            self.assertEqual(saved["finetune"]["fresh_transitions"], 3)
            cache.close()


if __name__ == "__main__":
    unittest.main()
