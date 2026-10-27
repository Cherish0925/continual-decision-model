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
                                                run_episode, save_checkpoint)
from experiments.discoveryworld.train import load_checkpoint


class FineTuneTests(unittest.TestCase):
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
        config = DigitalConfig(width=16, text_dim=256, image_dim=256, outcome_dim=14)
        model = DigitalPolicy(config, ContinualCore(core_config, torch.zeros(1, 1)))
        critic = StateCritic(16)
        optimizer = torch.optim.AdamW([*model.parameters(), *critic.parameters()], lr=.001)
        previous = model.score[-1].weight.detach().clone()
        inputs = []
        handle = model.core.gru.register_forward_pre_hook(lambda _, args: inputs.append(args[1].detach().clone()))
        with TemporaryDirectory() as folder:
            root = Path(folder)
            cache = FeatureCache(root / "features", FeatureConfig(mode="lexical"))
            env = DiscoveryWorldEnv(root / "episode", max_steps=3)
            with contextlib.redirect_stdout(io.StringIO()):
                result = run_episode(model, critic, optimizer, cache, env, scenario="Reactor Lab",
                                     difficulty="Normal", seed=100, sequence_length=2)
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
