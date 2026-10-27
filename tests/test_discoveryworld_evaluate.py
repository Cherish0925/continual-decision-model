"""Actual spawned episode isolation and unchanged serial sampling semantics."""
import contextlib
from dataclasses import asdict
from importlib.util import find_spec
import io
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import torch

from cdm.core import ContinualCore, CoreConfig
from cdm.digital import DigitalConfig, DigitalPolicy
from experiments.discoveryworld.evaluate import main
from experiments.discoveryworld.features import FeatureConfig


@unittest.skipUnless(find_spec("discoveryworld"), "Install the pinned DiscoveryWorld simulator")
class ParallelEvaluationTests(unittest.TestCase):
    def test_two_real_spawned_episodes_and_serial_shared_rng(self):
        torch.set_num_threads(1)
        torch.manual_seed(17)
        core = CoreConfig(event_dim=16, record_dim=16, actions=1, width=16,
                          heads=2, slots=2, hidden=16, action_width=16)
        model = DigitalPolicy(DigitalConfig(width=16, text_dim=256, image_dim=256, outcome_dim=14),
                              ContinualCore(core, torch.zeros(1, 1)))
        with TemporaryDirectory() as folder:
            root = Path(folder)
            checkpoint = root / "policy.pt"
            torch.save({"model": model.state_dict(), "core_config": asdict(core),
                        "digital_config": asdict(model.config), "features": asdict(FeatureConfig(mode="lexical")),
                        "training_args": {"mode": "cdm"}, "epoch": 0,
                        "train_instances": [], "validation_instances": []}, checkpoint)
            common = ["--checkpoint", str(checkpoint), "--feature-cache", str(root / "features"),
                      "--seeds", "110", "111", "--max-steps", "3", "--threads", "1",
                      "--temperature", ".7", "--random-seed", "40"]
            with contextlib.redirect_stdout(io.StringIO()):
                main(common + ["--workers", "2", "--devices", "cpu", "cpu", "--output", str(root / "parallel")])
            parallel = json.loads((root / "parallel" / "summary.json").read_text())
            self.assertEqual(parallel["workers"], 2)
            self.assertEqual(parallel["rng_mode"], "per_episode_rng")
            self.assertEqual([row["random_seed"] for row in parallel["episodes"]], [40, 41])
            worker_pids = {row["worker_pid"] for row in parallel["episodes"]}
            self.assertEqual(len(worker_pids), 2)
            self.assertNotIn(os.getpid(), worker_pids)
            for seed in (110, 111):
                directory = root / "parallel" / f"seed{seed}"
                rows = [json.loads(line) for line in (directory / "trajectory.jsonl").read_text().splitlines()]
                run = json.loads((directory / "run.json").read_text())
                self.assertEqual(run["rng_mode"], "per_episode_rng")
                self.assertEqual(len(rows), 3)
                self.assertEqual([row["index"] for row in rows], [0, 1, 2])
                self.assertEqual(rows[0]["observation"]["ui"]["world_steps"], 1)
                self.assertTrue(rows[-1]["truncated"])
                for ref in rows[-1]["next_observation"]["vision"].values():
                    self.assertTrue((directory / ref["path"]).read_bytes().startswith(b"\x89PNG\r\n\x1a\n"))
            original_seed = torch.manual_seed
            with contextlib.redirect_stdout(io.StringIO()), patch(
                "experiments.discoveryworld.evaluate.torch.manual_seed", wraps=original_seed
            ) as seeded:
                # Default workers=1 must not reseed between stochastic episodes.
                main(common + ["--output", str(root / "serial")])
                seeded.assert_called_once_with(40)
            serial = json.loads((root / "serial" / "summary.json").read_text())
            self.assertEqual(serial["workers"], 1)
            self.assertEqual(serial["rng_mode"], "serial_shared_rng")
            self.assertEqual([row["random_seed"] for row in serial["episodes"]], [40, 40])
            self.assertTrue(all(row["worker_pid"] == os.getpid() for row in serial["episodes"]))


if __name__ == "__main__":
    unittest.main()
