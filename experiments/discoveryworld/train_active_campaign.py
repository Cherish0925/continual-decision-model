"""Repeat the two fixed active-core training phases for one seed and architecture.

Launch one process per available GPU. The phases run consecutively and retain
the optimizer between datasets. This entry point does not allocate devices or
change the numerical tool, training objectives, or checkpoint selection.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys


def commands(args):
    common = [sys.executable, "-u", "-m", "experiments.discoveryworld.train_active_core",
              "--mode", args.mode, "--device", args.device, "--seed", str(args.seed),
              "--updates", "1200", "--unroll", "8", "--max-steps", "160", "--eval-every", "100",
              "--core-checkpoint", str(args.core_checkpoint),
              "--mixture-checkpoint", str(args.mixture_checkpoint), "--bf16"]
    first = common + ["--data", str(args.data_root / "data_expanded"),
                      "--output", str(args.output / "phase1"), "--warmup", "400", "--batch", "128",
                      "--test-retry-rule", "any_measurement"]
    second = common + ["--data", str(args.data_root / "data_large"),
                       "--output", str(args.output / "phase2"), "--warmup", "0", "--batch", "512",
                       "--resume", str(args.output / "phase1" / "update1200.pt"),
                       "--test-retry-rule", "changed_evidence"]
    return [first, second]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("persistent", "ledger"), required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--seed", type=int, default=182)
    parser.add_argument("--data-root", type=Path, default=Path("runs/discoveryworld/active_experiments"))
    parser.add_argument("--core-checkpoint", type=Path,
                        default=Path("runs/discoveryworld/measurement_memory/action_run/last.pt"))
    parser.add_argument("--mixture-checkpoint", type=Path,
                        default=Path("runs/discoveryworld/calibration/mixture/shared_train_only_seed31/last.pt"))
    parser.add_argument("--print-commands", action="store_true")
    args = parser.parse_args()
    plan = commands(args)
    if args.print_commands:
        print("\n".join(shlex.join(command) for command in plan))
        return
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "campaign.json").write_text(json.dumps({"seed": args.seed, "mode": args.mode,
                                                          "commands": plan}, indent=2) + "\n")
    for phase, command in enumerate(plan, start=1):
        print(json.dumps({"starting_phase": phase, "command": command}), flush=True)
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
