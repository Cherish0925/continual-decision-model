"""Finish paired-action collection, learn from it, and execute fresh instances."""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import time


SEEDS = [s for s in range(2600, 2620) if s % 5 != 4] + list(range(2604, 2640, 5))


def launch(module, arguments, log):
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w") as stream:
        return subprocess.Popen([sys.executable, "-m", module, *arguments], stdout=stream, stderr=subprocess.STDOUT,
                                env={**os.environ, "TOKENIZERS_PARALLELISM": "false", "RAYON_NUM_THREADS": "2"})


def finished(process, name):
    if process is None:
        return
    if process.wait() != 0:
        raise RuntimeError(f"{name} failed; its log retains the error")
    print(json.dumps({"completed": name}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("runs/discoveryworld/consequence_learning"))
    parser.add_argument("--collection", type=Path, default=Path("runs/discoveryworld/action_consequences"))
    parser.add_argument("--checkpoint", type=Path, default=Path("runs/discoveryworld/predictive_actions/feedback/last.pt"))
    parser.add_argument("--reference", type=Path, default=Path("runs/discoveryworld/belief_actions/full_seed61/last.pt"))
    parser.add_argument("--prepared", type=Path, default=Path("runs/discoveryworld/predictive_actions/prepared"))
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.root.exists() and not args.resume:
        parser.error("Choose a fresh campaign directory")
    args.root.mkdir(parents=True, exist_ok=True)
    config = {k: str(v) for k, v in vars(args).items()}
    config.update(seeds=SEEDS, epochs=2, seed=161, core_lr=1e-6, lr=1e-5,
                  replay_stride=16, action_cost=.001,
                  comparison="paired native consequences versus matched replay; final epoch; new instances")
    if not (args.root / "config.json").exists():
        (args.root / "config.json").write_text(json.dumps(config, indent=2) + "\n")

    def evaluate(arm, checkpoint, kind):
        if args.resume and (args.root / "native" / arm / "summary.json").exists():
            return None
        arguments = ["--policy-kind", kind, "--checkpoint", str(checkpoint),
                     "--feature-cache", "runs/discoveryworld/features", "--encoder-path", "data/cache/siglip2",
                     "--output", str(args.root / "native" / arm), "--seeds", *map(str, SEEDS),
                     "--max-steps", "400", "--devices", "cuda:0", "cuda:1", "--workers", "4", "--threads", "2", "--local-probes"]
        if args.resume:
            arguments.append("--resume")
        if arm != "reference":
            arguments += ["--initial-rng-root", str(args.root / "native/reference")]
        return launch("experiments.discoveryworld.evaluate", arguments, args.root / f"native_{arm}.log")

    reference = evaluate("reference", args.reference, "belief")
    # Collection runs independently. All selected pairs must finish before an
    # outcome is consumed; no successful subset is used for training.
    expected = 2 * sum(len(json.loads(p.read_text())["states"]) for p in args.collection.glob("seed*/selection.json"))
    if not expected:
        raise ValueError("The collection must finish state selection first")
    deadline = time.monotonic() + 3600
    while True:
        try:
            rows = json.loads((args.collection / "comparison.json").read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            rows = []
        if len(rows) == expected:
            break
        if time.monotonic() > deadline:
            raise TimeoutError("Paired collection has not completed; examine its execution logs")
        time.sleep(10)
    print(json.dumps({"paired_native_branches": expected}), flush=True)
    contexts = args.root / "contexts.pt"
    if not contexts.exists():
        finished(launch("experiments.discoveryworld.train_consequences", ["--prepare", "--checkpoint", str(args.checkpoint),
                        "--branches", str(args.collection), "--prepared", str(args.prepared), "--output", str(contexts)],
                        args.root / "prepare.log"), "context preparation")
    trainings = []
    for device, arm in enumerate(("paired", "replay")):
        if args.resume and (args.root / arm / "last.pt").exists():
            saved_training = [json.loads(line) for line in (args.root / arm / "training.jsonl").read_text().splitlines()]
            if saved_training[-1]["epoch"] != 2:
                raise ValueError("The saved training is incomplete; resume its optimizer before evaluation")
            continue
        command = ["--checkpoint", str(args.checkpoint), "--contexts", str(contexts),
                   "--output", str(args.root / arm), "--device", f"cuda:{device}"]
        if arm == "replay":
            command.append("--no-paired")
        trainings.append((arm, launch("experiments.discoveryworld.train_consequences", command, args.root / f"{arm}.log")))
    for arm, process in trainings:
        finished(process, arm + " training")
    finished(reference, "reference execution")
    executions = [(arm, evaluate(arm, checkpoint, "predictive")) for arm, checkpoint in
                  (("parent", args.checkpoint), ("replay", args.root / "replay/last.pt"), ("paired", args.root / "paired/last.pt"))]
    for arm, process in executions:
        finished(process, arm + " execution")
    rows = []
    for arm in ("reference", "parent", "replay", "paired"):
        for seed in SEEDS:
            summary = json.loads((args.root / "native" / arm / f"seed{seed}/summary.json").read_text())
            rows.append({"arm": arm, "seed": seed, "cohort": "heldout_rule" if seed % 5 == 4 else "new_instance",
                         "completed": int(summary["completed_successfully"]), "progress": summary["progress"], "actions": summary["steps"]})
        for cohort in ("new_instance", "heldout_rule"):
            group = [r for r in rows if r["arm"] == arm and r["cohort"] == cohort]
            print(json.dumps({"arm": arm, "cohort": cohort, "completed": sum(r["completed"] for r in group),
                              "instances": len(group), "progress": sum(r["progress"] for r in group) / len(group),
                              "actions": sum(r["actions"] for r in group) / len(group)}), flush=True)
    with (args.root / "comparison.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
