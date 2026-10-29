"""Queue the additional recurrent reference after the primary learned runs."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import subprocess
import sys
import time


def wait_for_primary(root, seed):
    names = [f"{mode}_seed{seed}" for mode in ("full", "discard_physical", "task_reset")]
    while not all((root / name / "finished.json").exists() for name in names):
        completed = root / "completed.json"
        if completed.exists():
            # The primary writer replaces this short list as jobs finish.
            try:
                rows = json.loads(completed.read_text())
            except json.JSONDecodeError:
                rows = []
            failed = [row["name"] for row in rows if row["name"] in names and row["exit_code"]]
            if failed:
                raise RuntimeError(f"Primary training failed: {failed}")
        time.sleep(20)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--primary", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--environment-truth", type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    jobs = []
    for gpu, seed, parent_seed, campaign in ((0, 231, 221, "joint_20261029"),
                                            (1, 232, 222, "joint_repeat_20261029")):
        parent = Path("runs/physical_belief") / campaign / f"carry_seed{parent_seed}"
        name = f"recurrent_reference_seed{seed}"
        command = ["-m", "experiments.continual_physics.train",
            "--data", "runs/discoveryworld/active_experiments/data_large",
            "--checkpoint", str(parent / "policy.pt"),
            "--belief-checkpoint", str(parent / "belief.pt"),
            "--mixture-checkpoint", "runs/discoveryworld/calibration/mixture/shared_train_only_seed31/last.pt",
            "--environment-truth", str(args.environment_truth),
            "--output", str(args.output / name), "--mode", "recurrent_reference",
            "--device", f"cuda:{gpu}", "--seed", str(seed),
            "--updates", "200", "--batch", "32", "--external-per-stream", "32"]
        jobs.append(dict(name=name, seed=seed, gpu=gpu, parent_seed=parent_seed, arguments=command))
    (args.output / "plan.json").write_text(json.dumps(dict(
        method="recurrent policy with shared physical tools", jobs=jobs,
        role="additional method reference, separate from the primary eight trainings",
        external_requests_per_training=204800,
        computation="one receive update per new observation; zero additional refinement",
        online_neural_updates="disabled for physical and cost adapters",
        retained_state="h/z and shared Bayesian physical memory across tasks",
        selection="fixed final update; no development-set selection"), indent=2) + "\n")

    def run(job):
        wait_for_primary(args.primary, job["seed"])
        started = time.monotonic()
        with (args.output / f"{job['name']}.log").open("w") as output:
            result = subprocess.run([sys.executable, *job["arguments"]], stdout=output, stderr=subprocess.STDOUT)
        return dict(name=job["name"], exit_code=result.returncode, seconds=time.monotonic() - started)

    completed = []
    with ThreadPoolExecutor(max_workers=2) as executor:
        for future in as_completed([executor.submit(run, job) for job in jobs]):
            row = future.result()
            completed.append(row)
            print(json.dumps(row), flush=True)
            (args.output / "completed.json").write_text(json.dumps(completed, indent=2) + "\n")
    if any(row["exit_code"] for row in completed):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
