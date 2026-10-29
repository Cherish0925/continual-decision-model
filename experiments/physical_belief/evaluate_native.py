"""Run learned physical beliefs with the same native task and action policy.

The policy core, numerical tools, initial randomness, and action budget are
shared. Physical memory may continue, reset, or be computed and discarded.
Omit --belief-checkpoint for the unchanged numerical-prior reference.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import multiprocessing
from pathlib import Path

from experiments.discoveryworld.evaluate_active_stream import run_stream


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", required=True, help="Frozen active policy shared by every comparison")
    parser.add_argument("--mixture-checkpoint", required=True)
    parser.add_argument("--belief-checkpoint")
    parser.add_argument("--belief-mode", choices=("carry", "reset", "discard"), default="carry")
    parser.add_argument("--initial-rng-root", required=True)
    parser.add_argument("--state-mode", choices=("carry", "reset"), default="reset",
                        help="Neural policy state, independent of physical memory")
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-actions", type=int, default=400)
    parser.add_argument("--max-decisions", type=int, default=160)
    parser.add_argument("--refinements", type=int, default=1)
    parser.add_argument("--save-boundary-states", action="store_true")
    args = parser.parse_args()
    streams = json.loads(args.manifest.read_text())["streams"]
    names = [stream["name"] for stream in streams]
    if len(set(names)) != len(names) or any(Path(name).name != name for name in names):
        parser.error("Stream names must be unique directory names")
    args.output.mkdir(parents=True, exist_ok=True)
    options = {key: str(value) if isinstance(value, Path) else value
               for key, value in vars(args).items()}
    options.update(continual=False, observation_mode="structured")
    (args.output / "configuration.json").write_text(json.dumps({**options, "streams": streams}, indent=2) + "\n")
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        futures = [pool.submit(run_stream, (options, stream)) for stream in streams]
        for future in as_completed(futures):
            rows.append(future.result())
            (args.output / "summary.json").write_text(json.dumps(rows, indent=2) + "\n")


if __name__ == "__main__":
    main()
