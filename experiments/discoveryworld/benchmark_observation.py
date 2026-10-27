"""Measure observation-read latency on the native simulator, without a model."""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
from pathlib import Path
import platform
from statistics import median
from tempfile import TemporaryDirectory
from time import perf_counter

from .adapter import UPSTREAM_REVISION
from .observation import get_observation


def main():
    from discoveryworld.DiscoveryWorldAPI import DiscoveryWorldAPI

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=24)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.steps < 4:
        parser.error("Use at least four steps")
    report = {"upstream_revision": UPSTREAM_REVISION, "scenario": "Reactor Lab",
              "difficulty": "Normal", "seed": 100,
              "scope": "CPU observation reads only; no model inference, training, or world ticks in the timings",
              "samples_per_mode": args.steps, "modes": {}}
    report["runtime"] = {"python": platform.python_version(), "machine": platform.machine(),
                         "logical_cpus": os.cpu_count()}
    with TemporaryDirectory() as temporary:
        for mode in ("native", "png", "rgb"):
            with contextlib.redirect_stdout(io.StringIO()):
                api = DiscoveryWorldAPI(threadID=0)
                api.FRAME_DIR = str(Path(temporary) / mode)
                if not api.loadScenario("Reactor Lab", "Normal", randomSeed=100, numUserAgents=1):
                    raise RuntimeError("Could not initialize benchmark scenario")
            elapsed = []
            image_bytes = []
            for step in range(args.steps + 2):
                action = {"action": "ROTATE_DIRECTION", "arg1": ("north", "east", "south", "west")[step % 4]}
                outcome = api.performAgentAction(agentIdx=0, actionJSON=action)
                if not outcome["success"] or not api.tick()["success"]:
                    raise RuntimeError("Could not advance benchmark world")
                started = perf_counter()
                raw = api.getAgentObservation(0) if mode == "native" else get_observation(api, rgb=mode == "rgb")
                duration = perf_counter() - started
                if step >= 2:
                    elapsed.append(duration)
                    image_bytes.append(sum(value.nbytes if mode == "rgb" else len(value) for value in raw["vision"].values()))
            report["modes"][mode] = {"median_seconds": median(elapsed),
                                      "mean_seconds": sum(elapsed) / len(elapsed),
                                      "median_vision_bytes": median(image_bytes),
                                      "frame_files": len(list(Path(api.FRAME_DIR).glob("*.png")))}
    native = report["modes"]["native"]["median_seconds"]
    for stats in report["modes"].values():
        stats["observation_speedup"] = native / stats["median_seconds"]
    rendered = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()

