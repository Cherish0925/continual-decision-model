#!/usr/bin/env python3
"""Planners under a partial view: does knowing how old each part of the map is help?

Every planner meets the same held-out episodes with the same scripted activities:
a full view every tick (``full``), only the local window (``move``), or the local
window plus a survey every 16 ticks (``survey16``). Planners trained on full maps
read the agent's belief as if it were current; planners trained on partial views
read the belief, and with an age input also how long ago each cell was seen.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from active import REGIME_ORDER, make_world, run_policy  # noqa: E402
from evaluate import load  # noqa: E402


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, nargs="+", required=True)
    p.add_argument("--rooms", type=int, default=5)
    p.add_argument("--regimes", nargs="+", default=["partial_return", "continuous"])
    p.add_argument("--episodes", type=int, default=1024)
    p.add_argument("--ticks", type=int, default=96)
    p.add_argument("--radius", type=int, default=2)
    p.add_argument("--seed", type=int, default=424242)
    p.add_argument("--device", default="cuda")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args(argv)
    device = torch.device(args.device)
    rows = []
    for path in args.model:
        model = load(path, device)
        if model.config.mode in ("carry", "restart", "goal_restart"):
            model.config = replace(model.config, memory_size=0)
        for regime in args.regimes:
            world = make_world(args.rooms, regime, args.episodes, args.ticks,
                               args.seed + 1000 * args.rooms + 100 * REGIME_ORDER.index(regime), (12, 24), device)
            for policy, every in (("full", 8), ("move", 8), ("survey16", 16)):
                out = run_policy(model, world, policy, args.radius, 0, every=every)
                rows.append({"model": str(path), "regime": regime, "policy": policy,
                             "return": float(out["return"].mean()),
                             "return_sem": float(out["return"].std(ddof=1) / np.sqrt(len(out["return"])))})
                print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in rows[-1].items()}), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"args": {k: (str(v) if isinstance(v, Path) else [str(x) for x in v]
                                                if k == "model" else v) for k, v in vars(args).items()},
                                       "runs": rows}, indent=1))


if __name__ == "__main__":
    main()
