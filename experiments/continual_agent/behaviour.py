#!/usr/bin/env python3
"""When does a trained activity head compute, look and move?

Runs the greedy activity head saved by ``agent.py`` on fresh sites and reports
refinements, surveys and moves per tick split by whether a request is pending,
by ticks since the request arrived, and by the age of the agent's map.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from agent import ACTIONS, REFINEMENTS, Policy, act, begin, features, make_sites, perceive  # noqa: E402
from evaluate import load  # noqa: E402


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--policy", type=Path, required=True)
    p.add_argument("--sites", type=int, default=2)
    p.add_argument("--streams", type=int, default=2048)
    p.add_argument("--site-ticks", type=int, default=128)
    p.add_argument("--device", default="cuda")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args(argv)
    device = torch.device(args.device)
    model = load(args.model, device)
    if model.config.mode in ("carry", "restart", "goal_restart"):
        model.config = replace(model.config, memory_size=0)
    blob = torch.load(args.policy, map_location=device, weights_only=False)
    run_args = argparse.Namespace(**blob["args"])
    policy = Policy(blob["policy"]["norm.weight"].shape[0]).to(device)
    policy.load_state_dict(blob["policy"])
    policy.eval()
    table = torch.tensor(ACTIONS, device=device)
    sums = defaultdict(lambda: np.zeros(4))      # key -> [count, refinements, surveys, moves]
    for site in range(args.sites):
        blocked, goal, start = make_sites(args.streams, run_args.rooms, args.site_ticks, 6_000_000 + site, device)
        s = begin(model, blocked, goal, start)
        for t in range(1, args.site_ticks + 1):
            s = perceive(model, s, blocked, goal, t, run_args.radius)
            with torch.no_grad():
                choice = policy(features(s))[0].argmax(1)
            n = torch.tensor(REFINEMENTS, device=device)[table[choice, 0]].float()
            survey, step = table[choice, 1].float(), table[choice, 2].float()
            pending = ~s.done
            since = s.since_goal.clamp(max=8).long()
            age = (s.age.flatten(1).mean(1) / 4).floor().clamp(max=6).long()
            for key, mask in [("idle", ~pending), ("pending", pending)] + \
                    [(f"pending, {int(k)} ticks since request" + ("+" if k == 8 else ""), pending & (since == k))
                     for k in range(9)] + \
                    [(f"pending, mean map age {4 * int(a)}-{4 * int(a) + 3}", pending & (age == a)) for a in range(7)]:
                m = mask.float()
                sums[key] += np.array([float(m.sum()), float((n * m).sum()), float((survey * m).sum()),
                                       float((step * m).sum())])
            s, *_ = act(model, s, blocked, goal, t, choice, run_args)
    out = {k: {"ticks": int(v[0]), "refinements_per_tick": v[1] / max(v[0], 1),
               "surveys_per_100_ticks": 100 * v[2] / max(v[0], 1), "moves_per_tick": v[3] / max(v[0], 1)}
           for k, v in sums.items() if v[0] > 0}
    for k, v in out.items():
        print(json.dumps({"state": k, **{a: round(b, 3) if isinstance(b, float) else b for a, b in v.items()}}))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
