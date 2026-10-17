#!/usr/bin/env python3
"""How much the values of activity options measured on copies of a process are worth as a training signal.

From one trained model running ``--devices`` processes, every ``--every`` ticks the option values of each process
(``runtime.branch_values``) are measured twice with independent randomness, for several horizons. Within each
activity (measurement; compute), values are centred over the allowed options. Reported per horizon: the spread of
centred values, the noise of one measurement (from the difference between the two), the share of the spread that
is repeatable (the correlation of the two measurements), how often both name the same best option, and, against
each measurement's best option, the regret and agreement of the policy's own probabilities. ``--greedy`` runs the
copies on the policy's most probable choices; ``--replicates`` averages that many measurements into each of the two.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.lifetime.evaluate import load, runtime_options, training_costs  # noqa: E402
from experiments.lifetime.runtime import (branch_values, configure_backends, distributions, event_features,  # noqa: E402
                                          goal_offsets, make_fleet, masks, new_agent, scalar_features, tick)
from experiments.lifetime.train import FACTORS, centred, first_observation  # noqa: E402


@torch.no_grad()
def policy_probabilities(model, agent, env, obs, mode, every):
    """Activity probabilities [n, options] within each activity, at the agent's current state."""
    goals, pending = model.config.goals, env.pending.clone()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        h = model.fast(agent.h, event_features(obs, agent, pending, goals))
        mean, logvar = model.predict_actions(model.state_features(h, agent.z), agent.A, agent.B)
        offset, logvar = goal_offsets(mean.float(), obs, goals), logvar.float()
        _, activity, _ = model.heads(model.trunk(h, agent.z, offset, logvar, scalar_features(obs, agent, pending, goals)),
                                     offset, logvar)
    allowed = masks(agent, pending, mode, env.tick, every)
    _, measure, compute = distributions(torch.zeros_like(offset), activity, allowed)
    return torch.cat((measure.probs, compute.probs), 1)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--devices", type=int, default=128)
    p.add_argument("--warmup", type=int, default=400)
    p.add_argument("--points", type=int, default=6, help="measurements per process")
    p.add_argument("--every", type=int, default=50)
    p.add_argument("--horizons", type=int, nargs="+", default=[4, 8, 16, 32])
    p.add_argument("--seed", type=int, default=777)
    p.add_argument("--greedy", action="store_true")
    p.add_argument("--replicates", type=int, default=1)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    configure_backends()
    device = torch.device("cuda")
    model, targs, specs = load(args.model, device)
    mode, costs, options = targs["mode"], training_costs(targs), runtime_options(targs)
    torch.manual_seed(args.seed)
    env = make_fleet(targs.get("family", "instrument"), args.devices, args.seed, device, specs,
                     targs.get("thermal_share", 0.5))
    agent, obs = new_agent(model, args.devices, targs["buffer"], device), first_observation(env)
    rows = torch.arange(args.devices, device=device)
    run = lambda a, o: tick(model, a, env, o, mode, costs, targs["adapter_lr"], targs["every"],
                            learn_ticks=targs["learn_ticks"], **options)[:2]
    for _ in range(args.warmup):
        agent, obs = run(agent, obs)
    pairs = {h: [] for h in args.horizons}
    probs = []
    for point in range(args.points):
        probs.append(policy_probabilities(model, agent, env, obs, mode, targs["every"]).cpu())
        for h in args.horizons:
            measured = []
            for r in (0, 1):
                values = [branch_values(model, agent, env, obs, rows, mode, costs, targs["adapter_lr"], targs["every"],
                                        h, targs["gamma"], targs["reward_scale"],
                                        seed=args.seed + 100000 * point + 1000 * h + 10 * k + r, greedy=args.greedy,
                                        learn_ticks=targs["learn_ticks"], **options) for k in range(args.replicates)]
                measured.append((torch.stack([q for q, _ in values]).mean(0), values[0][1]))
            valid = measured[0][1].float()
            pairs[h].append(tuple(centred(q, valid).cpu() for q, _ in measured) + (valid.cpu(),))
        for _ in range(args.every):
            agent, obs = run(agent, obs)
    probs = torch.cat(probs)
    out = {"model": str(args.model), "devices": args.devices, "points": args.points, "greedy": args.greedy,
           "replicates": args.replicates, "horizons": {}}
    for h in args.horizons:
        a, b, valid = (torch.cat(x) for x in zip(*pairs[h]))
        row = {}
        for f, name in zip(FACTORS, ("measurement", "compute")):
            va, vb, m = a[:, f], b[:, f], valid[:, f] > 0
            two = m.sum(1) >= 2                                   # states with a choice in this activity
            xa, xb = va[m & two.unsqueeze(1)].numpy(), vb[m & two.unsqueeze(1)].numpy()
            best_a = va.masked_fill(~m, -1e9).argmax(1)
            best_b = vb.masked_fill(~m, -1e9).argmax(1)
            pr = probs[:, f]
            regret = ((pr * va * m).sum(1) - va.masked_fill(~m, -1e9).max(1).values)[two]
            row[name] = {"states": int(two.sum()), "spread": float(np.std(xa)),
                         "noise": float(np.std(xa - xb) / np.sqrt(2)),
                         "repeatable_share": float(np.corrcoef(xa, xb)[0, 1]) if len(xa) > 2 else None,
                         "same_best": float((best_a == best_b)[two].float().mean()),
                         "policy_regret": float(regret.mean()),
                         "policy_agrees": float((pr.argmax(1) == best_a)[two].float().mean())}
        out["horizons"][h] = row
        print(h, json.dumps(row), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
