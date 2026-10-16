#!/usr/bin/env python3
"""Evaluate trained continual agents over long uninterrupted lifetimes.

Each of ``--devices`` instances runs one lifetime of ``--ticks`` ticks with its
own state, experience and adapters; nothing is reset. Reported:

- tracking error while a request is active, reward, and the rates of
  refinements, measurements and learning steps;
- time to bring the output within 0.1 of a new setpoint;
- tracking error, prediction error, learning and refinement as functions of
  ticks since the last condition switch, separately for new conditions and
  conditions met before in the lifetime (acquisition and retention);
- tracking and prediction error as functions of time in the lifetime.

``--also`` evaluates each model that learns a second time under other modes,
for example with learning switched off.
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
sys.path.insert(0, str(Path(__file__).resolve().parent))
from cdm.core import CoreConfig, ContinualCore  # noqa: E402
from cdm.instrument import InstrumentFleet, InstrumentSpec  # noqa: E402
from runtime import MODES, Costs, configure_backends, new_agent, tick  # noqa: E402
from train import first_observation  # noqa: E402

SWITCH_BINS = (0, 10, 25, 50, 100, 200, 400, 800, 10 ** 9)


def load(path: Path, device):
    blob = torch.load(path, map_location=device, weights_only=False)
    config = CoreConfig(**blob["config"])
    model = ContinualCore(config, torch.zeros(config.actions, config.action_dim)).to(device)
    model.load_state_dict(blob["model"])
    return model.eval(), blob["args"], InstrumentSpec(**blob["instrument"])


@torch.no_grad()
def evaluate(model, args_train, spec, devices, ticks, seed, device, mode, greedy=False):
    env = InstrumentFleet(devices, spec, seed=seed, device=device)
    agent = new_agent(model, devices, args_train["buffer"], device)
    obs = first_observation(env)
    costs = Costs()
    nb = len(SWITCH_BINS) - 1
    life_bins = max(1, ticks // 1000)
    acc = {key: torch.zeros(2, nb, device=device) for key in ("err", "active", "pred", "all", "learn", "refine")}
    life = {key: torch.zeros(life_bins, device=device) for key in ("err", "active", "pred", "reward")}
    totals = dict.fromkeys(("reward", "refine", "measure", "learn", "err", "active", "pred"), 0.0)
    request_start = torch.full((devices,), -1, dtype=torch.long, device=device)
    reached = torch.zeros(devices, dtype=torch.bool, device=device)
    response = []
    bins = torch.tensor(SWITCH_BINS[1:-1], device=device)
    for t in range(ticks):
        agent, obs, step = tick(model, agent, env, obs, mode, costs, args_train["adapter_lr"], args_train["every"],
                                learn_ticks=args_train["learn_ticks"], greedy=greedy)
        active, err, perr = step["active"].float(), step["error"], step["prediction_error"]
        where = (step["returned"].long(), torch.bucketize(step["since_switch"], bins, right=True))
        for key, value in (("err", err * active), ("active", active), ("pred", perr), ("all", torch.ones_like(perr)),
                           ("learn", step["learn"].float()), ("refine", step["counts"].float())):
            acc[key].index_put_(where, value, accumulate=True)
        lb = min(t // 1000, life_bins - 1)
        for key, value in (("err", err * active), ("active", active), ("pred", perr), ("reward", step["reward"])):
            life[key][lb] += value.sum()
        totals["reward"] += float(step["reward"].sum())
        totals["refine"] += float(step["counts"].float().sum())
        totals["measure"] += float(step["accepted"].float().sum())
        totals["learn"] += float(step["learn"].float().sum())
        totals["err"] += float((err * active).sum())
        totals["active"] += float(active.sum())
        totals["pred"] += float(perr.sum())
        started, on = step["started"], step["active"]
        finished = (request_start >= 0) & ~reached & on & (err < 0.1)
        if finished.any():
            response.extend((t - request_start[finished]).tolist())
        reached |= finished
        ended = (request_start >= 0) & ~on
        if ended.any():
            response.extend([spec.request_ticks] * int((ended & ~reached).sum()))
        request_start = torch.where(started, torch.full_like(request_start, t),
                                    torch.where(ended, torch.full_like(request_start, -1), request_start))
        reached = torch.where(started | ended, torch.zeros_like(reached), reached)
    n = devices * ticks
    ratio = lambda a, b: (a / b.clamp_min(1)).cpu().tolist()
    return {"reward_per_100_ticks": 100 * totals["reward"] / n,
            "error_while_active": totals["err"] / max(totals["active"], 1),
            "prediction_error": totals["pred"] / n,
            "refinements_per_tick": totals["refine"] / n, "measurements_per_100_ticks": 100 * totals["measure"] / n,
            "learning_per_100_ticks": 100 * totals["learn"] / n,
            "response_ticks_median": float(np.median(response)) if response else None,
            "response_ticks_mean": float(np.mean(response)) if response else None,
            "requests": len(response),
            "switch_bins": list(SWITCH_BINS),
            **{f"{name}_after_switch_{group}": ratio(acc[a][g], acc[b][g])
               for name, a, b in (("error", "err", "active"), ("prediction_error", "pred", "all"),
                                  ("learning_rate", "learn", "all"), ("refinements", "refine", "all"))
               for g, group in enumerate(("new", "returned"))},
            "active_ticks_after_switch_new": acc["active"][0].cpu().tolist(),
            "active_ticks_after_switch_returned": acc["active"][1].cpu().tolist(),
            "error_by_thousand_ticks": ratio(life["err"], life["active"]),
            "prediction_error_by_thousand_ticks": (life["pred"] / (devices * 1000)).cpu().tolist(),
            "reward_per_100_by_thousand_ticks": (100 * life["reward"] / (devices * 1000)).cpu().tolist()}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, nargs="+", required=True)
    p.add_argument("--also", nargs="*", default=[], choices=MODES, help="further modes to evaluate each model under")
    p.add_argument("--devices", type=int, default=1024)
    p.add_argument("--ticks", type=int, default=20000)
    p.add_argument("--seed", type=int, default=777)
    p.add_argument("--greedy", action="store_true")
    p.add_argument("--device", default="cuda")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args(argv)
    device = torch.device(args.device)
    configure_backends()
    rows = []
    for path in args.model:
        model, targs, spec = load(path, device)
        learns = targs["mode"] in ("full", "fixed_learn", "world_model")
        for mode in dict.fromkeys([targs["mode"], *(args.also if learns else [])]):
            row = {"model": str(path), "variant": path.parent.parent.name, "trained": targs["mode"], "mode": mode,
                   "seed": targs["seed"],
                   **evaluate(model, targs, spec, args.devices, args.ticks, args.seed, device, mode, args.greedy)}
            rows.append(row)
            print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()
                              if not isinstance(v, list)}), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"args": {k: ([str(x) for x in v] if k == "model" else
                                                    str(v) if isinstance(v, Path) else v)
                                                for k, v in vars(args).items()},
                                       "runs": rows}, indent=1))


if __name__ == "__main__":
    main()
