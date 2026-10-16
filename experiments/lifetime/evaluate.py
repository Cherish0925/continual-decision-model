#!/usr/bin/env python3
"""Evaluate trained continual agents over long uninterrupted lifetimes.

Each of ``--devices`` instances runs one lifetime of ``--ticks`` ticks with its
own state, experience and adapters; nothing is reset. Reported:

- tracking error while a request is active, reward, and the rates of
  refinements, measurements and learning steps;
- prediction error of the next reading for the action taken, with the instance
  adapter and, on the same states, with the shared predictor alone;
- time to bring the output within 0.1 of a new setpoint;
- tracking error, prediction error, learning and refinement as functions of
  ticks since the last condition switch, separately for new conditions and
  conditions met before in the lifetime (acquisition and retention);
- tracking and prediction error as functions of time in the lifetime.

``--also`` evaluates each model that learns a second time under other modes,
for example with learning switched off. ``--instrument`` changes the instrument
family, for example ``time_constant=8,16 width=0.15,0.25`` for dynamics outside
the training ranges.
"""
from __future__ import annotations

import argparse
import dataclasses
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


class ResponseTimes:
    """Count executed controls until first reaching a request, with failures capped at its duration."""

    def __init__(self, devices, device, request_ticks):
        self.start = torch.full((devices,), -1, dtype=torch.long, device=device)
        self.reached = torch.zeros(devices, dtype=torch.bool, device=device)
        self.request_ticks = request_ticks
        self.values = []
        self.successes = 0

    def add(self, t, step):
        # started belongs to the request served by this action; next_started
        # belongs to a request that can first be served on the following tick.
        started, active = step["started"], step["active"]
        self.start = torch.where(started, torch.full_like(self.start, t), self.start)
        self.reached = torch.where(started, torch.zeros_like(self.reached), self.reached)
        hit = (self.start >= 0) & ~self.reached & active & (step["error"] < 0.1)
        if hit.any():
            times = (t - self.start[hit] + 1).tolist()
            self.values.extend(times)
            self.successes += len(times)
        self.reached |= hit
        ended = active & (~step["next_active"] | step["next_started"])
        failed = ended & (self.start >= 0) & ~self.reached
        if failed.any():
            self.values.extend([self.request_ticks] * int(failed.sum()))
        self.start = torch.where(ended, torch.full_like(self.start, -1), self.start)
        self.reached &= ~ended

    def summary(self):
        return {"response_ticks_median": float(np.median(self.values)) if self.values else None,
                "response_ticks_mean": float(np.mean(self.values)) if self.values else None,
                "requests": len(self.values), "successful_requests": self.successes,
                "unfinished_requests": int(((self.start >= 0) & ~self.reached).sum())}


def load(path: Path, device):
    blob = torch.load(path, map_location=device, weights_only=False)
    weights = blob["model"]       # earlier checkpoints predate these options; their weights tell which they used
    config = CoreConfig(**{**blob["config"], "goal_term": "goal_weight" in weights,
                           "bounded_workspace": "norm_slots.weight" in weights})
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
    life_bins = max(1, (ticks + 999) // 1000)
    acc = {key: torch.zeros(2, nb, device=device) for key in ("err", "active", "pred", "shared", "all", "learn", "refine")}
    life = {key: torch.zeros(life_bins, device=device) for key in ("err", "active", "pred", "reward")}
    totals = dict.fromkeys(("reward", "refine", "measure", "learn", "err", "active", "pred", "shared"), 0.0)
    response = ResponseTimes(devices, device, spec.request_ticks)
    bins = torch.tensor(SWITCH_BINS[1:-1], device=device)
    for t in range(ticks):
        agent, obs, step = tick(model, agent, env, obs, mode, costs, args_train["adapter_lr"], args_train["every"],
                                learn_ticks=args_train["learn_ticks"], greedy=greedy, shared=True)
        active, err, perr = step["active"].float(), step["error"], step["prediction_error"]
        serr = step.get("shared_prediction_error", perr)
        where = (step["returned"].long(), torch.bucketize(step["since_switch"], bins, right=True))
        for key, value in (("err", err * active), ("active", active), ("pred", perr), ("shared", serr),
                           ("all", torch.ones_like(perr)),
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
        totals["shared"] += float(serr.sum())
        response.add(t, step)
    n = devices * ticks
    ratio = lambda a, b: (a / b.clamp_min(1)).cpu().tolist()
    predicts = mode != "recurrent"
    life_samples = devices * torch.tensor([min(1000, ticks - 1000 * i) for i in range(life_bins)],
                                          device=device).clamp_min(1)
    return {"reward_per_100_ticks": 100 * totals["reward"] / n,
            "error_while_active": totals["err"] / max(totals["active"], 1),
            "prediction_error": totals["pred"] / n if predicts else None,
            "shared_prediction_error": totals["shared"] / n if predicts else None,
            "refinements_per_tick": totals["refine"] / n, "measurements_per_100_ticks": 100 * totals["measure"] / n,
            "learning_per_100_ticks": 100 * totals["learn"] / n,
            **response.summary(),
            "switch_bins": list(SWITCH_BINS),
            **{f"{name}_after_switch_{group}": ratio(acc[a][g], acc[b][g])
               for name, a, b in (("error", "err", "active"), ("prediction_error", "pred", "all"),
                                  ("shared_prediction_error", "shared", "all"),
                                  ("learning_rate", "learn", "all"), ("refinements", "refine", "all"))
               for g, group in enumerate(("new", "returned"))},
            "active_ticks_after_switch_new": acc["active"][0].cpu().tolist(),
            "active_ticks_after_switch_returned": acc["active"][1].cpu().tolist(),
            "error_by_thousand_ticks": ratio(life["err"], life["active"]),
            "prediction_error_by_thousand_ticks": (life["pred"] / life_samples).cpu().tolist(),
            "reward_per_100_by_thousand_ticks": (100 * life["reward"] / life_samples).cpu().tolist()}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, nargs="+", required=True)
    p.add_argument("--also", nargs="*", default=[], choices=MODES, help="further modes to evaluate each model under")
    p.add_argument("--devices", type=int, default=1024)
    p.add_argument("--ticks", type=int, default=20000)
    p.add_argument("--seed", type=int, default=777)
    p.add_argument("--greedy", action="store_true")
    p.add_argument("--instrument", nargs="*", default=[], metavar="FIELD=VALUE",
                   help="instrument fields to change; a pair of values for a range")
    p.add_argument("--device", default="cuda")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args(argv)
    changes = {}
    for item in args.instrument:
        key, value = item.split("=")
        values = tuple(float(v) for v in value.split(","))
        changes[key] = values if len(values) > 1 else values[0]
    device = torch.device(args.device)
    configure_backends()
    rows = []
    for path in args.model:
        model, targs, spec = load(path, device)
        spec = dataclasses.replace(spec, **changes)
        learns = targs["mode"] in ("full", "fixed_learn", "world_model")
        for mode in dict.fromkeys([targs["mode"], *(args.also if learns else [])]):
            row = {"model": str(path), "variant": path.parent.parent.name, "trained": targs["mode"], "mode": mode,
                   "seed": targs["seed"], "instrument": changes,
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

