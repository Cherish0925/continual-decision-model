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

``--also`` evaluates each model that learns a second time under other modes:
``frozen`` switches learning off, so the compute it used is chosen anew;
``discard`` keeps every chosen learning step, its time and its cost, but throws
its adapter update away, which isolates what the updates themselves contribute;
``norefine`` keeps the model's own setting but removes the refinement options,
so its compute goes to learning or nothing, which isolates what refinement
contributes; ``noidle`` removes refinement and learning while no request is
active, which isolates what computation between requests contributes to later
ones. Errors are also reported separately for setpoint requests and requests
for the largest output, and activities separately for ticks with and without
an active request.
Every setting of a model starts from the same random state, so the settings are
compared under common random numbers. ``--family`` selects the process family
(by default the one trained on); ``--instrument`` changes fields of its
specification, for example ``time_constant=8,16 width=0.15,0.25`` for dynamics
outside the training ranges.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from cdm.core import CoreConfig, ContinualCore  # noqa: E402
from cdm.instrument import InstrumentSpec  # noqa: E402
from cdm.navigation import NavigationSpec  # noqa: E402
from cdm.thermal import ThermalSpec  # noqa: E402
from experiments.lifetime.runtime import FAMILIES, MODES, Costs, configure_backends, make_fleet, new_agent, tick  # noqa: E402
from experiments.lifetime.train import first_observation  # noqa: E402

SWITCH_BINS = (0, 10, 25, 50, 100, 200, 400, 800, 10 ** 9)


class ResponseTimes:
    """Count executed controls until first reaching a request, with failures capped at its duration.

    Counts accumulate on the device as a histogram, so recording a tick needs no host synchronisation.
    """

    def __init__(self, devices, device, request_ticks):
        self.start = torch.full((devices,), -1, dtype=torch.long, device=device)
        self.reached = torch.zeros(devices, dtype=torch.bool, device=device)
        self.request_ticks = request_ticks
        self.counts = torch.zeros(request_ticks + 1, device=device)   # by response time in ticks
        self.successes = torch.zeros((), device=device)

    def add(self, t, step):
        # started belongs to the request served by this action; next_started
        # belongs to a request that can first be served on the following tick.
        started, active = step["started"], step["active"]
        self.start = torch.where(started, torch.full_like(self.start, t), self.start)
        self.reached = torch.where(started, torch.zeros_like(self.reached), self.reached)
        hit = (self.start >= 0) & ~self.reached & active & (step["error"] < 0.1)
        self.counts.index_put_(((t - self.start + 1).clamp(0, self.request_ticks),), hit.float(), accumulate=True)
        self.successes += hit.sum()
        self.reached |= hit
        ended = active & (~step["next_active"] | step["next_started"])
        failed = ended & (self.start >= 0) & ~self.reached
        self.counts[self.request_ticks] += failed.sum()
        self.start = torch.where(ended, torch.full_like(self.start, -1), self.start)
        self.reached &= ~ended

    @property
    def values(self) -> list[int]:
        """Response times of completed requests, in increasing order."""
        return [k for k, c in enumerate(self.counts.round().long().tolist()) for _ in range(c)]

    def summary(self):
        values = self.values
        return {"response_ticks_median": float(np.median(values)) if values else None,
                "response_ticks_mean": float(np.mean(values)) if values else None,
                "requests": len(values), "successful_requests": int(self.successes),
                "unfinished_requests": int(((self.start >= 0) & ~self.reached).sum())}


def block_use(use: torch.Tensor) -> dict:
    """How consistently each operating condition is served by one adapter block, from ticks [n, conditions, blocks]:
    the share of ticks on a condition's most used block, and the share of instances in which two conditions
    met in the lifetime have the same most used block."""
    met = use.sum(-1) > 0
    top = use.argmax(-1)
    same = (top.unsqueeze(2) == top.unsqueeze(1)) & met.unsqueeze(2) & met.unsqueeze(1)
    same &= ~torch.eye(use.shape[1], dtype=torch.bool, device=use.device)
    return {"block_purity": float(use.max(-1).values.sum() / use.sum().clamp_min(1)),
            "block_shared_by_conditions": float(same.flatten(1).any(1).float().mean())}


def runtime_options(args_train: dict) -> dict:
    """Options of ``runtime.tick`` a model was trained with; earlier checkpoints predate them."""
    return {"support": args_train.get("support", "sample"), "precise_targets": args_train.get("precise_targets", False)}


def training_costs(args_train: dict) -> Costs:
    return Costs(*(args_train.get(f"cost_{k}", getattr(Costs, k)) for k in ("refinement", "measurement", "learning")))


def load(path: Path, device):
    blob = torch.load(path, map_location=device, weights_only=False)
    weights = blob["model"]       # earlier checkpoints predate these options; their weights tell which they used
    config = {"goal_term": "goal_weight" in weights, "bounded_workspace": "norm_slots.weight" in weights,
              "learned_step": "log_step" in weights, **blob["config"]}
    config = CoreConfig(**config)
    model = ContinualCore(config, torch.zeros(config.actions, config.action_dim)).to(device)
    model.load_state_dict(blob["model"])
    instrument = InstrumentSpec(**blob["instrument"])
    shape = {"dims": instrument.dims, "levels": instrument.levels}
    thermal = ThermalSpec(**blob.get("thermal", shape))
    navigation = NavigationSpec(**blob.get("navigation", {**shape, "search_share": instrument.search_share}))
    return model.eval(), blob["args"], {"instrument": instrument, "thermal": thermal, "navigation": navigation}


@torch.no_grad()
def evaluate(model, args_train, specs, family, devices, ticks, seed, device, mode, greedy=False):
    torch.manual_seed(seed)
    apply_updates, refinements, idle_compute = mode != "discard", mode != "norefine", mode != "noidle"
    mode = args_train["mode"] if mode in ("discard", "norefine", "noidle") else mode
    env = make_fleet(family, devices, seed, device, specs, args_train.get("thermal_share", 0.5))
    spec = env.spec
    agent = new_agent(model, devices, args_train["buffer"], device)
    obs = first_observation(env)
    costs = training_costs(args_train)
    nb = len(SWITCH_BINS) - 1
    life_bins = max(1, (ticks + 999) // 1000)
    acc = {key: torch.zeros(2, nb, device=device) for key in ("err", "active", "pred", "shared", "all", "learn", "refine")}
    life = {key: torch.zeros(life_bins, device=device) for key in ("err", "active", "pred", "reward")}
    totals = torch.zeros(8, device=device)           # reward, refine, measure, learn, err, active, pred, shared
    kinds = torch.zeros(2, 2, device=device)         # [setpoint, largest output] x [error, active ticks]
    by_request = torch.zeros(2, 4, device=device)    # [idle, active] x [ticks, refinements, measurements, learning]
    blocks = model.config.adapter_blocks
    use = torch.zeros(devices, spec.conditions, blocks, device=device)   # ticks each condition gave each block most
    response = ResponseTimes(devices, device, spec.request_ticks)
    bins = torch.tensor(SWITCH_BINS[1:-1], device=device)
    for t in range(ticks):
        agent, obs, step = tick(model, agent, env, obs, mode, costs, args_train["adapter_lr"], args_train["every"],
                                learn_ticks=args_train["learn_ticks"], greedy=greedy, shared=True,
                                apply_updates=apply_updates, refinements=refinements,
                                idle_compute=idle_compute, **runtime_options(args_train))
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
        totals += torch.stack((step["reward"].sum(), step["counts"].float().sum(), step["accepted"].float().sum(),
                               step["learn"].float().sum(), (err * active).sum(), active.sum(), perr.sum(), serr.sum()))
        if "block" in step:
            use[torch.arange(devices, device=device), env.condition, step["block"]] += 1
        search = step["transition_search"].long()
        kinds.index_put_((search,), torch.stack((err * active, active), 1), accumulate=True)
        by_request.index_put_((step["active"].long(),), torch.stack((torch.ones_like(active), step["counts"].float(),
                                                                   step["accepted"].float(),
                                                                   step["learn"].float()), 1), accumulate=True)
        response.add(t, step)
    n = devices * ticks
    reward, refine, measure, learn, err_sum, active_sum, pred, shared = totals.tolist()
    ratio = lambda a, b: (a / b.clamp_min(1)).cpu().tolist()
    predicts = mode != "recurrent"
    life_samples = devices * torch.tensor([min(1000, ticks - 1000 * i) for i in range(life_bins)],
                                          device=device).clamp_min(1)
    return {"reward_per_100_ticks": 100 * reward / n,
            "error_while_active": err_sum / max(active_sum, 1),
            "prediction_error": pred / n if predicts else None,
            "shared_prediction_error": shared / n if predicts else None,
            "refinements_per_tick": refine / n, "measurements_per_100_ticks": 100 * measure / n,
            "learning_per_100_ticks": 100 * learn / n,
            "error_setpoint_requests": float(kinds[0, 0] / kinds[0, 1].clamp_min(1)),
            "error_largest_output_requests": float(kinds[1, 0] / kinds[1, 1]) if kinds[1, 1] > 0 else None,
            **{f"{what}_per_tick_{state}": float(by_request[i, j] / by_request[i, 0].clamp_min(1))
               for i, state in enumerate(("idle", "active"))
               for j, what in ((1, "refinements"), (2, "measurements"), (3, "learning"))},
            "idle_share": float(by_request[0, 0] / n),
            **(block_use(use) if blocks > 1 else {}),
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
    p.add_argument("--also", nargs="*", default=[], choices=(*MODES, "discard", "norefine", "noidle"),
                   help="further settings to evaluate each learning model under")
    p.add_argument("--family", choices=FAMILIES, help="process family to evaluate on; by default the training family")
    p.add_argument("--adapter-lr", type=float, help="learning-step size during evaluation; by default the training one")
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
        model, targs, specs = load(path, device)
        if args.adapter_lr is not None:
            targs = {**targs, "adapter_lr": args.adapter_lr}
        family = args.family or targs.get("family", "instrument")
        specs = {name: dataclasses.replace(spec, **{k: v for k, v in changes.items() if hasattr(spec, k)})
                 for name, spec in specs.items()}
        learns = targs["mode"] in ("full", "fixed_learn", "world_model")
        refines = targs["mode"] in ("full", "fixed_learn", "frozen")
        # a model trained on fixed schedules has untrained activity heads: switch its learning off with discard
        also = [m for m in args.also if (refines if m == "norefine" else (refines or learns) if m == "noidle" else
                                         learns and not (targs["mode"] == "world_model" and m == "frozen"))]
        for mode in dict.fromkeys([targs["mode"], *also]):
            row = {"model": str(path), "variant": path.parent.parent.name, "trained": targs["mode"], "mode": mode,
                   "seed": targs["seed"], "family": family, "instrument": changes, "adapter_lr": targs["adapter_lr"],
                   **evaluate(model, targs, specs, family, args.devices, args.ticks, args.seed, device, mode,
                              args.greedy)}
            rows.append(row)
            print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()
                              if not isinstance(v, list)}), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    args.output.write_text(json.dumps({"args": {k: ([str(x) for x in v] if k == "model" else
                                                    str(v) if isinstance(v, Path) else v)
                                                for k, v in vars(args).items()},
                                       "revision": revision or None, "runs": rows}, indent=1))


if __name__ == "__main__":
    main()

