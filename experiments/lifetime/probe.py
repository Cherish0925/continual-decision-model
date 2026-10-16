#!/usr/bin/env python3
"""What a trained model's predictions are worth for control, with and without its own learning.

``readout`` evaluates the model's policy and a greedy readout of its predictions,
which takes the action whose predicted next reading is closest to the setpoint,
each with learning on and off.

``decisions`` follows one trajectory of the model in its own mode and scores, at
every tick of an active request, the action the adapted predictor and the shared
predictor alone would choose greedily, by the true distance of that action's
response from the setpoint. The simulator's response is used only for scoring.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from cdm.instrument import InstrumentFleet, response  # noqa: E402
from evaluate import SWITCH_BINS, evaluate, load  # noqa: E402
from runtime import Costs, configure_backends, event_features, new_agent, tick  # noqa: E402
from train import first_observation  # noqa: E402

KEYS = ("error_while_active", "prediction_error", "shared_prediction_error", "refinements_per_tick",
        "learning_per_100_ticks", "response_ticks_median", "reward_per_100_ticks")


def greedy_control(model):
    heads = model.heads

    def readout(trunk, offset, logvar):
        _, activity, value = heads(trunk, offset, logvar)
        return -100.0 * offset.abs(), activity, value
    model.heads = readout
    return model


def readouts(path, devices, ticks, seed, device):
    rows = []
    for control in ("policy", "greedy"):
        for learning in (True, False):
            model, targs, spec = load(path, device)
            if control == "greedy":
                greedy_control(model)
            mode = targs["mode"] if learning else "frozen"
            r = evaluate(model, targs, spec, devices, ticks, seed, device, mode)
            rows.append({"control": control, "learning": learning, **{k: r[k] for k in KEYS}})
            print(json.dumps(rows[-1]), flush=True)
    return rows


@torch.no_grad()
def decisions(path, devices, ticks, seed, device):
    model, targs, spec = load(path, device)
    env = InstrumentFleet(devices, spec, seed=seed, device=device)
    agent, obs = new_agent(model, devices, targs["buffer"], device), first_observation(env)
    bins = torch.tensor(SWITCH_BINS[1:-1], device=device)
    nb, rows = len(SWITCH_BINS) - 1, torch.arange(devices, device=device)
    acc = {k: torch.zeros(nb, device=device) for k in ("adapted", "shared", "best", "count")}
    for _ in range(ticks):
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            h = model.fast(agent.h, event_features(obs, agent, env.pending.clone()))
            state = model.state_features(h, agent.z)
            adapted = model.predict_actions(state, agent.A, agent.B)[0].float()
            shared = model.predict_actions(state)[0].float()
        r = response(spec, env.current().unsqueeze(1), env.table.unsqueeze(0), env.drift.unsqueeze(1))
        goal = obs["setpoint"].unsqueeze(1)
        gap = (r - goal).abs()
        active = obs["active"].float()
        b = torch.bucketize(env.since_switch, bins, right=True)
        for key, value in (("adapted", gap[rows, (adapted - goal).abs().argmin(1)]),
                           ("shared", gap[rows, (shared - goal).abs().argmin(1)]),
                           ("best", gap.min(1).values), ("count", torch.ones_like(active))):
            acc[key].index_add_(0, b, value * active)
        agent, obs, _ = tick(model, agent, env, obs, targs["mode"], Costs(), targs["adapter_lr"], targs["every"],
                             learn_ticks=targs["learn_ticks"])
    count = acc["count"].clamp_min(1)
    return {"switch_bins": list(SWITCH_BINS), **{k: (acc[k] / count).tolist() for k in ("adapted", "shared", "best")},
            **{f"{k}_overall": float(acc[k].sum() / acc["count"].sum()) for k in ("adapted", "shared", "best")}}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--probes", nargs="+", default=["readout", "decisions"], choices=["readout", "decisions"])
    p.add_argument("--devices", type=int, default=256)
    p.add_argument("--ticks", type=int, default=4000)
    p.add_argument("--seed", type=int, default=777)
    p.add_argument("--device", default="cuda")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    configure_backends()
    device = torch.device(args.device)
    out = {"model": args.model.parent.parent.name, "seed": args.model.parent.name, "devices": args.devices,
           "ticks": args.ticks}
    if "readout" in args.probes:
        out["readout"] = readouts(args.model, args.devices, args.ticks, args.seed, device)
    if "decisions" in args.probes:
        out["decisions"] = decisions(args.model, args.devices, args.ticks, args.seed, device)
        print(json.dumps({k: v for k, v in out["decisions"].items() if k.endswith("overall")}), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
