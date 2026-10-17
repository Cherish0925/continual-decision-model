#!/usr/bin/env python3
"""What a trained model's predictions are worth for control, with and without its own learning.

``readout`` evaluates the model's policy and a greedy readout of its predictions,
which takes the action whose predicted next reading is closest to the setpoint,
each with learning on and off.

``decisions`` follows one trajectory of the model in its own mode and scores, at
every tick of an active request, the action the adapted predictor and the shared
predictor alone would choose greedily, by the distance of that action's steady
response from the setpoint. The predictors target the next reading, so this
scores where an action leads rather than its one-step effect; the simulator's
response is used only for scoring.

``workspace`` follows one trajectory of a model with a gated workspace and
records, at the first refinement of every tick, the probabilities with which the
gate keeps, revises or restarts the slots, together with the refinement rate,
by ticks since the active request began (idle ticks apart) and by ticks since
the operating condition changed.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.lifetime.evaluate import SWITCH_BINS, evaluate, load, runtime_options, training_costs  # noqa: E402
from experiments.lifetime.runtime import configure_backends, event_features, make_fleet, new_agent, tick  # noqa: E402
from experiments.lifetime.train import first_observation  # noqa: E402

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
            model, targs, specs = load(path, device)
            if control == "greedy":
                greedy_control(model)
            mode = targs["mode"] if learning else "frozen"
            r = evaluate(model, targs, specs, targs.get("family", "instrument"), devices, ticks, seed, device, mode)
            rows.append({"control": control, "learning": learning, **{k: r[k] for k in KEYS}})
            print(json.dumps(rows[-1]), flush=True)
    return rows


@torch.no_grad()
def decisions(path, devices, ticks, seed, device):
    """Same trajectory, two predictors: the instance's adapted predictor and the shared predictor alone.

    Against the noise-free next output of every action (from the simulator's state), each tick records the
    residual error of the action taken and the mean residual over all actions; while a request is active it
    records how often the two predictors would choose different actions greedily, the one-step regret of each
    choice, and the steady-response gap of each choice.
    """
    model, targs, specs = load(path, device)
    family = targs.get("family", "instrument")
    assert family != "mixed", "decisions scores one process family at a time"
    env = make_fleet(family, devices, seed, device, specs)
    agent, obs = new_agent(model, devices, targs["buffer"], device), first_observation(env)
    bins = torch.tensor(SWITCH_BINS[1:-1], device=device)
    nb, rows, table = len(SWITCH_BINS) - 1, torch.arange(devices, device=device), env.table.expand(devices, -1, -1)
    keys = ("taken_adapted", "taken_shared", "map_adapted", "map_shared", "all",
            "differ", "regret_adapted", "regret_shared", "steady_adapted", "steady_shared", "steady_best", "active")
    acc = {k: torch.zeros(nb, device=device) for k in keys}
    torch.manual_seed(seed)
    for _ in range(ticks):
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            h = model.fast(agent.h, event_features(obs, agent, env.pending.clone(), model.config.goals))
            state = model.state_features(h, agent.z)
            adapted = model.predict_actions(state, agent.A, agent.B)[0].float()
            shared = model.predict_actions(state)[0].float()
        truth = env.expected_next(table)
        goal = obs["setpoint"].unsqueeze(1)
        steady = (env.steady(env.current(), table) - goal).abs()
        one_step = (truth - goal).abs()
        pick_a, pick_s = (adapted - goal).abs().argmin(1), (shared - goal).abs().argmin(1)
        b = torch.bucketize(env.since_switch, bins, right=True)
        agent, nobs, step = tick(model, agent, env, obs, targs["mode"], training_costs(targs), targs["adapter_lr"],
                                 targs["every"], learn_ticks=targs["learn_ticks"], **runtime_options(targs))
        taken = step["action"]
        active = obs["active"].float()
        for key, value in (("taken_adapted", (adapted - truth).abs()[rows, taken]),
                           ("taken_shared", (shared - truth).abs()[rows, taken]),
                           ("map_adapted", (adapted - truth).abs().mean(1)), ("map_shared", (shared - truth).abs().mean(1)),
                           ("all", torch.ones_like(active)), ("active", active),
                           ("differ", (pick_a != pick_s).float() * active),
                           ("regret_adapted", (one_step[rows, pick_a] - one_step.min(1).values) * active),
                           ("regret_shared", (one_step[rows, pick_s] - one_step.min(1).values) * active),
                           ("steady_adapted", steady[rows, pick_a] * active), ("steady_shared", steady[rows, pick_s] * active),
                           ("steady_best", steady.min(1).values * active)):
            acc[key].index_add_(0, b, value)
        obs = nobs
    per_tick = ("taken_adapted", "taken_shared", "map_adapted", "map_shared")
    out = {"switch_bins": list(SWITCH_BINS)}
    for key in keys[:4] + keys[5:11]:
        base = acc["all"] if key in per_tick else acc["active"]
        out[key] = (acc[key] / base.clamp_min(1)).tolist()
        out[f"{key}_overall"] = float(acc[key].sum() / base.sum().clamp_min(1))
    return out


REQUEST_BINS = (0, 1, 3, 6, 11, 21, 10 ** 9)           # ticks since the request began; idle ticks apart


@torch.no_grad()
def workspace(path, devices, ticks, seed, device):
    model, targs, specs = load(path, device)
    assert model.gate is not None, "the model has no workspace gate"
    family = targs.get("family", "instrument")
    env = make_fleet(family, devices, seed, device, specs, targs.get("thermal_share", 0.5))
    agent, obs = new_agent(model, devices, targs["buffer"], device), first_observation(env)
    request_edges = torch.tensor(REQUEST_BINS[1:-1], device=device)
    switch_edges = torch.tensor(SWITCH_BINS[1:-1], device=device)
    by_request = torch.zeros(len(REQUEST_BINS), 5, device=device)   # [bins + idle] x [keep, revise, restart, refining, ticks]
    by_switch = torch.zeros(2, len(SWITCH_BINS) - 1, 5, device=device)
    torch.manual_seed(seed)
    for _ in range(ticks):
        active, since_request = obs["active"], agent.since_request.long()
        since_switch, returned = env.since_switch.clone(), env.returned.long()
        agent, obs, step = tick(model, agent, env, obs, targs["mode"], training_costs(targs), targs["adapter_lr"],
                                targs["every"], learn_ticks=targs["learn_ticks"], **runtime_options(targs))
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            gate = model.gate_probabilities(step["z_prev"].float(), agent.h, step["records"]).mean(1)   # [n, 3]
        refining = (step["counts"] > 0).float()
        row = torch.cat((gate * refining.unsqueeze(1), torch.stack((refining, torch.ones_like(refining)), 1)), 1)
        b = torch.where(active, torch.bucketize(since_request, request_edges, right=True),
                        torch.full_like(since_request, len(REQUEST_BINS) - 1))
        by_request.index_put_((b,), row, accumulate=True)
        by_switch.index_put_((returned, torch.bucketize(since_switch, switch_edges, right=True)), row, accumulate=True)

    def summary(acc):
        refining = acc[..., 3].clamp_min(1)
        return {"keep": (acc[..., 0] / refining).tolist(), "revise": (acc[..., 1] / refining).tolist(),
                "restart": (acc[..., 2] / refining).tolist(),
                "refining_share": (acc[..., 3] / acc[..., 4].clamp_min(1)).tolist(), "ticks": acc[..., 4].tolist()}
    return {"request_bins": [*REQUEST_BINS[:-1], "idle"], "by_request": summary(by_request),
            "switch_bins": list(SWITCH_BINS), "by_switch_new": summary(by_switch[0]),
            "by_switch_returned": summary(by_switch[1])}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--probes", nargs="+", default=["readout", "decisions"], choices=["readout", "decisions", "workspace"])
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
    if "workspace" in args.probes:
        out["workspace"] = workspace(args.model, args.devices, args.ticks, args.seed, device)
        print(json.dumps(out["workspace"]["by_request"]), flush=True)
    if "decisions" in args.probes:
        out["decisions"] = decisions(args.model, args.devices, args.ticks, args.seed, device)
        print(json.dumps({k: v for k, v in out["decisions"].items() if k.endswith("overall")}), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
