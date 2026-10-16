#!/usr/bin/env python3
"""Reference tracking errors for an instrument family: random actions, and a greedy controller that knows
the true response (current condition and drift) and takes the action whose response is closest to the setpoint.
The greedy controller ignores relaxation and backlash; it is a reference point, not a bound."""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from cdm.instrument import InstrumentFleet, InstrumentSpec, response  # noqa: E402


@torch.no_grad()
def reference(spec: InstrumentSpec, policy: str, devices: int, ticks: int, seed: int, device) -> float:
    env = InstrumentFleet(devices, spec, seed=seed, device=device)
    gen = torch.Generator(device=device).manual_seed(seed + 1)
    err = active = 0.0
    for _ in range(ticks):
        if policy == "random":
            action = torch.randint(0, spec.actions, (devices,), generator=gen, device=device)
        else:
            r = response(spec, env.current().unsqueeze(1), env.table.unsqueeze(0), env.drift.unsqueeze(1))
            action = (r - env.setpoint.unsqueeze(1)).abs().argmin(1)
        out = env.step(action, torch.zeros(devices, dtype=torch.bool, device=device))
        err += float(out["error"].sum())
        active += float(out["active"].sum())
    return err / active


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--family", nargs=2, action="append", metavar=("NAME", "CHANGES"), default=[],
                   help='a name and instrument changes, e.g. slow_narrow "time_constant=8,16 width=0.15,0.25"')
    p.add_argument("--devices", type=int, default=1024)
    p.add_argument("--ticks", type=int, default=3000)
    p.add_argument("--seed", type=int, default=3)
    p.add_argument("--device", default="cuda")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    device = torch.device(args.device)
    out = {}
    for name, text in [("training", "")] + args.family:
        changes = {}
        for item in text.split():
            key, value = item.split("=")
            values = tuple(float(v) for v in value.split(","))
            changes[key] = values if len(values) > 1 else values[0]
        spec = dataclasses.replace(InstrumentSpec(), **changes)
        out[name] = {"instrument": changes, **{policy: reference(spec, policy, args.devices, args.ticks, args.seed, device)
                                              for policy in ("random", "true_response_greedy")}}
        print(name, json.dumps(out[name]), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
