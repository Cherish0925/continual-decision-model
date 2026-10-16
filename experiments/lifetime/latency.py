#!/usr/bin/env python3
"""Wall-clock cost of control, refinement and learning, and control latency while learning runs alongside.

Part one times each operation for a number of instances served together:
the control path (event, fast state, predictions for every action under the
instance adapters, heads, sampling), one workspace refinement, and one
learning step on 32 stored ticks.

Part two runs an actor on its own CUDA stream that issues control every
``--period`` milliseconds, first alone and then while a learner thread runs
learning steps back to back on a second stream, and reports the actor's
latency from tick start to control available on the host.
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from cdm.core import adapter_step  # noqa: E402
from runtime import configure_backends  # noqa: E402


def operations(model, n, device):
    c = model.config
    s = model.initial(n, device)
    h, z, A, B = s["h"], s["z"], s["A"], s["B"]
    event = torch.randn(n, c.event_dim, device=device)
    scalars = torch.randn(n, c.scalar_dim, device=device)
    records = torch.randn(n, c.records, c.record_dim, device=device)
    xs = torch.randn(n, 32, 3 * c.width, device=device)
    ys = torch.randn(n, 32, device=device)
    amp = torch.autocast(device.type, dtype=torch.bfloat16)

    def control():
        with amp:
            hn = model.fast(h, event)
            mean, logvar = model.predict_actions(model.state_features(hn, z), A, B)
            mean, logvar = mean.float(), logvar.float()
            logits, activity, _ = model.heads(model.trunk(hn, z, mean, logvar, scalars), mean, logvar)
        return torch.distributions.Categorical(logits=logits.float()).sample()

    def refine():
        with amp:
            return model.refine(z, h, records, torch.ones(n, dtype=torch.long, device=device))

    def learn():
        return adapter_step(model, A, B, xs, ys, 0.05)[0]

    return {"control": control, "refinement": refine, "learning": learn}


def time_op(fn, repeats=50):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    out = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        out.append(1000 * (time.perf_counter() - t0))
    return float(np.median(out))


def clock_statistics(service_ms, response_ms, period_ms):
    """Service time excludes queueing; scheduled response time includes any accumulated lateness."""
    service, response = np.asarray(service_ms), np.asarray(response_ms)
    return {"latency_ms_median": float(np.median(service)),
            "latency_ms_p99": float(np.percentile(service, 99)),
            "response_ms_median": float(np.median(response)),
            "response_ms_p95": float(np.percentile(response, 95)),
            "response_ms_p99": float(np.percentile(response, 99)),
            "missed_periods": int((response > period_ms).sum())}


def actor_latency(model, n, device, period_ms, ticks, with_learner):
    ops = operations(model, n, device)
    actor, learner = torch.cuda.Stream(device), torch.cuda.Stream(device)
    stop, steps = threading.Event(), [0]

    def learn_loop():
        with torch.cuda.stream(learner):
            while not stop.is_set():
                ops["learning"]()
                learner.synchronize()
                steps[0] += 1

    thread = threading.Thread(target=learn_loop) if with_learner else None
    if thread:
        thread.start()
        time.sleep(0.5)
    latency, response, start = [], [], time.perf_counter()
    with torch.cuda.stream(actor):
        for t in range(ticks):
            due = start + t * period_ms / 1000
            while time.perf_counter() < due:
                time.sleep(0.0002)
            t0 = time.perf_counter()
            ops["control"]().cpu()                      # control on the host: waits only for the actor stream
            completed = time.perf_counter()
            latency.append(1000 * (completed - t0))
            response.append(1000 * (completed - due))
    elapsed = time.perf_counter() - start
    if thread:
        stop.set()
        thread.join()
    return {"instances": n, "period_ms": period_ms, "learner": with_learner,
            **clock_statistics(latency, response, period_ms), "ticks": ticks,
            "learning_steps_per_tick": steps[0] / ticks if with_learner else 0.0,
            "seconds": round(elapsed, 2)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--instances", type=int, nargs="+", default=[1, 16, 256, 1024])
    p.add_argument("--period", type=float, default=10.0, help="actor period in milliseconds")
    p.add_argument("--ticks", type=int, default=1000)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    configure_backends()
    device = torch.device("cuda")
    from evaluate import load
    model, _, _ = load(args.model, device)
    model.requires_grad_(False)
    timing = []
    with torch.no_grad():
        for n in args.instances:
            ops = operations(model, n, device)
            row = {"instances": n, **{f"{k}_ms": time_op(f) for k, f in ops.items()}}
            timing.append(row)
            print(json.dumps(row), flush=True)
        clock = []
        for n in args.instances:
            for with_learner in (False, True):
                row = actor_latency(model, n, device, args.period, args.ticks, with_learner)
                clock.append(row)
                print(json.dumps(row), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"gpu": torch.cuda.get_device_name(device), "timing": timing,
                                       "independent_clock": clock}, indent=1))


if __name__ == "__main__":
    main()

