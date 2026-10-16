#!/usr/bin/env python3
"""Act and learn against processes that advance on their own wall clock.

A fleet steps every ``--period`` milliseconds on its own thread and CUDA stream,
applying the latest action it has received and holding the previous one when
no new action has arrived. The agent runs the control path as soon as an
observation is published and publishes its action at once; then it spends the
time until the next observation on its chosen compute: refinements on its own
stream, or a learning step handed to a learner thread on a third stream, whose
updated adapter is installed when it returns. The control path never waits for
refinement or learning, so the workspace reads experience records one tick
older than in the simulated runtime, and a learning step takes as long as it
takes.

Reported per instance count: response time from the published observation to
the published action (median, p95, p99), fleet steps that ran with a held
action, tracking error, and refinements and learning steps per second.
"""
from __future__ import annotations

import argparse
import json
import queue
import sys
import threading
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from cdm.core import REFINEMENTS, adapter_step  # noqa: E402
from experiments.lifetime.evaluate import load  # noqa: E402
from experiments.lifetime.runtime import (LEARN, configure_backends, distributions, event_features, make_fleet, masks,  # noqa: E402
                     new_agent, scalar_features)
from experiments.lifetime.train import first_observation  # noqa: E402


class Fleet(threading.Thread):
    """Steps the processes on a fixed schedule and publishes each observation with its time."""

    def __init__(self, env, period_ms, steps):
        super().__init__(daemon=True)
        self.env, self.period, self.steps = env, period_ms / 1000, steps
        self.stream = torch.cuda.Stream(env.device)
        self.cond = threading.Condition()
        self.action = torch.full((env.n,), env.spec.actions // 2, dtype=torch.long, device=env.device)
        self.measure = torch.zeros(env.n, dtype=torch.bool, device=env.device)
        self.answered = -1                      # index of the observation the current action responds to
        self.obs, self.index, self.published = None, -1, 0.0
        self.held = self.err = self.active = 0.0
        self.done = False

    def submit(self, action, measure, index):
        with self.cond:
            self.action, self.measure, self.answered = action, measure, index

    def run(self):
        with torch.cuda.stream(self.stream):
            obs = first_observation(self.env)
            self.stream.synchronize()
            start = time.perf_counter()
            self._publish(obs, 0)
            for k in range(1, self.steps + 1):
                while time.perf_counter() < start + k * self.period:
                    time.sleep(0.0001)
                with self.cond:
                    action, measure, fresh = self.action, self.measure, self.answered == k - 1
                    self.measure = torch.zeros_like(self.measure)        # a measurement request is sent once
                self.held += 0.0 if fresh else 1.0
                obs = self.env.step(action, measure)
                self.stream.synchronize()
                self.err += float((obs["error"] * obs["transition_active"]).sum())
                self.active += float(obs["transition_active"].sum())
                self._publish(obs, k)
        with self.cond:
            self.done = True
            self.cond.notify_all()

    def _publish(self, obs, k):
        obs = {**obs, "pending": self.env.pending.clone()}
        with self.cond:
            self.obs, self.index, self.published = obs, k, time.perf_counter()
            self.cond.notify_all()

    def next_observation(self, after):
        with self.cond:
            self.cond.wait_for(lambda: self.index > after or self.done)
            return (None, None, None) if self.index <= after else (self.obs, self.index, self.published)


class Learner(threading.Thread):
    """Takes learning steps on its own stream and returns the updated adapters."""

    def __init__(self, model, learning_rate, device):
        super().__init__(daemon=True)
        self.model, self.lr = model, learning_rate
        self.stream = torch.cuda.Stream(device)
        self.jobs, self.results = queue.Queue(), queue.Queue()
        self.completed = 0

    def run(self):
        with torch.cuda.stream(self.stream):
            while (job := self.jobs.get()) is not None:
                idx, A, B, xs, ys, ready = job
                self.stream.wait_event(ready)
                newA, newB, _ = adapter_step(self.model, A, B, xs, ys, self.lr)
                self.stream.synchronize()
                self.completed += len(idx)
                self.results.put((idx, newA, newB))


@torch.no_grad()
def run(model, targs, specs, family, n, period_ms, steps, seed, device, learn: bool):
    torch.manual_seed(seed)
    mode = targs["mode"] if learn else "frozen"
    env = make_fleet(family, n, seed, device, specs["instrument"], specs["thermal"], targs.get("thermal_share", 0.5))
    fleet, learner = Fleet(env, period_ms, steps), Learner(model, targs["adapter_lr"], device)
    agent = new_agent(model, n, targs["buffer"], device)
    stream, amp = torch.cuda.Stream(device), torch.autocast("cuda", dtype=torch.bfloat16)
    counts_of = torch.tensor(REFINEMENTS + (0,), device=device)
    rows, T = torch.arange(n, device=device), targs["buffer"]
    learning = torch.zeros(n, dtype=torch.bool, device=device)        # a step for this instance is under way
    response, refinements = [], 0.0
    learner.start()
    fleet.start()
    index, previous = -1, None                                         # previous: (state, action) awaiting its reading
    with torch.cuda.stream(stream):
        while True:
            obs, index, published = fleet.next_observation(index)
            if obs is None:
                break
            pending = obs["pending"]
            event = event_features(obs, agent, pending)
            scalars = scalar_features(obs, agent, pending)
            agent.busy = learning.long()
            allowed = masks(agent, pending, mode, index, targs["every"])
            with amp:
                h = model.fast(agent.h, event)
                state = model.state_features(h, agent.z)
                mean, logvar = model.predict_actions(state, agent.A, agent.B)
                offset, logvar = mean.float() - obs["setpoint"].unsqueeze(1), logvar.float()
                logits, activity, _ = model.heads(model.trunk(h, agent.z, offset, logvar, scalars), offset, logvar)
            action, measure, compute = [d.sample() for d in distributions(logits, activity, allowed)]
            stream.synchronize()
            fleet.submit(action, measure.bool(), index)
            response.append(1000 * (time.perf_counter() - published))
            # off the control path: record the reading that answered the previous action, refine, learn
            if previous is not None:
                s_prev, a_prev, u_prev = previous
                record = torch.cat((u_prev, torch.stack((obs["cheap"], obs["precise"], obs["arrived"].float(),
                                                         obs["setpoint"], obs["active"].float()), 1)), 1)
                agent.records = torch.cat((agent.records[:, 1:], record.unsqueeze(1)), 1)
                agent.learn_s = torch.cat((agent.learn_s[:, 1:], s_prev.unsqueeze(1).to(agent.learn_s.dtype)), 1)
                agent.learn_a = torch.cat((agent.learn_a[:, 1:], a_prev.unsqueeze(1)), 1)
                agent.learn_y = torch.cat((agent.learn_y[:, 1:], obs["cheap"].unsqueeze(1)), 1)
                agent.filled = (agent.filled + 1).clamp(max=T)
            previous = (state, action, env.table[action])
            counts = counts_of[compute]
            with amp:
                agent.z = model.refine(agent.z, h, agent.records, counts).float()
            refinements += float(counts.sum())
            while not learner.results.empty():
                idx, newA, newB = learner.results.get()
                for tensor in (idx, newA, newB):
                    tensor.record_stream(stream)
                agent.A, agent.B = agent.A.clone(), agent.B.clone()
                agent.A[idx], agent.B[idx] = newA, newB
                learning[idx] = False
            start = (compute == LEARN) & ~learning
            if start.any():
                idx = start.nonzero().squeeze(1)
                pick = T - 1 - (torch.randint(0, T, (len(idx), 32), device=device) % agent.filled[idx].unsqueeze(1))
                sel = (idx.unsqueeze(1), pick)
                xs = torch.cat((agent.learn_s[sel].float(), model.action_features(agent.learn_a[sel])), -1)
                job = (idx, agent.A[idx].clone(), agent.B[idx].clone(), xs, agent.learn_y[sel])
                for tensor in job:
                    tensor.record_stream(learner.stream)
                ready = torch.cuda.Event()
                ready.record(stream)
                learner.jobs.put((*job, ready))
                learning |= start
            pred = mean.float()[rows, action]
            agent.h = h.float()
            agent.last = torch.cat((env.table[action], torch.stack((measure.float(), counts.float() / 4,
                                                                    (compute == LEARN).float()), 1)), 1)
            agent.since_measure = torch.where(measure.bool() & (pending == 0), torch.zeros_like(agent.since_measure),
                                              agent.since_measure + 1)
            agent.since_learn = torch.where(compute == LEARN, torch.zeros_like(agent.since_learn), agent.since_learn + 1)
            agent.since_request = torch.where(obs["started"], torch.zeros_like(agent.since_request),
                                              agent.since_request + 1)
            agent.error_ema = 0.95 * agent.error_ema + 0.05 * (obs["cheap"] - agent.predicted).abs()
            agent.predicted = pred
    learner.jobs.put(None)
    fleet.join()
    learner.join()
    seconds = steps * period_ms / 1000
    lat = np.asarray(response[10:] if len(response) > 20 else response)          # the first actions warm up
    quantile = lambda q: float(np.percentile(lat, q)) if len(lat) else None
    return {"instances": n, "period_ms": period_ms, "learning": learn, "fleet_steps": steps, "actions": len(response),
            "response_ms_median": quantile(50), "response_ms_p95": quantile(95),
            "response_ms_p99": quantile(99), "held_steps": int(fleet.held),
            "error_while_active": fleet.err / max(fleet.active, 1),
            "refinements_per_second": refinements / seconds, "learning_steps_per_second": learner.completed / seconds}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--family", help="process family; by default the training family")
    p.add_argument("--instances", type=int, nargs="+", default=[1, 64, 512])
    p.add_argument("--period", type=float, default=10.0, help="fleet step period in milliseconds")
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--seed", type=int, default=777)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    configure_backends()
    device = torch.device("cuda")
    model, targs, specs = load(args.model, device)
    model.requires_grad_(False)
    family = args.family or targs.get("family", "instrument")
    rows = []
    for n in args.instances:
        for learn in (True, False):
            rows.append(run(model, targs, specs, family, n, args.period, args.steps, args.seed, device, learn))
            print(json.dumps(rows[-1]), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"model": args.model.parent.parent.name, "seed": args.model.parent.name,
                                       "family": family, "gpu": torch.cuda.get_device_name(device),
                                       "runs": rows}, indent=1))


if __name__ == "__main__":
    main()
