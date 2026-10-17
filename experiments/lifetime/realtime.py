#!/usr/bin/env python3
"""Act and learn against processes that advance on their own wall clock.

A fleet steps every ``--period`` milliseconds on its own thread and CUDA stream,
applying the latest action it has received and holding the previous one when
no new action has arrived. Every observation is queued with its step, the
action that was actually applied, and its publication time. Each cycle of the
agent first absorbs every queued observation in order: the fast state takes
each event, experience records and the learning buffer pair each reading with
the action that produced it, the prediction error compares each reading with
the prediction made for the action actually applied, and elapsed-time features
count fleet steps.
Then the agent decides on its latest state and publishes the action at once.
The time until the next cycle goes to its chosen compute: refinements on its
own stream, or a learning step handed to a learner thread on a third stream,
whose updated adapter is installed when it returns. The control path never
waits for refinement or learning, so the workspace reads experience records
one step older than in the simulated runtime.

Reported per instance count: response time from the newest published
observation to the published action (median, p95, p99), observations absorbed
per decision, fleet steps that ran on a held action, tracking error, and
refinements and learning steps per second of measured wall time.
"""
from __future__ import annotations

import argparse
import json
import queue
import sys
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from cdm.core import REFINEMENTS, adapter_update  # noqa: E402
from experiments.lifetime.evaluate import load, runtime_options  # noqa: E402
from experiments.lifetime.runtime import (LEARN, configure_backends, distributions, event_features,  # noqa: E402
                                          experience_record, goal_offsets, make_fleet, masks, new_agent,
                                          precise_arrivals, scalar_features)
from experiments.lifetime.train import first_observation  # noqa: E402


class Fleet(threading.Thread):
    """Steps the processes on a fixed schedule and queues every observation."""

    def __init__(self, env, period_ms, steps):
        super().__init__(daemon=True)
        self.env, self.period, self.steps = env, period_ms / 1000, steps
        self.stream = torch.cuda.Stream(env.device)
        self.cond = threading.Condition()
        self.action = torch.full((env.n,), env.spec.actions // 2, dtype=torch.long, device=env.device)
        self.measure = torch.zeros(env.n, dtype=torch.bool, device=env.device)
        self.answered = -1                      # index of the observation the current action responds to
        self.queue = deque()
        self.held = self.err = self.active = 0.0
        self.started = self.finished = 0.0
        self.done = False

    def submit(self, action, measure, index):
        with self.cond:
            self.action, self.measure, self.answered = action, measure, index

    def run(self):
        with torch.cuda.stream(self.stream):
            applied = self.action
            obs = first_observation(self.env)
            self.stream.synchronize()
            self.started = time.perf_counter()
            self._publish(0, obs, applied, -1)
            for k in range(1, self.steps + 1):
                while time.perf_counter() < self.started + k * self.period:
                    time.sleep(0.0001)
                with self.cond:
                    applied, measure, answered = self.action, self.measure, self.answered
                    self.measure = torch.zeros_like(self.measure)        # a measurement request is sent once
                self.held += 0.0 if answered == k - 1 else 1.0
                obs = self.env.step(applied, measure)
                self.stream.synchronize()
                self.err += float((obs["error"] * obs["transition_active"]).sum())
                self.active += float(obs["transition_active"].sum())
                self._publish(k, obs, applied, answered)
            self.finished = time.perf_counter()
        with self.cond:
            self.done = True
            self.cond.notify_all()

    def _publish(self, k, obs, applied, answered):
        """answered: index of the observation the applied action responded to."""
        obs = {**obs, "applied": applied, "answered": answered, "pending": self.env.pending.clone()}
        with self.cond:
            self.queue.append((k, obs, time.perf_counter()))
            self.cond.notify_all()

    def take(self):
        """Every observation not yet taken, in order; empty once the fleet has finished."""
        with self.cond:
            self.cond.wait_for(lambda: self.queue or self.done)
            items = list(self.queue)
            self.queue.clear()
            return items


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
                idx, A, B, xs, ys, ps, held, ready = job
                self.stream.wait_event(ready)
                newA, newB, _ = adapter_update(self.model, A, B, xs, ys, ps, held, self.lr)
                self.stream.synchronize()
                self.completed += len(idx)
                self.results.put((idx, newA, newB))


@torch.no_grad()
def run(model, targs, specs, family, n, period_ms, steps, seed, device, learn: bool):
    torch.manual_seed(seed)
    mode = targs["mode"] if learn else "frozen"
    goals, options = model.config.goals, runtime_options(targs)
    env = make_fleet(family, n, seed, device, specs["instrument"], specs["thermal"], targs.get("thermal_share", 0.5))
    fleet, learner = Fleet(env, period_ms, steps), Learner(model, targs["adapter_lr"], device)
    agent = new_agent(model, n, targs["buffer"], device)
    stream, amp = torch.cuda.Stream(device), torch.autocast("cuda", dtype=torch.bfloat16)
    counts_of = torch.tensor(REFINEMENTS + (0,), device=device)
    rows, T = torch.arange(n, device=device), targs["buffer"]
    learning = torch.zeros(n, dtype=torch.bool, device=device)        # a step for this instance is under way
    flags = torch.zeros(n, 3, device=device)                          # activities of the latest decision
    decided, attached = -2, True                                      # observation the latest decision answered
    state = expect = None                                             # latest state; its predictions for each action
    response, absorbed, refinements = [], [], 0.0
    learner.start()
    fleet.start()
    with torch.cuda.stream(stream):
        while items := fleet.take():
            for i, (k, obs, published) in enumerate(items):           # absorb every observation in order
                if state is not None:                                 # the reading that answered the applied action
                    agent.predicted = expect[rows, obs["applied"]]
                    agent.records = torch.cat((agent.records[:, 1:], experience_record(obs, goals).unsqueeze(1)), 1)
                    agent.learn_s = torch.cat((agent.learn_s[:, 1:], state.unsqueeze(1).to(agent.learn_s.dtype)), 1)
                    agent.learn_a = torch.cat((agent.learn_a[:, 1:], obs["applied"].unsqueeze(1)), 1)
                    agent.learn_y = torch.cat((agent.learn_y[:, 1:], obs["cheap"].unsqueeze(1)), 1)
                    agent.learn_p = torch.cat((agent.learn_p[:, 1:], torch.zeros_like(agent.learn_p[:, :1])), 1)
                    if options["precise_targets"]:
                        agent.learn_y, agent.learn_p = precise_arrivals(agent, obs, agent.learn_y, agent.learn_p)
                    agent.filled = (agent.filled + 1).clamp(max=T)
                    agent.error_ema = 0.95 * agent.error_ema + 0.05 * (obs["cheap"] - agent.predicted).abs()
                    agent.since_measure = torch.where(obs["accepted"], torch.zeros_like(agent.since_measure),
                                                      agent.since_measure + 1)
                    agent.since_request = torch.where(obs["started"], torch.zeros_like(agent.since_request),
                                                      agent.since_request + 1)
                    agent.since_learn = agent.since_learn + 1
                    first = obs["answered"] == decided and not attached    # the step that applied the latest decision
                    agent.last = torch.cat((obs["control"], flags if first else torch.zeros_like(flags)), 1)
                    attached = attached or first
                agent.busy = learning.long()
                event = event_features(obs, agent, obs["pending"], goals)
                with amp:
                    agent.h = model.fast(agent.h, event).float()
                    state = model.state_features(agent.h, agent.z)
                    if i < len(items) - 1:
                        expect = model.predict_actions(state, agent.A, agent.B)[0].float()
            k, obs, published = items[-1]
            absorbed.append(len(items))
            pending = obs["pending"]
            scalars = scalar_features(obs, agent, pending, goals)
            allowed = masks(agent, pending, mode, k, targs["every"])
            with amp:
                mean, logvar = model.predict_actions(state, agent.A, agent.B)
                expect = mean.float()
                offset, logvar = goal_offsets(expect, obs, goals), logvar.float()
                logits, activity, _ = model.heads(model.trunk(agent.h, agent.z, offset, logvar, scalars), offset,
                                                  logvar)
            action, measure, compute = [d.sample() for d in distributions(logits, activity, allowed)]
            stream.synchronize()
            fleet.submit(action, measure.bool(), k)
            response.append(1000 * (time.perf_counter() - published))
            # off the control path: refine, start a learning step, install finished ones
            counts = counts_of[compute]
            with amp:
                agent.z = model.refine(agent.z, agent.h, agent.records, counts).float()
            refinements += float(counts.sum())
            while not learner.results.empty():
                idx, newA, newB = learner.results.get()
                for tensor in (idx, newA, newB):
                    tensor.record_stream(stream)
                agent.A, agent.B = agent.A.clone(), agent.B.clone()
                agent.A[idx], agent.B[idx] = newA, newB
                learning[idx] = False
            start = (compute == LEARN) & ~learning & (agent.filled > 0)
            if start.any():
                idx = start.nonzero().squeeze(1)
                if options["support"] == "buffer":                    # every filled entry of the buffer
                    sel = (idx.unsqueeze(1), torch.arange(T, device=device).expand(len(idx), T))
                    held = (sel[1] >= (T - agent.filled[idx]).unsqueeze(1)).float()
                else:
                    pick = torch.randint(0, T, (len(idx), 32), device=device) % agent.filled[idx].unsqueeze(1)
                    sel, held = (idx.unsqueeze(1), T - 1 - pick), torch.ones(len(idx), 32, device=device)
                xs = torch.cat((agent.learn_s[sel].float(), model.action_features(agent.learn_a[sel])), -1)
                job = (idx, agent.A[idx].clone(), agent.B[idx].clone(), xs, agent.learn_y[sel], agent.learn_p[sel],
                       held)
                for tensor in job:
                    tensor.record_stream(learner.stream)
                ready = torch.cuda.Event()
                ready.record(stream)
                learner.jobs.put((*job, ready))
                learning |= start
            agent.since_learn = torch.where(compute == LEARN, torch.zeros_like(agent.since_learn), agent.since_learn)
            flags = torch.stack((measure.float(), counts.float() / 4, (compute == LEARN).float()), 1)
            decided, attached = k, False
    learner.jobs.put(None)
    fleet.join()
    learner.join()
    seconds = fleet.finished - fleet.started
    lat = np.asarray(response[10:] if len(response) > 20 else response)          # the first actions warm up
    quantile = lambda q: float(np.percentile(lat, q)) if len(lat) else None
    return {"instances": n, "period_ms": period_ms, "learning": learn, "fleet_steps": steps, "decisions": len(response),
            "observations_per_decision": float(np.mean(absorbed)) if absorbed else None,
            "response_ms_median": quantile(50), "response_ms_p95": quantile(95), "response_ms_p99": quantile(99),
            "held_steps": int(fleet.held), "error_while_active": fleet.err / max(fleet.active, 1),
            "wall_seconds": round(seconds, 3), "refinements_per_second": refinements / seconds,
            "learning_steps_per_second": learner.completed / seconds}


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
