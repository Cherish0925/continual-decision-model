"""Measured task streams with private transitions and delayed public feedback.

An external request changes a private snapshot. Its measurements, success flag,
and executed setting become visible only when the request returns. No clock
tick duplicates a measurement, an outcome, or a task transition.
"""
from __future__ import annotations

import numpy as np
import torch

from experiments.discoveryworld.train_active_core import (
    ActiveTableEnvironment, measurement_event, test_event,
)
from experiments.discoveryworld.train_active_lifetime import _TASK_FIELDS
from experiments.physical_belief.task_streams import PhysicalTaskEnvironment


class ExactPhysicalTasks(PhysicalTaskEnvironment):
    def __init__(self, pool, plan, mixture, *, truth_path=None, max_steps=160):
        # Keep task creation, public scalar settings and private resonances in
        # one dtype across later task boundaries as well as initialization.
        pool.labels = pool.labels.double()
        pool.data["working_frequency_hz"] = pool.data["working_frequency_hz"].double()
        self.exact_native = None
        if truth_path is not None:
            with np.load(truth_path) as data:
                resonance = data["resonance_hz"]
            if resonance.shape != (pool.size, 4):
                raise ValueError("Environment truth must match the training world/object order")
            self.exact_native = torch.as_tensor(resonance, dtype=torch.float64, device=pool.device)
        super().__init__(pool, plan, mixture, max_steps=max_steps, tolerance_hz=2.)

    @torch.no_grad()
    def _begin_tasks(self, rows, initial=False):
        super()._begin_tasks(rows, initial=initial)
        # Public support settings remain exactly those actually read. Only the
        # simulator's transition target is replaced by its private resonance.
        if initial:
            self.table.targets = self.table.targets.double()
        if self.exact_native is not None:
            selected = self.plan.native[rows, self.lifetime[rows]]
            native = rows[selected]
            if len(native):
                self.table.targets[native] = self.exact_native[self.table.indices[native]].gather(
                    1, self.table.permutation[native])


@torch.no_grad()
def execute_request(table, action, predictions):
    """One measured acquisition or one actual scalar setting; no hidden probe.

    The environment alone reads the resonance. A successful setting becomes a
    public support at the frequency executed, not at the hidden target value.
    """
    rows = torch.arange(table.batch, device=table.device)
    reward = torch.zeros(table.batch, device=table.device)
    success = torch.zeros(table.batch, dtype=torch.bool, device=table.device)
    frequency = predictions["means"][rows, (action - 20).clamp_min(0)].double().clamp(0., 10000.)
    table.steps += 1
    acquire = action < 20
    if acquire.any():
        selected = rows[acquire]
        obj, group = action[acquire] // 5, action[acquire] % 5
        index = table.group_counts[selected, obj, group]
        returned = table.samples[selected, obj, group, index]
        present = table.sample_present[selected, obj, group, index]
        previous, old = table.counts[selected, obj], table.values[selected, obj]
        new = (old * previous + returned * present) / (previous + present).clamp_min(1)
        new[:, 2] = torch.where(present[:, 2], torch.where(previous[:, 2] > 0,
            torch.minimum(old[:, 2], returned[:, 2]), returned[:, 2]), old[:, 2])
        revised = present & ((previous == 0) | (new != old))
        relevant = table.known[selected, obj] | (obj == table.query[selected])
        table.failed[selected] &= ~(revised & relevant[:, None])
        table.values[selected, obj] = new
        table.counts[selected, obj] += present.long()
        table.group_counts[selected, obj, group] += 1
        table.event[selected] = measurement_event(obj, group, returned, present,
            table.query[selected], table.steps[selected], table.max_steps, table.mixture)
        reward[selected] -= table.measurement_cost
        table.acquisitions[selected] += 1
    test = ~acquire
    if test.any():
        selected, hypothesis = rows[test], action[test] - 20
        target = table.targets[selected, table.query[selected]]
        error = (frequency[selected] - target).abs()
        success[selected] = error < 2.
        reward[selected] += success[selected].float() - table.test_cost
        table.best_error[selected, table.query[selected]] = torch.minimum(
            table.best_error[selected, table.query[selected]], error.to(table.best_error))
        # Frequency feedback, not permanent relation rejection, governs retry.
        # Online learning may produce a different useful setting for this same
        # relation. The proposal layer uses the public interval history.
        table.event[selected] = test_event(table.query[selected], hypothesis, success[selected],
            table.steps[selected], table.max_steps, table.device)
        completed = selected[success[selected]]
        table.known_hz[completed, table.query[completed]] = frequency[completed].to(table.known_hz)
        table.known[completed, table.query[completed]] = True
        table.successes[completed] += 1
        table.tests[selected] += 1
        remaining = table.target_mask & ~table.known
        table.query[completed] = remaining.float().argmax(-1)[completed]
        table.failed[completed] = False
        table.event[completed, 32:36] = 0
        table.event[completed, 32 + table.query[completed]] = 1
    done = ~(table.target_mask & ~table.known).any(-1) | (table.steps >= table.max_steps)
    table.alive &= ~done
    done |= ~table.observation()["action_mask"].any(-1)
    table.alive &= ~done
    return reward, done, success, frequency


class DelayedExperiments:
    """One pending request per stream with reproducible 1--4 tick returns."""

    def __init__(self, environment, *, seed, max_delay=4):
        self.env = environment
        self.batch, self.device = environment.batch, environment.device
        self.seed, self.max_delay, self.tick = seed, max_delay, 0
        self.pending = torch.zeros(self.batch, dtype=torch.bool, device=self.device)
        self.due = torch.zeros(self.batch, dtype=torch.long, device=self.device)
        self.dispatched = torch.zeros_like(self.due)
        self.returned = torch.zeros_like(self.due)
        self.snapshot = {key: getattr(environment.table, key).clone() for key in _TASK_FIELDS}
        self.pending_reward = torch.zeros(self.batch, device=self.device)
        self.pending_done = torch.zeros_like(self.pending)
        self.pending_success = torch.zeros_like(self.pending)
        self.pending_test = torch.zeros_like(self.pending)
        self.pending_frequency = torch.zeros(self.batch, dtype=torch.float64, device=self.device)
        self.pending_latency = torch.zeros_like(self.due)
        self.pending_query = torch.zeros_like(self.due)
        self.failed_settings = torch.zeros(self.batch, 4, environment.table.max_steps,
                                          dtype=torch.float64, device=self.device)
        self.failed_counts = torch.zeros(self.batch, 4, dtype=torch.long, device=self.device)

    def query_exclusions(self):
        rows = torch.arange(self.batch, device=self.device)
        values = self.failed_settings[rows, self.env.table.query]
        count = self.failed_counts[rows, self.env.table.query]
        mask = torch.arange(values.shape[-1], device=self.device)[None] < count[:, None]
        return {"physical_exclusions": torch.stack((values - 2., values + 2.), -1),
                "physical_exclusion_mask": mask}

    @torch.no_grad()
    def reset_history(self, selected):
        self.failed_counts[selected] = 0

    @torch.no_grad()
    def dispatch(self, actions, predictions, selected):
        if (selected & self.pending).any():
            raise ValueError("A stream cannot issue another request before its result returns")
        rows = selected.nonzero().squeeze(1)
        if not len(rows):
            return
        table = ActiveTableEnvironment.__new__(ActiveTableEnvironment)
        table.__dict__ = {**self.env.table.__dict__, "batch": len(rows)}
        for key in _TASK_FIELDS:
            setattr(table, key, getattr(self.env.table, key)[rows].clone())
        belief = {key: value[rows].detach() for key, value in predictions.items()}
        reward, done, success, frequency = execute_request(table, actions[rows], belief)
        for key in _TASK_FIELDS:
            self.snapshot[key][rows] = getattr(table, key)
        # Delay depends on the stream and external-request index, not policy RNG.
        latency = 1 + ((rows * 1103515245 + self.dispatched[rows] * 12345 + self.seed) % self.max_delay)
        self.due[rows] = self.tick + latency
        self.pending_latency[rows] = latency
        self.pending_reward[rows], self.pending_done[rows] = reward, done
        self.pending_success[rows] = success
        self.pending_test[rows] = actions[rows] >= 20
        self.pending_frequency[rows] = frequency
        self.pending_query[rows] = self.env.table.query[rows]
        self.pending[rows] = True
        self.dispatched[rows] += 1

    @torch.no_grad()
    def advance(self):
        self.tick += 1
        arrived = self.pending & (self.due <= self.tick)
        for key in _TASK_FIELDS:
            getattr(self.env.table, key)[arrived] = self.snapshot[key][arrived]
        reward = self.pending_reward * arrived
        self.env.task_rewards += reward
        done = self.pending_done & arrived
        rows = torch.arange(self.batch, device=self.device)
        lengths = self.env.plan.lengths[rows, self.env.lifetime]
        lifetime_done = done & (self.env.task + 1 == lengths)
        self.pending &= ~arrived
        self.returned += arrived.long()
        failed = arrived & self.pending_test & ~self.pending_success
        indices = failed.nonzero().flatten()
        query = self.pending_query[indices]
        self.failed_settings[indices, query, self.failed_counts[indices, query]] = self.pending_frequency[indices]
        self.failed_counts[indices, query] += 1
        targets = torch.stack((self.pending_success.float(),
                               torch.full_like(self.pending_reward, 2.).log()), -1)
        setting = self.pending_frequency
        feedback = {"intervals": torch.stack((setting - 2., setting + 2.), -1)[:, None],
                    "success": self.pending_success[:, None],
                    "mask": (arrived & self.pending_test)[:, None]}
        return {"arrived": arrived, "reward": reward, "task_done": done,
                "lifetime_done": lifetime_done, "targets": targets,
                "physical_feedback": feedback}
