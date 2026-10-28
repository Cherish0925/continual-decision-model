"""Jointly train experimental acquisition and the continuing core from native readings.

The batched environment replays recorded instrument returns. Its activation
reward is a frequency-tolerance proxy; native execution remains a separate
evaluation. Future readings and working settings are used only by the
environment, outcome targets and hindsight warm-start supervision.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch import Tensor
import torch.nn.functional as F

from cdm.active_experiment import ActiveExperimentConfig, ActiveExperimentPolicy, load_active_checkpoint
from .calibration_mixture import load_mixture_checkpoint

GROUP_QUANTITIES = ((0,), (1,), (2,), (3,), (4, 5, 6, 7, 8))
QUANTITY_GROUP = (0, 1, 2, 3, 4, 4, 4, 4, 4)


@torch.no_grad()
def visible_beliefs(values, seen, known_hz, known, query, mixture):
    """Nine affine conditionals, each fitted only to received readings/labels."""
    n = len(values)
    x = torch.where(seen, values.double(), mixture.x_mean).double()
    x = (x - mixture.x_mean) / mixture.x_scale
    y = (torch.where(known, known_hz.double(), mixture.y_mean) - mixture.y_mean) / mixture.y_scale
    means = mixture.prior_mean.expand(9, -1)[None].expand(n, -1, -1).clone()
    cov = torch.diag_embed((2 * mixture.log_coefficient_scale).exp().expand(9, -1))[None].expand(n, -1, -1, -1).clone()
    weights = mixture.mixture_logits.log_softmax(-1)[None].expand(n, -1).clone()
    noise = (mixture.noise_std_hz / mixture.y_scale).square()[None]
    support = seen & known[..., None]
    for obj in range(values.shape[1]):
        phi = torch.stack((torch.ones_like(x[:, obj]), x[:, obj]), -1)
        cp = (cov @ phi[..., None])[..., 0]
        variance = (phi * cp).sum(-1) + noise
        error = y[:, obj, None] - (phi * means).sum(-1)
        valid = support[:, obj]
        weights += torch.where(valid, -.5 * (error.square() / variance + variance.log() + np.log(2 * np.pi)), 0.)
        gain = cp / variance[..., None]
        means += torch.where(valid[..., None], gain * error[..., None], 0.)
        cov -= torch.where(valid[..., None, None], gain[..., None] * cp[..., None, :], 0.)
    rows = torch.arange(n, device=values.device)
    q = torch.stack((torch.ones_like(x[rows, query]), x[rows, query]), -1)
    hz = (q * means).sum(-1) * mixture.y_scale + mixture.y_mean
    variance = (q[..., None, :] @ cov @ q[..., None])[..., 0, 0].clamp_min(0) + noise
    count = support.sum(1)
    eligible = (count >= 2) & seen[rows, query]
    return {"means": hz.float(), "std": (variance.sqrt() * mixture.y_scale).float(),
            "weights": weights.softmax(-1).float(), "support_counts": count, "eligible": eligible}


def _vector(value, n, device, dtype=torch.long):
    return torch.as_tensor(value, device=device, dtype=dtype).expand(n)


def measurement_event(object_index, group, returned_values, returned_present, query,
                      step, max_steps, mixture, cost=1.):
    n, device = len(returned_values), returned_values.device
    rows = torch.arange(n, device=device)
    event = torch.zeros(n, 36, device=device)
    event[rows, _vector(object_index, n, device)] = 1
    event[rows, 4 + _vector(group, n, device)] = 1
    normalized = (returned_values - mixture.x_mean.float()) / mixture.x_scale.float()
    event[:, 9:18] = torch.where(returned_present, normalized, 0.)
    event[:, 18:27] = returned_present.float()
    event[:, 27] = 1
    event[:, 30] = _vector(cost, n, device, torch.float32) / 20
    event[:, 31] = _vector(step, n, device, torch.float32) / max_steps
    event[rows, 32 + _vector(query, n, device)] = 1
    return event


def test_event(query, hypothesis, success, step, max_steps, device, cost=5.):
    query = torch.as_tensor(query, device=device, dtype=torch.long).reshape(-1)
    n, rows = len(query), torch.arange(len(query), device=device)
    event = torch.zeros(n, 36, device=device)
    event[rows, query] = 1
    event[rows, 18 + _vector(hypothesis, n, device)] = 1
    event[:, 28] = 1
    event[:, 29] = _vector(success, n, device, torch.float32)
    event[:, 30] = _vector(cost, n, device, torch.float32) / 20
    event[:, 31] = _vector(step, n, device, torch.float32) / max_steps
    event[rows, 32 + query] = 1
    return event


def changed_hypotheses(previous_values, previous_counts, values, counts, known, query):
    """A failed setting becomes eligible only when its own evidence changes."""
    changed = (counts > 0) & ((previous_counts == 0) | (values != previous_values))
    relevant = known | (torch.arange(values.shape[1], device=values.device)[None] == query[:, None])
    return (changed & relevant[..., None]).any(1)


@torch.no_grad()
def counterfactual_calibration(full_values, full_present, settings, supports, targets, quantity):
    """Fit two observed supports and generate environment-only query settings.

    The returned validity flag concerns physical control range and available
    measurements. No observed native query outcome is used to choose a fit.
    """
    n, device = len(full_values), full_values.device
    rows = torch.arange(n, device=device)
    x = full_values.gather(2, quantity[:, None, None].expand(-1, 4, 1))[..., 0].double()
    present = full_present.gather(2, quantity[:, None, None].expand(-1, 4, 1))[..., 0]
    support_indices = supports.long().argsort(-1, descending=True)[:, :2]
    x0, x1 = (x[rows, support_indices[:, index]] for index in range(2))
    y0, y1 = (settings[rows, support_indices[:, index]].double() for index in range(2))
    delta = x1 - x0
    scale = torch.maximum(torch.maximum(x0.abs(), x1.abs()), torch.ones_like(x0))
    finite = ((torch.isfinite(x) & present) | ~(supports | targets)).all(-1)
    nondegenerate = (supports.sum(-1) == 2) & (delta.abs() > 1e-6 * scale)
    slope = (y1 - y0) / torch.where(nondegenerate, delta, torch.ones_like(delta))
    intercept = y0 - slope * x0
    generated = slope[:, None] * x + intercept[:, None]
    valid_range = ((torch.isfinite(generated) & (generated >= 0) & (generated <= 10000)) | ~targets).all(-1)
    accepted = finite & nondegenerate & valid_range & targets.any(-1)
    labels = torch.where(targets, generated.to(settings.dtype), settings)
    return labels, accepted, {"missing": ~finite, "degenerate": finite & ~nondegenerate,
                              "range": finite & nondegenerate & ~valid_range,
                              "slope": slope, "intercept": intercept}


@torch.no_grad()
def make_observation(values, counts, known_hz, known, query, failed, event, mixture,
                     step, max_steps, acquire_mask=None):
    n, device = len(values), values.device
    seen = counts > 0
    beliefs = visible_beliefs(values, seen, known_hz, known, query, mixture)
    labels = torch.where(known, (known_hz - mixture.y_mean.float()) / mixture.y_scale.float(), 0.)
    normalized = torch.where(seen, (values - mixture.x_mean.float()) / mixture.x_scale.float(), 0.)
    objects = torch.eye(4, device=device)[None, :, None].expand(n, -1, 9, -1)
    quantities = torch.eye(9, device=device)[None, None].expand(n, 4, -1, -1)
    query_flag = torch.arange(4, device=device)[None] == query[:, None]
    evidence = torch.cat((objects, quantities, normalized[..., None], counts.float().log1p()[..., None] / 4,
                          seen[..., None].float(), known[:, :, None, None].expand(-1, -1, 9, -1).float(),
                          labels[:, :, None, None].expand(-1, -1, 9, -1),
                          query_flag[:, :, None, None].expand(-1, -1, 9, -1).float(),
                          known.float().mean(-1)[:, None, None, None].expand(-1, 4, 9, -1)), -1).reshape(n, 36, 20)
    candidates = torch.zeros(n, 29, 30, device=device)
    group_counts = torch.stack([counts[..., indices[0]] for indices in GROUP_QUANTITIES], -1)
    limits = torch.tensor([1, 1, 32, 1, 1], device=device)
    available = group_counts < limits
    if acquire_mask is not None:
        available &= acquire_mask.reshape(n, 4, 5)
    for obj in range(4):
        for group, axes in enumerate(GROUP_QUANTITIES):
            k = obj * 5 + group
            candidates[:, k, 0] = 1
            candidates[:, k, 2 + obj] = 1
            candidates[:, k, 6 + group] = 1
            candidates[:, k, [11 + q for q in axes]] = 1
            candidates[:, k, 20] = group_counts[:, obj, group] / 32
            candidates[:, k, 21] = available[:, obj, group]
            candidates[:, k, 22] = known[:, obj]
            candidates[:, k, 23] = query_flag[:, obj]
    rows = torch.arange(n, device=device)
    for q, group in enumerate(QUANTITY_GROUP):
        k = 20 + q
        candidates[:, k, 1] = 1
        candidates[rows, k, 2 + query] = 1
        candidates[:, k, 6 + group] = 1
        candidates[:, k, 11 + q] = 1
        candidates[:, k, 20] = counts[rows, query, q] / 32
        candidates[:, k, 21] = beliefs["eligible"][:, q] & ~failed[:, q]
        candidates[:, k, 23] = 1
        eligible = beliefs["eligible"][:, q]
        candidates[:, k, 24] = torch.where(eligible, (beliefs["means"][:, q] - mixture.y_mean.float()) / mixture.y_scale.float(), 0.)
        candidates[:, k, 25] = torch.where(eligible, beliefs["std"][:, q] / mixture.y_scale.float(), 0.)
        candidates[:, k, 26] = torch.where(eligible, beliefs["weights"][:, q], 0.)
        candidates[:, k, 27] = beliefs["support_counts"][:, q] / 4
        candidates[:, k, 28] = failed[:, q]
    candidates[:, :, 29] = _vector(step, n, device, torch.float32)[:, None] / max_steps
    mask = torch.cat((available.reshape(n, 20), beliefs["eligible"] & ~failed), -1)
    return {"event": event.float().clone(), "evidence": evidence.float(),
            "evidence_mask": torch.ones(n, 36, dtype=torch.bool, device=device),
            "candidates": candidates, "action_mask": mask}


class ActiveTableEnvironment:
    """Acquisition over literal native returns, with an explicit activation proxy."""
    def __init__(self, path, mixture, *, batch=64, max_steps=160, tolerance_hz=8., measurement_cost=.005,
                 test_cost=.025, device="cpu", permute=True, test_retry_rule="changed_evidence",
                 counterfactual_fraction=0.):
        source = np.load(path)
        fields = ("sample_values", "sample_present", "sample_count", "working_frequency_hz", "initial_support_mask", "target_mask", "label_present")
        self.data = {k: torch.as_tensor(source[k], device=device) for k in fields}
        self.size = len(source["x"])
        self.mixture, self.batch, self.max_steps = mixture, batch, max_steps
        self.tolerance, self.measurement_cost, self.test_cost = tolerance_hz, measurement_cost, test_cost
        self.device, self.permute = torch.device(device), permute
        self.test_retry_rule = test_retry_rule
        self.counterfactual_fraction = counterfactual_fraction
        self.counterfactual_counts = {key: torch.zeros(9, dtype=torch.long, device=device)
                                      for key in ("proposed", "accepted", "missing", "degenerate", "range")}
        self.sampled_worlds = torch.zeros((), dtype=torch.long, device=device)
        self.reset()

    @torch.no_grad()
    def reset(self, indices=None):
        b, device = self.batch, self.device
        self.indices = torch.randint(self.size, (b,), device=device) if indices is None else indices.to(device)
        self.permutation = torch.rand(b, 4, device=device).argsort(-1) if self.permute else torch.arange(4, device=device)[None].expand(b, -1).clone()
        self._load_data()
        self._counterfactual_targets()
        self.values = torch.zeros(b, 4, 9, device=device)
        self.counts = torch.zeros(b, 4, 9, device=device, dtype=torch.long)
        self.group_counts = torch.zeros(b, 4, 5, device=device, dtype=torch.long)
        self.known = self.initial_support.clone()
        self.known_hz = torch.where(self.known, self.targets, 0.)
        self.query = self.target_mask.float().argmax(-1)
        self.failed = torch.zeros(b, 9, dtype=torch.bool, device=device)
        self.event = torch.zeros(b, 36, device=device)
        self.event[torch.arange(b, device=device), 32 + self.query] = 1
        self.steps = torch.zeros(b, dtype=torch.long, device=device)
        self.alive = torch.ones(b, dtype=torch.bool, device=device)
        self.successes = torch.zeros(b, device=device)
        self.acquisitions = torch.zeros(b, device=device)
        self.tests = torch.zeros(b, device=device)
        self.best_quality = torch.zeros(b, 4, device=device)
        self.best_error = torch.full((b, 4), torch.inf, device=device)
        return self.observation()

    def _load_data(self):
        rows = torch.arange(self.batch, device=self.device)[:, None]
        def select(name):
            return self.data[name][self.indices][rows, self.permutation]
        self.samples = select("sample_values")
        self.sample_present = select("sample_present")
        self.sample_count = select("sample_count")
        self.targets = select("working_frequency_hz")
        self.initial_support = select("initial_support_mask")
        self.target_mask = select("target_mask") & select("label_present")
        sums = torch.where(self.sample_present, self.samples, 0.).sum(3)
        count = self.sample_present.sum(3).clamp_min(1)
        self.full_values = sums / count
        radiation = torch.where(self.sample_present[:, :, 2, :, 2], self.samples[:, :, 2, :, 2], torch.inf).min(-1).values
        self.full_values = self.full_values.sum(2)
        self.full_values[:, :, 2] = radiation
        self.full_present = self.sample_present.any(3).any(2)

    @torch.no_grad()
    def _counterfactual_targets(self):
        self.sampled_worlds.add_(self.batch)
        self.counterfactual_quantity = torch.full((self.batch,), -1, dtype=torch.long, device=self.device)
        if self.counterfactual_fraction == 0:
            return
        selected = torch.rand(self.batch, device=self.device) < self.counterfactual_fraction
        quantity = torch.randint(9, (self.batch,), device=self.device)
        generated, valid, checks = counterfactual_calibration(
            self.full_values, self.full_present, self.targets, self.initial_support, self.target_mask, quantity)
        accepted = selected & valid
        self.targets = torch.where(accepted[:, None], generated, self.targets)
        self.counterfactual_quantity[accepted] = quantity[accepted]
        self.counterfactual_counts["proposed"].add_(torch.bincount(quantity[selected], minlength=9))
        self.counterfactual_counts["accepted"].add_(torch.bincount(quantity[accepted], minlength=9))
        for key in ("missing", "degenerate", "range"):
            self.counterfactual_counts[key].add_(torch.bincount(quantity[selected & checks[key]], minlength=9))

    def counterfactual_summary(self):
        counts = {key: value.tolist() for key, value in self.counterfactual_counts.items()}
        total = int(self.sampled_worlds)
        return {"sampled_worlds": total, "proposal_fraction": self.counterfactual_fraction,
                "accepted_worlds": sum(counts["accepted"]),
                "effective_fraction": sum(counts["accepted"]) / max(total, 1), "by_quantity": counts}

    @torch.no_grad()
    def reset_rows(self, reset):
        if not reset.any():
            return
        fresh = ActiveTableEnvironment.__new__(ActiveTableEnvironment)
        fresh.__dict__ = {**self.__dict__, "batch": int(reset.sum())}
        fresh.reset()
        for name in ("indices", "permutation", "samples", "sample_present", "sample_count", "targets", "initial_support", "target_mask",
                     "full_values", "full_present", "counterfactual_quantity", "values", "counts", "group_counts", "known", "known_hz", "query", "failed", "event",
                     "steps", "alive", "successes", "acquisitions", "tests", "best_quality", "best_error"):
            getattr(self, name)[reset] = getattr(fresh, name)

    @torch.no_grad()
    def observation(self):
        obs = make_observation(self.values, self.counts, self.known_hz, self.known, self.query, self.failed,
                               self.event, self.mixture, self.steps, self.max_steps,
                               self.group_counts < self.sample_count)
        # Terminal rows are ignored by the vectorized rollout until reset.
        obs["action_mask"][~self.alive, 0] = True
        return obs

    @torch.no_grad()
    def oracle_values(self):
        """Hindsight warm-start targets. They never enter policy observations."""
        n, device = self.batch, self.device
        rows = torch.arange(n, device=device)
        full = visible_beliefs(self.full_values, torch.ones_like(self.counts, dtype=torch.bool),
                               self.known_hz, self.known, self.query, self.mixture)
        target = self.targets[rows, self.query, None]
        error = (full["means"] - target).abs()
        quality = torch.exp(-error / self.tolerance)
        terminal = (error <= self.tolerance).float() + .2 * quality - self.test_cost
        needed = self.known.clone()
        needed[rows, self.query] = True
        limits = torch.tensor([1, 1, 32, 1, 1], device=device)
        destination = torch.minimum(self.sample_count, limits)
        missing = (destination - self.group_counts).clamp_min(0) * needed[..., None]
        costs = missing.sum(1)
        future = terminal - self.measurement_cost * costs[:, torch.tensor(QUANTITY_GROUP, device=device)]
        scores = torch.full((n, 29), -.5, device=device)
        for obj in range(4):
            for group, axes in enumerate(GROUP_QUANTITIES):
                value = future[:, list(axes)].max(-1).values
                contributes = needed[:, obj] & (missing[:, obj, group] > 0)
                scores[:, obj * 5 + group] = torch.where(contributes, value, value - .1)
        current = visible_beliefs(self.values, self.counts > 0, self.known_hz, self.known, self.query, self.mixture)
        error = (current["means"] - target).abs()
        scores[:, 20:] = (error <= self.tolerance).float() + .2 * torch.exp(-error / self.tolerance) - self.test_cost
        mask = self.observation()["action_mask"]
        return scores.masked_fill(~mask, -1.)

    @torch.no_grad()
    def step(self, action):
        n, device = self.batch, self.device
        rows = torch.arange(n, device=device)
        reward = torch.zeros(n, device=device)
        self.steps += self.alive.long()
        acquire = self.alive & (action < 20)
        if acquire.any():
            selected = rows[acquire]
            obj, group = action[acquire] // 5, action[acquire] % 5
            index = self.group_counts[selected, obj, group]
            returned = self.samples[selected, obj, group, index]
            present = self.sample_present[selected, obj, group, index]
            previous = self.counts[selected, obj]
            old = self.values[selected, obj]
            new = (old * previous + returned * present) / (previous + present).clamp_min(1)
            new[:, 2] = torch.where(present[:, 2], torch.where(previous[:, 2] > 0,
                                      torch.minimum(old[:, 2], returned[:, 2]), returned[:, 2]), old[:, 2])
            revised = present & ((previous == 0) | (new != old))
            relevant = self.known[selected, obj] | (obj == self.query[selected])
            if self.test_retry_rule == "any_measurement":
                self.failed[selected] = False
            else:
                self.failed[selected] &= ~(revised & relevant[:, None])
            self.values[selected, obj] = new
            self.counts[selected, obj] += present.long()
            self.group_counts[selected, obj, group] += 1
            self.event[selected] = measurement_event(obj, group, returned, present, self.query[selected],
                                                     self.steps[selected], self.max_steps, self.mixture)
            reward[selected] -= self.measurement_cost
            self.acquisitions[selected] += 1
        test = self.alive & (action >= 20)
        if test.any():
            selected, hypothesis = rows[test], action[test] - 20
            belief = visible_beliefs(self.values, self.counts > 0, self.known_hz, self.known, self.query, self.mixture)
            target = self.targets[selected, self.query[selected]]
            error = (belief["means"][selected, hypothesis] - target).abs()
            quality = torch.exp(-error / self.tolerance)
            previous_quality = self.best_quality[selected, self.query[selected]]
            reward[selected] += .2 * (quality - previous_quality).clamp_min(0) - self.test_cost
            self.best_quality[selected, self.query[selected]] = torch.maximum(quality, previous_quality)
            self.best_error[selected, self.query[selected]] = torch.minimum(error, self.best_error[selected, self.query[selected]])
            success = error <= self.tolerance
            reward[selected] += success.float()
            self.failed[selected, hypothesis] = True
            self.event[selected] = test_event(self.query[selected], hypothesis, success, self.steps[selected],
                                              self.max_steps, device)
            completed = selected[success]
            self.known_hz[completed, self.query[completed]] = target[success]
            self.known[completed, self.query[completed]] = True
            self.successes[completed] += 1
            self.tests[selected] += 1
            remaining = self.target_mask & ~self.known
            next_query = remaining.float().argmax(-1)
            self.query[completed] = next_query[completed]
            self.failed[completed] = False
            self.event[completed, 32:36] = 0
            self.event[completed, 32 + self.query[completed]] = 1
        done = ~(self.target_mask & ~self.known).any(-1) | (self.steps >= self.max_steps)
        self.alive &= ~done
        exhausted = ~self.observation()["action_mask"].any(-1)
        done |= exhausted
        self.alive &= ~done
        return reward, done


@torch.no_grad()
def evaluate(model, env, *, refinements=1):
    env.reset(torch.arange(env.size, device=env.device))
    model.eval()
    state = model.initial(env.batch, env.device)
    total = torch.zeros(env.batch, device=env.device)
    for _ in range(env.max_steps):
        out = model(**env.observation(), state=state, refinements=refinements)
        state = out["state"]
        reward, done = env.step(out["logits"].argmax(-1))
        total += reward
        if done.all():
            break
    errors = env.best_error[env.target_mask]
    finite = torch.isfinite(errors)
    return {"worlds": env.batch, "proxy_completed_worlds": int((~(env.target_mask & ~env.known).any(-1)).sum()),
            "proxy_completed_queries": int(env.successes.sum()), "queries": int(env.target_mask.sum()),
            "mean_reward": float(total.mean()), "mean_acquisitions": float(env.acquisitions.mean()),
            "mean_tests": float(env.tests.mean()), "tested_queries": int(finite.sum()),
            "mean_best_error_hz_tested": float(errors[finite].mean()) if finite.any() else None,
            "within_2hz": int((errors <= 2).sum()), "proxy_tolerance_hz": env.tolerance}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--core-checkpoint", type=Path, default=Path("runs/discoveryworld/measurement_memory/action_run/last.pt"))
    parser.add_argument("--mixture-checkpoint", type=Path, default=Path("runs/discoveryworld/calibration/mixture/shared_train_only_seed31/last.pt"))
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--mode", choices=("persistent", "ledger"), default="persistent")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=181)
    parser.add_argument("--updates", type=int, default=1200)
    parser.add_argument("--warmup", type=int, default=400)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--unroll", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=160)
    parser.add_argument("--refinements", type=int, default=1)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--tolerance-hz", type=float, default=8.)
    parser.add_argument("--test-retry-rule", choices=("changed_evidence", "any_measurement"), default="changed_evidence",
                        help="The historical any_measurement rule is retained only for exact phase-one repetitions.")
    parser.add_argument("--counterfactual-fraction", type=float, default=0.,
                        help="Training-world probability of proposing a two-support affine counterfactual task.")
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--head-lr", type=float, default=3e-4)
    parser.add_argument("--bf16", action="store_true")
    args = parser.parse_args()
    if not 0 <= args.counterfactual_fraction <= 1:
        parser.error("--counterfactual-fraction must be between zero and one")
    if args.output.exists():
        parser.error("Choose a fresh output directory")
    args.output.mkdir(parents=True)
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    mixture, _ = load_mixture_checkpoint(args.mixture_checkpoint, args.device)
    mixture.requires_grad_(False)
    env = ActiveTableEnvironment(args.data / "train.npz", mixture, batch=args.batch, max_steps=args.max_steps,
                                 tolerance_hz=args.tolerance_hz, device=args.device, test_retry_rule=args.test_retry_rule,
                                 counterfactual_fraction=args.counterfactual_fraction)
    validation_size = len(np.load(args.data / "validation.npz")["x"])
    validation = ActiveTableEnvironment(args.data / "validation.npz", mixture, batch=validation_size,
                                        max_steps=args.max_steps, tolerance_hz=args.tolerance_hz,
                                        device=args.device, permute=False, test_retry_rule=args.test_retry_rule)
    if args.resume:
        model, saved = load_active_checkpoint(args.resume, args.device)
        if model.config.mode != args.mode:
            raise ValueError("Resume mode differs from the selected architecture")
    else:
        config = ActiveExperimentConfig(36, 20, 30, mode=args.mode)
        model = ActiveExperimentPolicy.from_native_checkpoint(args.core_checkpoint, config).to(args.device)
    parameters = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW([
        {"params": [p for name, p in parameters if name.startswith("core.")], "lr": args.lr},
        {"params": [p for name, p in parameters if not name.startswith("core.")], "lr": args.head_lr}], weight_decay=.01)
    if args.resume and "optimizer" in saved:
        optimizer.load_state_dict(saved["optimizer"])
        optimizer.param_groups[0]["lr"] = args.lr
        optimizer.param_groups[1]["lr"] = args.head_lr
    settings = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    settings.update(train_worlds=env.size, train_queries=int(env.data["target_mask"].sum()), validation_worlds=validation.size,
                    trainable_parameters=sum(p.numel() for _, p in parameters),
                    trainable_core_parameters=sum(p.numel() for name, p in parameters if name.startswith("core.")),
                    scope="native-recorded measurement replay; activation tolerance proxy, not native task completion",
                    radiation_summary="running minimum of individually acquired recorded returns",
                    counterfactual_design="uniform quantity proposal; two original supports; generated query labels only; public0–10000Hz range; rejected fits retain native labels; validation unchanged",
                    checkpoint_selection="final update; intermediate validation is recorded")
    (args.output / "config.json").write_text(json.dumps(settings, indent=2) + "\n")
    state = model.initial(env.batch, env.device)
    history, started = [], time.monotonic()
    for update in range(args.updates):
        model.train()
        state = {key: value.detach() for key, value in state.items()}
        warm = update < args.warmup
        rewards, dones, values, log_probs, entropies, aux_losses, imitation = [], [], [], [], [], [], []
        for _ in range(args.unroll):
            obs = env.observation()
            with torch.autocast(torch.device(args.device).type, dtype=torch.bfloat16, enabled=args.bf16):
                out = model(**obs, state=state, refinements=args.refinements)
            oracle = env.oracle_values()
            oracle_logits = (oracle / .1).masked_fill(~obs["action_mask"], -torch.inf)
            teacher = oracle_logits.softmax(-1)
            distribution = torch.distributions.Categorical(logits=out["logits"].float())
            action = oracle_logits.argmax(-1) if warm else distribution.sample()
            reward, done = env.step(action)
            rewards.append(reward); dones.append(done); values.append(out["value"].float())
            log_probs.append(distribution.log_prob(action)); entropies.append(distribution.entropy())
            pred = out["outcomes"][..., 0].float()
            aux_losses.append(F.smooth_l1_loss(pred[obs["action_mask"]], oracle[obs["action_mask"]]))
            logp = out["logits"].float().log_softmax(-1).masked_fill(~obs["action_mask"], 0.)
            imitation.append(-(teacher * logp).sum(-1).mean())
            state = model.reset_rows(out["state"], done)
            env.reset_rows(done)
        with torch.no_grad(), torch.autocast(torch.device(args.device).type, dtype=torch.bfloat16, enabled=args.bf16):
            bootstrap = model(**env.observation(), state=state, refinements=args.refinements)["value"].float()
        returns, target = [], bootstrap
        for reward, done in zip(reversed(rewards), reversed(dones)):
            target = reward + .99 * target * (~done)
            returns.append(target)
        returns = torch.stack(list(reversed(returns)))
        predicted_values = torch.stack(values)
        advantages = returns - predicted_values
        value_loss = F.smooth_l1_loss(predicted_values, returns)
        policy_loss = -(torch.stack(log_probs) * advantages.detach()).mean()
        imitation_loss = torch.stack(imitation).mean()
        auxiliary_loss = torch.stack(aux_losses).mean()
        entropy = torch.stack(entropies).mean()
        loss = (imitation_loss + .5 * auxiliary_loss + .25 * value_loss if warm else
                policy_loss + .5 * value_loss + .1 * auxiliary_loss + .02 * imitation_loss - .01 * entropy)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_([p for _, p in parameters], 1.)
        optimizer.step()
        if update == 0 or (update + 1) % 10 == 0:
            row = {"update": update + 1, "phase": "hindsight_warm_start" if warm else "actor_critic",
                   "loss": float(loss.detach()), "imitation": float(imitation_loss.detach()),
                   "outcome_loss": float(auxiliary_loss.detach()), "policy_loss": float(policy_loss.detach()),
                   "value_loss": float(value_loss.detach()), "entropy": float(entropy.detach()),
                   "mean_step_reward": float(torch.stack(rewards).mean()), "gradient_norm": float(gradient_norm),
                   "elapsed_seconds": time.monotonic() - started}
            print(json.dumps(row), flush=True)
            history.append(row)
        if (update + 1) % args.eval_every == 0 or update + 1 == args.updates:
            result = evaluate(model, validation, refinements=args.refinements)
            row = {"update": update + 1, "validation": result, "elapsed_seconds": time.monotonic() - started}
            print(json.dumps(row), flush=True); history.append(row)
            settings["counterfactual_counts"] = env.counterfactual_summary()
            (args.output / "config.json").write_text(json.dumps(settings, indent=2) + "\n")
            saved = {**model.configuration(), "model": model.state_dict(), "optimizer": optimizer.state_dict(), "training": settings, "update": update + 1,
                     "validation": result}
            torch.save(saved, args.output / "last.pt")
            if update + 1 == args.warmup or update + 1 == args.updates:
                torch.save(saved, args.output / f"update{update + 1}.pt")
            (args.output / "training.json").write_text(json.dumps({"config": settings, "history": history}, indent=2) + "\n")
    print(json.dumps({"finished": True, "updates": args.updates,
                      "elapsed_seconds": time.monotonic() - started}), flush=True)


if __name__ == "__main__":
    main()
