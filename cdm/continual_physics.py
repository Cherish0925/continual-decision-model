"""Continuing action and computation with deployment-time physical learning.

A received setting outcome is an interval event, not an exact resonance label.
The model learns residual corrections to its numerical physical belief from
forecasts saved BEFORE that outcome arrived. Those corrections feed both action
scores and the settings executed by the environment. The analytic belief still
fits the public ledger once; replay here optimizes a prediction model and does
not multiply the same observation into the Bayesian ledger a second time.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn
import torch.nn.functional as F

from .active_experiment import ActiveExperimentConfig, ActiveExperimentPolicy
from .continual_experiment import (Activity, ContinualExperimentConfig,
                                  ContinualExperimentPolicy, load_continual_checkpoint,
                                  select_rows)
from .core import CoreConfig, ContinualCore
from .censored_observation import censored_gaussian_observation, _complement, _log_interval_mass
from .experimental_belief import ExperimentalBelief, load_experimental_belief_checkpoint


@dataclass(frozen=True)
class ContinualPhysicsConfig:
    memory_size: int = 16
    max_feedback_intervals: int = 64
    test_action_start: int = 20
    mean_column: int = 24
    std_column: int = 25
    weight_column: int = 26
    max_mean_shift: float = 16.
    max_log_scale: float = 3.
    gradient_norm: float = 10.
    max_exclusions: int = 128
    test_tolerance_hz: float = 2.


def propose_physical_settings(means, std, exclusions, mask, *, tolerance=2., current=None,
                              minimum_step=None, lower=None, upper=None):
    """Choose a setting with greatest remaining success mass for each relation.

    Failed settings exclude intervals of the unknown target, not intervals of
    allowable control settings. For each surviving target segment, optimize the
    setting's success window, then compare its probability with other segments.
    A setting may be inside an excluded target interval if its window still
    reaches possible targets. Optional control grids use the CURRENT public
    setting as their origin. The default control is continuous, with no assumed
    one-Hz resolution. Gaussian means remain unchanged by this action proposal.
    """
    low, high, present = _complement(exclusions.double(), mask.bool())
    batch = len(means)
    def vector(value, default):
        return (means.new_full((batch,), default) if value is None else
                torch.as_tensor(value, dtype=means.dtype, device=means.device).expand(batch))
    tolerance = vector(tolerance, 2.)
    current, step = vector(current, 0.), vector(minimum_step, 0.)
    lower, upper = vector(lower, -torch.inf), vector(upper, torch.inf)
    # For a short segment a range of centers covers its entire mass; for a
    # long segment the best full-width window is closest to the Gaussian mode.
    left, right = low + tolerance[:, None], high - tolerance[:, None]
    center_low, center_high = torch.minimum(left, right), torch.maximum(left, right)
    centers = means[..., None].clamp(min=center_low[:, None], max=center_high[:, None])
    centers = centers.clamp(min=lower[:, None, None], max=upper[:, None, None])
    safe_step = step.clamp_min(1e-12)
    grid = (centers - current[:, None, None]) / safe_step[:, None, None]
    floor = current[:, None, None] + grid.floor() * safe_step[:, None, None]
    ceil = current[:, None, None] + grid.ceil() * safe_step[:, None, None]
    candidates = torch.stack((floor, ceil), -1)
    candidates = torch.where((step > 0)[:, None, None, None], candidates, centers[..., None])
    valid = present[:, None, :, None] & (candidates >= lower[:, None, None, None]) & (candidates <= upper[:, None, None, None])
    lo = torch.maximum(candidates - tolerance[:, None, None, None], low[:, None, :, None])
    hi = torch.minimum(candidates + tolerance[:, None, None, None], high[:, None, :, None])
    valid = valid & (lo < hi)
    # Invalid padded candidates use a harmless interval before CDF arithmetic.
    a = (torch.where(valid, lo, means[..., None, None] - std[..., None, None]) - means[..., None, None]) / std[..., None, None]
    b = (torch.where(valid, hi, means[..., None, None] + std[..., None, None]) - means[..., None, None]) / std[..., None, None]
    mass = _log_interval_mass(a, b).masked_fill(~valid, -torch.inf)
    best = mass.flatten(2).argmax(-1)
    setting = candidates.flatten(2).gather(-1, best[..., None]).squeeze(-1)
    log_mass = mass.flatten(2).gather(-1, best[..., None]).squeeze(-1)
    return {"settings": setting, "log_success_mass": log_mass, "valid": torch.isfinite(log_mass)}


class ContinualPhysicsPolicy(ContinualExperimentPolicy):
    """One continuing state owns neural, physical and episodic learning state."""

    def __init__(self, base: ActiveExperimentPolicy, belief: ExperimentalBelief,
                 config=ContinualExperimentConfig(), physics_config=ContinualPhysicsConfig()):
        super().__init__(base, config)
        if physics_config.memory_size < 1 or physics_config.max_feedback_intervals < 1:
            raise ValueError("Physical replay and feedback sizes must be positive")
        self.belief, self.physics_config = belief, physics_config
        self.retain_physical_adaptation = True
        r, w = config.adapter_width, self.core.config.width
        self.physical_predictor = nn.Linear(r, 2)
        nn.init.zeros_(self.physical_predictor.weight)
        nn.init.zeros_(self.physical_predictor.bias)
        self.physical_fast_step = nn.Parameter(torch.tensor(-2.))
        self.physical_slow_step = nn.Parameter(torch.tensor(-4.))
        self.physical_memory_token = nn.Linear(r + 3, w)
        self.physical_activity = nn.Linear(4, len(Activity), bias=False)
        self.physical_value = nn.Linear(4, 1, bias=False)
        nn.init.zeros_(self.physical_activity.weight)
        nn.init.zeros_(self.physical_value.weight)

    def initial(self, batch, device):
        state = super().initial(batch, device)
        p, r, q = self.physics_config, self.continual_config.adapter_width, self.belief.features
        m, l = p.memory_size, p.max_feedback_intervals + 2
        def zeros(*shape, dtype=torch.float32):
            return torch.zeros(batch, *shape, dtype=dtype, device=device)
        state.update(physical_means=zeros(q, dtype=torch.float64),
                     physical_std=torch.ones(batch, q, dtype=torch.float64, device=device),
                     physical_weights=torch.full((batch, q), 1 / q, dtype=torch.float64, device=device),
                     physical_eligible=zeros(q, dtype=torch.bool),
                     physics_fast=zeros(r, 2), physics_slow=zeros(r, 2),
                     physics_memory_x=zeros(m, q, r),
                     physics_memory_means=zeros(m, q, dtype=torch.float64),
                     physics_memory_std=torch.ones(batch, m, q, dtype=torch.float64, device=device),
                     physics_memory_weights=torch.full((batch, m, q), 1 / q, dtype=torch.float64, device=device),
                     physics_memory_eligible=zeros(m, q, dtype=torch.bool),
                     physics_memory_intervals=zeros(m, l, 2, dtype=torch.float64),
                     physics_memory_mask=zeros(m, l, dtype=torch.bool),
                     physics_memory_valid=zeros(m, dtype=torch.bool),
                     physics_memory_cursor=zeros(dtype=torch.long),
                     physics_pending_x=zeros(q, r),
                     physics_pending_means=zeros(q, dtype=torch.float64),
                     physics_pending_std=torch.ones(batch, q, dtype=torch.float64, device=device),
                     physics_pending_weights=torch.full((batch, q), 1 / q, dtype=torch.float64, device=device),
                     physics_pending_eligible=zeros(q, dtype=torch.bool),
                     physics_pending=zeros(dtype=torch.bool),
                     physical_exclusions=zeros(p.max_exclusions, 2, dtype=torch.float64),
                     physical_exclusion_mask=zeros(p.max_exclusions, dtype=torch.bool),
                     physical_current_hz=zeros(dtype=torch.float64),
                     physical_minimum_step_hz=zeros(dtype=torch.float64),
                     physical_lower_hz=torch.full((batch,), -torch.inf, dtype=torch.float64, device=device),
                     physical_upper_hz=torch.full((batch,), torch.inf, dtype=torch.float64, device=device))
        state.update({"belief_" + k: v for k, v in self.belief.initial(batch, device).items()})
        return state

    def belief_memory(self, state):
        return {k.removeprefix("belief_"): v for k, v in state.items() if k.startswith("belief_")}

    def remember_belief(self, state, posterior, selected=None, retain=True):
        revised = self.belief.remember(self.belief_memory(state), posterior, selected)
        return {**state, **{"belief_" + k: v for k, v in revised.items()}} if retain else state

    def receive(self, state, observation, selected=None):
        if selected is not None:
            rows = selected.nonzero().flatten()
            if not len(rows):
                return state
            if len(rows) != len(selected):
                revised = self.receive({k: v[rows] for k, v in state.items()},
                                       {k: v[rows] for k, v in observation.items()})
                return {k: v.index_copy(0, rows, revised[k]) for k, v in state.items()}
        revised = super().receive(state, observation)
        for key in ("physical_means", "physical_std", "physical_weights", "physical_eligible"):
            if key in observation:
                revised[key] = observation[key].to(state[key])
        # These are the complete received exclusions for the CURRENT target,
        # replacing the previous query's history rather than accumulating twice.
        if "physical_exclusions" in observation:
            intervals = observation["physical_exclusions"].double()
            mask = observation["physical_exclusion_mask"].bool()
            lo, hi, _ = _complement(intervals, mask)
            union = torch.stack((hi[:, :-1], lo[:, 1:]), -1)
            valid = union[..., 0] < union[..., 1]
            count = valid.sum(-1).max().item()
            if count > self.physics_config.max_exclusions:
                raise ValueError("Increase max_exclusions to retain the current public failure union")
            union, valid = union[:, :count], valid[:, :count]
            padding = self.physics_config.max_exclusions - count
            revised["physical_exclusions"] = F.pad(union, (0, 0, 0, padding)).to(state["physical_exclusions"])
            revised["physical_exclusion_mask"] = F.pad(valid, (0, padding))
        elif "task_boundary" in observation:
            revised["physical_exclusion_mask"] = state["physical_exclusion_mask"] & ~observation["task_boundary"].bool()[:, None]
        for key in ("physical_current_hz", "physical_minimum_step_hz", "physical_lower_hz", "physical_upper_hz"):
            if key in observation:
                revised[key] = observation[key].to(state[key])
        revised["physics_fast"] = state["physics_fast"] * self.continual_config.fast_decay
        return revised

    def _correction(self, x, means, std, eligible, fast, slow):
        raw = self.physical_predictor(x) + torch.einsum("b...qr,brd->b...qd", x, fast + slow)
        p = self.physics_config
        mean_raw, scale_raw = raw[..., 0].double(), raw[..., 1].double()
        shift = p.max_mean_shift * (mean_raw / p.max_mean_shift).tanh()
        log_scale = p.max_log_scale * (scale_raw / p.max_log_scale).tanh()
        mean = means + std * shift * eligible
        scale = std * (log_scale * eligible).exp()
        return mean, scale, raw

    def _activity_features(self, state):
        return torch.stack((state["physics_memory_valid"].float().mean(-1),
                            state["physics_fast"].flatten(1).norm(dim=-1) / state["physics_fast"][0].numel() ** .5,
                            state["physics_slow"].flatten(1).norm(dim=-1) / state["physics_slow"][0].numel() ** .5,
                            state["physical_eligible"].float().mean(-1)), -1)

    def readout(self, state, candidates, action_mask):
        original = super().readout(state, candidates, action_mask)
        p, q = self.physics_config, self.belief.features
        sl = slice(p.test_action_start, p.test_action_start + q)
        x = original["features"][:, sl]
        if x.shape[1] != q:
            raise ValueError("Each physical relation needs its corresponding candidate action")
        means, std, _ = self._correction(x, state["physical_means"], state["physical_std"],
                                         state["physical_eligible"], state["physics_fast"], state["physics_slow"])
        proposal = propose_physical_settings(means, std, state["physical_exclusions"],
            state["physical_exclusion_mask"], tolerance=p.test_tolerance_hz,
            current=state["physical_current_hz"], minimum_step=state["physical_minimum_step_hz"],
            lower=state["physical_lower_hz"], upper=state["physical_upper_hz"])
        settings = proposal["settings"]
        corrected = candidates.clone()
        # Preserve the original encoding for ineligible actions. At initialization
        # these replacements equal the existing physical candidate representation.
        eligible = state["physical_eligible"]
        corrected[:, sl, p.mean_column] = torch.where(eligible,
            ((settings - self.belief.y_mean) / self.belief.y_scale).to(candidates), candidates[:, sl, p.mean_column])
        corrected[:, sl, p.std_column] = torch.where(eligible,
            (std / self.belief.y_scale).to(candidates), candidates[:, sl, p.std_column])
        corrected[:, sl, p.weight_column] = torch.where(eligible,
            state["physical_weights"].to(candidates), candidates[:, sl, p.weight_column])
        execution_mask = action_mask.clone()
        execution_mask[:, sl] &= proposal["valid"]
        out = super().readout(state, corrected, execution_mask)
        activity_features = self._activity_features(state)
        out["activity_logits"] = out["activity_logits"] + self.physical_activity(activity_features)
        out["value"] = out["value"] + self.physical_value(activity_features).squeeze(-1)
        return {**out, "physical_means": means, "physical_std": std,
                "physical_weights": state["physical_weights"], "physical_eligible": eligible,
                "physical_settings": settings, "physical_proposal_valid": proposal["valid"],
                "physical_features": x, "physical_base_means": state["physical_means"],
                "physical_base_std": state["physical_std"], "physical_candidates": corrected}

    def begin_action(self, state, features, actions, selected=None, physical=None):
        selected = torch.ones_like(actions, dtype=torch.bool) if selected is None else selected
        revised = super().begin_action(state, features, actions, selected)
        p, q = self.physics_config, self.belief.features
        active = selected & (actions >= p.test_action_start) & (actions < p.test_action_start + q)
        if physical is None:
            if active.any():
                raise ValueError("Pass the readout as physical= when dispatching a physical test")
            return revised
        mapping = {"x": "physical_features", "means": "physical_base_means", "std": "physical_base_std",
                   "weights": "physical_weights", "eligible": "physical_eligible"}
        for suffix, key in mapping.items():
            old = state["physics_pending_" + suffix]
            revised["physics_pending_" + suffix] = torch.where(
                active.reshape(-1, *([1] * (old.ndim - 1))), physical[key].to(old), old)
        revised["physics_pending"] = state["physics_pending"] | active
        return revised

    def _event_exclusions(self, feedback, state):
        intervals = feedback["intervals"].to(state["physical_means"])
        mask, success = feedback["mask"].bool(), feedback["success"].bool()
        limit = self.physics_config.max_feedback_intervals
        if intervals.shape[1] > limit:
            raise ValueError("Increase max_feedback_intervals to retain all returned setting outcomes")
        if bool(((intervals[..., 0] >= intervals[..., 1]) & mask).any()):
            raise ValueError("Setting outcome intervals must have positive width")
        padding = limit - intervals.shape[1]
        intervals = F.pad(intervals, (0, 0, 0, padding))
        mask, success = F.pad(mask, (0, padding)), F.pad(success, (0, padding))
        valid_success = mask & success
        lower = intervals[..., 0].masked_fill(~valid_success, -torch.inf).max(-1).values
        upper = intervals[..., 1].masked_fill(~valid_success, torch.inf).min(-1).values
        if bool((lower >= upper).any()):
            raise ValueError("Public successful intervals have an empty intersection")
        failure = mask & ~success
        outside = torch.stack((torch.stack((torch.full_like(lower, -torch.inf), lower), -1),
                               torch.stack((upper, torch.full_like(upper, torch.inf)), -1)), 1)
        has_success = valid_success.any(-1)
        return (torch.cat((intervals, outside), 1),
                torch.cat((failure, has_success[:, None].expand(-1, 2)), 1), mask.any(-1))

    def feedback(self, state, targets, selected=None, physical_feedback=None):
        selected = state["pending"] if selected is None else selected & state["pending"]
        revised = super().feedback(state, targets, selected)
        if physical_feedback is not None:
            intervals, mask, present = self._event_exclusions(physical_feedback, state)
            active = selected & state["physics_pending"] & present
            slot = F.one_hot(state["physics_memory_cursor"] % self.physics_config.memory_size,
                             self.physics_config.memory_size).bool() & active[:, None]
            for key in ("x", "means", "std", "weights", "eligible"):
                old = state["physics_memory_" + key]
                revised["physics_memory_" + key] = torch.where(
                    slot.reshape(*slot.shape, *([1] * (old.ndim - 2))),
                    state["physics_pending_" + key][:, None], old)
            revised["physics_memory_intervals"] = torch.where(slot[..., None, None], intervals[:, None],
                                                              state["physics_memory_intervals"])
            revised["physics_memory_mask"] = torch.where(slot[..., None], mask[:, None], state["physics_memory_mask"])
            revised["physics_memory_valid"] = state["physics_memory_valid"] | slot
            revised["physics_memory_cursor"] = state["physics_memory_cursor"] + active.long()
        revised["physics_pending"] = state["physics_pending"] & ~selected
        return revised

    def physical_replay(self, state):
        """Prequential event likelihood and exact adapter gradient, per stream.

        For E = successful interval minus the union of failed intervals,
        d(-log P(E))/d mu = responsibility * (mu - E[Y|E]) / sigma**2.
        The scale derivative is responsibility * (1 - E[(Y-mu)**2|E]/sigma**2).
        These operations also support outer gradients through online updates.
        """
        b, r = len(state["h"]), self.continual_config.adapter_width
        indices = state["physics_memory_valid"].nonzero()
        gradient = state["physics_fast"].new_zeros(b, r, 2)
        loss = state["physical_means"].new_zeros(b)
        if not len(indices):
            return loss, gradient
        rows, slots = indices.unbind(-1)
        x = state["physics_memory_x"][rows, slots]
        base_mean, base_std = (state["physics_memory_" + key][rows, slots] for key in ("means", "std"))
        eligible = state["physics_memory_eligible"][rows, slots]
        mean, std, raw = self._correction(x, base_mean, base_std, eligible,
                                         state["physics_fast"][rows], state["physics_slow"][rows])
        event = censored_gaussian_observation(mean, std, state["physics_memory_intervals"][rows, slots],
                                             state["physics_memory_mask"][rows, slots])
        log_joint = state["physics_memory_weights"][rows, slots].clamp_min(1e-300).log() + event["log_survival"]
        log_probability = log_joint.logsumexp(-1)
        responsibility = log_joint.softmax(-1)
        difference = mean - event["mean"]
        mean_gradient = responsibility * difference / std.square()
        scale_gradient = responsibility * (1 - (event["variance"] + difference.square()) / std.square())
        p = self.physics_config
        chain_mean = base_std * (1 - (raw[..., 0].double() / p.max_mean_shift).tanh().square())
        chain_scale = 1 - (raw[..., 1].double() / p.max_log_scale).tanh().square()
        grad_raw = torch.stack((mean_gradient * chain_mean, scale_gradient * chain_scale), -1) * eligible[..., None]
        per_record = torch.einsum("nqr,nqd->nrd", x, grad_raw.to(x))
        counts = torch.bincount(rows, minlength=b).clamp_min(1)
        gradient = gradient.index_add(0, rows, per_record) / counts[:, None, None]
        loss = loss.index_add(0, rows, -log_probability) / counts
        return loss, gradient

    def internal(self, state, activity, observation):
        revised = super().internal(state, activity, observation)
        retrieve = (activity == Activity.RETRIEVE) & state["physics_memory_valid"].any(-1)
        if retrieve.any():
            weights = state["physics_memory_weights"].to(state["physics_memory_x"])
            records = (state["physics_memory_x"] * weights[..., None]).sum(-2)
            query = F.normalize(self.memory_query(self.core.state_features(state["h"], state["z"])), dim=-1)
            scores = torch.einsum("br,bmr->bm", query, records) * 8
            valid = state["physics_memory_valid"]
            attention = scores.masked_fill(~valid, -1e4).softmax(-1) * valid
            attention = attention / attention.sum(-1, keepdim=True).clamp_min(1e-8)
            intervals, mask = state["physics_memory_intervals"], state["physics_memory_mask"]
            has_success = mask[..., -1]
            # The last two excluded half-lines encode the successful interval.
            lo = torch.where(has_success, intervals[..., -2, 1], 0.)
            hi = torch.where(has_success, intervals[..., -1, 0], 0.)
            failures = mask[..., :-2]
            count = failures.sum(-1).clamp_min(1)
            failed_mid = (intervals[..., :-2, :].mean(-1) * failures).sum(-1) / count
            center = torch.where(has_success, (lo + hi) / 2, failed_mid)
            width = torch.where(has_success, hi - lo, torch.zeros_like(lo))
            summary = torch.stack(((center - self.belief.y_mean) / self.belief.y_scale,
                                   width / self.belief.y_scale, has_success.double()), -1).to(records)
            token = self.physical_memory_token((torch.cat((records, summary), -1) * attention[..., None]).sum(1))[:, None]
            revised = self._refine(revised, token, valid.any(-1, keepdim=True), retrieve)
        for kind, name, step in ((Activity.LEARN, "physics_fast", self.physical_fast_step),
                                 (Activity.CONSOLIDATE, "physics_slow", self.physical_slow_step)):
            selected = (activity == kind) & state["physics_memory_valid"].any(-1)
            if selected.any():
                rows = selected.nonzero().flatten()
                _, gradient = self.physical_replay({k: v[rows] for k, v in revised.items()})
                norm = gradient.flatten(1).norm(dim=-1).clamp_min(1e-8)
                gradient = gradient * (self.physics_config.gradient_norm / norm).clamp_max(1)[:, None, None]
                updated = revised[name][rows] - step.sigmoid() * gradient
                if self.retain_adaptation and self.retain_physical_adaptation:
                    revised[name] = revised[name].index_copy(0, rows, updated)
        return revised

    def configuration(self):
        return {**super().configuration(), "kind": "continual_physics_v1",
                "physics_config": asdict(self.physics_config), "belief_config": self.belief.configuration()}

    def forward(self, event, evidence, evidence_mask, candidates, action_mask, state,
                refinements=1, feedback_targets=None, activity_generator=None,
                physical_feedback=None, **physical_observation):
        if refinements != 1:
            raise ValueError("Use internal activities for additional computation")
        if feedback_targets is not None:
            state = self.feedback(state, feedback_targets, physical_feedback=physical_feedback)
        observation = dict(event=event, evidence=evidence, evidence_mask=evidence_mask,
                           candidates=candidates, action_mask=action_mask, **physical_observation)
        state = self.receive(state, observation)
        finished = torch.zeros(len(event), dtype=torch.bool, device=event.device)
        act = finished.clone()
        for _ in range(self.continual_config.max_internal_steps + 1):
            out = self.readout(state, candidates, action_mask)
            choices = (out["activity_logits"].argmax(-1) if activity_generator is None else
                       torch.multinomial(out["activity_logits"].softmax(-1), 1, generator=activity_generator).squeeze(-1))
            choices = torch.where(finished, int(Activity.WAIT), choices)
            previous = state
            state = select_rows(previous, self.internal(state, choices, observation), ~finished)
            act |= choices == Activity.ACT
            finished |= (choices == Activity.ACT) | (choices == Activity.WAIT)
            if finished.all():
                break
        out = self.readout(state, candidates, action_mask)
        state = self.begin_action(state, out["features"], out["logits"].argmax(-1), act, physical=out)
        return {**out, "state": state}


def load_continual_physics_checkpoint(path, device="cpu", belief_checkpoint=None,
                                      config=None, physics_config=None):
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if saved.get("kind") != "continual_physics_v1":
        if belief_checkpoint is None:
            raise ValueError("Initializing a physical runtime requires a physical belief checkpoint")
        parent, _ = load_continual_checkpoint(path, device, config=config)
        belief, _ = load_experimental_belief_checkpoint(belief_checkpoint, device)
        model = ContinualPhysicsPolicy(parent.base, belief, parent.continual_config,
                                       physics_config or ContinualPhysicsConfig()).to(device)
        # Keep any already-trained activity/cost learning modules from the parent.
        model.load_state_dict(parent.state_dict(), strict=False)
        return model, saved
    c = saved["belief_config"]
    belief = ExperimentalBelief(c["normalization"], memory_slots=c["memory_slots"], noise_floor_hz=c["noise_floor_hz"])
    core = ContinualCore(CoreConfig(**saved["core_config"]), saved["model"]["base.core.action_values"])
    base = ActiveExperimentPolicy(ActiveExperimentConfig(**saved["active_config"]), core)
    model = ContinualPhysicsPolicy(base, belief, ContinualExperimentConfig(**saved["continual_config"]),
                                   ContinualPhysicsConfig(**saved["physics_config"]))
    model.load_state_dict(saved["model"], strict=True)
    if saved.get("training", {}).get("mode") == "discard_physical":
        model.retain_physical_adaptation = False
    return model.to(device), saved
