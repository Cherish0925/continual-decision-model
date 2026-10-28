"""Activity selection and deployment-time adaptation around the active core.

Received observations update the GRU once. Internal activities update workspace
or per-stream adapter weights without pretending another observation arrived.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import IntEnum

import torch
from torch import nn
import torch.nn.functional as F

from .active_experiment import ActiveExperimentPolicy, ActiveExperimentConfig, load_active_checkpoint
from .core import ContinualCore, CoreConfig


class Activity(IntEnum):
    ACT = 0
    REFINE = 1
    RETRIEVE = 2
    LEARN = 3
    CONSOLIDATE = 4
    WAIT = 5


@dataclass(frozen=True)
class ContinualExperimentConfig:
    memory_size: int = 128
    adapter_width: int = 32
    max_internal_steps: int = 4
    modality_dim: int = 0
    fast_decay: float = .99
    activity_mode: str = "learned"
    fixed_refinements: int = 1


def select_rows(old, new, mask):
    return {key: torch.where(mask.reshape(-1, *([1] * (value.ndim - 1))), new[key], value)
            for key, value in old.items()}


def detach_state(state):
    return {key: value.detach() for key, value in state.items()}


class ContinualExperimentPolicy(nn.Module):
    def __init__(self, base: ActiveExperimentPolicy, config=ContinualExperimentConfig()):
        super().__init__()
        if base.config.mode != "persistent":
            raise ValueError("The continuing runtime requires a persistent base core")
        if config.memory_size < 1 or config.adapter_width < 1 or config.max_internal_steps < 0:
            raise ValueError("Invalid memory, adapter, or compute dimensions")
        if not 0 <= config.fast_decay <= 1:
            raise ValueError("fast_decay must be in [0, 1]")
        if config.activity_mode not in {"learned", "fixed", "fixed_schedule", "no_adaptation"}:
            raise ValueError("Unknown activity mode")
        if config.activity_mode == "fixed" and not 0 <= config.fixed_refinements <= config.max_internal_steps:
            raise ValueError("Fixed refinements must fit the internal budget")
        if config.activity_mode == "fixed_schedule" and config.max_internal_steps < 4:
            raise ValueError("The complete fixed schedule needs four internal activities")
        self.base, self.continual_config = base, config
        self.retain_adaptation = True  # Deployment intervention; shared weights stay unchanged.
        w, a, r = base.core.config.width, base.config.action_width, config.adapter_width
        self.activity = nn.Sequential(nn.Linear(2 * w + 6, a), nn.GELU(), nn.Linear(a, len(Activity)))
        # Warm-start deployment with the inherited policy; training still samples all available activities.
        nn.init.zeros_(self.activity[-1].weight)
        nn.init.zeros_(self.activity[-1].bias)
        with torch.no_grad():
            self.activity[-1].bias[Activity.ACT] = 2.
        self.adapter_features = nn.Linear(a, r)
        self.predictor = nn.Linear(r, 2)
        self.memory_query = nn.Linear(2 * w, r)
        self.memory_token = nn.Linear(r + 2, w)
        self.prediction_to_action = nn.Linear(2, 1, bias=False)
        nn.init.zeros_(self.prediction_to_action.weight)
        self.activity_value = nn.Sequential(nn.Linear(2 * w + 6, a), nn.GELU(), nn.Linear(a, 1))
        self.fast_step = nn.Parameter(torch.tensor(-2.))
        self.slow_step = nn.Parameter(torch.tensor(-4.))
        self.modality_projection = nn.Linear(config.modality_dim, w) if config.modality_dim else None

    @property
    def core(self):
        return self.base.core

    @property
    def config(self):
        return self.base.config

    def initial(self, batch, device):
        c = self.continual_config
        state = self.base.initial(batch, device)
        state.update(memory_x=torch.zeros(batch, c.memory_size, c.adapter_width, device=device),
                     memory_y=torch.zeros(batch, c.memory_size, 2, device=device),
                     memory_valid=torch.zeros(batch, c.memory_size, dtype=torch.bool, device=device),
                     memory_cursor=torch.zeros(batch, dtype=torch.long, device=device),
                     fast=torch.zeros(batch, c.adapter_width, 2, device=device),
                     slow=torch.zeros(batch, c.adapter_width, 2, device=device),
                     pending_x=torch.zeros(batch, c.adapter_width, device=device),
                     pending=torch.zeros(batch, dtype=torch.bool, device=device),
                     internal_steps=torch.zeros(batch, dtype=torch.long, device=device),
                     observation_count=torch.zeros(batch, dtype=torch.long, device=device),
                     activity_counts=torch.zeros(batch, len(Activity), dtype=torch.long, device=device))
        return state

    def reset_rows(self, state, reset):
        return select_rows(state, self.initial(len(reset), reset.device), reset)

    def _context(self, observation):
        tokens = self.base.evidence(observation["evidence"])
        mask = observation["evidence_mask"]
        if "modality_tokens" in observation:
            if self.modality_projection is None:
                raise ValueError("Configure modality_dim before supplying encoded image/text/audio tokens")
            tokens = torch.cat((tokens, self.modality_projection(observation["modality_tokens"])), 1)
            mask = torch.cat((mask, observation["modality_mask"]), 1)
        return tokens, mask

    def _refine(self, state, tokens, mask, selected):
        empty = state["h"].new_empty(len(selected), 0, self.core.config.record_dim)
        z = self.core.refine(state["z"], state["h"], empty, selected.long(), tokens, mask)
        return {**state, "z": z}

    def receive(self, state, observation, selected=None):
        """Consume newly received input, not a clock tick or internal computation."""
        if selected is not None:
            rows = selected.nonzero().squeeze(1)
            if len(rows) == 0:
                return state
            if len(rows) != len(selected):
                revised = self.receive({k: v[rows] for k, v in state.items()},
                                       {k: v[rows] for k, v in observation.items()})
                return {k: v.index_copy(0, rows, revised[k]) for k, v in state.items()}
        selected = (torch.ones(len(state["h"]), dtype=torch.bool, device=state["h"].device)
                    if selected is None else selected)
        h = self.core.fast_embeddings(state["h"], self.base.event(observation["event"]).unsqueeze(0))[0]
        updated = {**state, "h": h, "fast": state["fast"] * self.continual_config.fast_decay,
                   "internal_steps": torch.zeros_like(state["internal_steps"]),
                   "observation_count": state["observation_count"] + 1}
        tokens, mask = self._context(observation)
        updated = self._refine(updated, tokens, mask, selected)
        return select_rows(state, updated, selected)

    def feedback(self, state, targets, selected=None):
        """Record returned [test completion, log(1 + execution cost)] for a pending action.

        No parameter update happens until LEARN or CONSOLIDATE is selected.
        """
        selected = state["pending"] if selected is None else selected & state["pending"]
        slot = F.one_hot(state["memory_cursor"] % self.continual_config.memory_size,
                         self.continual_config.memory_size).bool() & selected[:, None]
        return {**state,
                "memory_x": torch.where(slot[..., None], state["pending_x"][:, None], state["memory_x"]),
                "memory_y": torch.where(slot[..., None], targets[:, None], state["memory_y"]),
                "memory_valid": state["memory_valid"] | slot,
                "memory_cursor": state["memory_cursor"] + selected.long(),
                "pending": state["pending"] & ~selected}

    def begin_action(self, state, features, actions, selected=None):
        selected = (torch.ones(len(actions), dtype=torch.bool, device=actions.device)
                    if selected is None else selected)
        if (selected & state["pending"]).any():
            raise ValueError("Return the pending action's feedback before dispatching another action")
        x = features[torch.arange(len(actions), device=actions.device), actions]
        return {**state, "pending_x": torch.where(selected[:, None], x, state["pending_x"]),
                "pending": state["pending"] | selected,
                "internal_steps": torch.where(selected, 0, state["internal_steps"])}

    def replay_error(self, state):
        """Current prediction error on received outcomes, visible to the scheduler.

        Updating fast or slow weights changes this readout immediately, before
        any new external observation or workspace refinement.
        """
        x = state["memory_x"]
        predicted = self.predictor(x) + torch.bmm(x, state["fast"] + state["slow"])
        valid = state["memory_valid"].float()
        return ((predicted - state["memory_y"]).square() * valid[..., None]).sum(1) / valid.sum(-1, keepdim=True).clamp_min(1)

    def readout(self, state, candidates, action_mask):
        f = self.core.state_features(state["h"], state["z"])
        q = self.base.state_query(f)[:, None].expand(-1, candidates.shape[1], -1)
        c = self.base.candidate(candidates)
        joint = self.base.interaction(torch.cat((q, c, q * c), -1))
        x = F.normalize(self.adapter_features(joint).float(), dim=-1)
        predictions = self.predictor(x) + torch.bmm(x, state["fast"] + state["slow"])
        logits = self.base.action(joint).squeeze(-1) + self.prediction_to_action(predictions).squeeze(-1)
        scalars = torch.stack((state["memory_valid"].float().mean(-1), state["pending"].float(),
                              state["internal_steps"].float() / max(1, self.continual_config.max_internal_steps),
                              action_mask.any(-1).float()), -1)
        activity_input = torch.cat((f, scalars, self.replay_error(state)), -1)
        mask = torch.ones(len(f), len(Activity), dtype=torch.bool, device=f.device)
        mask[:, Activity.ACT] = action_mask.any(-1) & ~state["pending"]
        has_memory = state["memory_valid"].any(-1)
        mask[:, Activity.RETRIEVE:Activity.WAIT] = has_memory[:, None]
        mask[:, Activity.WAIT] = ~mask[:, Activity.ACT]
        exhausted = state["internal_steps"] >= self.continual_config.max_internal_steps
        mask[:, Activity.REFINE:Activity.WAIT] &= ~exhausted[:, None]
        if self.continual_config.activity_mode == "no_adaptation":
            mask[:, Activity.LEARN:Activity.CONSOLIDATE + 1] = False
        if self.continual_config.activity_mode == "fixed":
            fixed = torch.where(state["internal_steps"] < self.continual_config.fixed_refinements,
                                int(Activity.REFINE), int(Activity.ACT))
            fixed = torch.where(mask[:, Activity.ACT], fixed, int(Activity.WAIT))
            mask &= F.one_hot(fixed, len(Activity)).bool()
        elif self.continual_config.activity_mode == "fixed_schedule":
            schedule = torch.tensor([Activity.REFINE, Activity.RETRIEVE, Activity.LEARN,
                                     Activity.CONSOLIDATE, Activity.ACT], device=f.device)
            fixed = schedule[state["internal_steps"].clamp_max(4)]
            needs_memory = (fixed >= Activity.RETRIEVE) & (fixed <= Activity.CONSOLIDATE)
            fixed = torch.where(needs_memory & ~has_memory, int(Activity.ACT), fixed)
            available = mask.gather(1, fixed[:, None]).squeeze(1)
            fixed = torch.where(available, fixed, int(Activity.WAIT))
            mask &= F.one_hot(fixed, len(Activity)).bool()
        return {"logits": logits.masked_fill(~action_mask, -torch.inf), "features": x,
                "predictions": predictions, "activity_mask": mask,
                "activity_logits": self.activity(activity_input).masked_fill(~mask, -torch.inf),
                "value": self.base.value(f).squeeze(-1) + self.activity_value(activity_input).squeeze(-1)}

    def internal(self, state, activity, observation):
        """Execute batched internal choices; ACT and WAIT leave the model state intact."""
        original = state
        refine = activity == Activity.REFINE
        if refine.any():
            tokens, mask = self._context(observation)
            state = self._refine(state, tokens, mask, refine)
        retrieve = activity == Activity.RETRIEVE
        if retrieve.any():
            f = self.core.state_features(state["h"], state["z"])
            q = F.normalize(self.memory_query(f).float(), dim=-1)
            scores = torch.einsum("br,bmr->bm", q, state["memory_x"]) * 8
            valid = state["memory_valid"]
            weights = scores.masked_fill(~valid, -1e4).softmax(-1) * valid
            weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-8)
            records = torch.cat((state["memory_x"], state["memory_y"]), -1)
            token = self.memory_token((records * weights[..., None]).sum(1))[:, None]
            state = self._refine(state, token, valid.any(-1, keepdim=True), retrieve)
        for kind, name, step in ((Activity.LEARN, "fast", self.fast_step),
                                 (Activity.CONSOLIDATE, "slow", self.slow_step)):
            selected = activity == kind
            if selected.any():
                x, y = state["memory_x"], state["memory_y"]
                predicted = self.predictor(x) + torch.bmm(x, state["fast"] + state["slow"])
                weights = state["memory_valid"].float()
                residual = (predicted - y) * weights[..., None]
                # Exact gradient of 1/(2N) sum ||prediction - target||^2.
                gradient = torch.bmm(x.transpose(1, 2), residual) / weights.sum(-1)[:, None, None].clamp_min(1)
                revised = state[name] - step.sigmoid() * gradient
                if not self.retain_adaptation:
                    revised = state[name]
                state = {**state, name: torch.where(selected[:, None, None], revised, state[name])}
        internal = (activity > Activity.ACT) & (activity < Activity.WAIT)
        return {**state, "internal_steps": original["internal_steps"] + internal.long(),
                "activity_counts": original["activity_counts"] + F.one_hot(activity, len(Activity))}

    def configuration(self):
        return {**self.base.configuration(), "continual_config": asdict(self.continual_config)}

    def advance(self, event, evidence, evidence_mask, state, refinements=1, feedback_targets=None):
        """v0.9 evaluator compatibility; feedback is the received event, once."""
        if refinements != 1:
            raise ValueError("Use learned internal activities; base assimilation uses one refinement")
        targets = (torch.stack((event[:, 29], (event[:, 30] * 20).clamp_min(0).log1p()), -1)
                   if feedback_targets is None else feedback_targets)
        state = self.feedback(state, targets)
        return self.receive(state, {"event": event, "evidence": evidence, "evidence_mask": evidence_mask})

    def forward(self, event, evidence, evidence_mask, candidates, action_mask, state, refinements=1,
                feedback_targets=None):
        """Greedy synchronous deployment for the existing native stream evaluator."""
        observation = dict(event=event, evidence=evidence, evidence_mask=evidence_mask,
                           candidates=candidates, action_mask=action_mask)
        state = self.advance(event, evidence, evidence_mask, state, refinements, feedback_targets)
        finished = torch.zeros(len(event), dtype=torch.bool, device=event.device)
        for _ in range(self.continual_config.max_internal_steps + 1):
            out = self.readout(state, candidates, action_mask)
            choices = out["activity_logits"].argmax(-1)
            choices = torch.where(finished, int(Activity.WAIT), choices)
            previous = state
            state = self.internal(state, choices, observation)
            state = select_rows(previous, state, ~finished)
            finished |= (choices == Activity.ACT) | (choices == Activity.WAIT)
            if finished.all():
                break
        out = self.readout(state, candidates, action_mask)
        state = self.begin_action(state, out["features"], out["logits"].argmax(-1), action_mask.any(-1))
        return {**out, "state": state}


def load_continual_checkpoint(path, device="cpu", config=None):
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if "continual_config" not in saved:
        base, saved = load_active_checkpoint(path, device)
        return ContinualExperimentPolicy(base, config or ContinualExperimentConfig()).to(device), saved
    core = ContinualCore(CoreConfig(**saved["core_config"]), saved["model"]["base.core.action_values"])
    base = ActiveExperimentPolicy(ActiveExperimentConfig(**saved["active_config"]), core)
    model = ContinualExperimentPolicy(base, ContinualExperimentConfig(**saved["continual_config"]))
    model.load_state_dict(saved["model"], strict=True)
    return model.to(device), saved
