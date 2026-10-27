"""Joint continuing state, action readout and observation-conditioned computation.

One external observation advances the GRU once. A learned allocation selects
0, 1, 2 or 4 shared workspace refinements before a native action. Offline
branch training and actual acting use the same state transitions; acting only
executes the selected branch. Numerical execution remains the documented
primitive, with its evidence acquired through actual tool actions.
"""
from __future__ import annotations

from time import perf_counter

import torch
from torch import nn

from cdm.digital import public_candidates
from .belief_actions import BELIEF_DIM, BeliefActionHead, BeliefPolicy
from .features import batch_features, load_features


DEPTHS = (0, 1, 2, 4)


class ComputeAllocator(nn.Module):
    def __init__(self, state_dim, width=128):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(state_dim + 2 * BELIEF_DIM),
                                 nn.Linear(state_dim + 2 * BELIEF_DIM, width), nn.GELU(),
                                 nn.Linear(width, len(DEPTHS)))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        with torch.no_grad():
            self.net[-1].bias[1] = 2.0

    def forward(self, state_features, belief, present):
        count = present.sum(-1, keepdim=True).clamp_min(1)
        mean = (belief * present[..., None]).sum(1) / count
        maximum = belief.masked_fill(~present[..., None], -torch.inf).max(1).values
        return self.net(torch.cat((state_features, mean, maximum), -1))


class JointPolicy(nn.Module):
    def __init__(self, digital, reader):
        super().__init__()
        self.digital, self.reader = digital, reader
        self.allocator = ComputeAllocator(2 * digital.config.width)

    def proposals(self, observations, candidates, belief, state, *, depths=DEPTHS):
        event, state = self.digital.assimilate(observations, state)
        allocation = self.allocator(self.digital.core.state_features(state["h"], state["z"]),
                                    belief, candidates["present"])
        states, logits = [], []
        previous = 0
        for depth in depths:
            state = self.digital.refine_state(event, state, depth - previous)
            base, _ = self.digital.readout(event, candidates, state, predict_outcomes=False)
            features = self.digital.core.state_features(state["h"], state["z"])
            logits.append(self.reader(features, candidates["text"], belief, base))
            states.append(state)
            previous = depth
        return allocation, torch.stack(logits, 1), states

    def act(self, observations, candidates, belief, state, *, refinements=None):
        event, state = self.digital.assimilate(observations, state)
        allocation = self.allocator(self.digital.core.state_features(state["h"], state["z"]),
                                    belief, candidates["present"])
        index = allocation.argmax(-1)
        counts = torch.tensor(DEPTHS, device=index.device)[index] if refinements is None else \
            torch.full_like(index, refinements)
        state = self.digital.refine_state(event, state, counts)
        base, _ = self.digital.readout(event, candidates, state, predict_outcomes=False)
        logits = self.reader(self.digital.core.state_features(state["h"], state["z"]),
                             candidates["text"], belief, base)
        return logits, state, counts, allocation


def from_reader(checkpoint, device):
    from .calibration_mixture import load_mixture_checkpoint
    from .train import load_checkpoint
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    _, mixture = load_mixture_checkpoint(saved["mixture_checkpoint"], "cpu")
    digital, metadata = load_checkpoint(mixture["source_checkpoint"], device)
    digital.requires_grad_(True)
    reader = BeliefActionHead(**saved["config"])
    reader.load_state_dict(saved["model"], strict=True)
    return JointPolicy(digital, reader).to(device), metadata, saved


class AdaptivePolicy(BeliefPolicy):
    policy_type = "joint_cdm_action_and_compute_policy_with_explicit_numeric_primitive"

    def __init__(self, checkpoint, feature_cache, **kwargs):
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        super().__init__(saved["base_checkpoint"], feature_cache, **kwargs)
        self.joint = JointPolicy(self.model, self.actor).to(self.device)
        self.joint.load_state_dict(saved["model"], strict=True)
        self.joint.eval()
        self.joint_training = saved["training"]
        self.checkpoint = {**self.checkpoint, "epoch": saved["epoch"]}

    @property
    def evaluation_metadata(self):
        return {**super().evaluation_metadata, "source_core_training": "joint_sequence_training",
                "compute_choices": list(DEPTHS), "compute_policy": "learned" if self.refinements is None else "fixed",
                "joint_training": self.joint_training,
                "numeric_update_policy": "unchanged_received_support_update"}

    @property
    def refinement_schedule(self):
        return "learned_0_1_2_4" if self.refinements is None else self.refinements

    @torch.no_grad()
    def __call__(self, observation):
        return self.resolve_action(self.propose(observation), observation)

    @torch.no_grad()
    def prime(self, observation):
        return self.propose(observation)

    def propose(self, observation):
        if self.mode != "cdm":
            raise ValueError("The joint policy requires its trained CDM state; use a separately trained GRU comparator")
        start = perf_counter()
        self.calibration.observe_initial(observation)
        if self.measurements is not None:
            observation = {**observation, "measurement_records": self.measurements.summaries()}
            if self.features.config.object_relations:
                observation["measurement_dialog_owner"] = self.measurements.dialog_owner
        observation = self.memory.observe(observation)
        candidates = public_candidates(observation)
        path = self.features.encode(observation, candidates)
        obs, actions = batch_features([load_features(str(path))], self.device)
        evidence = self.beliefs.encode(observation, candidates).to(self.device)[None]
        features_done = perf_counter()
        # Keep the existing state-reset intervention available for comparisons.
        state = self.state
        if not self.persistent or self.restart_workspace:
            state = {**state, "h": state["h"] if self.persistent else torch.zeros_like(state["h"]),
                     "z": self.model.core.slots[None].expand(state["h"].shape[0], -1, -1)}
        logits, self.state, counts, allocation = self.joint.act(obs, actions, evidence, state,
                                                               refinements=self.refinements)
        selected = int(torch.multinomial((logits[0] / self.temperature).softmax(-1), 1)) \
            if self.temperature > 0 else int(logits[0].argmax())
        done = perf_counter()
        self.decisions.append({"candidates": len(candidates), "selected": selected,
                               "feature_seconds": features_done - start, "policy_seconds": done - features_done,
                               "decision_seconds": done - start, "refinements": int(counts[0]),
                               "compute_probabilities": allocation[0].softmax(-1).tolist()})
        return dict(candidates[selected].action)


def create(config):
    return AdaptivePolicy(**config)
