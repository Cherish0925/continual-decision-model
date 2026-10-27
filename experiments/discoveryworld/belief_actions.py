"""Trainable action reading of a CDM state and received numerical beliefs.

No action is removed or selected by these features. Object handles perform
joins only. All measurements, success flags and relation posteriors precede
the decision being represented. Numerical setting execution remains an
explicit primitive inherited from MixturePolicy.
"""
from __future__ import annotations

from collections import defaultdict
import math
import re

import numpy as np
import torch
from torch import nn

from cdm.digital import _objects
from .calibration import FEATURE_QUANTITIES
from .features import canonical_action
from .mixture_policy import MixturePolicy
from .numeric_reference import quantity_vector
from .quantitative_policy import _DELTA


def magnitude(value):
    return math.copysign(math.log1p(abs(float(value))) / 5, float(value))


class BeliefFeatures:
    """A public evidence view used identically in offline replay and acting."""

    def __init__(self, calibration, mixture):
        self.calibration, self.mixture = calibration, mixture
        self.last_action = None
        self.repeat_count = 0
        self.last_success = 0.0
        self._support_key = None
        self.posterior = None

    def observe(self, transition):
        action = canonical_action(transition["action"])
        self.repeat_count = self.repeat_count + 1 if action == self.last_action else 1
        self.last_action = action
        self.last_success = float(bool(transition["action_result"].get("success")))

    @torch.no_grad()
    def encode(self, observation, candidates):
        log, mixture = self.calibration, self.mixture
        support = log.support()
        key = tuple((row.label.object_handle,
                     tuple(quantity_vector(row.query)), row.label.frequency_hz) for row in support)
        if key != self._support_key:
            self.posterior = mixture.fit_posterior(
                torch.as_tensor(np.asarray([quantity_vector(row.query) for row in support]), dtype=torch.float64,
                                device=mixture.device).reshape(-1, 9),
                torch.tensor([row.label.frequency_hz for row in support], dtype=torch.float64,
                             device=mixture.device))
            self._support_key = key
        weights = self.posterior["log_weights"].exp()
        common = [magnitude(log.sequence + 1), magnitude(len(log.labels)), magnitude(len(support)),
                  float(-(weights * weights.clamp_min(1e-30).log()).sum() / math.log(9)),
                  float(weights.max()), self.last_success, magnitude(self.repeat_count)]
        objects = _objects(observation)
        inventory = {x["uuid"] for x in observation["ui"].get("inventoryObjects", [])}
        current = {x["uuid"]: x for x in observation["ui"].get("accessibleEnvironmentObjects", [])}
        current.update((x["uuid"], x) for x in observation["ui"].get("inventoryObjects", []))
        parents, children = {}, defaultdict(list)
        for handle, obj in current.items():
            match = re.search(r"\(in [^\n]*?\[uuid:\s*(\d+)\]", obj.get("description", ""))
            if match:
                parent = next((x for x in objects if str(x) == match.group(1)), None)
                parents[handle] = parent
                children[parent].append(handle)
        active = lambda handle: "(activated)" in current.get(handle, {}).get("name", "")
        labels = {row.object_handle: row.frequency_hz for row in log.labels}
        summaries = log.measurements.summaries()
        by_object, by_pair = defaultdict(list), defaultdict(list)
        for row in summaries:
            by_object[row.target_handle].append(row)
            by_pair[(row.instrument_handle, row.target_handle)].append(row)
        encoded, predictions = {}, {}
        for handle in {None, *objects}:
            rows = by_object[handle]
            latest = {}
            for row in rows:
                if row.quantity not in latest or row.last_sequence > latest[row.quantity].last_sequence:
                    latest[row.quantity] = row
            base = [float(handle is not None), float(handle in inventory),
                    float(objects.get(handle, {}).get("accessible", False)), float(active(handle)),
                    float(active(parents.get(handle))), magnitude(len(children[handle])),
                    float(any(active(child) for child in children[handle])), float(handle in labels),
                    (labels.get(handle, float(mixture.y_mean)) - float(mixture.y_mean)) / float(mixture.y_scale),
                    magnitude(sum(row.count for row in rows)), magnitude(len(latest)),
                    magnitude(log.sequence - max((r.last_sequence for r in rows), default=log.sequence))]
            for index, quantity in enumerate(FEATURE_QUANTITIES):
                row = latest.get(quantity)
                base.extend([float(row is not None), magnitude(row.count) if row else 0.0,
                             (row.mean - float(mixture.x_mean[index])) / float(mixture.x_scale[index]) if row else 0.0,
                             (row.maximum - row.minimum) / float(mixture.x_scale[index]) if row else 0.0])
            complete = all(q in latest for q in FEATURE_QUANTITIES)
            if complete:
                query = log.query(handle)
                prediction = mixture.predict(torch.as_tensor(quantity_vector(query), dtype=torch.float64,
                                                             device=mixture.device)[None], self.posterior)
                expert = int(prediction["map_expert_index"][0])
                mean = float(prediction["map_expert_mean_hz"][0])
                sd = float(prediction["expert_std_hz"][0, expert])
                spread = float(prediction["mixture_std_hz"][0])
                predictions[handle] = (mean, sd, spread)
                base.extend([1., (mean - float(mixture.y_mean)) / float(mixture.y_scale),
                             magnitude(sd), magnitude(spread)])
            else:
                base.extend([0., 0., 0., 0.])
            encoded[handle] = base
        dialog = log.current_dialog()
        result = []
        for candidate in candidates:
            first, second = candidate.object_arguments
            if dialog is not None and "chosen_dialog_option_int" in candidate.action:
                first, second = dialog["reactor_handle"], dialog["object_handle"]
            pair = by_pair[(first, second)]
            specific = [magnitude(sum(row.count for row in pair)), magnitude(len({row.quantity for row in pair})),
                        float(parents.get(first) == second and second is not None),
                        float(parents.get(second) == first and first is not None),
                        float(canonical_action(candidate.action) == self.last_action)]
            control = [0.] * 9
            if dialog is not None:
                options = observation["ui"]["dialog_box"]["dialogOptions"]
                choice = candidate.action.get("chosen_dialog_option_int")
                text = options.get(str(choice), options.get(choice, ""))
                match = _DELTA.fullmatch(str(text).strip())
                delta = (float(match[2].replace(",", "")) * (1 if match[1].lower() == "increase" else -1)) if match else 0.
                known = predictions.get(dialog["object_handle"])
                frequency = dialog["frequency_hz"]
                control = [1., float(dialog["activated"] is True), float(text.strip().lower() == "exit"),
                           magnitude(delta), (frequency - float(mixture.y_mean)) / float(mixture.y_scale),
                           float(known is not None), magnitude(frequency - known[0]) if known else 0.,
                           magnitude(frequency + delta - known[0]) if known else 0.,
                           float(abs(frequency + delta - known[0]) < abs(frequency - known[0])) if known else 0.]
            result.append(common + encoded.get(first, encoded[None]) + encoded.get(second, encoded[None]) + specific + control)
        return torch.tensor(result, dtype=torch.float32)


BELIEF_DIM = 125  # 7 common + 2 * (12 + 9 * 4 + 4) object + 5 pair + 9 control


class BeliefActionHead(nn.Module):
    """Residual action ranking; one unchanged CDM owns the continuing state."""

    def __init__(self, state_dim, text_dim, *, width=128, belief_mode="full"):
        super().__init__()
        self.state_dim, self.text_dim, self.width, self.belief_mode = state_dim, text_dim, width, belief_mode
        self.state = nn.Linear(state_dim, width)
        self.text = nn.Linear(text_dim, width)
        self.belief = nn.Sequential(nn.Linear(BELIEF_DIM, width), nn.GELU(), nn.Linear(width, width))
        self.score = nn.Sequential(nn.Linear(width * 3, width), nn.GELU(), nn.Linear(width, 1))
        nn.init.zeros_(self.score[-1].weight)
        nn.init.zeros_(self.score[-1].bias)

    def forward(self, state, text, belief, base_logits):
        if self.belief_mode == "none":
            belief = torch.zeros_like(belief)
        elif self.belief_mode == "observations":
            belief = belief.clone()
            belief[..., 3:5] = 0  # Candidate probabilities, not received facts.
            belief[..., 55:59] = 0  # First object's predicted setting/uncertainty.
            belief[..., 107:111] = 0  # Second object's predicted setting/uncertainty.
            belief[..., 121:125] = 0  # Proposed control relative to that prediction.
        context = self.state(state).unsqueeze(-2).expand(*text.shape[:-1], self.width)
        candidate = self.text(text)
        evidence = self.belief(belief)
        residual = self.score(torch.cat((context + candidate, context * candidate, evidence), -1)).squeeze(-1)
        return base_logits + residual


class BeliefPolicy(MixturePolicy):
    policy_type = "cdm_belief_action_policy_with_explicit_numeric_primitive"

    def __init__(self, checkpoint, feature_cache, **kwargs):
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        super().__init__(saved["mixture_checkpoint"], feature_cache, **kwargs)
        self.actor = BeliefActionHead(**saved["config"]).to(self.device).eval()
        self.actor.load_state_dict(saved["model"], strict=True)
        self.beliefs = BeliefFeatures(self.calibration, self.predictor.mixture)
        self.reader_training = {"checkpoint": str(checkpoint),
                                "source_training_epoch": self.checkpoint["epoch"],
                                "imitation_epochs": saved["epoch"],
                                "on_policy_update": saved.get("on_policy_training"),
                                "branch_training": saved.get("branch_training")}
        self.checkpoint = {**self.checkpoint, "train_instances": saved["train_instances"],
                           "validation_instances": saved["validation_instances"], "epoch": saved["epoch"]}

    def reset(self):
        super().reset()
        self.beliefs = BeliefFeatures(self.calibration, self.predictor.mixture)

    def observe(self, transition):
        super().observe(transition)
        self.beliefs.observe(transition)

    @property
    def evaluation_metadata(self):
        return {**super().evaluation_metadata,
                "action_reader": "trained_residual_over_continuing_source_and_public_beliefs",
                "action_reader_beliefs": self.actor.belief_mode,
                "action_reader_training": self.reader_training,
                "source_core_training": "frozen_for_this_action_reader_comparison"}

    def action_logits(self, logits, observation, candidates, features):
        if not hasattr(self, "actor"):
            return logits
        evidence = self.beliefs.encode(observation, candidates).to(self.device)
        state = self.model.core.state_features(self.state["h"], self.state["z"])
        return self.actor(state, features["text"], evidence[None], logits)


def create(config):
    return BeliefPolicy(**config)
