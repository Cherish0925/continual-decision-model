"""Action-conditioned forecasts read and revised by the continuing workspace.

Forecasts describe returned observations and subsequent behavior returns. They
are predictions, not simulator queries. The GRU consumes each real observation
once; additional computation lets the workspace compare predicted consequences
of its leading actions and revise their ranking. The numerical execution tool
and its received-evidence updates are unchanged.
"""
from __future__ import annotations

from time import perf_counter

import torch
from torch import nn

from cdm.digital import public_candidates
from cdm.multimodal import EncodedEvent, join_events
from .adaptive_compute import from_reader
from .belief_actions import BELIEF_DIM, BeliefPolicy
from .features import batch_features, canonical_action, load_features

HORIZONS = (1, 8, 32, 128)
FORECAST_DIM = 14 + len(HORIZONS)


class ConsequenceModel(nn.Module):
    def __init__(self, state_dim, text_dim, width=256):
        super().__init__()
        self.state = nn.Sequential(nn.LayerNorm(state_dim), nn.Linear(state_dim, width))
        self.action = nn.Linear(text_dim + 8, width)
        self.evidence = nn.Sequential(nn.Linear(BELIEF_DIM, width), nn.GELU(), nn.Linear(width, width))
        self.predict = nn.Sequential(nn.LayerNorm(3 * width), nn.Linear(3 * width, width),
                                     nn.GELU(), nn.Linear(width, FORECAST_DIM))

    def forward(self, state, candidates, belief):
        action = self.action(torch.cat((candidates["text"].float(), candidates["numeric"].float()), -1))
        context = self.state(state.float())[:, None].expand_as(action)
        return self.predict(torch.cat((context, action, self.evidence(belief.float())), -1)), action


class PredictiveModel(nn.Module):
    def __init__(self, digital, reader, *, feedback=True, candidates_per_cycle=8):
        super().__init__()
        self.digital, self.reader = digital, reader
        self.feedback, self.candidates_per_cycle = feedback, candidates_per_cycle
        self.forecaster = ConsequenceModel(2 * digital.config.width, reader.text_dim)
        self.write = nn.Sequential(nn.Linear(256 + FORECAST_DIM, digital.config.width),
                                   nn.LayerNorm(digital.config.width))
        self.prediction_type = nn.Parameter(torch.randn(digital.config.width) * .02)
        self.forecast_score = nn.Linear(FORECAST_DIM, 1, bias=False)
        nn.init.zeros_(self.forecast_score.weight)

    def proposals(self, observations, candidates, belief, state, *, cycles=2, use_forecasts=True):
        event, state = self.digital.assimilate(observations, state)
        # The established policy performs one ordinary workspace update. Extra
        # cycles are explicitly separate from that observation-processing step.
        state = self.digital.refine_state(event, state, 1)
        outputs, forecasts, states = [], [], []
        for cycle in range(cycles + 1):
            features = self.digital.core.state_features(state["h"], state["z"])
            base, _ = self.digital.readout(event, candidates, state, predict_outcomes=False)
            prediction, action = self.forecaster(features, candidates, belief)
            scores = self.reader(features, candidates["text"], belief, base)
            if use_forecasts:
                scores = scores + self.forecast_score(prediction).squeeze(-1)
            scores = scores.masked_fill(~candidates["present"], -torch.inf)
            outputs.append(scores)
            forecasts.append(prediction)
            states.append(state)
            if cycle == cycles:
                break
            refinement_event = event
            if self.feedback and use_forecasts:
                k = min(self.candidates_per_cycle, scores.shape[1])
                indices = scores.detach().topk(k, dim=-1).indices
                token = self.write(torch.cat((action, prediction), -1)) + self.prediction_type
                token = token.gather(1, indices[..., None].expand(-1, -1, token.shape[-1]))
                present = candidates["present"].gather(1, indices)
                refinement_event = join_events(event, EncodedEvent(token, present))
            state = self.digital.refine_state(refinement_event, state, 1)
        return torch.stack(outputs, 1), torch.stack(forecasts, 1), states


def build(checkpoint, device, *, feedback=True):
    original, source, reader = from_reader(checkpoint, device)
    return PredictiveModel(original.digital, original.reader, feedback=feedback).to(device), source, reader


class PredictivePolicy(BeliefPolicy):
    policy_type = "predictive_cdm_with_explicit_numeric_primitive"

    def __init__(self, checkpoint, feature_cache, *, use_forecasts=True, **kwargs):
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        super().__init__(saved["base_checkpoint"], feature_cache, **kwargs)
        self.predictive = PredictiveModel(self.model, self.actor, **saved["model_config"]).to(self.device)
        self.predictive.load_state_dict(saved["model"], strict=True)
        self.predictive.eval()
        self.predictive_training = saved["training"]
        self.consequence_training = saved.get("consequence_training")
        self.use_forecasts = use_forecasts
        self.checkpoint = {**self.checkpoint, "epoch": saved["epoch"],
                           "train_instances": saved["training"]["train_instances"],
                           "validation_instances": saved["training"]["validation_instances"]}

    @property
    def refinement_schedule(self):
        return {"observation_updates": 1, "prediction_cycles": self.refinements if self.refinements is not None else 2}

    @property
    def evaluation_metadata(self):
        return {**super().evaluation_metadata, "source_core_training": "joint_prediction_and_action_training",
                "prediction_feedback": self.predictive.feedback and self.use_forecasts,
                "use_forecasts": self.use_forecasts, "return_horizons": list(HORIZONS),
                "prediction_targets": "received_result_and_behavior_return; no_future_inputs",
                "predictive_training": self.predictive_training,
                "consequence_training": self.consequence_training}

    @torch.no_grad()
    def __call__(self, observation):
        action = self.resolve_action(self.propose(observation), observation)
        executed = [i for i, candidate in enumerate(self._last_candidates)
                    if canonical_action(candidate.action) == canonical_action(action)]
        if len(executed) != 1:
            raise ValueError("Executed action must match its public prediction candidate")
        self.decisions[-1]["executed_candidate"] = executed[0]
        self.decisions[-1]["executed_forecast_by_cycle"] = self._last_forecasts[:, executed[0]].float().tolist()
        return action

    @torch.no_grad()
    def prime(self, observation):
        return self.propose(observation)

    def propose(self, observation):
        if self.mode != "cdm":
            raise ValueError("Prediction feedback requires the trained CDM workspace")
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
        belief = self.beliefs.encode(observation, candidates).to(self.device)[None]
        feature_done = perf_counter()
        state = self.state
        if not self.persistent or self.restart_workspace:
            state = {**state, "h": state["h"] if self.persistent else torch.zeros_like(state["h"]),
                     "z": self.model.core.slots[None].expand(state["h"].shape[0], -1, -1)}
        cycles = 2 if self.refinements is None else self.refinements
        scores, forecasts, states = self.predictive.proposals(obs, actions, belief, state,
                                                             cycles=cycles, use_forecasts=self.use_forecasts)
        self.state = states[-1]
        self._last_candidates, self._last_forecasts = candidates, forecasts[0]
        self._last_scores = scores[0]
        logits = scores[0, -1].float()
        selected = int(torch.multinomial((logits / self.temperature).softmax(-1), 1)) \
            if self.temperature > 0 else int(logits.argmax())
        done = perf_counter()
        self.decisions.append({"candidates": len(candidates), "selected": selected,
                               "selected_by_cycle": scores[0].argmax(-1).tolist(),
                               "forecast_by_cycle": forecasts[0, :, selected].float().tolist(),
                               "feature_seconds": feature_done - start, "policy_seconds": done - feature_done,
                               "decision_seconds": done - start, "refinements": 1 + cycles})
        return dict(candidates[selected].action)


def create(config):
    return PredictivePolicy(**config)
