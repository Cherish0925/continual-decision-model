"""Choose and carry out complete, observation-conditioned numerical experiments.

The continuing CDM handles perception and ordinary actions. A plan chooses one
of the existing measured-coordinate hypotheses, reaches its predicted setting,
and tests nearby settings with the existing bounded numerical tool. The plan
persists across low-level adjustments and ends on received activation or probe
exhaustion. No hidden rule, task score or future trajectory enters a choice.
"""
from __future__ import annotations

import math
from pathlib import Path

import torch
from torch import nn

from .predictive_actions import PredictivePolicy
from .numeric_reference import quantity_vector
from .quantitative_policy import scalar_dialog_action
from .scalar_probe import ScalarProbe, public_scalar_control


PLAN_DIM = 10


class PlanValue(nn.Module):
    """Predict success and execution cost for each complete experimental plan."""

    def __init__(self, state_dim, width=128, use_state=True):
        super().__init__()
        self.state_dim, self.width, self.use_state = state_dim, width, use_state
        self.context = nn.Sequential(nn.LayerNorm(state_dim), nn.Linear(state_dim, width), nn.GELU())
        self.plan = nn.Sequential(nn.Linear(PLAN_DIM, width), nn.GELU())
        self.result = nn.Sequential(nn.Linear(3 * width, width), nn.GELU(), nn.Linear(width, 2))

    def forward(self, state, plans):
        if not self.use_state:
            state = torch.zeros_like(state)
        c = self.context(state).unsqueeze(-2).expand(*plans.shape[:-1], self.width)
        p = self.plan(plans)
        return self.result(torch.cat((c, p, c * p), -1))


class ExperimentalPolicy(PredictivePolicy):
    policy_type = "cdm_with_persistent_experimental_plans"

    def __init__(self, checkpoint, feature_cache, *, plan_checkpoint=None,
                 forced_plan=None, one_trial=False, **kwargs):
        self.forced_plan, self.one_trial = forced_plan, one_trial
        self.plan_value = None
        super().__init__(checkpoint, feature_cache, **kwargs)
        if plan_checkpoint:
            saved = torch.load(plan_checkpoint, map_location="cpu", weights_only=True)
            self.plan_value = PlanValue(**saved["config"]).to(self.device).eval()
            self.plan_value.load_state_dict(saved["model"])
            self.plan_training = saved["training"]
        else:
            self.plan_training = None

    def reset(self):
        super().reset()
        self.plan = None
        self.tested_plans = {}
        self.trials = []
        self.stop_requested = False

    @property
    def evaluation_metadata(self):
        return {**super().evaluation_metadata,
                "experimental_selection": "learned_plan_value" if self.plan_value is not None else "posterior_order",
                "experimental_plan": "one_affine_hypothesis_then_existing_local_probe",
                "plan_training": self.plan_training,
                "forced_plan": self.forced_plan, "one_trial": self.one_trial,
                "failed_plan_retention": "per_object_and_reactor; reset_per_instance"}

    def candidates(self, observation, dialog):
        query = self.calibration.query(dialog["object_handle"])
        if len(self.calibration.support()) < 2 or not all(query.presence.get(q, False)
                                                         for q in self.calibration.required_quantities):
            return None
        mixture = self.predictor.mixture
        x = torch.as_tensor(quantity_vector(query), dtype=torch.float64, device=mixture.device)[None]
        prediction = mixture.predict(x, self.adapters.get("relation_posterior"))
        means, sd, probs = (prediction[key][0].detach().cpu() for key in
                            ("expert_mean_hz", "expert_std_hz", "expert_posterior"))
        control = public_scalar_control(observation["ui"]["dialog_box"])
        if control is None:
            return None
        lower, upper = float(control.lower_hz), float(control.upper_hz)
        current = dialog["frequency_hz"]
        scale = float(mixture.y_scale)
        center = float(mixture.y_mean)
        ent = float(-(probs * probs.clamp_min(1e-30).log()).sum()) / math.log(len(probs))
        tried = self.tested_plans.get((dialog["reactor_handle"], dialog["object_handle"]), set())
        features = []
        for mean, spread, prob in zip(means, sd, probs):
            features.append([float((mean-center)/scale), float(spread/scale), float(prob),
                             float((mean-current)/scale), abs(float(mean-current))/scale,
                             (current-center)/scale, ent, len(self.calibration.support())/4,
                             len(tried)/9, float(control.minimum_step_hz)/scale])
        return {"means": means.tolist(), "std": sd.tolist(), "probabilities": probs.tolist(),
                "features": features, "valid": [lower <= float(m) <= upper and i not in tried
                                                  for i, m in enumerate(means)]}

    def _finish(self, succeeded, reason):
        if self.plan is None:
            return
        plan = self.plan
        self.tested_plans.setdefault(tuple(plan["identity"]), set()).add(plan["hypothesis"])
        self.trials.append({**{k:v for k,v in plan.items() if k != "probe"},
                            "success": bool(succeeded), "ending": reason,
                            "end_sequence": self.calibration.sequence})
        self.plan = None
        if self.one_trial:
            self.stop_requested = True

    def observe(self, transition):
        super().observe(transition)
        if self.plan is None:
            return
        self.plan["actions"] += 1
        dialog = self.calibration.current_dialog()
        if dialog is None or [dialog["reactor_handle"], dialog["object_handle"]] != self.plan["identity"]:
            self._finish(False, "dialogue_changed")
            return
        if dialog["activated"]:
            self._finish(True, "observed_activation")
        elif self.plan["probe"].started:
            self.plan["probe"].observe_attempt(dialog["frequency_hz"], activated=dialog["activated"])

    def resolve_action(self, action, observation):
        dialog = self.calibration.current_dialog()
        if self.stop_requested:
            return action
        if dialog is None or dialog["activated"] is not False:
            return super().resolve_action(action, observation)
        if any(label.object_handle == dialog["object_handle"] for label in self.calibration.labels):
            return super().resolve_action(action, observation)
        if self.plan is None:
            plans = self.candidates(observation, dialog)
            if plans is None or not any(plans["valid"]):
                return super().resolve_action(action, observation)
            state = self.model.core.state_features(self.state["h"], self.state["z"])
            scores = torch.tensor(plans["probabilities"], device=self.device)
            if self.plan_value is not None:
                value = self.plan_value(state.float(), torch.tensor([plans["features"]], device=self.device))[0]
                scores = value[:, 0].sigmoid() - .001 * value[:, 1].clamp_min(0) * 64
            scores = scores.masked_fill(~torch.tensor(plans["valid"], device=self.device), -torch.inf)
            chosen = int(scores.argmax()) if self.forced_plan is None else self.forced_plan
            if not plans["valid"][chosen]:
                raise ValueError("The requested experiment is not available in the observed state")
            self.plan = {"identity": [dialog["reactor_handle"], dialog["object_handle"]],
                         "hypothesis": chosen, "mean_hz": plans["means"][chosen],
                         "std_hz": plans["std"][chosen], "actions": 0,
                         "start_sequence": self.calibration.sequence,
                         "state": state[0].float().cpu().tolist(), "candidates": plans,
                         "probe": ScalarProbe()}
        plan = self.plan
        public_dialog = observation["ui"]["dialog_box"]
        current, target = dialog["frequency_hz"], plan["mean_hz"]
        selected = scalar_dialog_action(public_dialog, current, target)
        options = public_dialog.get("dialogOptions", {})
        key = (selected or {}).get("chosen_dialog_option_int")
        text = str(options.get(str(key), options.get(key, ""))).strip().casefold()
        if text == "exit" and not plan["probe"].started:
            plan["probe"].observe_attempt(current, activated=False, point_attempt=True)
        if plan["probe"].started:
            target = plan["probe"].next_target(current, plan["mean_hz"], plan["std_hz"], public_dialog)
            if target is None:
                self._finish(False, "local_experiment_exhausted")
                # Return the public exit; a later encounter can select another
                # hypothesis. The one-trial collector stops before this action.
                return scalar_dialog_action(public_dialog, current, current)
            selected = scalar_dialog_action(public_dialog, current, target)
        if selected is None:
            raise ValueError("Experimental plan lacks an executable public scalar action")
        self.decisions[-1].update(action_source="committed_hypothesis_experiment",
                                 hypothesis=plan["hypothesis"], plan_target_hz=plan["mean_hz"],
                                 selected_action=selected)
        return selected


def create(config):
    return ExperimentalPolicy(**config)
