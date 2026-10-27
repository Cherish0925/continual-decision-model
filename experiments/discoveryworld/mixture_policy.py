"""Execute a distribution over measured relations through native scalar control.

The continued DigitalPolicy owns perception, navigation and recurrent state.
An instance-owned posterior over affine hypotheses supplies numerical settings.
The hypotheses and scalar controller are explicit inductive choices, not actions
or physical laws discovered by the original CDM action head.
"""
from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from .quantitative_policy import QuantitativePolicy


class MixtureTaskPredictor(nn.Module):
    def __init__(self, source, mixture):
        super().__init__()
        self.source = source.requires_grad_(False).eval()
        self.mixture = mixture.eval()
        self.last_prediction = None

    def context(self, state):
        return self.source.core.state_features(state["h"], state["z"])

    def values(self, queries):
        from .numeric_reference import quantity_vector
        return torch.stack([torch.as_tensor(quantity_vector(query), dtype=torch.float64,
                                           device=next(self.mixture.parameters()).device)
                            for query in queries])

    def predict(self, queries, contexts, adapters):
        prediction = self.mixture.predict(self.values(queries), adapters.get("relation_posterior"))
        self.last_prediction = prediction
        # Each candidate is a coherent relation; averaging incompatible settings
        # can produce a setting supported by none of them.
        return {"mean_hz": prediction["map_expert_mean_hz"],
                "std_hz": prediction["mixture_std_hz"],
                "local_std_hz": prediction["expert_std_hz"].gather(
                    1, prediction["map_expert_index"][:, None])[:, 0]}

    def update(self, support_queries, support_contexts, support_hz, adapters, *, differentiable=False):
        values = self.values(support_queries)
        target = torch.as_tensor(support_hz, dtype=values.dtype, device=values.device)
        with torch.set_grad_enabled(differentiable):
            posterior = self.mixture.fit_posterior(values, target)
            prediction = self.mixture.predict(values, posterior)
            loss = (prediction["map_expert_mean_hz"] - target).square().mean()
        # The current buffer replaces the old fit. A repeated reading or label
        # is never accumulated a second time as independent calibration evidence.
        return {**adapters, "relation_posterior": posterior}, loss


class MixturePolicy(QuantitativePolicy):
    policy_type = "learned_relation_mixture_with_native_scalar_control"

    def __init__(self, checkpoint: str | Path, feature_cache: str | Path, *, device="cpu",
                 feature_device=None, encoder_path=None, retain_updates=True,
                 refinements=None, temperature=0.0, local_probes=False, mixture_probes=False):
        from .calibration_mixture import load_mixture_checkpoint
        from .train import load_checkpoint
        mixture, metadata = load_mixture_checkpoint(checkpoint, device)
        source, source_metadata = load_checkpoint(metadata["source_checkpoint"], device)
        predictor = MixtureTaskPredictor(source, mixture)
        wrapped = {"source": source_metadata,
                   "source_mode": source_metadata["training_args"]["mode"],
                   "training": {"adapter_initialization_seed": 0}}
        self.mixture_probes = mixture_probes
        self._configure(predictor, wrapped, feature_cache, device=device,
                        feature_device=feature_device, encoder_path=encoder_path,
                        retain_updates=retain_updates, refinements=refinements,
                        temperature=temperature, local_probes=local_probes or mixture_probes)

    def make_probe(self):
        if not self.mixture_probes:
            return super().make_probe()
        from .mixture_probe import MixtureProbe
        distribution = self.predictor.last_prediction
        return MixtureProbe(distribution["expert_posterior"][0], distribution["expert_mean_hz"][0],
                            distribution["expert_std_hz"][0], tolerance_hz=2., max_adjustment_actions=16)

    @property
    def evaluation_metadata(self):
        return {**super().evaluation_metadata,
                "prediction": "posterior_over_named_quantity_affine_relations",
                "control": "public_additive_choices_from_conditional_mixture" if self.mixture_probes else "public_additive_choices_toward_one_prediction_per_dialogue",
                "online_update": "replace_instance_posterior_from_then_available_support",
                "hypothesis_family": "nine_explicit_affine_quantity_relations",
                "action_point_prediction": "largest_posterior_expert_mean",
                "source_core_adapter_update": False,
                "probe_selection": "remaining_mixture_mass_per_native_action" if self.mixture_probes else "nearest_untried_local_setting",
                "probe_success_tolerance_hz": 2. if self.mixture_probes else None,
                "adapter_initialization": "source_A_B_unchanged; numerical_state_starts_at_learned_prior",
                "support_context": "stationary_public_measurements_with_named_units"}

    @torch.no_grad()
    def __call__(self, observation):
        action = super().__call__(observation)
        if self.decisions[-1].get("action_source") == "explicit_native_scalar_control":
            prediction = self.predictor.last_prediction
            if prediction is not None:
                self.decisions[-1]["relation_prediction"] = {
                    key: value.detach().cpu().tolist() for key, value in prediction.items()
                    if torch.is_tensor(value)}
        return action


def create(config):
    return MixturePolicy(**config)
