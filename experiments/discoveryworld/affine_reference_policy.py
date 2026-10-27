"""Frozen affine-expert reference with explicit native scalar execution.

This is a standard numerical baseline, not a learned CDM action or newly
discovered solver. Its prior and temperatures are fitted on training episodes
only. Deployment fits the complete actually received support, without changing
the source core's A/B or any shared model parameter.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import gzip
from hashlib import sha256
import json
import os
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .numeric_reference import QUANTITIES, TEMPERATURES, affine_fit, learned_prior, mixture, quantity_vector
from .quantitative_policy import QuantitativePolicy


@dataclass(frozen=True)
class AffinePosterior:
    values: torch.Tensor
    targets: torch.Tensor
    slope: torch.Tensor
    intercept: torch.Tensor
    support_mse: torch.Tensor
    weights: torch.Tensor
    temperature: torch.Tensor


class FrozenAffineReference(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        if config.get("kind") != "frozen_affine_reference_v1" or config["quantities"] != list(QUANTITIES):
            raise ValueError("Expected the frozen nine-quantity affine reference configuration")
        self.register_buffer("priors", torch.tensor([config["priors"][str(n)] for n in (2, 3)], dtype=torch.float64))
        self.register_buffer("temperatures", torch.tensor([config["temperatures_hz"][str(n)] for n in (2, 3)], dtype=torch.float64))
        self.register_buffer("global_mean", torch.tensor(config["global_prior_mean_hz"], dtype=torch.float64))
        self.register_buffer("global_std", torch.tensor(config["global_prior_std_hz"], dtype=torch.float64))

    @torch.no_grad()
    def fit_posterior(self, values, targets) -> AffinePosterior:
        values = torch.as_tensor(values, dtype=torch.float64, device=self.priors.device)
        targets = torch.as_tensor(targets, dtype=torch.float64, device=self.priors.device)
        if values.ndim != 2 or values.shape[1] != len(QUANTITIES) or targets.shape != values.shape[:1] or not len(targets):
            raise ValueError("Expected nonempty support values [S,9] and targets [S]")
        center, mean = values.mean(0), targets.mean()
        centered = values - center
        variance = centered.square().sum(0)
        covariance = (centered * (targets - mean)[:, None]).sum(0)
        slope = torch.zeros_like(covariance)
        nonconstant = variance > 0
        slope[nonconstant] = covariance[nonconstant] / variance[nonconstant]
        intercept = mean - center * slope
        mse = ((intercept + values * slope - targets[:, None]).square()).mean(0)
        index = int(len(targets) >= 3)
        temperature = self.temperatures[index]
        weights = (self.priors[index].log() - mse / (2 * temperature.square())).softmax(0)
        return AffinePosterior(values.clone(), targets.clone(), slope, intercept, mse, weights, temperature)

    @torch.no_grad()
    def predict(self, values, posterior: AffinePosterior | None = None) -> dict:
        values = torch.as_tensor(values, dtype=torch.float64, device=self.priors.device)
        if values.ndim != 2 or values.shape[1] != len(QUANTITIES):
            raise ValueError("Expected query values [Q,9]")
        if posterior is None:
            proposals = self.global_mean.expand(len(values), len(QUANTITIES))
            weights = self.priors[0].expand_as(proposals)
            mean = self.global_mean.expand(len(values))
            std = self.global_std.expand(len(values))
            count = 0
        else:
            proposals = posterior.intercept + values * posterior.slope
            weights = posterior.weights.expand_as(proposals)
            mean = (weights * proposals).sum(-1)
            std = ((weights * (proposals - mean[:, None]).square()).sum(-1) + posterior.temperature.square()).sqrt()
            count = len(posterior.targets)
        selected = weights.argmax(-1)
        return {"map_expert_mean_hz": proposals.gather(1, selected[:, None])[:, 0],
                "mixture_mean_hz": mean, "mixture_std_hz": std,
                "expert_predictions_hz": proposals, "weights": weights, "selected_expert": selected,
                "support_count": torch.full((len(values),), count, device=values.device, dtype=torch.long)}


def load_affine_reference(path: str | Path, device="cpu") -> tuple[FrozenAffineReference, dict]:
    config = json.loads(Path(path).read_text())
    return FrozenAffineReference(config).to(device).eval(), config


class AffineTaskPredictor(nn.Module):
    def __init__(self, source, reference: FrozenAffineReference):
        super().__init__()
        self.source = source.requires_grad_(False).eval()
        self.reference = reference.eval()
        self.last_prediction = None

    def context(self, state):
        return self.source.core.state_features(state["h"], state["z"])

    def values(self, queries):
        return torch.as_tensor(np.stack([quantity_vector(query) for query in queries]),
                               dtype=torch.float64, device=self.reference.priors.device)

    def predict(self, queries, contexts, adapters):
        prediction = self.reference.predict(self.values(queries), adapters.get("affine_reference_state"))
        self.last_prediction = prediction
        return {"mean_hz": prediction["map_expert_mean_hz"], "std_hz": prediction["mixture_std_hz"]}

    def update(self, support_queries, support_contexts, support_hz, adapters, *, differentiable=False):
        if differentiable:
            raise ValueError("The frozen affine reference is not a trainable CDM adapter update")
        values = self.values(support_queries)
        target = torch.as_tensor(support_hz, dtype=values.dtype, device=values.device)
        posterior = self.reference.fit_posterior(values, target)
        fit = self.reference.predict(values, posterior)["map_expert_mean_hz"]
        return {**adapters, "affine_reference_state": posterior}, (fit - target).square().mean()


class AffineReferencePolicy(QuantitativePolicy):
    policy_type = "affine_reference_with_native_scalar_control"

    def __init__(self, checkpoint: str | Path, feature_cache: str | Path, *, device="cpu",
                 feature_device=None, encoder_path=None, retain_updates=True, refinements=None, temperature=0.0):
        from .train import load_checkpoint
        reference, config = load_affine_reference(checkpoint, device)
        source, source_metadata = load_checkpoint(config["source_checkpoint"], device)
        self.reference_config = config
        predictor = AffineTaskPredictor(source, reference)
        self._configure(predictor, {"source": source_metadata,
                                   "source_mode": source_metadata["training_args"]["mode"],
                                   "training": {"adapter_initialization_seed": 0}}, feature_cache,
                        device=device, feature_device=feature_device, encoder_path=encoder_path,
                        retain_updates=retain_updates, refinements=refinements, temperature=temperature)

    @property
    def evaluation_metadata(self):
        return {**super().evaluation_metadata, "prediction": "frozen_standard_affine_expert_reference",
                "online_update": "replace_complete_public_support_affine_fit",
                "source_core_adapter_update": False, "action_point_prediction": "largest_weight_expert_prediction",
                "support_context": "stationary_named_public_quantities",
                "uncertainty": "weighted_expert_spread_plus_selected_temperature; not calibrated",
                "reference_config": self.reference_config}

    @torch.no_grad()
    def __call__(self, observation):
        action = super().__call__(observation)
        if self.decisions[-1].get("action_source") == "explicit_native_scalar_control":
            prediction = self.predictor.last_prediction
            if prediction is not None:
                self.decisions[-1]["affine_reference_prediction"] = {
                    key: value.detach().cpu().tolist() for key, value in prediction.items()}
                self.decisions[-1]["affine_reference_quantities"] = list(QUANTITIES)
        return action


def fit_training_config(episodes: list[dict], source_checkpoint: str | Path, *, project_root=None) -> dict:
    """Use only the supplied training archive's future queries to freeze settings."""
    data, labels, instances = [], {}, set()
    for episode in episodes:
        samples = episode["samples"]
        if not samples:
            continue
        identity = samples[0]["seed"]
        if identity in instances:
            raise ValueError("Use one source trajectory per training instance")
        instances.add(identity)
        for sample in samples:
            row = {"X": np.stack([quantity_vector(s["query"]) for s in sample["support"]]),
                   "y": np.array([s["label"]["frequency_hz"] for s in sample["support"]]),
                   "x": quantity_vector(sample["query"]), "support_count": sample["support_count"],
                   "target": sample["evaluation_only_target"]["frequency_hz"]}
            row["affine"] = affine_fit(row["X"], row["y"], row["x"])
            data.append(row)
            for label in [s["label"] for s in sample["support"]] + [sample["evaluation_only_target"]]:
                key = (identity, label["object_handle"])
                if key not in labels or label["received_sequence"] < labels[key][0]:
                    labels[key] = (label["received_sequence"], label["frequency_hz"])
    temperatures, priors, selection = {}, {}, {}
    for count in (2, 3):
        examples = [row for row in data if row["support_count"] == count]
        if not examples:
            raise ValueError(f"No training queries with {count} available calibrations")
        errors = {str(t): float(np.mean([abs(mixture(row, t)[0] - row["target"]) for row in examples]))
                  for t in TEMPERATURES}
        temperature = min(TEMPERATURES, key=lambda t: errors[str(t)])
        prior = learned_prior(examples, temperature)
        temperatures[str(count)], priors[str(count)] = temperature, prior.tolist()
        selection[str(count)] = {"queries": len(examples), "uniform_mixture_train_mae_hz_by_temperature": errors,
                                 "empirical_prior_map_train_mae_hz": float(np.mean([
                                     abs(row["affine"][0][mixture(row, temperature, prior)[2].argmax()] - row["target"])
                                     for row in examples]))}
    values = np.array([value for _, value in labels.values()], dtype=np.float64)
    source = Path(source_checkpoint).resolve()
    root = Path.cwd() if project_root is None else Path(project_root).resolve()
    return {"kind": "frozen_affine_reference_v1", "source_checkpoint": os.path.relpath(source, root),
            "source_checkpoint_sha256": sha256(source.read_bytes()).hexdigest(),
            "quantities": list(QUANTITIES), "priors": priors, "temperatures_hz": temperatures,
            "global_prior_mean_hz": float(values.mean()), "global_prior_std_hz": float(values.std()),
            "training_instances": len(instances), "training_queries": len(data), "unique_training_labels": len(labels),
            "training_seeds": sorted(instances), "training_selection": selection,
            "selection_rule": "Original fixed grid minimizes all-training uniform-mixture future MAE separately for support2/3; then original 30-step EM with 0.1 pseudocount fits expert prior",
            "support_count_rule": "up to two supports uses support2 settings; three or more uses support3 settings",
            "native_prediction": "MAP expert; complete proposals and posterior weights recorded",
            "discarded_prediction": "global training native-Hz prior; same support fit computed and discarded",
            "scope": "Frozen standard numerical reference; training labels only; no native/validation/held-out selection; not learned CDM core or an original method"}


def create(config: dict):
    return AffineReferencePolicy(**config)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--public-inputs", type=Path, required=True)
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    raw = args.public_inputs.read_bytes()
    episodes = json.loads(gzip.decompress(raw))
    # Archive extraction itself enforces split=train. Also verify any source
    # run metadata available here, without reading its trajectory or outcomes.
    for episode in episodes:
        run = Path(episode["source"]).parent / "run.json"
        if run.exists() and json.loads(run.read_text()).get("split") != "train":
            raise ValueError(f"Non-training source in reference archive: {run}")
    config = fit_training_config(episodes, args.source_checkpoint, project_root=args.project_root)
    config["training_archive"] = {"path": os.path.relpath(args.public_inputs.resolve(), args.project_root.resolve()),
                                  "sha256": sha256(raw).hexdigest()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(config, indent=2) + "\n")
    print(json.dumps({"config": str(args.output), "instances": config["training_instances"],
                      "queries": config["training_queries"], "temperatures_hz": config["temperatures_hz"]}))


if __name__ == "__main__":
    main()
