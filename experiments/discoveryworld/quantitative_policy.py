"""A learned scalar prediction executed by an explicit native control primitive.

The existing DigitalPolicy still advances its state and supplies other actions.
This wrapper does not claim that its action head learned numeric adjustment.
It uses only returned measurements and currently visible dialogue choices.
"""
from __future__ import annotations

from dataclasses import asdict
import math
from pathlib import Path
import re
from time import perf_counter

import torch

from .evaluate import LearnedPolicy
from .features import FeatureCache, FeatureConfig


_DELTA = re.compile(r"^(increase|decrease)\s+.+?\s+by\s+([\d,.]+)\s+(\S+)\s*$", re.I)
_RANGE = re.compile(r"allowable range is\s+([\d,.]+)\s+to\s+([\d,.]+)\s+(\S+)", re.I)


def scalar_dialog_action(dialog: dict, current: float, desired: float, *, unit: str = "Hz",
                         activated: bool = False) -> dict | None:
    """Take the public additive choice closest to a requested scalar setpoint.

    Reaching a setpoint does not establish scientific correctness. Exit when
    no offered adjustment improves the distance, or activation is observed.
    No action number, allowed range, step size or scientific law is assumed.
    """
    if not dialog.get("is_in_dialog") or not all(math.isfinite(x) for x in (current, desired)):
        return None
    bounds = _RANGE.search(dialog.get("dialogIn", ""))
    limits = None
    if bounds and bounds.group(3).casefold().rstrip(".") == unit.casefold():
        limits = tuple(float(value.replace(",", "")) for value in bounds.group(1, 2))
    choices, exit_choice = [], None
    for key, text in dialog.get("dialogOptions", {}).items():
        if str(text).strip().casefold() == "exit":
            exit_choice = int(key)
        match = _DELTA.fullmatch(str(text).strip())
        if match is None or match.group(3).casefold() != unit.casefold():
            continue
        delta = float(match.group(2).replace(",", "")) * (1 if match.group(1).casefold() == "increase" else -1)
        value = current + delta
        if limits is not None:
            value = min(max(value, limits[0]), limits[1])
        choices.append((abs(value - desired), abs(delta), int(key)))
    if not activated and choices:
        distance, _, selected = min(choices)
        if distance < abs(current - desired) - 1e-9:
            return {"chosen_dialog_option_int": selected}
    return {"chosen_dialog_option_int": exit_choice} if exit_choice is not None else None


class QuantitativePolicy(LearnedPolicy):
    """One source core, public ledger and deployed adapter pair per episode."""

    def __init__(self, checkpoint: str | Path, feature_cache: str | Path, *, device: str = "cpu",
                 feature_device: str | None = None, encoder_path: str | Path | None = None,
                 retain_updates: bool = True, refinements: int | None = None,
                 temperature: float = 0.0, local_probes: bool = False):
        from .calibration_predictor import load_calibration_checkpoint
        predictor, metadata = load_calibration_checkpoint(checkpoint, device)
        self._configure(predictor, metadata, feature_cache, device=device, feature_device=feature_device,
                        encoder_path=encoder_path, retain_updates=retain_updates,
                        refinements=refinements, temperature=temperature, local_probes=local_probes)

    def _configure(self, predictor, metadata: dict, feature_cache: str | Path, *, device="cpu",
                   feature_device=None, encoder_path=None, retain_updates=True, refinements=None,
                    temperature=0.0, local_probes=False):
        # Initialize the existing LearnedPolicy around the already loaded source;
        # loading a second checkpoint here would create a different state owner.
        self.predictor, self.model = predictor, predictor.source
        self.predictor.eval()
        self.checkpoint = metadata["source"]
        self.device = torch.device(device)
        self.features = FeatureCache(feature_cache, FeatureConfig(**{
            "schema": 1, "time_basis": "wall_clock", **self.checkpoint["features"]}),
                                     feature_device or device, encoder_path)
        self.mode = metadata["source_mode"]
        self.persistent = not self.checkpoint["training_args"].get("restart_state", False)
        self.restart_workspace = self.checkpoint["training_args"].get("restart_workspace", False)
        self.refinements, self.temperature = refinements, temperature
        self.retain_updates = retain_updates
        self.local_probes = local_probes
        self.adapter_initialization_seed = int(metadata.get("training", {}).get("adapter_initialization_seed", 0))
        initial = metadata.get("initial_adapters")
        if initial is not None and set(initial) != {"A", "B"}:
            raise ValueError("A saved adapter initialization must contain exactly A and B")
        self._initial_adapters = None if initial is None else {
            key: value.detach().cpu().clone() for key, value in initial.items()}
        self.reset()

    @property
    def evaluation_metadata(self):
        return {"prediction": "learned_working_scalar_with_instance_adapters",
                "control": "public_additive_choices_toward_one_prediction_per_dialogue",
                "numeric_control_is_learned_action_head": False,
                "shared_source_core_state": True, "updates_retained": self.retain_updates,
                "adapter_initialization": "checkpoint_tensors" if self._initial_adapters is not None else "seed_fallback",
                "adapter_initialization_seed": self.adapter_initialization_seed,
                "support_context": "last_actual_forward_before_received_feedback",
                "update_schedule": "new_support_physical_record_or_label; current_support_buffer",
                "local_probes": self.local_probes,
                "features": asdict(self.features.config)}

    def reset(self):
        from .calibration import CalibrationLog
        # Initializing a policy must not advance the caller's sampling stream.
        # Seed only generators used here; other CUDA devices keep their state.
        devices = [self.device] if self.device.type == "cuda" else []
        with torch.random.fork_rng(devices=devices):
            torch.random.default_generator.manual_seed(self.adapter_initialization_seed)
            if self.device.type == "cuda":
                with torch.cuda.device(self.device):
                    torch.cuda.manual_seed(self.adapter_initialization_seed)
            super().reset()
        if self._initial_adapters is not None:
            for key, value in self._initial_adapters.items():
                if value.shape != self.state[key].shape or value.dtype != self.state[key].dtype:
                    raise ValueError(f"Saved initial adapter {key} does not match the source model")
                self.state[key] = value.to(self.device).clone()
        self.calibration = CalibrationLog()
        if self.features.config.measurement_memory:
            self.measurements = self.calibration.measurements
        self.adapters = {key: self.state[key] for key in ("A", "B")}
        self._seen_support = {}
        self._setpoint = None
        self.update_events = []
        self._probes, self._probing = {}, set()
        self._pending_probe = None

    def observe(self, transition: dict):
        # This also updates the very MeasurementLog read by LearnedPolicy.
        self.calibration.append_transition(transition)
        if self._pending_probe is not None:
            dialog = self.calibration.current_dialog()
            if dialog is not None and (dialog["reactor_handle"], dialog["object_handle"]) == self._pending_probe:
                self._probes[self._pending_probe].observe_attempt(
                    dialog["frequency_hz"], activated=dialog["activated"], adjustment_actions=1)
            self._pending_probe = None
        supports = self.calibration.support()
        new_support = []
        for support in supports:
            identity = support.label.object_handle
            revision = (support.label.received_sequence, max(record.sequence for record in support.query.records))
            if self._seen_support.get(identity) != revision:
                self._seen_support[identity] = revision
                new_support.append(identity)
        if not new_support:
            return
        started = perf_counter()
        context = self.predictor.context(self.state).detach()
        updated, loss = self.predictor.update(
            [support.query for support in supports], context.expand(len(supports), -1),
            torch.tensor([support.label.frequency_hz for support in supports], device=self.device,
                         dtype=torch.float64),
            self.adapters, differentiable=False)
        if self.retain_updates:
            self.adapters = updated
            self.state = {**self.state, **updated}
        self.update_events.append({"sequence": transition["index"], "new_support_objects": new_support,
                                   "support_count": len(supports), "retained": self.retain_updates,
                                   "support_versions": [
                                       {"object_handle": support.label.object_handle,
                                        "label_sequence": support.label.received_sequence,
                                        "last_physical_sequence": max(record.sequence for record in support.query.records)}
                                       for support in supports],
                                   "loss": float(loss.detach().mean()), "seconds": perf_counter() - started})

    @torch.no_grad()
    def prime(self, observation: dict) -> dict:
        """Assimilate an externally executed prefix without proposing trials.

        Actual returned transitions still update the evidence and numerical
        learner through observe(). Unexecuted proposals must not start probes
        or consume their per-target budget.
        """
        self.calibration.observe_initial(observation)
        return LearnedPolicy.__call__(self, observation)

    def make_probe(self):
        from .scalar_probe import ScalarProbe
        return ScalarProbe(max_adjustment_actions=16)

    @torch.no_grad()
    def __call__(self, observation: dict) -> dict:
        self.calibration.observe_initial(observation)
        action = super().__call__(observation)
        return self.resolve_action(action, observation)

    def resolve_action(self, action: dict, observation: dict) -> dict:
        """Execute the numeric primitive after the neural policy has acted."""
        decision = self.decisions[-1]
        decision["action_source"] = "source_digital_policy"
        dialog = self.calibration.current_dialog()
        if dialog is None:
            self._setpoint = None
            return action
        identity = (dialog["reactor_handle"], dialog["object_handle"])
        if self._setpoint is not None and self._setpoint["identity"] != identity:
            self._setpoint = None
        if self._setpoint is None:
            known = {label.object_handle for label in self.calibration.labels}
            query = self.calibration.query(dialog["object_handle"])
            if dialog["activated"] is not False or dialog["object_handle"] in known or not all(
                    query.presence.get(quantity, False) for quantity in self.calibration.required_quantities):
                return action
            prediction = self.predictor.predict([query], self.predictor.context(self.state), self.adapters)
            self._setpoint = {"identity": identity, "mean_hz": float(prediction["mean_hz"][0]),
                              "std_hz": float(prediction["std_hz"][0]),
                              "probe_std_hz": float(prediction.get("local_std_hz", prediction["std_hz"])[0]),
                              "query_after_sequence": query.available_after_sequence}
            if not all(math.isfinite(self._setpoint[key]) for key in ("mean_hz", "std_hz")) or self._setpoint["std_hz"] < 0:
                raise ValueError("The learned scalar prediction must have a finite mean and nonnegative finite standard deviation")
        selected = scalar_dialog_action(observation["ui"]["dialog_box"], dialog["frequency_hz"],
                                        self._setpoint["mean_hz"], unit=dialog["unit"], activated=dialog["activated"])
        if self.local_probes and not dialog["activated"]:
            public_dialog = observation["ui"]["dialog_box"]
            options = public_dialog.get("dialogOptions", {})
            choice = (selected or {}).get("chosen_dialog_option_int")
            option = options.get(str(choice), options.get(choice, ""))
            if identity not in self._probing and option.strip().casefold() == "exit":
                if identity not in self._probes:
                    self._probes[identity] = self.make_probe()
                probe = self._probes[identity]
                probe.observe_attempt(dialog["frequency_hz"], activated=False, point_attempt=True)
                self._probing.add(identity)
            if identity in self._probing:
                target = self._probes[identity].next_target(
                    dialog["frequency_hz"], self._setpoint["mean_hz"],
                    self._setpoint["probe_std_hz"], public_dialog)
                if target is not None:
                    selected = scalar_dialog_action(public_dialog, dialog["frequency_hz"], target,
                                                    unit=dialog["unit"])
                    self._pending_probe = identity
                    decision["local_probe"] = {"target_hz": target,
                                               "mean_hz": self._setpoint["mean_hz"],
                                               "std_hz": self._setpoint["probe_std_hz"]}
                    if getattr(self._probes[identity], "last_choice", None) is not None:
                        decision["mixture_probe"] = self._probes[identity].last_choice
                else:
                    # Finish the bounded experiment rather than returning to a
                    # setting already observed to be inactive.
                    selected = scalar_dialog_action(public_dialog, dialog["frequency_hz"],
                                                    dialog["frequency_hz"], unit=dialog["unit"])
        if selected is not None:
            decision.update(action_source="explicit_native_scalar_control", source_proposal=action,
                            selected_action=selected, scalar_prediction=dict(self._setpoint),
                            current_hz=dialog["frequency_hz"], observed_activated=dialog["activated"])
            return selected
        return action


def create(config: dict):
    return QuantitativePolicy(**config)
