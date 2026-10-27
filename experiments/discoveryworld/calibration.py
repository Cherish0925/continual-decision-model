"""Causal calibration evidence from executed public DiscoveryWorld events.

Working settings require a fresh Hz return and an activated reactor with one
publicly identified crystal in that same observation. Instrument readings keep
their units and conditions. This module fits no rule and chooses no actions.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import re

from .measurements import Measurement, MeasurementLog, MeasurementSummary, _FREQUENCY
from .reference import observed_objects

Handle = int | str
FEATURE_QUANTITIES = ("density_g_cm3", "temperature_c", "radiation_usv_h", "quantum_gap_nm",
                      *(f"spectrum_channel_{i}" for i in range(1, 6)))


@dataclass(frozen=True)
class CrystalQuery:
    object_handle: Handle
    object_name: str
    available_after_sequence: int
    features: dict[str, tuple[MeasurementSummary, ...]]
    presence: dict[str, bool]
    records: tuple[Measurement, ...]


@dataclass(frozen=True)
class CalibrationLabel:
    object_handle: Handle
    reactor_handle: Handle
    frequency_hz: float
    unit: str
    received_sequence: int
    activation_sequence: int | None
    measurement: Measurement


@dataclass(frozen=True)
class CalibrationSupport:
    label: CalibrationLabel
    query: CrystalQuery
    first_available_sequence: int
    available_after_sequence: int


@dataclass(frozen=True)
class QueryOpportunity:
    decision_sequence: int
    reactor_handle: Handle
    query: CrystalQuery
    support: tuple[CalibrationSupport, ...]
    clean: bool
    exclusion_reason: str | None


@dataclass(frozen=True)
class CalibrationQueryExample:
    opportunity: QueryOpportunity
    label: CalibrationLabel


@dataclass(frozen=True)
class ActivationObservation:
    sequence: int
    object_handle: Handle
    reactor_handle: Handle
    kind: str  # new_activation, restoration, or first_observed_active


class CalibrationLog:
    """Episode-owned facts, then-available support, and pre-intervention queries.

    Call ``observe_initial`` at reset, then ``observe`` only after actual actions.
    ``append_transition`` replays the same public boundary. Snapshots are copies.
    Support joins an earlier working-frequency label to properties received
    later, never retroactively changing an earlier snapshot.

    A query opportunity is frozen after the first sufficiently measured,
    uncalibrated crystal's actual TALK/USE frequency return, before the next
    decision. Prior frequency intervention marks it unsuitable for the clean
    first-query comparison. A later active Hz return supplies only its label.
    """

    def __init__(self, *, required_quantities=FEATURE_QUANTITIES):
        self.required_quantities = tuple(required_quantities)
        self.measurements = MeasurementLog()
        self.sequence = -1
        self._observation = {"ui": {}}
        self._active: dict[Handle, tuple[Handle, bool | None]] = {}
        self._ever_active: set[Handle] = set()
        self._activation_sequence: dict[Handle, int] = {}
        self._activations: list[ActivationObservation] = []
        self._labels: list[CalibrationLabel] = []
        self._label_keys = set()
        self._tuned_crystals: set[Handle] = set()
        self._unbound_tuned_reactors: set[Handle] = set()
        self._opportunities: dict[Handle, QueryOpportunity] = {}
        self._examples: dict[Handle, CalibrationQueryExample] = {}

    @staticmethod
    def _bindings(observation):
        objects = observed_objects(observation)
        handles = {str(handle): handle for handle in objects}
        result = {}
        for crystal, obj in objects.items():
            if not obj.get("name", "").startswith("quantum crystal"):
                continue
            parent = re.search(r"\(in [^\n]*?\[uuid:\s*(\d+)\]", obj.get("description", ""))
            reactor = handles.get(parent.group(1)) if parent else None
            name = objects.get(reactor, {}).get("name", "")
            if name.startswith("crystal reactor"):
                result[crystal] = (reactor, name == "crystal reactor (activated)")
            elif not parent or reactor is not None:
                result[crystal] = (reactor, False)
            else:
                result[crystal] = (reactor, None)  # Parent state is not currently visible.
        return result

    def _ingest(self, observation, *, initial=False):
        bindings = self._bindings(observation)
        for crystal, (reactor, active) in bindings.items():
            previous = self._active.get(crystal)
            if active:
                if not initial and previous != (reactor, True):
                    kind = "restoration" if crystal in self._ever_active else \
                        "new_activation" if previous is not None and previous[1] is False else "first_observed_active"
                    self._activations.append(ActivationObservation(self.sequence, crystal, reactor, kind))
                    if kind == "new_activation":
                        self._activation_sequence.setdefault(crystal, self.sequence)
                self._ever_active.add(crystal)
            self._active[crystal] = (reactor, active)
        self._observation = deepcopy(observation)
        return bindings

    def observe_initial(self, observation: dict):
        self.measurements.observe_initial(observation)
        self._ingest(observation, initial=True)

    def query(self, object_handle: Handle) -> CrystalQuery:
        records = tuple(record for record in self.measurements.records
                        if record.kind == "instrument" and record.target_handle == object_handle)
        quantities = dict.fromkeys((*FEATURE_QUANTITIES, *(record.quantity for record in records)))
        summaries = self.measurements.summaries()
        features = {quantity: tuple(summary for summary in summaries
                                     if summary.target_handle == object_handle and summary.quantity == quantity)
                    for quantity in quantities}
        return deepcopy(CrystalQuery(object_handle, self.measurements.objects.get(object_handle, {}).get("name", ""),
                                    self.sequence, features, {key: bool(value) for key, value in features.items()}, records))

    def current_dialog(self) -> dict | None:
        """Current public control binding; no remembered location or stale Hz."""
        dialog = self._observation.get("ui", {}).get("dialog_box", {})
        owner = self.measurements.dialog_owner
        if owner is None or not dialog.get("is_in_dialog") or not dialog.get("dialogOptions"):
            return None
        match = _FREQUENCY.search(dialog.get("dialogIn", ""))
        bindings = self._bindings(self._observation)
        crystals = [crystal for crystal, (reactor, _) in bindings.items() if reactor == owner]
        if match is None or len(crystals) != 1:
            return None
        crystal = crystals[0]
        return {"reactor_handle": owner, "object_handle": crystal, "frequency_hz": float(match.group(1)),
                "unit": "Hz", "activated": bindings[crystal][1], "available_after_sequence": self.sequence}

    def support(self, required_quantities=None) -> tuple[CalibrationSupport, ...]:
        required = self.required_quantities if required_quantities is None else tuple(required_quantities)
        result, used = [], set()
        for label in self._labels:
            if label.object_handle in used:
                continue  # Repeated working settings are not independent crystals.
            query = self.query(label.object_handle)
            if not query.records or not all(query.presence.get(quantity, False) for quantity in required):
                continue
            first = [min(row.sequence for row in query.records if row.quantity == quantity) for quantity in required]
            if not required:
                first = [min(row.sequence for row in query.records)]
            result.append(CalibrationSupport(label, query, max(label.received_sequence, *first), self.sequence))
            used.add(label.object_handle)
        return deepcopy(tuple(result))

    def _record_intervention(self, action, result):
        if result.get("success") is not True or "chosen_dialog_option_int" not in action:
            return
        dialog = self._observation.get("ui", {}).get("dialog_box", {})
        options = dialog.get("dialogOptions", {})
        choice = action["chosen_dialog_option_int"]
        option = options.get(choice, options.get(str(choice), ""))
        if not re.search(r"(?:increase|decrease)\s+frequency", str(option), re.I):
            return
        owner = self.measurements.dialog_owner
        crystals = [crystal for crystal, (parent, _) in self._bindings(self._observation).items() if parent == owner]
        if owner is not None and len(crystals) == 1:
            self._tuned_crystals.add(crystals[0])
        elif owner is not None:
            self._unbound_tuned_reactors.add(owner)

    def observe(self, action: dict, result: dict, next_observation: dict, *, sequence: int | None = None):
        sequence = self.sequence + 1 if sequence is None else sequence
        if sequence <= self.sequence:
            raise ValueError("Calibration events must arrive in increasing sequence order")
        self._record_intervention(action, result)
        self.sequence = sequence
        records = self.measurements.observe(action, result, next_observation, sequence=sequence)
        bindings = self._ingest(next_observation)
        new_labels = []
        for record in records:
            if record.kind != "setting" or not record.bound or record.quantity != "resonance_frequency_hz":
                continue
            candidates = [crystal for crystal in record.contained_objects if crystal in bindings
                          and bindings[crystal][0] == record.target_handle]
            if len(candidates) != 1:
                continue
            crystal = candidates[0]
            if bindings[crystal][1] is True:
                key = (crystal, record.value)
                if key not in self._label_keys:
                    label = CalibrationLabel(crystal, record.target_handle, record.value, "Hz", sequence,
                                             self._activation_sequence.get(crystal), deepcopy(record))
                    self._labels.append(label)
                    self._label_keys.add(key)
                    new_labels.append(label)
                    if crystal in self._opportunities and crystal not in self._examples:
                        self._examples[crystal] = CalibrationQueryExample(self._opportunities[crystal], label)
            elif bindings[crystal][1] is False and crystal not in self._ever_active \
                    and crystal not in self._opportunities and action.get("action") in ("TALK", "USE"):
                query = self.query(crystal)
                if all(query.presence.get(quantity, False) for quantity in self.required_quantities):
                    clean = crystal not in self._tuned_crystals and record.target_handle not in self._unbound_tuned_reactors
                    self._opportunities[crystal] = QueryOpportunity(sequence + 1, record.target_handle, query, self.support(),
                                                                  clean, None if clean else "prior_frequency_intervention")
        return deepcopy(tuple(new_labels))

    def append_transition(self, transition: dict):
        self.observe_initial(transition["observation"])
        return self.observe(transition["action"], transition["action_result"], transition["next_observation"],
                            sequence=transition.get("index"))

    @property
    def labels(self):
        return deepcopy(tuple(self._labels))

    @property
    def query_examples(self):
        return deepcopy(tuple(self._examples.values()))

    def snapshot(self) -> dict:
        crystals = [handle for handle, obj in self.measurements.objects.items()
                    if obj.get("name", "").startswith("quantum crystal")]
        return {"available_after_sequence": self.sequence, "queries": tuple(self.query(handle) for handle in crystals),
                "support": self.support(), "labels": self.labels, "query_opportunities": deepcopy(tuple(self._opportunities.values())),
                "query_examples": self.query_examples, "activations": deepcopy(tuple(self._activations))}
