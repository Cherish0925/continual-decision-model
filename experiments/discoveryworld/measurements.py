"""Records of measurements actually returned through the public native API.

This module extracts and groups observations. It selects no actions, fits no
scientific hypothesis, and defines no sampling/stopping rule. Object handles
serve only as routing/equality keys. Repeated successful measurements, even
with identical text and values, remain separate records.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, replace
import json
import re

from .reference import INSTRUMENTS, observed_objects, parse_measurement

_UNITS = {"density_g_cm3": "g/cm^3", "temperature_c": "degC",
          "radiation_usv_h": "uSv/h", "quantum_gap_nm": "nm"}
_FREQUENCY = re.compile(r"current resonance frequenc[ey] is:\s*([-+]?\d+(?:\.\d+)?)\s*Hertz", re.I)
_SAID = re.compile(r"(?:^|\n)([^\n]+?) said:")
NUMERIC_COLUMNS = ("count", "minimum", "maximum", "mean", "latest", "range", "last_change", "sequence_span")


def _conditions(observation: dict) -> dict:
    """Observed spatial/object conditions, excluding clocks and result text.

    Equal keys mean equal recorded public UI conditions, not a guarantee that
    every physical influence is observed or that samples are independent.
    Handles in this structure identify objects; they are not numeric features.
    """
    ui = observation["ui"]
    location = ui.get("agentLocation", {})

    def obj(item):
        return {key: item[key] for key in ("uuid", "name", "description", "distance") if key in item}

    def ordered(items):
        return sorted((obj(item) for item in items), key=lambda item: json.dumps(item, sort_keys=True))

    nearby = ui.get("nearbyObjects", {}).get("objects", {})
    return {"scope": "public_ui", "agent_location": {key: location[key] for key in ("x", "y", "faceDirection") if key in location},
            "inventory": ordered(ui.get("inventoryObjects", [])),
            "accessible": ordered(ui.get("accessibleEnvironmentObjects", [])),
            "nearby": {direction: ordered(items) for direction, items in sorted(nearby.items())}}


@dataclass(frozen=True)
class Measurement:
    sequence: int
    world_step: int | None
    kind: str
    instrument_handle: int | str | None
    instrument_name: str
    target_handle: int | str | None
    target_name: str
    quantity: str
    value: float
    unit: str | None
    text: str
    text_source: str
    action: dict
    conditions: dict
    contained_objects: tuple[int | str, ...] = ()

    @property
    def bound(self) -> bool:
        return self.instrument_handle is not None and self.target_handle is not None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class MeasurementSummary:
    instrument_handle: int | str
    instrument_name: str
    target_handle: int | str
    target_name: str
    quantity: str
    unit: str | None
    conditions: dict
    first_sequence: int
    last_sequence: int
    count: int
    minimum: float
    maximum: float
    mean: float
    first: float
    latest: float
    last_change: float
    # Historical public contents at measurement time; names/descriptions are
    # retained in conditions, not inferred from the object's current location.
    contained_objects: tuple[int | str, ...] = ()

    def numeric_features(self) -> dict[str, float]:
        """Only observed values/counts: no object-handle magnitudes or hashes."""
        return {"count": float(self.count), "minimum": self.minimum, "maximum": self.maximum,
                "mean": self.mean, "range": self.maximum - self.minimum,
                "first": self.first, "latest": self.latest,
                "total_change": self.latest - self.first, "last_change": self.last_change,
                "sequence_span": float(self.last_sequence - self.first_sequence)}

    def numeric_row(self) -> tuple[float, ...]:
        """Eight raw numeric columns in NUMERIC_COLUMNS order, with no handles."""
        features = self.numeric_features()
        return tuple(features[key] for key in NUMERIC_COLUMNS)

    def text(self) -> str:
        """Semantic labels and statistics for the existing frozen text encoder."""
        location = self.conditions.get("agent_location", {})
        return (f"{self.instrument_name} measuring {self.target_name}: {self.quantity}; "
                f"unit {self.unit or 'not specified'}; count {self.count}; "
                f"min {self.minimum:g}; max {self.maximum:g}; mean {self.mean:g}; latest {self.latest:g}; "
                f"first sequence {self.first_sequence}; last sequence {self.last_sequence}; "
                f"observed agent location {location.get('x', 'unknown')}, {location.get('y', 'unknown')}")


class MeasurementLog:
    """One episode's public object bindings, measurement records and summaries.

    ``observe_initial`` supplies an optional initial public snapshot.
    ``observe(action, action_result, next_observation)`` consumes an executed
    action and its received result. ``append_transition`` replays raw adapter
    trajectories; it reads no expert annotations, scorecard or teacher ledger.
    An unbound dialogue reading is retained but excluded from grouped summaries.
    """

    def __init__(self):
        self.records: list[Measurement] = []
        self.objects: dict[int | str, dict] = {}
        self.dialog_owner: int | str | None = None
        self.sequence = 0
        self._groups: dict[str, MeasurementSummary] = {}

    def observe_initial(self, observation: dict):
        self.objects.update(deepcopy(observed_objects(observation)))

    def append_transition(self, transition: dict) -> list[Measurement]:
        self.observe_initial(transition["observation"])
        return self.observe(transition["action"], transition["action_result"],
                            transition["next_observation"], sequence=transition.get("index"))

    def _message_sources(self, result: dict, observation: dict):
        ui = observation["ui"]
        # A popup may outlive its originating action. Never search an old
        # extended popup after a fresh action message reports no measurement.
        for source, container, key in (("action_result.message", result, "message"),
                                       ("action_result.successMessage", result, "successMessage"),
                                       ("ui.lastActionMessage", ui, "lastActionMessage"),
                                       ("ui.extended_action_message", ui, "extended_action_message")):
            if isinstance(container.get(key), str) and container[key]:
                yield source, container[key]
                return

    def _dialog_binding(self, action: dict, message: str):
        if action.get("action") == "TALK":
            return action.get("arg1")
        if action.get("action") == "USE":
            # The native USE handler can swap its arguments. Bind a newly
            # opened conversation to the publicly named speaker, not arg order.
            speakers = {speaker.group(1).strip() for speaker in _SAID.finditer(message)}
            if speakers:
                matches = [action[key] for key in ("arg1", "arg2")
                           if action.get(key) in self.objects and self.objects[action[key]].get("name") in speakers]
                if len(matches) == 1:
                    return matches[0]
            return None
        return self.dialog_owner if "chosen_dialog_option_int" in action else None

    def observe(self, action: dict, action_result: dict, next_observation: dict,
                *, sequence: int | None = None) -> list[Measurement]:
        sequence = self.sequence if sequence is None else sequence
        self.sequence = max(self.sequence, sequence + 1)
        self.observe_initial(next_observation)
        if action_result.get("success") is not True:
            return []
        ui = next_observation["ui"]
        conditions = _conditions(next_observation)
        messages = list(self._message_sources(action_result, next_observation))
        new = []

        def add(kind, instrument, target, quantity, value, unit, text, source, contained=()):
            new.append(Measurement(sequence, ui.get("world_steps"), kind, instrument,
                                   self.objects.get(instrument, {}).get("name", "dialog" if kind == "setting" else ""),
                                   target, self.objects.get(target, {}).get("name", ""), quantity,
                                   value, unit, text, source, deepcopy(action), deepcopy(conditions), contained))

        if action.get("action") == "USE":
            # A known instrument and a measurement-specific returned message
            # are both required. Ordinary USE success is not itself a reading.
            for tool_key, target_key in (("arg1", "arg2"), ("arg2", "arg1")):
                tool, target = action.get(tool_key), action.get(target_key)
                name = self.objects.get(tool, {}).get("name")
                if name not in INSTRUMENTS or target is None:
                    continue
                for source, message in messages:
                    try:
                        values = parse_measurement(name, message)
                    except ValueError:
                        continue
                    for quantity, value in values.items():
                        add("instrument", tool, target, quantity, value, _UNITS.get(quantity), message, source)
                    break  # The UI can duplicate this return in an extended popup.
                if new:
                    break

        dialog = ui.get("dialog_box", {})
        dialog_sources = [("ui.dialog_box.dialogIn", dialog["dialogIn"])] \
            if dialog.get("is_in_dialog") and dialog.get("dialogIn") else []
        dialog_sources += messages
        joined = "\n".join(message for _, message in messages)
        is_dialog_action = action.get("action") == "TALK" or "chosen_dialog_option_int" in action \
            or (action.get("action") == "USE" and _SAID.search(joined) is not None)
        if is_dialog_action:
            owner = self._dialog_binding(action, joined)
            for source, message in dialog_sources:
                match = _FREQUENCY.search(message)
                if match:
                    self.dialog_owner = owner
                    contained = tuple(obj["uuid"] for obj in observed_objects(next_observation).values()
                                      if owner is not None and f"[uuid: {owner}]" in obj.get("description", "")
                                      and obj.get("description", "").find("(in ") >= 0)
                    add("setting", owner, owner, "resonance_frequency_hz", float(match.group(1)),
                        "Hz", message, source, contained)
                    break
            else:
                if not dialog.get("dialogOptions"):
                    self.dialog_owner = None

        for record in new:
            self.records.append(record)
            if record.bound:
                key = json.dumps([record.instrument_handle, record.target_handle, record.quantity,
                                  record.unit, record.conditions,
                                  sorted(record.contained_objects, key=json.dumps)], sort_keys=True)
                previous = self._groups.get(key)
                if previous is None:
                    self._groups[key] = MeasurementSummary(record.instrument_handle, record.instrument_name,
                                                          record.target_handle, record.target_name,
                                                          record.quantity, record.unit, record.conditions,
                                                          record.sequence, record.sequence, 1,
                                                          record.value, record.value, record.value,
                                                          record.value, record.value, 0.0,
                                                          contained_objects=record.contained_objects)
                else:
                    count = previous.count + 1
                    self._groups[key] = replace(previous, count=count, last_sequence=record.sequence,
                                               minimum=min(previous.minimum, record.value),
                                               maximum=max(previous.maximum, record.value),
                                               mean=previous.mean + (record.value - previous.mean) / count,
                                               latest=record.value, last_change=record.value - previous.latest)
        return new

    def summaries(self) -> list[MeasurementSummary]:
        return deepcopy(list(self._groups.values()))
