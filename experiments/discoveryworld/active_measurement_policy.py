"""Execute selected measurements and setting experiments using public actions.

The learned policy chooses a request. This operator handles its native action
sequence and returns the received evidence and cost. It makes no measurement
selection and receives no simulator internals or task score.
"""
from __future__ import annotations

from copy import deepcopy
import re

from .calibration import CalibrationLog
from .reference import INSTRUMENTS, current_frequency, observed_objects
from .quantitative_policy import scalar_dialog_action
from .scalar_probe import ScalarProbe, public_scalar_control


class NativeExperimentOperator:
    """One pending request, explicit public observations, and received facts.

    ``request`` accepts ``measure`` (group_index, target_handle),
    ``read_setting`` (reactor_handle), or ``test`` (target_handle,
    reactor_handle, mean_hz, std_hz). Call ``next_action`` on each observation,
    execute its returned action, then pass the actual transition to ``observe``.
    A None action means the request ended; ``completed[-1]`` contains its result.
    The five instrument groups follow reference.INSTRUMENTS. Spectrometry
    returns five channels together in one USE action.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.calibration = CalibrationLog()
        self.observation = None
        self.objects = {}
        self.pending = None
        self.program = None
        self.completed = []
        self._actions = 0
        self._first_record = 0
        self._last_transition = None
        self._isolated = None

    def request(self, request):
        if self.pending is not None:
            raise RuntimeError("Finish the current experimental request before selecting another")
        if request["kind"] not in ("measure", "read_setting", "test"):
            raise ValueError("Unknown experimental request")
        self.pending = deepcopy(request)
        self.program = None
        self._actions = 0
        self._first_record = len(self.calibration.measurements.records)

    def _ingest(self, observation):
        self.observation = observation
        self.objects.update(deepcopy(observed_objects(observation)))
        self.calibration.observe_initial(observation)

    def next_action(self, observation):
        self._ingest(observation)
        if self.pending is None:
            return None
        if self.program is None:
            self.program = self._execute(self.pending)
        try:
            return next(self.program)
        except StopIteration as done:
            self._complete(done.value or {})
            return None

    def _complete(self, result):
        self.completed.append({"request": self.pending, "actions": self._actions,
                               "end_sequence": self.calibration.sequence,
                               "measurements": [r.to_dict() for r in
                                                self.calibration.measurements.records[self._first_record:]],
                               **result})
        self.pending = self.program = None

    def finish_at_environment_end(self):
        """Record the public result when the caller receives native termination."""
        if self.pending is None:
            return
        dialog = self.calibration.current_dialog()
        activated = bool(dialog is not None and dialog["activated"]
                         and dialog["object_handle"] == self.pending.get("target_handle"))
        self._complete({"success": activated, "ending": "environment_ended",
                        "setting": deepcopy(dialog)})

    def observe(self, transition):
        self.calibration.append_transition(transition)
        self.objects.update(deepcopy(observed_objects(transition["next_observation"])))
        self.observation = transition["next_observation"]
        self._last_transition = transition
        if self.pending is not None:
            self._actions += 1

    def _action(self, kind, first=None, second=None):
        if kind not in self.observation["actions"]:
            raise ValueError(f"Undocumented native action: {kind}")
        action = {"action": kind}
        if first is not None:
            action["arg1"] = first
        if second is not None:
            action["arg2"] = second
        yield action

    def _exit_dialog(self):
        dialog = self.observation["ui"].get("dialog_box", {})
        if dialog.get("is_in_dialog") and dialog.get("dialogOptions"):
            choices = [(key, text) for key, text in dialog["dialogOptions"].items()
                       if str(text).strip().casefold() == "exit"]
            if not choices:
                raise ValueError("Current dialogue has no public exit option")
            yield {"chosen_dialog_option_int": int(choices[0][0])}

    def _approach(self, handle):
        if handle not in self.objects:
            raise ValueError("An experimental target must have been publicly observed")
        ui = self.observation["ui"]
        reachable = {o["uuid"] for field in ("accessibleEnvironmentObjects", "inventoryObjects")
                     for o in ui.get(field, [])}
        if handle not in reachable:
            yield from self._action("TELEPORT_TO_OBJECT", handle)

    def _instrument(self, group):
        name = INSTRUMENTS[group]
        inventory = self.observation["ui"].get("inventoryObjects", [])
        held = next((o["uuid"] for o in inventory if o["name"] == name), None)
        if held is not None:
            return held
        known = next((handle for handle, obj in self.objects.items() if obj["name"] == name), None)
        if known is None:
            locations = [name for name in self.observation["teleport_locations"] if "instruments" in name]
            if len(locations) != 1:
                raise ValueError("No unique public instrument location")
            yield from self._action("TELEPORT_TO_LOCATION", locations[0])
            known = next((handle for handle, obj in self.objects.items() if obj["name"] == name), None)
        if known is None:
            raise ValueError(f"Instrument {name} was not present in the received observation")
        yield from self._approach(known)
        yield from self._action("PICKUP", known)
        return known

    def _restore_isolated(self):
        if self._isolated is None:
            return
        target, container = self._isolated
        yield from self._approach(container)
        yield from self._action("PUT", target, container)
        self._isolated = None

    def _isolate(self, target):
        if self._isolated is not None and self._isolated[0] == target:
            return
        yield from self._restore_isolated()
        yield from self._approach(target)
        description = self.objects[target].get("description", "")
        parent = re.search(r"\((?:in|on) [^\n]*?\[uuid:\s*(\d+)\]", description)
        handles = {str(handle): handle for handle in self.objects}
        container = handles.get(parent.group(1)) if parent else None
        if container is None:
            raise ValueError("Radiation isolation needs a publicly identified return container")
        yield from self._action("PICKUP", target)
        location = next(name for name in self.observation["teleport_locations"] if "instruments" in name)
        yield from self._action("TELEPORT_TO_LOCATION", location)
        nearby = self.observation["ui"]["nearbyObjects"]["objects"]
        grasses = [(obj["distance"], direction, obj["uuid"])
                   for direction, objects in nearby.items() for obj in objects
                   if obj["name"] == "grass" and direction != "same_location"]
        if not grasses:
            raise ValueError("No observed clearing for the radiation measurement")
        grass = min(grasses, key=lambda item: item[:2])[2]
        yield from self._action("TELEPORT_TO_OBJECT", grass)
        locations = list(self.observation["teleport_locations"].values())
        visited = {grass}
        for _ in range(24):
            position = self.observation["ui"]["agentLocation"]
            distance = min(max(abs(position["x"] - place["gridX"]),
                               abs(position["y"] - place["gridY"])) for place in locations)
            if distance >= 9:
                self._isolated = (target, container)
                return
            choices = []
            for direction, objects in self.observation["ui"]["nearbyObjects"]["objects"].items():
                if direction == "same_location":
                    continue
                axes = direction.split("-")
                for obj in objects:
                    if obj["name"] != "grass" or obj["uuid"] in visited:
                        continue
                    span = obj["distance"] / len(axes)
                    x = position["x"] + span * (("east" in axes) - ("west" in axes))
                    y = position["y"] + span * (("south" in axes) - ("north" in axes))
                    separation = min(max(abs(x - place["gridX"]), abs(y - place["gridY"]))
                                     for place in locations)
                    choices.append((separation, obj["distance"], obj["uuid"]))
            if not choices:
                raise ValueError("No unvisited public clearing for the radiation measurement")
            grass = max(choices, key=lambda item: item[:2])[2]
            visited.add(grass)
            yield from self._action("TELEPORT_TO_OBJECT", grass)
        raise ValueError("Could not isolate the crystal from observed lab locations")

    def _execute(self, request):
        yield from self._exit_dialog()
        is_radiation = request["kind"] == "measure" and int(request["group_index"]) == 2
        if self._isolated is not None and not (is_radiation and self._isolated[0] == request["target_handle"]):
            yield from self._restore_isolated()
        if request["kind"] == "measure":
            group = int(request["group_index"])
            if group not in range(len(INSTRUMENTS)):
                raise ValueError("A measurement selects one of the five instrument groups")
            instrument = yield from self._instrument(group)
            target = request["target_handle"]
            if is_radiation:
                yield from self._isolate(target)
            else:
                yield from self._approach(target)
            yield from self._action("USE", instrument, target)
            return {"success": bool(self._last_transition["action_result"].get("success")),
                    "instrument": INSTRUMENTS[group], "target_handle": target,
                    "measurement_condition": "isolated_single_return" if is_radiation else "native_in_place"}
        reactor = request["reactor_handle"]
        if request["kind"] == "read_setting":
            yield from self._approach(reactor)
            yield from self._action("TALK", reactor)
            dialog = self.calibration.current_dialog()
            ui = self.observation["ui"]["dialog_box"]
            control = public_scalar_control(ui)
            result = {"success": dialog is not None, "setting": deepcopy(dialog),
                      "frequency_hz": current_frequency(ui),
                      "control": None if control is None else {
                          "lower_hz": float(control.lower_hz), "upper_hz": float(control.upper_hz),
                          "minimum_step_hz": control.minimum_step_hz}}
            yield from self._exit_dialog()
            return result
        target = request["target_handle"]
        binding = self.calibration._bindings(self.observation).get(target)
        if binding is None or binding[0] != reactor:
            yield from self._approach(target)
            inventory = {o["uuid"] for o in self.observation["ui"].get("inventoryObjects", [])}
            if target not in inventory:
                yield from self._action("PICKUP", target)
            yield from self._approach(reactor)
            yield from self._action("PUT", target, reactor)
        else:
            yield from self._approach(reactor)
        yield from self._action("TALK", reactor)
        mean, spread = float(request["mean_hz"]), float(request.get("std_hz", 0.))
        probe = ScalarProbe()
        attempts = []
        while True:
            dialog = self.calibration.current_dialog()
            if dialog is None or dialog["object_handle"] != target or dialog["reactor_handle"] != reactor:
                return {"success": False, "ending": "dialogue_changed", "attempts": attempts}
            attempts.append({"frequency_hz": dialog["frequency_hz"], "activated": dialog["activated"]})
            if dialog["activated"]:
                result = {"success": True, "ending": "observed_activation", "attempts": attempts}
                yield from self._exit_dialog()
                return result
            ui = self.observation["ui"]["dialog_box"]
            current = dialog["frequency_hz"]
            action = scalar_dialog_action(ui, current, mean)
            options = ui.get("dialogOptions", {})
            key = (action or {}).get("chosen_dialog_option_int")
            label = str(options.get(str(key), options.get(key, ""))).strip().casefold()
            if label == "exit" and not probe.started:
                probe.observe_attempt(current, activated=False, point_attempt=True)
            if probe.started:
                proposed = probe.next_target(current, mean, spread, ui)
                if proposed is None:
                    result = {"success": False, "ending": "local_experiment_exhausted", "attempts": attempts}
                    yield from self._exit_dialog()
                    return result
                action = scalar_dialog_action(ui, current, proposed)
            if action is None:
                raise ValueError("No native setting action for the requested experiment")
            yield action
            latest = self.calibration.current_dialog()
            if probe.started and latest is not None:
                probe.observe_attempt(latest["frequency_hz"], activated=latest["activated"])
