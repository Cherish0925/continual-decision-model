"""Reactor Lab expert driven by the actual public interaction history.

The scientific prior is the same as reference.py: measure five instruments,
compare affine hypotheses calibrated by working crystals, then intervene. This
module separates a pure action recommendation from observing executed actions.
It receives neither the environment nor its seed, scorecard, or hidden objects.
It is a scripted correction teacher, not a learned scientific agent.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
import re

from .reference import (INSTRUMENTS, current_frequency, fit_hypotheses,
                        observed_objects, parse_measurement, tuning_choice)


class RecoveryError(RuntimeError):
    """The observed history does not support a reliable next expert action."""


def _dialog(observation):
    dialog = observation["ui"].get("dialog_box", {})
    # Upstream can retain the final message after the conversation has ended.
    return dialog if dialog.get("is_in_dialog") and dialog.get("dialogOptions") else None


def _action(name, first=None, second=None):
    action = {"action": name}
    if first is not None:
        action["arg1"] = first
    if second is not None:
        action["arg2"] = second
    return action


class RecoveryExpert:
    """Keep an evidence ledger from reset; recommend without advancing it.

    Current access comes from the latest observation. Remembered UUIDs permit
    navigation, but are not treated as currently accessible. Measurements are
    credited only to the actual successful USE and its returned message.
    """

    def __init__(self, initial_observation: dict):
        self.observation = deepcopy(initial_observation)
        self.objects = {}
        self.parents = {}
        self.homes = {}
        self.anchor_pairs = {}
        self.nominal = {}
        self.calibration = {}
        self.readings = {}
        self.raw_measurements = []
        self.radiation_batches = {}
        self.radiation_complete = {}
        self.restore = []
        self.visited_grass = set()
        self.dialog_owner = None
        self.dialog_history = []
        self.reactor_frequencies = {}
        self.hypotheses = []
        self.selected = None
        self.trials = []
        self.trial_indices = {}
        self.phase = "acquire_instruments"
        self.events = 0
        self.completed = False
        self.blocked = None
        self._ingest_objects(initial_observation)
        self._advance()

    def _ids(self, name):
        return [uuid for uuid, obj in self.objects.items() if obj["name"].startswith(name)]

    def _inventory(self, observation=None):
        return {o["uuid"] for o in (observation or self.observation)["ui"]["inventoryObjects"]}

    def _accessible(self):
        return {o["uuid"] for o in self.observation["ui"]["accessibleEnvironmentObjects"]}

    def _ingest_objects(self, observation):
        current = observed_objects(observation)
        inventory = self._inventory(observation)
        self.objects.update(deepcopy(current))
        for uuid, obj in current.items():
            parent = re.search(r"\[uuid: (\d+)\]", obj.get("description", ""))
            self.parents[uuid] = "inventory" if uuid in inventory else int(parent.group(1)) if parent else None
            if obj["name"].startswith("quantum crystal") and parent and uuid not in inventory:
                container = int(parent.group(1))
                self.homes.setdefault(uuid, container)
                if (len(self.anchor_pairs) < 2 and
                        current.get(container, {}).get("name") == "crystal reactor (activated)"):
                    self.anchor_pairs.setdefault(container, uuid)

    def _location_name(self, fragment):
        names = [n for n in self.observation["teleport_locations"] if fragment in n]
        if len(names) != 1:
            raise RecoveryError(f"Expected one public location containing {fragment!r}")
        return names[0]

    def _location_action(self, fragment):
        return _action("TELEPORT_TO_LOCATION", self._location_name(fragment))

    def _reach(self, uuid):
        if uuid in self._inventory() or uuid in self._accessible():
            return None
        if uuid not in self.objects:
            raise RecoveryError(f"Object {uuid} was not observed")
        return _action("TELEPORT_TO_OBJECT", uuid)

    def _hold(self, uuid):
        if uuid in self._inventory():
            return None
        return self._reach(uuid) or _action("PICKUP", uuid)

    def _place(self, uuid, container):
        if self.parents.get(uuid) == container:
            return None
        return self._hold(uuid) or self._reach(container) or _action("PUT", uuid, container)

    def _parking_table(self):
        for crystal in self._ids("quantum crystal"):
            home = self.homes.get(crystal)
            if self.objects.get(home, {}).get("name") == "table":
                return home
        raise RecoveryError("No observed laboratory table for temporarily parking a crystal")

    def _install(self, crystal, reactor):
        for other in self._ids("quantum crystal"):
            if other != crystal and self.parents.get(other) == reactor:
                return self._place(other, self._parking_table())
        return self._place(crystal, reactor)

    def _working(self, crystal, reactor):
        return (self.parents.get(crystal) == reactor and
                self.objects[reactor]["name"] == "crystal reactor (activated)")

    def _separation(self, observation=None):
        obs = observation or self.observation
        p = obs["ui"]["agentLocation"]
        return min(max(abs(p["x"] - location["gridX"]), abs(p["y"] - location["gridY"]))
                   for location in obs["teleport_locations"].values())

    def _radiation_key(self, crystal, observation=None):
        obs = observation or self.observation
        carried = self._inventory(obs) & set(self._ids("quantum crystal"))
        crystals = self._ids("quantum crystal")
        if (len(crystals) != 4 or carried != {crystal} or self._separation(obs) < 9 or
                any(self.homes.get(c) is None or self.parents.get(c) != self.homes[c]
                    for c in crystals if c != crystal)):
            return None
        # Movement within a qualified isolated area does not erase valid
        # replicates. Record each exact location separately; aggregate only the
        # same crystal/instrument with the other crystals still parked at home.
        return (crystal, self._instrument("radiation meter"),
                tuple((c, self.parents[c]) for c in crystals if c != crystal))

    def _navigate_clearing(self):
        p = self.observation["ui"]["agentLocation"]
        choices = []
        locations = self.observation["teleport_locations"].values()
        for direction, group in self.observation["ui"]["nearbyObjects"]["objects"].items():
            if direction == "same_location":
                continue
            axes = direction.split("-")
            for obj in group:
                if obj["name"] != "grass" or obj["uuid"] in self.visited_grass:
                    continue
                span = obj["distance"] / len(axes)
                x = p["x"] + span * (("east" in axes) - ("west" in axes))
                y = p["y"] + span * (("south" in axes) - ("north" in axes))
                separation = min(max(abs(x - q["gridX"]), abs(y - q["gridY"])) for q in locations)
                choices.append((separation, obj["distance"], obj["uuid"]))
        if not choices:
            return self._location_action("instruments")
        return _action("TELEPORT_TO_OBJECT", max(choices, key=lambda item: item[:2])[2])

    def _radio_action(self, crystal):
        # Learner moves are reconciled before starting/continuing a batch.
        for other in self._ids("quantum crystal"):
            if other not in self.homes:
                raise RecoveryError("A crystal has no observed original container")
            if other != crystal and self.parents.get(other) != self.homes.get(other):
                home = self.homes.get(other)
                if home is None:
                    raise RecoveryError("A displaced crystal has no observed original container")
                return self._place(other, home)
        action = self._hold(crystal)
        if action:
            return action
        if self._separation() < 9:
            return self._navigate_clearing()
        return _action("USE", self._instrument("radiation meter"), crystal)

    def _instrument(self, name):
        return next((uuid for uuid, obj in self.objects.items() if obj["name"] == name), None)

    def _exit(self):
        dialog = _dialog(self.observation)
        for key, label in dialog["dialogOptions"].items():
            if label.casefold() == "exit":
                return {"chosen_dialog_option_int": int(key)}
        raise RecoveryError("The current public dialog exposes no exit option")

    def _talk(self, reactor):
        return self._reach(reactor) or _action("TALK", reactor)

    def _calibrate_action(self):
        for reactor, crystal in self.anchor_pairs.items():
            if crystal in self.calibration:
                continue
            installed = self._install(crystal, reactor)
            if installed:
                return installed
            dialog = _dialog(self.observation)
            if not dialog or self.dialog_owner != reactor:
                return self._talk(reactor)
            if crystal not in self.nominal:
                raise RecoveryError("Working-pair history has no recoverable displayed frequency")
            option = tuning_choice(dialog, self.nominal[crystal])
            if option is not None:
                return {"chosen_dialog_option_int": option}
            raise RecoveryError("Restoring the observed calibration frequency did not activate its crystal")
        raise RecoveryError("Fewer than two publicly identified calibration crystals")

    def _assignments(self):
        pairs = dict(self.anchor_pairs)
        assigned = set(pairs.values())
        loose = [c for c in self._ids("quantum crystal") if c not in assigned]
        free = [r for r in self._ids("crystal reactor") if r not in pairs]
        if len(loose) != len(free):
            raise RecoveryError("Public crystal/reactor assignment is incomplete")
        pairs.update(zip(free, loose))
        return pairs

    def _target(self, crystal):
        if crystal in self.calibration:
            return self.calibration[crystal], None
        candidates = self.hypotheses if self.selected is None else [self.selected] + [
            h for h in self.hypotheses if h != self.selected]
        candidates = [h for h in candidates if 0 <= h.predict(self._values(crystal)) <= 10000]
        index = self.trial_indices.get(crystal, 0)
        if index >= len(candidates):
            raise RecoveryError("No remaining measured-coordinate hypothesis for this crystal")
        hypothesis = candidates[index]
        return hypothesis.predict(self._values(crystal)), hypothesis

    def _accept_hypothesis(self, hypothesis):
        if self.selected != hypothesis:
            # Indices refer to the old ordering. A newly supported relation
            # leads the next trial for every still-unresolved crystal, including
            # one whose working reactor the learner subsequently changes.
            self.trial_indices.clear()
        self.selected = hypothesis

    def _tune_action(self):
        for reactor, crystal in self._assignments().items():
            if self._working(crystal, reactor):
                continue
            installed = self._install(crystal, reactor)
            if installed:
                return installed
            dialog = _dialog(self.observation)
            if not dialog or self.dialog_owner != reactor:
                return self._talk(reactor)
            target, _ = self._target(crystal)
            option = tuning_choice(dialog, target)
            if option is None:
                raise RecoveryError("A tested prediction has not been incorporated by observe")
            return {"chosen_dialog_option_int": option}
        raise RecoveryError("All remembered reactors work but public task completion is not yet observed")

    def _values(self, crystal):
        return {key: value for values in self.readings.get(crystal, {}).values() for key, value in values.items()}

    def _advance(self):
        task = self.observation["ui"].get("taskProgress", [])
        self.completed = bool(task) and all(t["completed"] for t in task)
        if self.completed:
            self.phase = "complete"
            return
        if any(self._instrument(name) not in self._inventory() for name in INSTRUMENTS):
            self.phase = "acquire_instruments"
        elif len(self._ids("quantum crystal")) < 4 or len(self._ids("crystal reactor")) < 4:
            self.phase = "survey_laboratory"
        elif len(self.calibration) < 2:
            self.phase = "read_calibration"
        elif any(len(self.readings.get(c, {})) < len(INSTRUMENTS) for c in self._ids("quantum crystal")):
            self.phase = "measure_crystals"
        else:
            self.phase = "test_predictions"
            if not self.hypotheses:
                self.hypotheses = fit_hypotheses([(self._values(c), f) for c, f in self.calibration.items()])

    def _plan(self):
        for crystal in self.restore:
            action = self._place(crystal, self.homes[crystal])
            if action:
                return action
        if self.phase == "acquire_instruments":
            for name in INSTRUMENTS:
                uuid = self._instrument(name)
                if uuid is None:
                    return self._location_action("instruments")
                action = self._hold(uuid)
                if action:
                    return action
        elif self.phase == "survey_laboratory":
            return self._location_action("reactor lab" if len(self._ids("crystal reactor")) < 4 else "crystal bench")
        elif self.phase == "read_calibration":
            return self._calibrate_action()
        elif self.phase == "measure_crystals":
            for crystal in self._ids("quantum crystal"):
                for name in INSTRUMENTS:
                    if name in self.readings.get(crystal, {}):
                        continue
                    if name == "radiation meter":
                        return self._radio_action(crystal)
                    return self._reach(crystal) or _action("USE", self._instrument(name), crystal)
        elif self.phase == "test_predictions":
            return self._tune_action()
        raise RecoveryError(f"No action supported in phase {self.phase}")

    def recommend(self, observation: dict) -> dict:
        """Pure query: call observe for every actual transition before querying."""
        if observation["ui"]["world_steps"] != self.observation["ui"]["world_steps"]:
            raise RecoveryError("A public transition is missing; call observe before recommend")
        if self.completed or self.blocked:
            raise RecoveryError(self.blocked or "The public task is complete")
        dialog = _dialog(self.observation)
        # A foreign conversation must be closed before ordinary tool/navigation
        # actions. A desired calibration/tuning conversation stays open.
        if dialog:
            wanted = None
            if not any(self.parents.get(c) != self.homes[c] for c in self.restore):
                if self.phase == "read_calibration":
                    wanted = next((r for r, c in self.anchor_pairs.items() if c not in self.calibration), None)
                elif self.phase == "test_predictions":
                    wanted = next((r for r, c in self._assignments().items() if not self._working(c, r)), None)
            if wanted != self.dialog_owner:
                return self._exit()
        action = self._plan()
        if dialog and "chosen_dialog_option_int" not in action:
            # Installation/placement can still be needed at the desired reactor.
            return self._exit()
        return deepcopy(action)

    def observe(self, executed_action: dict, next_observation: dict, action_result: dict):
        """Incorporate one actual transition; unexecuted suggestions have no effect."""
        before = self.observation
        if next_observation["ui"]["world_steps"] != before["ui"]["world_steps"] + 1:
            raise RecoveryError("Expected one contiguous native world tick")
        success = bool(action_result.get("success"))
        name = executed_action.get("action")
        first, second = executed_action.get("arg1"), executed_action.get("arg2")
        instrument = self.objects.get(first, {}).get("name")
        sample_key = self._radiation_key(second, before) if instrument == "radiation meter" else None
        before_objects, before_parents = observed_objects(before), dict(self.parents)
        # Attribute an outcome to the prediction pending before this action;
        # neither new readings nor another activation may relabel that trial.
        predictions = {}
        for crystal in self._ids("quantum crystal"):
            if self.hypotheses and crystal not in self.calibration:
                try:
                    predictions[crystal] = self._target(crystal)
                except RecoveryError:
                    # Exhausted hypotheses can block a recommendation, but
                    # cannot erase an actual result, even for another crystal.
                    predictions[crystal] = (None, None)
        self.events += 1
        self.observation = deepcopy(next_observation)
        self._ingest_objects(next_observation)
        if success and name == "TELEPORT_TO_OBJECT" and self.objects.get(first, {}).get("name") == "grass":
            self.visited_grass.add(first)
        if success and name == "USE" and instrument in INSTRUMENTS and second in self._ids("quantum crystal"):
            message = next_observation["ui"]["lastActionMessage"]
            try:
                values = parse_measurement(instrument, message)
            except ValueError:
                values = None  # E.g. USE entered a device dialog instead of measuring.
            if values:
                record = {"event": self.events, "world_step": next_observation["ui"]["world_steps"],
                          "instrument": instrument, "crystal": second, "readings": values,
                          "position": {key: before["ui"]["agentLocation"][key] for key in ("x", "y")},
                          "message": message, "qualified": instrument != "radiation meter" or sample_key is not None}
                self.raw_measurements.append(record)
                if instrument != "radiation meter":
                    self.readings.setdefault(second, {})[instrument] = values
                elif sample_key is not None and second not in self.radiation_complete:
                    batch = self.radiation_batches.setdefault(sample_key, [])
                    batch.append(record)
                    if len(batch) == 32:
                        minimum = min(item["readings"]["radiation_usv_h"] for item in batch)
                        self.readings.setdefault(second, {})[instrument] = {"radiation_usv_h": minimum}
                        self.radiation_complete[second] = [item["event"] for item in batch]
                        self.restore.append(second)

        dialog = _dialog(next_observation)
        if dialog:
            if success and name in ("TALK", "USE"):
                owners = [uuid for uuid in (first, second) if self.objects.get(uuid, {}).get("name", "").startswith("crystal reactor")]
                self.dialog_owner = owners[0] if len(owners) == 1 else None
            if self.dialog_owner is not None:
                frequency = current_frequency(dialog)
                reactor = self.dialog_owner
                self.dialog_history.append({"event": self.events, "reactor": reactor, "frequency_hz": frequency})
                self.reactor_frequencies[reactor] = (frequency, self.events)
                anchor = self.anchor_pairs.get(reactor)
                if anchor is not None:
                    self.nominal.setdefault(anchor, frequency)
                    if self._working(anchor, reactor) and anchor not in self.calibration:
                        self.calibration[anchor] = frequency
        else:
            self.dialog_owner = None
        self.restore = [c for c in self.restore if self.parents.get(c) != self.homes[c]]
        self._observe_trials(before_objects, before_parents, predictions)
        self._advance()

    def _observe_trials(self, before_objects, before_parents, predictions):
        """Log native activation changes before even a terminal phase advance."""
        dialog = _dialog(self.observation)
        for reactor, obj in observed_objects(self.observation).items():
            if not obj["name"].startswith("crystal reactor") or reactor not in before_objects:
                continue
            activated = obj["name"] == "crystal reactor (activated)"
            was_activated = before_objects[reactor]["name"] == "crystal reactor (activated)"
            changed = activated != was_activated
            contained = [c for c in self._ids("quantum crystal") if self.parents.get(c) == reactor]
            if not contained and changed:
                contained = [c for c in self._ids("quantum crystal") if before_parents.get(c) == reactor]
            crystal = contained[0] if len(contained) == 1 else None
            target, hypothesis = predictions.get(crystal, (None, None))
            last_frequency, frequency_event = self.reactor_frequencies.get(reactor, (None, None))
            applied = last_frequency if frequency_event == self.events else None
            current_dialog = dialog if self.dialog_owner == reactor else None
            reached = (current_dialog is not None and target is not None and
                       self.parents.get(crystal) == reactor and
                       tuning_choice(current_dialog, target) is None)
            if not changed and not (self.phase == "test_predictions" and reached and not activated):
                continue
            # Reactor Lab's native activation tolerance is strictly <2 Hz.
            # Activation elsewhere on the route is an observed success, but
            # does not support this prediction. A cached, older display alone
            # also cannot establish the frequency of a new activation.
            consistent = target is not None and applied is not None and abs(applied - target) < 2.0
            supported = activated and hypothesis is not None and consistent
            self.trials.append({"event": self.events,
                                "world_step": self.observation["ui"]["world_steps"],
                                "reactor": reactor, "crystal": crystal,
                                "hypothesis": asdict(hypothesis) if hypothesis is not None else None,
                                "prediction_hz": target, "applied_hz": applied,
                                "last_observed_frequency_hz": last_frequency,
                                "frequency_observed_event": frequency_event,
                                "previously_activated": was_activated, "activated": activated,
                                "activation_changed": changed, "prediction_reached": reached,
                                "prediction_consistent": consistent, "supports_hypothesis": supported})
            if supported:
                self._accept_hypothesis(hypothesis)
            elif not activated and reached:
                self.trial_indices[crystal] = self.trial_indices.get(crystal, 0) + 1

    def summary(self):
        return {"kind": "public_history_recovery_expert", "is_learned_policy": False,
                "phase": self.phase, "events": self.events, "completed": self.completed,
                "blocked": self.blocked, "known_objects": len(self.objects),
                "calibrations": {str(k): v for k, v in self.calibration.items()},
                "measurement_events": len(self.raw_measurements),
                "measured_coordinates": sum(len(self._values(c)) for c in self.readings),
                "radiation_completed": {str(k): v for k, v in self.radiation_complete.items()},
                "measurements": deepcopy(self.raw_measurements), "trials": deepcopy(self.trials)}
