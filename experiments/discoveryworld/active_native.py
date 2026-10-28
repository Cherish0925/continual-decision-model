"""From-reset native execution of measurements selected under incomplete evidence.

Public navigation and precise scalar adjustment are fixed execution tools. The
controller chooses a measurement (object and instrument) or a test of a currently
available numerical hypothesis. It receives no scenario seed or evaluator score.
The structured input path does not claim to evaluate the earlier visual policy.
"""
from __future__ import annotations

from copy import deepcopy

import torch

from .active_measurement_policy import NativeExperimentOperator
from .calibration import FEATURE_QUANTITIES


class NativeActiveEpisode:
    """Four public objects: two initial supports followed by two task targets."""

    def __init__(self, env, writer=None):
        self.env, self.writer = env, writer
        self.operator = NativeExperimentOperator()
        self.objects = []
        self.reactors = []
        self.controls = []
        self.values = torch.zeros(1, 4, 9, dtype=torch.float64)
        self.counts = torch.zeros(1, 4, 9, dtype=torch.long)
        self.known_hz = torch.zeros(1, 4, dtype=torch.float64)
        self.known = torch.zeros(1, 4, dtype=torch.bool)
        self.current_hz = torch.zeros(1, 4, dtype=torch.float64)
        self.query = torch.tensor([2], dtype=torch.long)
        self.failed = torch.zeros(1, 9, dtype=torch.bool)
        self.decisions = []
        self.completed_targets = set()
        self._record_cursor = 0
        self.bootstrap_actions = 0

    @property
    def seen(self):
        return self.counts > 0

    def _step(self, action):
        transition = self.env.step(action)
        self.operator.observe(transition)
        if self.writer is not None:
            self.writer.append(transition)
        return transition

    def execute(self, request):
        self.operator.request(request)
        while not self.env.ended:
            action = self.operator.next_action(self.env.policy_input())
            if action is None:
                break
            self._step(action)
        if self.env.ended:
            self.operator.finish_at_environment_end()
        self._refresh()
        return self.operator.completed[-1]

    def bootstrap(self):
        """Locate publicly listed stations and read initially working devices."""
        observation = self.env.policy_input()
        self.operator.next_action(observation)
        for fragment in ("instruments", "crystal bench", "reactor lab"):
            locations = [name for name in observation["teleport_locations"] if fragment in name]
            if len(locations) != 1:
                raise ValueError(f"Expected one public {fragment} location")
            self._step({"action": "TELEPORT_TO_LOCATION", "arg1": locations[0]})
            observation = self.env.policy_input()
            self.operator.next_action(observation)
        delivered = self.operator.objects
        working = [handle for handle, obj in delivered.items()
                   if obj["name"] == "crystal reactor (activated)"]
        empty = [handle for handle, obj in delivered.items() if "no crystal present" in obj["name"]]
        targets = [handle for handle, obj in delivered.items()
                   if obj["name"].startswith("quantum crystal") and "(on table" in obj["description"]]
        if len(working) != 2 or len(empty) != 2 or len(targets) != 2:
            raise ValueError("This native task interface expects two working and two uncalibrated devices")
        supports, frequencies, controls = [], [], []
        for reactor in working:
            result = self.execute({"kind": "read_setting", "reactor_handle": reactor})
            setting = result["setting"]
            if setting is None or setting["activated"] is not True:
                raise ValueError("An initially working device did not return a public setting")
            supports.append(setting["object_handle"])
            frequencies.append(setting["frequency_hz"])
            controls.append(result["control"])
        self.objects, self.reactors = supports + targets, working + empty
        self.known[0, :2] = True
        self.known_hz[0, :2] = torch.tensor(frequencies)
        self.current_hz[0, :2] = self.known_hz[0, :2]
        for index, reactor in enumerate(empty, start=2):
            result = self.execute({"kind": "read_setting", "reactor_handle": reactor})
            self.current_hz[0, index] = result["frequency_hz"]
            controls.append(result["control"])
        self.controls = controls
        self.bootstrap_actions = self.env.steps
        self._refresh()
        return self

    def _refresh(self):
        if not self.objects:
            return
        records = self.operator.calibration.measurements.records
        for row in records[self._record_cursor:]:
            if row.kind != "instrument" or row.target_handle not in self.objects:
                continue
            if row.quantity not in FEATURE_QUANTITIES:
                continue
            obj = self.objects.index(row.target_handle)
            quantity = FEATURE_QUANTITIES.index(row.quantity)
            # This lower envelope includes only samples requested and returned
            # so far. Every repeat remains a separately paid native action.
            if quantity == 2 and self.counts[0, obj, quantity] > 0:
                self.values[0, obj, quantity] = min(float(self.values[0, obj, quantity]), row.value)
            else:
                count = int(self.counts[0, obj, quantity])
                self.values[0, obj, quantity] = (self.values[0, obj, quantity] * count + row.value) / (count + 1)
            self.counts[0, obj, quantity] += 1
        self._record_cursor = len(records)
        for label in self.operator.calibration.labels:
            if label.object_handle in self.objects:
                obj = self.objects.index(label.object_handle)
                self.known[0, obj] = True
                self.known_hz[0, obj] = label.frequency_hz
        dialog = self.operator.calibration.current_dialog()
        if dialog is not None and dialog["object_handle"] in self.objects:
            self.current_hz[0, self.objects.index(dialog["object_handle"]) ] = dialog["frequency_hz"]

    def apply(self, action_index, *, mean_hz=None, std_hz=None):
        """Twenty object/instrument requests then nine current-target tests."""
        action_index = int(action_index)
        query = int(self.query[0])
        if action_index < 20:
            obj, group = divmod(action_index, 5)
            request = {"kind": "measure", "group_index": group,
                       "target_handle": self.objects[obj]}
        else:
            hypothesis = action_index - 20
            if hypothesis not in range(9) or mean_hz is None or std_hz is None:
                raise ValueError("A test needs an available hypothesis and visible-evidence prediction")
            request = {"kind": "test", "target_handle": self.objects[query],
                       "reactor_handle": self.reactors[query],
                       "mean_hz": float(mean_hz), "std_hz": float(std_hz)}
        start = self.env.steps
        result = self.execute(request)
        success = action_index >= 20 and bool(result["success"])
        if success:
            self.completed_targets.add(query)
        if action_index >= 20 and not success:
            self.failed[0, action_index - 20] = True
        elif action_index < 20:
            self.failed.zero_()
        if success and query == 2:
            self.query[0] = 3
            self.failed.zero_()
        decision = {"decision": len(self.decisions), "action_index": action_index,
                    "query_index": query, "native_start": start, "native_actions": self.env.steps - start,
                    "success": bool(success), "result": deepcopy(result)}
        self.decisions.append(decision)
        return decision

    def tensors(self, device):
        return {key: getattr(self, key).to(device) for key in
                ("values", "counts", "seen", "known_hz", "known", "current_hz", "query", "failed")}

    def summary(self):
        return {"native_actions": self.env.steps, "bootstrap_actions": self.bootstrap_actions,
                "completed_successfully": all(row["completed_successfully"] for row in self.env.metrics),
                "progress": sum(row["progress"] for row in self.env.metrics) / len(self.env.metrics),
                "measurement_requests": sum(row["action_index"] < 20 for row in self.decisions),
                "test_requests": sum(row["action_index"] >= 20 for row in self.decisions),
                "returned_sample_counts": self.counts[0].tolist(),
                "activated_targets": len(self.completed_targets),
                "target_working_settings": int(self.known[0, 2:].sum()),
                "task_metrics": self.env.metrics, "decisions": self.decisions}
