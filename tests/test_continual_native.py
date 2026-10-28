"""Check native consumer feedback and task boundaries without simulator sampling."""
import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from experiments.discoveryworld import evaluate_active_stream as native
from experiments.discoveryworld.active_native import NativeActiveEpisode
from experiments.discoveryworld.calibration import FEATURE_QUANTITIES
from experiments.discoveryworld.calibration_mixture import AffineMixture


class ScriptedEnvironment:
    def __init__(self, output, max_actions):
        self.steps, self.ended = 0, False
        self.configuration, self.actions, self.teleports = {}, {}, {}
        self.metrics = [{"completed": False, "completed_successfully": False, "progress": 0.}]

    def reset(self, scenario, difficulty, seed):
        self.configuration = {"scenario": scenario, "difficulty": difficulty, "seed": seed}

    def save_initial_rng(self, path):
        path.write_bytes(b"fixed test randomness")
        return {"source": "test"}

    def restore_initial_rng(self, path):
        return {"source": "test"}


class ScriptedEpisode(NativeActiveEpisode):
    def bootstrap(self):
        self.objects, self.reactors = list(range(4)), list(range(4, 8))
        self.operator.objects = {i: {"name": f"object {i}"} for i in range(4)}
        self.controls = [{"lower_hz": 0., "upper_hz": 1000.}] * 4
        self.known[0, :2] = True
        self.known_hz[0, :2] = torch.tensor([100., 200.])
        self.bootstrap_actions = self.env.steps = 3

    def apply(self, action_index, **kwargs):
        query, start = int(self.query[0]), self.env.steps
        cost = (7, 11, 13)[len(self.decisions)]
        success = action_index >= 20
        self.env.steps += cost
        measurements = []
        if success:
            self.known[0, query] = True
            self.known_hz[0, query] = 500.
            self.completed_targets.add(query)
            if query == 2:
                self.query[0] = 3
            else:
                self.env.ended = True
                self.env.metrics = [{"completed": True, "completed_successfully": True, "progress": 1.}]
        else:
            self.counts[0, 0, 0] += 1
            self.values[0, 0, 0] = 1.
            measurements = [{"kind": "instrument", "target_handle": 0,
                             "quantity": FEATURE_QUANTITIES[0], "value": 1.}]
        decision = {"action_index": action_index, "query_index": query, "success": success,
                    "native_start": start, "native_actions": cost,
                    "result": {"measurements": measurements}}
        self.decisions.append(decision)
        return decision


class RecordingPolicy:
    config = SimpleNamespace(mode="persistent")

    def __init__(self):
        self.calls = []

    def eval(self):
        return self

    def initial(self, batch, device):
        return {"h": torch.zeros(batch, 2, device=device), "z": torch.zeros(batch, 1, 2, device=device),
                "fast": torch.zeros(batch, 2, 2, device=device), "slow": torch.zeros(batch, 2, 2, device=device),
                "activity_counts": torch.zeros(batch, 6, device=device),
                "memory_valid": torch.zeros(batch, 4, dtype=torch.bool, device=device)}

    def _receive(self, kind, event, state, extra):
        feedback = extra.get("feedback_targets")
        self.calls.append({"kind": kind, "event": event.clone(), "has_feedback_argument": "feedback_targets" in extra,
                           "feedback": None if feedback is None else feedback.clone()})
        return {**state, "h": state["h"] + 1}

    def __call__(self, *, event, state, **extra):
        revised = self._receive("forward", event, state, extra)
        action = 20 if bool(event[0, 27:29].any()) else 0
        logits = torch.full((1, 29), -100.)
        logits[0, action] = 1.
        return {"state": revised, "logits": logits}

    def advance(self, event, evidence, evidence_mask, state, **extra):
        return self._receive("terminal", event, state, extra)


@pytest.mark.parametrize("continual,cost_mode,discard", [(False, "native", False), (True, "native", False),
                                                        (True, "replay", True)])
def test_feedback_reaches_next_action_and_terminal_once_without_changing_event(tmp_path, monkeypatch,
                                                                            continual, cost_mode, discard):
    model = RecordingPolicy()
    mixture = AffineMixture({"x_mean": [0.] * 9, "x_scale": [1.] * 9,
                             "y_mean_hz": 0., "y_scale_hz": 1.})
    monkeypatch.setattr(native, "load_continual_checkpoint", lambda *a: (model, {}))
    monkeypatch.setattr(native, "load_active_checkpoint", lambda *a: (model, {}))
    monkeypatch.setattr(native, "load_mixture_checkpoint", lambda *a: (mixture, {}))
    monkeypatch.setattr(native, "StructuredDiscoveryWorldEnv", ScriptedEnvironment)
    monkeypatch.setattr(native, "NativeActiveEpisode", ScriptedEpisode)

    def make_observation(values, counts, known_hz, known, query, failed, event, *args):
        return {"event": event, "evidence": torch.zeros(1, 4, 20), "evidence_mask": torch.ones(1, 4, dtype=torch.bool),
                "candidates": torch.zeros(1, 29, 30), "action_mask": torch.ones(1, 29, dtype=torch.bool)}

    monkeypatch.setattr(native, "make_observation", make_observation)
    monkeypatch.setattr(native, "visible_beliefs", lambda *a: {
        "means": torch.full((1, 9), 500.), "std": torch.ones(1, 9), "eligible": torch.ones(1, 9, dtype=torch.bool)})
    options = {"threads": 1, "device": "cpu", "continual": continual, "checkpoint": "fixture.pt",
               "mixture_checkpoint": "mixture.pt", "output": str(tmp_path / "output"), "state_mode": "carry",
               "observation_mode": "structured", "max_actions": 400, "max_decisions": 10, "refinements": 1,
               "initial_rng_root": str(tmp_path / "rng"), "save_boundary_states": False,
               "feedback_cost": cost_mode, "discard_adaptation": discard}
    result = native.run_stream((options, {"name": "development", "seeds": [2200, 2201]}))
    assert result["completed"] == 2
    assert len(model.calls) == 8
    if continual:
        assert model.retain_adaptation is not discard
    expected_costs = [7, 11, 13] if cost_mode == "native" else [1, 5, 5]
    for offset in (0, 4):
        calls = model.calls[offset:offset + 4]
        assert [row["kind"] for row in calls] == ["forward", "forward", "forward", "terminal"]
        assert calls[0]["feedback"] is None
        for row, proxy in zip(calls, (0., 1., 5., 5.)):
            assert float(row["event"][0, 30]) == pytest.approx(proxy / 20)
            assert row["has_feedback_argument"] is continual
        if continual:
            for row, success, cost in zip(calls[1:], (0., 1., 1.), expected_costs):
                torch.testing.assert_close(row["feedback"], torch.tensor([[success, math.log1p(cost)]]))
        assert calls[-1]["event"][0, :4].argmax() == 3  # last action's real object
        assert calls[-1]["event"][0, 32:36].argmax() == 0  # completed-task observation convention
    metadata = json.loads((tmp_path / "output/development/episode000_seed2200/run.json").read_text())
    assert metadata["event_cost"] == "measurement=1,test=5"
    assert metadata["adaptation_feedback_cost"] == (cost_mode if continual else None)


def test_fresh_native_schedule_covers_each_directed_pair_once():
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((root / "experiments/continual_experiment/native_comparison.json").read_text())
    prior = json.loads((root / "experiments/discoveryworld/active_lifetime_comparison.json").read_text())
    streams = manifest["streams"]
    assert len(streams) == 20
    assert {(s["relations"][0], s["relations"][1]) for s in streams} == {
        (a, b) for a in native.NATIVE_RELATIONS for b in native.NATIVE_RELATIONS if a != b}
    seeds = [seed for stream in streams for seed in stream["seeds"]]
    assert len(seeds) == len(set(seeds)) == 184
    assert not set(seeds) & {seed for stream in prior["streams"] for seed in stream["seeds"]}
    for stream in streams:
        assert all(2 <= n <= 4 for n in stream["segment_lengths"])
        assert stream["relations"][0] == stream["relations"][-1]
        relations = [relation for relation, length in zip(stream["relations"], stream["segment_lengths"])
                     for _ in range(length)]
        assert relations == [native.NATIVE_RELATIONS[seed % 5] for seed in stream["seeds"]]
