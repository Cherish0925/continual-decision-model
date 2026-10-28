"""Integration checks, not benchmark results. Fixtures are generated numerical tasks."""
from dataclasses import replace

import numpy as np
import pytest
import torch

from cdm.active_experiment import ActiveExperimentConfig, ActiveExperimentPolicy
from cdm.continual_experiment import (Activity, ContinualExperimentConfig, ContinualExperimentPolicy,
                                    detach_state, load_continual_checkpoint)
from cdm.core import CoreConfig, ContinualCore
from experiments.continual_experiment.runtime import ContinualSession
from experiments.continual_experiment.train import step_selected, train_window
from experiments.discoveryworld.calibration_mixture import AffineMixture, save_mixture_checkpoint
from experiments.discoveryworld.train_active_lifetime import CalibrationTaskPool, TaskStreamPlan, ActiveLifetimeEnvironment


@pytest.fixture(autouse=True)
def deterministic():
    torch.set_num_threads(1)
    torch.manual_seed(20)


def make_model(**kwargs):
    core = ContinualCore(CoreConfig(event_dim=3, record_dim=3, actions=2, width=32, slots=2,
                                    heads=4, hidden=32, action_width=16), torch.tensor([-1., 1.]))
    base = ActiveExperimentPolicy(ActiveExperimentConfig(36, 20, 30, action_width=16), core)
    return ContinualExperimentPolicy(base, ContinualExperimentConfig(adapter_width=8, memory_size=4, **kwargs))


def observation(batch=2):
    return {"event": torch.randn(batch, 36), "evidence": torch.randn(batch, 5, 20),
            "evidence_mask": torch.ones(batch, 5, dtype=torch.bool),
            "candidates": torch.randn(batch, 29, 30), "action_mask": torch.ones(batch, 29, dtype=torch.bool)}


def remember(model, state, obs):
    out = model.readout(state, obs["candidates"], obs["action_mask"])
    actions = torch.zeros(len(state["h"]), dtype=torch.long)
    state = model.begin_action(state, out["features"], actions)
    targets = torch.tensor([[1., 2.]]).expand(len(actions), -1)
    return model.feedback(state, targets), targets


def test_parent_policy_preserved_and_checkpoint_roundtrip(tmp_path):
    model = make_model()
    path = tmp_path / "parent.pt"
    torch.save({**model.base.configuration(), "model": model.base.state_dict()}, path)
    loaded, _ = load_continual_checkpoint(path, config=model.continual_config)
    obs = observation()
    parent = model.base(**obs, state=model.base.initial(2, "cpu"))
    result = loaded(**obs, state=loaded.initial(2, "cpu"))
    torch.testing.assert_close(parent["logits"], result["logits"])
    for key in ("h", "z"):
        torch.testing.assert_close(parent["state"][key], result["state"][key])
    torch.save({**loaded.configuration(), "model": loaded.state_dict()}, path)
    restored, _ = load_continual_checkpoint(path)
    for key, value in loaded.state_dict().items():
        torch.testing.assert_close(value, restored.state_dict()[key])


def test_internal_computation_does_not_reconsume_observations_and_accepts_modalities():
    model = make_model(modality_dim=12)
    obs = observation()
    obs.update(modality_tokens=torch.randn(2, 3, 12), modality_mask=torch.ones(2, 3, dtype=torch.bool))
    state = model.receive(model.initial(2, "cpu"), obs)
    revised = model.internal(state, torch.full((2,), int(Activity.REFINE)), obs)
    assert torch.equal(state["h"], revised["h"])
    assert torch.equal(state["observation_count"], revised["observation_count"])
    assert not torch.equal(state["z"], revised["z"])
    model.readout(revised, obs["candidates"], obs["action_mask"])["value"].sum().backward()
    assert model.modality_projection.weight.grad.abs().sum() > 0


def test_runtime_updates_reduce_replay_error_and_have_outer_gradients():
    model, obs = make_model(), observation()
    state, targets = remember(model, model.receive(model.initial(2, "cpu"), obs), obs)
    before = model.predictor(state["memory_x"][:, 0])
    learned = model.internal(state, torch.full((2,), int(Activity.LEARN)), obs)
    after = model.predictor(learned["memory_x"][:, 0]) + torch.bmm(
        learned["memory_x"][:, :1], learned["fast"] + learned["slow"])[:, 0]
    assert (after - targets).square().mean() < (before - targets).square().mean()
    (after - targets).square().mean().backward()
    assert model.fast_step.grad.abs() > 0
    assert model.adapter_features.weight.grad.abs().sum() > 0
    shared = {k: v.clone() for k, v in model.state_dict().items()}
    with torch.no_grad():
        consolidated = model.internal(detach_state(learned), torch.full((2,), int(Activity.CONSOLIDATE)), obs)
    assert consolidated["slow"].abs().sum() > 0
    for key, value in model.state_dict().items():
        assert torch.equal(value, shared[key])
    retrieved = model.internal(consolidated, torch.full((2,), int(Activity.RETRIEVE)), obs)
    assert not torch.equal(retrieved["z"], consolidated["z"])
    retrieved["z"].square().sum().backward()
    assert model.memory_query.weight.grad is not None


def test_async_observation_does_not_complete_pending_tool_and_budget_yields():
    model = make_model(max_internal_steps=1)
    session = ContinualSession(model)
    obs = observation(1)
    session.receive(0, obs)
    request = session.tick()
    assert request["activity"] == "ACT"
    session.receive(1, obs)  # a new sensor event while the tool is still running
    assert session.pending_request == request["request_id"]
    with torch.no_grad():
        model.activity[-1].bias[Activity.REFINE] = 10.
    assert session.tick()["activity"] == "REFINE"
    assert session.tick()["activity"] == "WAIT"
    assert int(session.state["observation_count"][0]) == 2
    session.receive(2, obs, request_id=request["request_id"], feedback=torch.tensor([[1., 2.]]))
    assert session.pending_request is None
    assert int(session.state["memory_valid"].sum()) == 1
    with pytest.raises(ValueError):
        session.receive(2, obs)


def make_pool(path):
    n = 16
    x = np.broadcast_to(np.arange(1, 5)[None, :, None], (n, 4, 9)).astype(np.float32).copy()
    x += np.arange(n)[:, None, None] * .01
    samples = np.zeros((n, 4, 5, 32, 9), dtype=np.float32)
    present = np.zeros_like(samples, dtype=bool)
    count = np.ones((n, 4, 5), dtype=np.int64)
    count[:, :, 2] = 32
    for group, axes in enumerate(((0,), (1,), (2,), (3,), (4, 5, 6, 7, 8))):
        for axis in axes:
            samples[:, :, group, :, axis] = x[:, :, axis, None]
            present[:, :, group, :32 if group == 2 else 1, axis] = True
    np.savez(path, x=x, sample_values=samples, sample_present=present, sample_count=count,
             working_frequency_hz=(1000 + 100 * x[:, :, 0]).astype(np.float32),
             initial_support_mask=np.tile([True, True, False, False], (n, 1)),
             target_mask=np.tile([False, False, True, True], (n, 1)), label_present=np.ones((n, 4), dtype=bool))
    mixture = AffineMixture({"x_mean": [0.] * 9, "x_scale": [1.] * 9,
                             "y_mean_hz": 1000., "y_scale_hz": 100.}).requires_grad_(False)
    return CalibrationTaskPool(path, mixture, "cpu"), mixture


def make_env(tmp_path, max_steps=3):
    pool, mixture = make_pool(tmp_path / "train.npz")
    plan = TaskStreamPlan(pool, streams=2, lifetimes=20, seed=4, device="cpu")
    return ActiveLifetimeEnvironment(pool, plan, mixture, max_steps=max_steps)


def test_internal_rows_do_not_advance_environment(tmp_path):
    env = make_env(tmp_path)
    before = {key: getattr(env.table, key).clone() for key in ("steps", "values", "counts", "event", "alive")}
    _, done, _ = step_selected(env, torch.tensor([0, 0]), torch.tensor([True, False]))
    assert int(env.table.steps[0]) == 1
    assert not done[1]
    for key, value in before.items():
        assert torch.equal(getattr(env.table, key)[1], value[1])


@pytest.mark.parametrize("mode", ["full", "fixed", "no_adaptation", "task_reset"])
def test_joint_training_crosses_task_boundaries_with_finite_gradients(tmp_path, mode):
    model, env = make_model(), make_env(tmp_path)
    # Explore the new activities in this integration test; production warm-start favors ACT.
    with torch.no_grad():
        model.activity[-1].bias.zero_()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    state = model.receive(model.initial(2, "cpu"), env.observation())
    for _ in range(3):
        loss, state, _ = train_window(model, env, detach_state(state), unroll=16, mode=mode)
        assert torch.isfinite(loss)
        optimizer.zero_grad()
        loss.backward()
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        assert model.base.core.gru.weight_ih_l0.grad.abs().sum() > 0
        optimizer.step()
    assert len(env.task_rows) > 2
    assert not state["pending"].any()
