"""Physical runtime integration; these checks are not task-performance results."""
import pytest
import torch

from cdm.active_experiment import ActiveExperimentConfig, ActiveExperimentPolicy
from cdm.continual_experiment import Activity, ContinualExperimentConfig
from cdm.continual_physics import ContinualPhysicsPolicy, ContinualPhysicsConfig
from cdm.core import ContinualCore, CoreConfig
from cdm.experimental_belief import ExperimentalBelief
from experiments.continual_physics.runtime import ContinualPhysicsSession, physical_feedback


def small_model():
    torch.manual_seed(73)
    torch.set_num_threads(1)
    core = ContinualCore(CoreConfig(event_dim=3, record_dim=3, actions=2, width=32,
                                    slots=2, heads=4, hidden=32, action_width=16), torch.tensor([-1., 1.]))
    base = ActiveExperimentPolicy(ActiveExperimentConfig(36, 20, 30, action_width=16), core)
    belief = ExperimentalBelief({"x_mean": [0.]*9, "x_scale": [1.]*9,
                                 "y_mean_hz": 1000., "y_scale_hz": 100.})
    return ContinualPhysicsPolicy(base, belief,
        ContinualExperimentConfig(adapter_width=8, memory_size=4, max_internal_steps=1),
        ContinualPhysicsConfig(memory_size=4)).eval()


def observation():
    mask = torch.zeros(1, 29, dtype=torch.bool)
    mask[:, 20] = True
    return {"event": torch.randn(1, 36), "evidence": torch.randn(1, 5, 20),
            "evidence_mask": torch.ones(1, 5, dtype=torch.bool),
            "candidates": torch.randn(1, 29, 30), "action_mask": mask,
            "physical_means": torch.full((1, 9), 1000., dtype=torch.float64),
            "physical_std": torch.full((1, 9), 10., dtype=torch.float64),
            "physical_weights": torch.full((1, 9), 1/9, dtype=torch.float64),
            "physical_eligible": torch.ones(1, 9, dtype=torch.bool)}


def test_result_waits_while_workspace_continues_and_is_consumed_once():
    model, obs = small_model(), observation()
    session = ContinualPhysicsSession(model)
    session.receive(0, obs)
    request = session.tick()
    assert request["action"] == 20
    delivered = {key: value.clone() for key, value in obs.items()}
    delivered["physical_means"].add_(100.)
    feedback = {"intervals": torch.tensor([[[998., 1002.]]]),
                "success": torch.tensor([[False]]), "mask": torch.tensor([[True]])}
    session.queue_result(1, request["request_id"], delivered, torch.tensor([[0., 2.]]),
                         feedback, delay_ticks=2)
    before_h, before_z = session.state["h"].clone(), session.state["z"].clone()
    with torch.no_grad():
        model.activity[-1].bias[Activity.REFINE] = 10.
    assert session.poll_result() == {"delivered": False, "activity": "REFINE"}
    assert session.poll_result() == {"delivered": False, "activity": "WAIT"}
    assert torch.equal(before_h, session.state["h"])
    assert not torch.equal(before_z, session.state["z"])
    assert session.state["observation_count"].item() == 1
    assert session.state["physical_means"][0, 0].item() == 1000.
    assert session.state["physics_memory_valid"].sum().item() == 0
    assert session.poll_result() == {"delivered": True}
    assert session.state["observation_count"].item() == 2
    assert session.state["physical_means"][0, 0].item() == 1100.
    assert session.state["physics_memory_valid"].sum().item() == 1
    assert session.delivered_results == 1 and session.pending_request is None
    after = {key: value.clone() for key, value in session.state.items()}
    with pytest.raises(ValueError, match="exactly once"):
        session.receive(1, delivered, request_id=request["request_id"],
                        feedback=torch.tensor([[0., 2.]]), physical_feedback=feedback)
    for key in after:
        assert torch.equal(after[key], session.state[key])


def test_setting_feedback_uses_received_values_and_matching_object():
    decision = {"action_index": 22, "result": {
        "request": {"target_handle": 7, "reactor_handle": 8, "mean_hz": 500.},
        "attempts": [{"frequency_hz": 100., "activated": False},
                     {"frequency_hz": 104., "activated": True}],
        "setting": {"object_handle": 9, "reactor_handle": 8,
                    "frequency_hz": 888., "activated": False}}}
    returned = physical_feedback(decision, "cpu")
    assert returned["intervals"].tolist() == [[[98., 102.], [102., 106.]]]
    assert returned["success"].tolist() == [[False, True]]
    assert physical_feedback({"action_index": 2, "result": {}}, "cpu")["mask"].shape == (1, 0)
