"""Feedback changes future recurrent state without a fictitious observation."""
from copy import deepcopy

import pytest
import torch

from cdm.continual_experiment import Activity, load_continual_checkpoint
from cdm.continual_physics import load_continual_physics_checkpoint
from cdm.recurrent_feedback import RecurrentFeedbackConfig, configure_recurrent_feedback
from test_continual_experiment import make_model, make_pool, observation
from test_continual_physics import example


@pytest.mark.parametrize('source', ['workspace', 'evidence', 'both'])
def test_zero_connection_preserves_physical_policy_and_rng(source):
    parent, obs = example()
    candidate = deepcopy(parent)
    rng = torch.get_rng_state().clone()
    configure_recurrent_feedback(candidate, RecurrentFeedbackConfig(source))
    assert torch.equal(torch.get_rng_state(), rng)
    left, right = parent.initial(2, 'cpu'), candidate.initial(2, 'cpu')
    selected = torch.tensor([True, False])
    for _ in range(2):
        left, right = parent.receive(left, obs, selected), candidate.receive(right, obs, selected)
        for key in left:
            torch.testing.assert_close(left[key], right[key], rtol=0, atol=0)
        a, b = parent.readout(left, obs['candidates'], obs['action_mask']), candidate.readout(right, obs['candidates'], obs['action_mask'])
        for key in ('logits', 'activity_logits', 'value', 'physical_settings', 'physical_means', 'physical_std'):
            torch.testing.assert_close(a[key], b[key], rtol=0, atol=0)
        activity = torch.tensor([int(Activity.REFINE), int(Activity.WAIT)])
        left, right = parent.internal(left, activity, obs), candidate.internal(right, activity, obs)
        selected = ~selected


def test_workspace_computation_reaches_future_h_after_one_learned_update():
    torch.set_num_threads(1)
    torch.manual_seed(12)
    model, obs = make_model(), observation()
    configure_recurrent_feedback(model)
    state = model.receive(model.initial(2, 'cpu'), obs)
    before_count, before_h = state['observation_count'].clone(), state['h'].clone()
    refined = model.internal(state, torch.full((2,), int(Activity.REFINE)), obs)
    assert torch.equal(refined['h'], before_h)
    assert torch.equal(refined['observation_count'], before_count)
    assert not torch.equal(refined['z'], state['z'])
    # Only loss on the FUTURE h can train the new projection in this check.
    loss = model.receive(refined, obs)['h'].square().mean()
    gradient, = torch.autograd.grad(loss, model.recurrent_feedback.workspace)
    assert torch.isfinite(gradient).all() and gradient.abs().sum() > 0
    with torch.no_grad():
        model.recurrent_feedback.workspace.add_(gradient, alpha=-.5)
    source = refined['z'].detach().requires_grad_(True)
    future = model.receive({**refined, 'z': source}, obs)
    source_gradient, = torch.autograd.grad(future['h'].square().sum(), source)
    assert torch.isfinite(source_gradient).all() and source_gradient.abs().sum() > 0
    unrefined = model.receive(state, obs)
    assert not torch.equal(future['h'], unrefined['h'])
    assert torch.equal(future['observation_count'], before_count + 1)
    frozen = model.receive(refined, obs, torch.zeros(2, dtype=torch.bool))
    assert all(torch.equal(frozen[key], refined[key]) for key in refined)


def test_evidence_control_respects_visibility_and_is_distinct_from_workspace():
    torch.set_num_threads(1)
    model, obs = make_model(), observation()
    configure_recurrent_feedback(model, RecurrentFeedbackConfig('evidence'))
    with torch.no_grad():
        model.recurrent_feedback.evidence.copy_(torch.eye(model.core.config.width) * .1)
    obs['evidence_mask'][:, -2:] = False
    evidence = obs['evidence'].detach().requires_grad_(True)
    state = model.initial(2, 'cpu')
    result = model.receive(state, {**obs, 'evidence': evidence})
    gradient, = torch.autograd.grad(result['h'].square().sum(), evidence)
    assert gradient[:, :-2].abs().sum() > 0
    assert torch.equal(gradient[:, -2:], torch.zeros_like(gradient[:, -2:]))
    changed_workspace = {**state, 'z': torch.randn_like(state['z'])}
    other = model.receive(changed_workspace, obs)
    torch.testing.assert_close(result['h'], other['h'], rtol=0, atol=0)


@pytest.mark.parametrize('physical', [False, True])
def test_trained_feedback_checkpoint_restores_source_and_weights(tmp_path, physical):
    model = example()[0] if physical else make_model()
    configure_recurrent_feedback(model, RecurrentFeedbackConfig('both'))
    with torch.no_grad():
        model.recurrent_feedback.workspace.fill_(.1)
        model.recurrent_feedback.evidence.fill_(.2)
    path = tmp_path / 'candidate.pt'
    torch.save({**model.configuration(), 'model': model.state_dict()}, path)
    loader = load_continual_physics_checkpoint if physical else load_continual_checkpoint
    restored, _ = loader(path)
    assert restored.recurrent_feedback.config.source == 'both'
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, restored.state_dict()[key], rtol=0, atol=0)


def test_feedback_trains_in_existing_delayed_cross_task_loop(tmp_path):
    from cdm.continual_experiment import ContinualExperimentConfig, detach_state
    from cdm.continual_physics import ContinualPhysicsPolicy, ContinualPhysicsConfig
    from cdm.experimental_belief import ExperimentalBelief
    from experiments.continual_physics.delayed import ExactPhysicalTasks, DelayedExperiments
    from experiments.continual_physics.train import observe, train_window
    from experiments.discoveryworld.train_active_lifetime import TaskStreamPlan

    torch.set_num_threads(1)
    torch.manual_seed(12)
    pool, mixture = make_pool(tmp_path / 'fixture.npz')
    plan = TaskStreamPlan(pool, streams=2, lifetimes=8, seed=5, device='cpu')
    plan.native = torch.zeros(plan.quantities.shape[:2], dtype=torch.bool)
    env = ExactPhysicalTasks(pool, plan, mixture, max_steps=3)
    delayed = DelayedExperiments(env, seed=5, max_delay=2)
    belief = ExperimentalBelief(mixture.normalization, memory_slots=2).initialize_from_mixture(mixture)
    model = ContinualPhysicsPolicy(make_model().base, belief,
        ContinualExperimentConfig(adapter_width=8, memory_size=4, max_internal_steps=1),
        ContinualPhysicsConfig(memory_size=2, max_feedback_intervals=2, max_exclusions=16))
    configure_recurrent_feedback(model)
    state = model.initial(2, 'cpu')
    obs, _ = observe(model, env, state, delayed=delayed)
    state = model.receive(state, obs)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    for _ in range(2):
        loss, state, _, stats = train_window(model, delayed, detach_state(state),
                                            external_per_stream=4, mode='full')
        assert torch.isfinite(loss)
        optimizer.zero_grad()
        loss.backward()
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        assert model.recurrent_feedback.workspace.grad.abs().sum() > 0
        optimizer.step()
        assert stats['external_requests'] == 8
        assert stats['streams_crossing_task_fraction'] == 1.
        assert not delayed.pending.any()
    assert model.recurrent_feedback.workspace.abs().sum() > 0
