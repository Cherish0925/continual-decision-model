"""Behavioral checks for public interval learning and actual physical readout."""
import torch

from cdm.active_experiment import ActiveExperimentConfig, ActiveExperimentPolicy
from cdm.continual_experiment import Activity, ContinualExperimentConfig, detach_state
from cdm.continual_physics import (ContinualPhysicsPolicy, ContinualPhysicsConfig,
                                  load_continual_physics_checkpoint)
from cdm.core import ContinualCore, CoreConfig
from cdm.experimental_belief import ExperimentalBelief


def example():
    torch.set_num_threads(1)
    torch.manual_seed(6)
    core = ContinualCore(CoreConfig(event_dim=3, record_dim=3, actions=2, width=32, slots=2,
                                    heads=4, hidden=32, action_width=16), torch.tensor([-1., 1.]))
    base = ActiveExperimentPolicy(ActiveExperimentConfig(36, 20, 30, action_width=16), core)
    belief = ExperimentalBelief(dict(x_mean=[0., 0.], x_scale=[1., 1.], y_mean_hz=100., y_scale_hz=100.),
                                memory_slots=3)
    model = ContinualPhysicsPolicy(base, belief, ContinualExperimentConfig(adapter_width=8, memory_size=4),
                                   ContinualPhysicsConfig(memory_size=3, max_feedback_intervals=8))
    obs = dict(event=torch.randn(2, 36), evidence=torch.randn(2, 5, 20),
               evidence_mask=torch.ones(2, 5, dtype=torch.bool), candidates=torch.randn(2, 22, 30),
               action_mask=torch.ones(2, 22, dtype=torch.bool),
               physical_means=torch.tensor([[100., 150.], [100., 150.]], dtype=torch.float64),
               physical_std=torch.full((2, 2), 10., dtype=torch.float64),
               physical_weights=torch.tensor([[.99, .01], [.99, .01]], dtype=torch.float64),
               physical_eligible=torch.ones(2, 2, dtype=torch.bool))
    obs['candidates'][:, 20:, 24] = ((obs['physical_means'] - 100) / 100).float()
    obs['candidates'][:, 20:, 25] = .1
    obs['candidates'][:, 20:, 26] = obs['physical_weights'].float()
    return model, obs


def returned(model, obs, success=True):
    state = model.receive(model.initial(2, 'cpu'), obs)
    out = model.readout(state, obs['candidates'], obs['action_mask'])
    state = model.begin_action(state, out['features'], torch.tensor([20, 20]), physical=out)
    feedback = dict(intervals=torch.tensor([[[116., 120.]], [[116., 120.]]]),
                    mask=torch.ones(2, 1, dtype=torch.bool), success=torch.full((2, 1), success))
    return model.feedback(state, torch.tensor([[1., 2.], [1., 2.]]), physical_feedback=feedback), feedback


def test_parent_predictions_actions_and_checkpoint_are_preserved(tmp_path):
    model, obs = example()
    state = model.receive(model.initial(2, 'cpu'), obs)
    out = model.readout(state, obs['candidates'], obs['action_mask'])
    parent = model.base.readout(state, obs['candidates'], obs['action_mask'])
    torch.testing.assert_close(out['logits'], parent['logits'], atol=0, rtol=0)
    torch.testing.assert_close(out['physical_means'], obs['physical_means'], atol=0, rtol=0)
    torch.testing.assert_close(out['physical_std'], obs['physical_std'], atol=0, rtol=0)
    path = tmp_path / 'model.pt'
    torch.save({**model.configuration(), 'model': model.state_dict()}, path)
    restored, _ = load_continual_physics_checkpoint(path)
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, restored.state_dict()[key])


def test_recurrent_reference_waits_for_feedback_without_neural_updates(tmp_path):
    from dataclasses import replace

    model, obs = example()
    model.continual_config = replace(model.continual_config, activity_mode="fixed",
                                    max_internal_steps=0, fixed_refinements=0)
    model.retain_adaptation = False
    model.retain_physical_adaptation = False
    state = model.receive(model.initial(2, "cpu"), obs)
    out = model.readout(state, obs["candidates"], obs["action_mask"])
    assert out["activity_logits"].argmax(-1).tolist() == [int(Activity.ACT)] * 2
    state = model.begin_action(state, out["features"], torch.tensor([20, 20]), physical=out)
    pending = model.readout(state, obs["candidates"], obs["action_mask"])
    assert pending["activity_logits"].argmax(-1).tolist() == [int(Activity.WAIT)] * 2
    waited = model.internal(state, torch.full((2,), int(Activity.WAIT)), obs)
    assert torch.equal(waited["h"], state["h"])
    assert torch.equal(waited["z"], state["z"])
    assert waited["observation_count"].tolist() == [1, 1]
    feedback = dict(intervals=torch.tensor([[[116., 120.]], [[116., 120.]]]),
                    mask=torch.ones(2, 1, dtype=torch.bool), success=torch.ones(2, 1, dtype=torch.bool))
    received = model.feedback(waited, torch.ones(2, 2), physical_feedback=feedback)
    for activity in (Activity.LEARN, Activity.CONSOLIDATE):
        # Flags also protect deployment if an external caller overrides activity.
        revised = model.internal(received, torch.full((2,), int(activity)), obs)
        for key in ("fast", "slow", "physics_fast", "physics_slow"):
            assert torch.equal(revised[key], received[key])
    state = model.receive(received, obs)
    assert state["observation_count"].tolist() == [2, 2]
    assert model.readout(state, obs["candidates"], obs["action_mask"])["activity_logits"].argmax(-1).tolist() == [0, 0]
    path = tmp_path / "reference.pt"
    torch.save({**model.configuration(), "model": model.state_dict()}, path)
    restored, _ = load_continual_physics_checkpoint(path)
    assert restored.continual_config.activity_mode == "fixed"
    assert restored.continual_config.fixed_refinements == 0
    assert not restored.retain_adaptation
    assert not restored.retain_physical_adaptation


def test_success_updates_executed_physics_and_outer_gradients():
    model, obs = example()
    state, _ = returned(model, obs)
    before_loss, gradient = model.physical_replay(state)
    # Analytic update agrees with autograd on the public event NLL.
    analytic_state = {**state, 'physics_fast': state['physics_fast'].detach().requires_grad_(True)}
    reference_loss, _ = model.physical_replay(analytic_state)
    automatic, = torch.autograd.grad(reference_loss.sum(), analytic_state['physics_fast'])
    torch.testing.assert_close(gradient, automatic, atol=1e-5, rtol=1e-4)
    updated = model.internal(state, torch.full((2,), int(Activity.LEARN)), obs)
    after_loss, _ = model.physical_replay(updated)
    assert (after_loss < before_loss).all()
    out = model.readout(updated, obs['candidates'], obs['action_mask'])
    assert (out['physical_means'][:, 0] > obs['physical_means'][:, 0]).all()
    torch.testing.assert_close(out['physical_candidates'][:, 20:, 24].double(),
                               (out['physical_means'] - 100) / 100)
    assert not torch.equal(out['logits'], model.readout(state, obs['candidates'], obs['action_mask'])['logits'])
    after_loss.mean().backward()
    for name, parameter in model.named_parameters():
        if parameter.grad is not None:
            assert torch.isfinite(parameter.grad).all(), name
    assert model.physical_fast_step.grad.abs() > 0
    assert model.adapter_features.weight.grad.abs().sum() > 0
    assert torch.equal(updated['h'], state['h'])
    assert torch.equal(updated['observation_count'], state['observation_count'])


def test_failure_union_is_not_double_counted_and_discard_is_isolated():
    model, obs = example()
    state, receipt = returned(model, obs, success=False)
    one_loss, _ = model.physical_replay(state)
    duplicate = {k: v.clone() for k, v in state.items()}
    duplicate['physics_memory_intervals'][:, 0, 1] = duplicate['physics_memory_intervals'][:, 0, 0]
    duplicate['physics_memory_mask'][:, 0, 1] = True
    duplicate_loss, _ = model.physical_replay(duplicate)
    torch.testing.assert_close(one_loss, duplicate_loss, atol=0, rtol=0)
    # A second delivery cannot insert the same pending action again.
    repeated = model.feedback(state, torch.ones(2, 2), physical_feedback=receipt)
    assert torch.equal(repeated['physics_memory_cursor'], state['physics_memory_cursor'])
    model.retain_physical_adaptation = False
    discarded = model.internal(state, torch.full((2,), int(Activity.LEARN)), obs)
    torch.testing.assert_close(discarded['physics_fast'], state['physics_fast'])
    assert not torch.equal(discarded['fast'], state['fast'])
    model.retain_physical_adaptation = True
    with torch.no_grad():
        updated = model.internal(detach_state(state), torch.full((2,), int(Activity.CONSOLIDATE)), obs)
    after_loss, _ = model.physical_replay(updated)
    assert (after_loss < one_loss).all()
    assert updated['physics_slow'].abs().sum() > 0


def test_empty_measurement_partial_delivery_and_reset():
    model, obs = example()
    state = model.receive(model.initial(2, 'cpu'), obs, torch.tensor([True, False]))
    assert state['observation_count'].tolist() == [1, 0]
    out = model.readout(state, obs['candidates'], obs['action_mask'])
    state = model.begin_action(state, out['features'], torch.tensor([0, 20]), physical=out)
    empty = dict(intervals=torch.empty(2, 0, 2), success=torch.empty(2, 0, dtype=torch.bool),
                 mask=torch.empty(2, 0, dtype=torch.bool))
    state = model.feedback(state, torch.zeros(2, 2), selected=torch.tensor([True, False]), physical_feedback=empty)
    assert state['pending'].tolist() == [False, True]
    assert not state['physics_memory_valid'].any()
    reset = model.reset_rows(state, torch.tensor([False, True]))
    assert not reset['pending'].any()
    assert reset['memory_valid'][0].any()
    assert not reset['memory_valid'][1].any()


def test_public_failures_change_settings_without_removing_relation():
    from cdm.continual_physics import propose_physical_settings
    # A failed setting excludes Y in [-2, 2]. Symmetry leaves conditional mean
    # at zero, but zero cannot succeed again. Propose a surviving success window.
    mean, std = torch.tensor([[0.]], dtype=torch.float64), torch.tensor([[3.]], dtype=torch.float64)
    proposal = propose_physical_settings(mean, std, torch.tensor([[[-2., 2.]]]), torch.tensor([[True]]))
    assert proposal['valid'].item()
    assert proposal['settings'].abs().item() == 4.
    # The setting itself need not be outside excluded target intervals. Here a
    # setting of -.5 covers all remaining high-probability targets in (0, 1).
    proposal = propose_physical_settings(torch.tensor([[-.5]]), torch.tensor([[.7]]),
        torch.tensor([[[-4., 0.], [1., 5.]]]), torch.tensor([[True, True]]))
    torch.testing.assert_close(proposal['settings'], torch.tensor([[-.5]], dtype=torch.float64))
    # Real control steps and origin are supplied by the caller, not assumed.
    grid = propose_physical_settings(mean, std, torch.tensor([[[-2., 2.]]]), torch.tensor([[True]]),
                                     current=torch.tensor([.3]), minimum_step=torch.tensor([1.5]),
                                     lower=torch.tensor([-20.]), upper=torch.tensor([20.]))
    grid_index = (grid['settings'] - .3) / 1.5
    torch.testing.assert_close(grid_index, grid_index.round())
    model, obs = example()
    obs.update(physical_exclusions=torch.tensor([[[98., 102.]], [[98., 102.]]]),
               physical_exclusion_mask=torch.ones(2, 1, dtype=torch.bool))
    state = model.receive(model.initial(2, 'cpu'), obs)
    out = model.readout(state, obs['candidates'], obs['action_mask'])
    torch.testing.assert_close(out['physical_means'], obs['physical_means'])
    assert (out['physical_settings'][:, 0] != 100).all()
    assert out['physical_proposal_valid'].all()
    assert torch.isfinite(out['logits'][:, 20]).all()
    # Changing the query replaces its exclusion history but retains learning.
    obs['physical_exclusions'] = torch.empty(2, 0, 2)
    obs['physical_exclusion_mask'] = torch.empty(2, 0, dtype=torch.bool)
    cleared = model.receive(state, obs)
    assert not cleared['physical_exclusion_mask'].any()
    assert cleared['h'].shape == state['h'].shape


def test_setting_probability_sums_multiple_surviving_segments():
    from cdm.continual_physics import propose_physical_settings
    # A narrower public exclusion allows one test window to cover both tails.
    result = propose_physical_settings(torch.tensor([[0.]], dtype=torch.float64),
        torch.tensor([[1.]], dtype=torch.float64), torch.tensor([[[-.2, .2]]]), torch.tensor([[True]]), tolerance=2.)
    assert result['settings'].item() == 0.
    normal = torch.distributions.Normal(0., 1.)
    expected = normal.cdf(torch.tensor(2.)) - normal.cdf(torch.tensor(-2.)) - normal.cdf(torch.tensor(.2)) + normal.cdf(torch.tensor(-.2))
    torch.testing.assert_close(result['log_success_mass'].exp().float().squeeze(), expected)
