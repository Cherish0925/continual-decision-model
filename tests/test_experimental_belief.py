"""Physical evidence, persistent posterior memory, and meta-gradient checks."""
import torch

from cdm.experimental_belief import (ExperimentalBelief, load_experimental_belief_checkpoint,
                                    save_experimental_belief_checkpoint)


def model():
    torch.set_num_threads(1)
    result = ExperimentalBelief({"x_mean": [0., 0.], "x_scale": [1., 1.],
                                "y_mean_hz": 0., "y_scale_hz": 1.},
                               memory_slots=3, noise_floor_hz=.05)
    with torch.no_grad():
        result.log_noise_extra_hz.fill_(-8)
        result.log_transition_scale.fill_(-2.5)
    return result


def ledger():
    x = torch.tensor([[[-1., 2.], [0., -1.], [1., 3.], [2., 0.5]]], dtype=torch.float64)
    y = 2 * x[..., 0] + 1
    return x, torch.ones_like(x, dtype=torch.bool), y


def test_only_public_values_enter_predictions_and_fit_is_order_independent():
    m = model()
    x, seen, y = ledger()
    seen[:, 1, 1] = False
    known = torch.tensor([[True, True, True, False]])
    memory = m.initial(1)
    out, post = m.observe_predict(x, seen, y, known, torch.tensor([3]), memory)
    hidden_x, hidden_y = x.clone(), y.clone()
    hidden_x[~seen] = float("nan")
    hidden_y[~known] = float("nan")
    hidden, repeated = m.observe_predict(hidden_x, seen, hidden_y, known, torch.tensor([3]), memory)
    for key in ("means", "std", "weights"):
        torch.testing.assert_close(hidden[key], out[key])
    assert not memory["valid"].any()
    perm = torch.tensor([2, 0, 3, 1])
    reordered = m.posterior(x[:, perm], seen[:, perm], y[:, perm], known[:, perm], memory)
    for key in ("coefficient_mean", "coefficient_covariance", "log_weights"):
        torch.testing.assert_close(reordered[key], post[key], atol=1e-10, rtol=1e-8)
        torch.testing.assert_close(repeated[key], post[key])


def test_new_evidence_improves_physical_prediction_before_it_is_remembered():
    m = model()
    x, seen, y = ledger()
    before, _ = m.observe_predict(x, seen, y, torch.tensor([[True, False, False, False]]),
                                 torch.tensor([3]), m.initial(1))
    after, _ = m.observe_predict(x, seen, y, torch.tensor([[True, True, True, False]]),
                                torch.tensor([3]), m.initial(1))
    assert m.nll(after, y[:, 3]) < m.nll(before, y[:, 3])
    assert (after["mixture_mean"] - y[:, 3]).abs() < .05
    assert after["weights"][0, 0] > .99


def test_returning_context_can_recover_after_an_intervening_relationship():
    m = model()
    x, seen, y = ledger()
    memory = m.initial(1)
    all_known = torch.ones_like(y, dtype=torch.bool)
    post_a = m.posterior(x, seen, y, all_known, memory)
    memory_a = m.remember(memory, post_a)
    y_b = -3 * x[..., 1] - 2
    post_b = m.posterior(x, seen, y_b, all_known, memory_a)
    memory_b = m.remember(memory_a, post_b)
    partial = torch.tensor([[True, False, False, False]])
    returned, post = m.observe_predict(x, seen, y, partial, torch.tensor([3]), memory_b)
    fresh, _ = m.observe_predict(x, seen, y, partial, torch.tensor([3]), memory_b, retrieve=False)
    assert m.nll(returned, y[:, 3]) < m.nll(fresh, y[:, 3])
    assert post["context_weights"][0, 1] > post["context_weights"][0, 2]
    assert memory_b["valid"].sum() == 2
    assert not memory["valid"].any()
    assert memory_a["valid"].sum() == 1
    # A dissimilar new rule remains representable through the shared prior.
    prediction, _ = m.observe_predict(x, seen, 6 * x[..., 0] - 5, all_known,
                                     torch.tensor([3]), memory_b)
    assert torch.isfinite(prediction["mixture_mean"]).all()
    assert (prediction["mixture_mean"] - (6 * x[:, 3, 0] - 5)).abs() < .1


def test_future_prediction_trains_persistent_update_and_roundtrips(tmp_path):
    m = model()
    x, seen, y = ledger()
    first_x = x.clone().requires_grad_()
    post = m.posterior(first_x, seen, y, torch.ones_like(y, dtype=torch.bool), m.initial(1))
    memory = m.remember(m.initial(1), post)
    shifted_y = 2.1 * x[..., 0] + .9
    out, _ = m.observe_predict(x, seen, shifted_y, torch.tensor([[True, True, False, False]]),
                               torch.tensor([3]), memory)
    m.nll(out, shifted_y[:, 3]).mean().backward()
    assert first_x.grad is not None and first_x.grad.abs().sum() > 0
    assert m.log_transition_scale.grad is not None and m.log_transition_scale.grad.abs().sum() > 0
    assert m.memory_logit.grad is not None and m.memory_logit.grad.abs() > 0
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters())
    path = tmp_path / "physical.pt"
    save_experimental_belief_checkpoint(m, path, training={"updates": 1})
    restored, raw = load_experimental_belief_checkpoint(path)
    assert raw["training"]["updates"] == 1
    for key, value in m.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[key], value)
