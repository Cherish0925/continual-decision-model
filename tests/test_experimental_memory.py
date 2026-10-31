import torch

from cdm.experimental_memory import ExperimentalMemory, ExperimentalMemoryConfig


def fixture():
    model = ExperimentalMemory(ExperimentalMemoryConfig(slots=2, capacity=3))
    state = model.initial(2, 'cpu')
    history = torch.zeros(2, 5, 20)
    history[:, :, 8:11] = 1
    history[0, :, 0] = torch.arange(5)
    history[1, :, 1] = torch.arange(5)
    state.update(reaction_history=history,
                 reaction_mask=torch.tensor([[1, 1, 1, 1, 1], [1, 0, 0, 0, 0]], dtype=torch.bool))
    return model, state


def test_archive_keeps_each_streams_latest_actual_returns():
    model, state = fixture()
    result = model.archive(state, torch.tensor([True, True]))
    assert result['archive_history'][0, 0, :, 0].tolist() == [2., 3., 4.]
    assert result['archive_mask'][1, 0].tolist() == [True, False, False]
    assert not state['archive_mask'].any()


def test_new_evidence_rejects_an_inapplicable_memory():
    model, state = fixture()
    state = model.archive(state, torch.tensor([True, True]))
    components = torch.tensor([[1, 1, 1, 0, 0, 0, 0, 0]] * 2, dtype=torch.bool)
    initial = model.weights(state, components).exp()
    probabilities = torch.tensor([[[.25, .25, .25, .25], [.97, .01, .01, .01], [.25, .25, .25, .25]]]*2)
    updated = model.record(state, probabilities.log(), torch.tensor([3, 0]))
    final = model.weights(updated, components).exp()
    assert final[0, 1] < initial[0, 1]
    assert final[1, 1] > initial[1, 1]
    incompatible = components.clone(); incompatible[:, 3] = True
    assert model.weights(updated, incompatible).exp()[:, 0].eq(1).all()


def test_future_prediction_trains_memory_read_and_empty_bank_is_identity():
    model, state = fixture()
    components = torch.tensor([[1, 1, 1, 0, 0, 0, 0, 0]]*2, dtype=torch.bool)
    probabilities = torch.tensor([[[[.25, .25, .25, .25], [.7, .1, .1, .1], [.1, .1, .1, .7]]]]*2)
    output = model.combine(state, components, probabilities.log())
    torch.testing.assert_close(output, probabilities[:, :, 0].log())
    state = model.archive(state, torch.tensor([True, True]))
    state = model.record(state, probabilities[:, 0].log(), torch.tensor([0, 0]))
    loss = -model.combine(state, components, probabilities.log())[:, :, 0].mean()
    loss.backward()
    assert model.compatibility[-1].weight.grad.abs().sum() > 0
    assert torch.isfinite(model.log_evidence_scale.grad)
