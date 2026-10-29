"""Queue only after complete dependencies; disable feedback without skipping it."""
import json

import pytest
import torch

from cdm.recurrent_feedback import RecurrentFeedback, RecurrentFeedbackConfig
from experiments.continual_physics.feedback_campaign import dependency_state


def test_dependency_requires_successful_exit_and_final_budget(tmp_path):
    items = [(tmp_path, 'full_seed231')]
    assert dependency_state(items) == ['full_seed231']
    (tmp_path / 'completed.json').write_text(json.dumps([{'name': 'full_seed231', 'exit_code': 1}]))
    with pytest.raises(RuntimeError, match='unsuccessfully'):
        dependency_state(items)
    directory = tmp_path / 'full_seed231'; directory.mkdir()
    (directory / 'last.pt').touch()
    (directory / 'finished.json').write_text(json.dumps(dict(updates=200, external_requests=204800)))
    # Even complete artifacts do not release the GPU before process exit.
    (tmp_path / 'completed.json').write_text('[]')
    assert dependency_state(items) == ['full_seed231']
    (tmp_path / 'completed.json').write_text(json.dumps([{'name': 'full_seed231', 'exit_code': 0}]))
    assert dependency_state(items) == []
    (directory / 'finished.json').write_text(json.dumps(dict(updates=50, external_requests=51200)))
    with pytest.raises(RuntimeError, match='fixed budget'):
        dependency_state(items)


@pytest.mark.parametrize('source', ['workspace', 'evidence'])
def test_disabled_connection_computes_projection_then_discards(source, monkeypatch):
    import cdm.recurrent_feedback as module
    layer = RecurrentFeedback(8, RecurrentFeedbackConfig(source))
    parameter = next(layer.parameters())
    with torch.no_grad():
        parameter.copy_(torch.eye(8))
    state, tokens = torch.randn(2, 3, 8), torch.randn(2, 5, 8)
    mask = torch.ones(2, 5, dtype=torch.bool)
    calls = []
    original = module.F.linear
    def measured(*args, **kwargs):
        calls.append(True)
        return original(*args, **kwargs)
    monkeypatch.setattr(module.F, 'linear', measured)
    enabled = layer(state, tokens, mask)
    snapshot = parameter.clone()
    layer.enabled = False
    disabled = layer(state, tokens, mask)
    assert len(calls) == 2
    assert enabled.abs().sum() > 0
    assert torch.equal(disabled, torch.zeros_like(disabled))
    assert torch.equal(parameter, snapshot)
