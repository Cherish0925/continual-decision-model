"""Physical predictions must reach both decision features and real execution."""
from types import SimpleNamespace

import numpy as np
import torch

from experiments.discoveryworld.active_native import NativeActiveEpisode
from experiments.discoveryworld.calibration_mixture import AffineMixture
from experiments.discoveryworld.evaluate_active_stream import observation
from experiments.discoveryworld.train_active_core import ActiveTableEnvironment


def mixture():
    return AffineMixture({"x_mean": [0.] * 9, "x_scale": [1.] * 9,
                          "y_mean_hz": 0., "y_scale_hz": 100.})


def test_override_is_the_prediction_seen_by_policy_and_executor():
    episode = NativeActiveEpisode(SimpleNamespace())
    episode.controls = [{"lower_hz": 0., "upper_hz": 10000.}] * 4
    episode.counts.fill_(1)
    episode.known[0, :2] = True
    means = torch.linspace(100., 900., 9)[None]
    expected = {"means": means, "std": torch.full((1, 9), 10.),
                "weights": torch.full((1, 9), 1 / 9),
                "support_counts": torch.full((1, 9), 2),
                "eligible": torch.ones(1, 9, dtype=torch.bool)}
    calls = []

    def source(tensors):
        calls.append(tensors)
        assert tensors["known"].sum() == 2
        return expected

    obs, execution_beliefs = observation(episode, torch.zeros(1, 36), mixture(), 0, 160,
                                         "cpu", belief_source=source)
    assert len(calls) == 1
    assert execution_beliefs is expected
    torch.testing.assert_close(obs["candidates"][:, 20:, 24], means / 100)
    torch.testing.assert_close(obs["candidates"][:, 20:, 25], expected["std"] / 100)
    torch.testing.assert_close(obs["candidates"][:, 20:, 26], expected["weights"])
    assert obs["action_mask"][:, 20:].all()


def test_replay_executes_supplied_pre_action_prediction(tmp_path):
    path = tmp_path / "readings.npz"
    np.savez(path, x=np.zeros((1, 4, 9)),
             sample_values=np.zeros((1, 4, 5, 1, 9), dtype=np.float32),
             sample_present=np.ones((1, 4, 5, 1, 9), dtype=bool),
             sample_count=np.ones((1, 4, 5), dtype=np.int64),
             working_frequency_hz=np.array([[10., 20., 3000., 4000.]], dtype=np.float32),
             initial_support_mask=np.array([[True, True, False, False]]),
             target_mask=np.array([[False, False, True, True]]),
             label_present=np.ones((1, 4), dtype=bool))
    unchanged = ActiveTableEnvironment(path, mixture(), batch=1, permute=False)
    updated = ActiveTableEnvironment(path, mixture(), batch=1, permute=False)
    unchanged.step(torch.tensor([20]))
    updated.step(torch.tensor([20]), beliefs={"means": torch.full((1, 9), 3000.)})
    assert unchanged.successes.item() == 0
    assert updated.successes.item() == 1
    assert updated.known_hz[0, 2].item() == 3000.
    assert updated.query.item() == 3
