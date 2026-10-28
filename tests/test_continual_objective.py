"""Return semantics for internal work, external progress and lifetime boundaries."""

import torch

from experiments.continual_experiment.train import activity_returns


def test_internal_delay_only_pays_explicit_compute_cost():
    # Two internal updates precede a successful external action. They do not
    # advance the environment, so success is not discounted merely for waiting
    # to act; each update still pays its explicit computation cost.
    result = activity_returns(
        [torch.tensor([-.1]), torch.tensor([-.1]), torch.tensor([1.])],
        [torch.tensor([False]), torch.tensor([False]), torch.tensor([True])],
        [torch.tensor([False]), torch.tensor([False]), torch.tensor([True])],
        torch.tensor([100.]), gamma=.5)
    torch.testing.assert_close(result[:, 0], torch.tensor([.8, .9, 1.]))


def test_external_progress_discounts_and_task_boundary_keeps_bootstrap():
    # Completing an ordinary task does not terminate the lifetime. Its next
    # task's value survives; only actual external progress applies gamma.
    result = activity_returns(
        [torch.tensor([1.]), torch.tensor([0.])],
        [torch.tensor([False]), torch.tensor([False])],
        [torch.tensor([True]), torch.tensor([False])],
        torch.tensor([4.]), gamma=.5)
    torch.testing.assert_close(result[:, 0], torch.tensor([3., 4.]))


def test_mixed_streams_stop_bootstrap_only_at_their_lifetime_end():
    result = activity_returns(
        [torch.tensor([2., 2., 2.])],
        [torch.tensor([True, False, False])],
        [torch.tensor([True, True, False])],
        torch.tensor([10., 10., 10.]), gamma=.5)
    torch.testing.assert_close(result[0], torch.tensor([2., 7., 12.]))
