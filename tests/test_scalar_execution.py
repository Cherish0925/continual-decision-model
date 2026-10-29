"""Execution semantics that affect feedback and the task's real action budget."""
import torch
from experiments.continual_physics.scalar_execution import ScalarExecution
from experiments.discoveryworld.scalar_probe import next_target


def test_intermediate_activation_and_environment_end():
    execution = ScalarExecution()
    tensor = lambda x: torch.tensor(x, dtype=torch.float64)
    result = execution(tensor([5000., 5000.]), tensor([3000., 3000.]), tensor([2., 2.]),
                       tensor([4000.5, 4000.5]), terminate_on_success=torch.tensor([False, True]))
    assert result['success'].tolist() == [True, True]
    assert result['actions'].tolist() == [3, 2]  # TALK, -1000, optionally EXIT.
    assert result['frequencies'][0, result['mask'][0]].tolist() == [5000., 4000.5]


def test_budget_stops_after_returned_setting_without_fake_exit():
    execution = ScalarExecution()
    f = lambda x: torch.tensor([x], dtype=torch.float64)
    result = execution(f(5000), f(3432), f(2), f(7000), torch.tensor([3]))
    assert result['actions'].item() == 3
    assert result['frequencies'][0, result['mask'][0]].tolist() == [5000., 4000., 3000.]
    assert not result['completed'].item()
    assert not result['success'].item()


def test_batched_probe_choice_matches_public_bfs():
    torch.manual_seed(41)
    execution = ScalarExecution()
    n = 24
    current = torch.randint(0, 10001, (n,)).double()
    mean = torch.rand(n).double() * 10000
    spread = torch.rand(n).double() * 30
    failed = torch.zeros(n, 10001, dtype=torch.bool)
    failed[torch.arange(n), current.long()] = True
    target, possible = execution._probe(current, mean, spread, failed, torch.full((n,), 16))
    options = {str(i): f'{"Increase" if delta > 0 else "Decrease"} frequency by {abs(delta)} Hz'
               for i, delta in enumerate((1000, 100, 10, 1, -1000, -100, -10, -1), 1)}
    dialog = {'is_in_dialog': True, 'dialogIn': 'The allowable range is 0 to 10,000 Hz.', 'dialogOptions': options}
    for i in range(n):
        expected = next_target(float(current[i]), float(mean[i]), float(spread[i]), dialog,
                               [float(current[i])], max_actions=16)
        assert bool(possible[i]) == (expected is not None)
        if expected is not None:
            assert float(target[i]) == expected
