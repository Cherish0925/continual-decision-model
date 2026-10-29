"""Verify matched initial weights and one full external-request window on CPU."""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path

import torch

from cdm.recurrent_feedback import RecurrentFeedbackConfig, configure_recurrent_feedback
from .train import initialize_experiment, observe, reference_receive, training_parser, train_window


def arguments_from_settings(settings, **overrides):
    values = {**settings, **overrides}
    arguments = []
    for action in training_parser()._actions:
        value = values.get(action.dest)
        if value is None or action.dest == 'help':
            continue
        option = next(name for name in action.option_strings if name.startswith('--'))
        if action.nargs == 0:
            if value:
                arguments.append(option)
        else:
            arguments += [option, str(value)]
    return arguments


def verify(primary, seed):
    root = primary / f'full_seed{seed}'
    settings = json.loads((root / 'config.json').read_text())
    args = training_parser().parse_args(arguments_from_settings(settings, device='cpu',
                                                               initial_state=root / 'initial.pt'))
    model, _, _, environment, delayed = initialize_experiment(args)
    saved = torch.load(root / 'initial.pt', map_location='cpu', weights_only=True)
    shared = model.state_dict()
    if set(shared) != set(saved['model']):
        raise ValueError('Shared state entries differ from the original training initialization')
    for key, value in shared.items():
        torch.testing.assert_close(value, saved['model'][key], rtol=0, atol=0)
    candidates = {}
    for source in ('workspace', 'evidence'):
        candidate = deepcopy(model)
        before_rng = torch.get_rng_state().clone()
        configure_recurrent_feedback(candidate, RecurrentFeedbackConfig(source))
        assert torch.equal(before_rng, torch.get_rng_state())
        for key, value in shared.items():
            torch.testing.assert_close(value, candidate.state_dict()[key], rtol=0, atol=0)
        extra = [value for key, value in candidate.state_dict().items() if key not in shared]
        assert len(extra) == 1 and not bool(extra[0].any())
        assert extra[0].numel() == model.core.config.width ** 2
        candidates[source] = candidate
    calls = dict(receive=0, readout=0)
    original_receive, original_readout = model.receive, model.readout

    def same(actual, expected):
        assert actual.keys() == expected.keys()
        for key in actual:
            torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)

    def receive(*items, **options):
        expected = original_receive(*items, **options)
        for candidate in candidates.values():
            same(candidate.receive(*items, **options), expected)
        calls['receive'] += 1
        return expected

    def readout(*items, **options):
        expected = original_readout(*items, **options)
        for candidate in candidates.values():
            same(candidate.readout(*items, **options), expected)
        calls['readout'] += 1
        return expected

    model.receive, model.readout = receive, readout
    reference = deepcopy(model.base).eval().requires_grad_(False)
    with torch.no_grad():
        state = model.initial(args.batch, 'cpu')
        obs, _ = observe(model, environment, state, delayed=delayed)
        state = model.receive(state, obs)
        reference_state = reference_receive(reference, reference.initial(args.batch, 'cpu'), obs,
                                            torch.ones(args.batch, dtype=torch.bool))
        _, _, _, stats = train_window(model, delayed, state,
            external_per_stream=args.external_per_stream, mode=args.mode,
            compute_cost=args.compute_cost, latency_cost=args.latency_cost, physical_weight=args.physical_weight,
            activity_entropy_weight=args.activity_entropy_initial, action_entropy_weight=args.action_entropy_weight,
            reference=reference, reference_state=reference_state, anchor_weight=args.anchor_weight)
    return dict(seed=seed, original_state_entries=len(shared), shared_weights_exact=True,
                extra_parameters_per_candidate=model.core.config.width ** 2,
                extra_weights_zero=True, install_preserves_random_state=True,
                first_window_external_requests=stats['external_requests'], comparison_calls=calls,
                initialization='exact update-zero weights from the original full condition',
                same_observation_outputs_exact=True, device='cpu',
                scope='One untrained matched window; initialization and execution equivalence, not performance')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--primary', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for seed in (231, 232):
        rows.append(verify(args.primary, seed))
        args.output.write_text(json.dumps(rows, indent=2) + '\n')
        print(json.dumps(rows[-1]), flush=True)


if __name__ == '__main__':
    main()
