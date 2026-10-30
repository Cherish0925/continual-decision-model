"""Compare batched transitions and public observations with native records."""
import argparse
import json
from pathlib import Path
import torch

from .active_environment import ActiveChemicalEnvironment
from .observations import observe


def check(root):
    torch.set_num_threads(1)
    env = ActiveChemicalEnvironment(root, 1, 1, 'cpu')
    returned = observations = 0
    for entry in env.entries:
        path = root / entry['directory'] / 'experiments.jsonl'
        records = [json.loads(line) for line in path.read_text().splitlines()]
        env.lifetime[0] = [entry, entry]; env.task_index[0] = 1
        env.terminal[:] = True; env.reset_finished()
        allowed = env.allowed[0].nonzero().flatten()
        n = len(records[0]['component_counts'])
        choices = [tuple(q[:n].int().tolist()) for q in env.counts[0, allowed]]
        # Full vectorized observation equality at three histories per episode;
        # every actual returned transition is checked below.
        positions = {0, len(records) // 2, len(records) - 1}
        for index, record in enumerate(records):
            if index in positions:
                predicted = env.observation()
                native, _ = observe(record['state'], elapsed_actions=record['transition_start'],
                                    candidate_recipes=choices)
                torch.testing.assert_close(predicted['event'], native['event'])
                length = native['evidence'].shape[1]
                torch.testing.assert_close(predicted['evidence'][:, :length], native['evidence'])
                torch.testing.assert_close(predicted['candidates'][:, allowed], native['candidates'])
                torch.testing.assert_close(predicted['action_mask'][:, allowed], native['action_mask'])
                observations += 1
            action = allowed[choices.index(tuple(record['component_counts']))][None]
            result = env.step(action)
            assert int(result['after'][0]) == record['rust_after']
            assert int(result['cost'][0]) == record['native_actions']
            assert bool(result['returned'][0])
            returned += 1
    return {'episodes': len(env.entries), 'actual_returns_matched': returned,
            'public_prefixes_matched': observations}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(check(args.data)))
