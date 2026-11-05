"""Compare fixed-data training with and without executed-action history."""
import argparse
from collections import Counter
import json
from pathlib import Path


def summarize(folder, mode):
    history = [json.loads(line) for line in (folder / mode / 'training.jsonl').read_text().splitlines()]
    if len(history) != 8 or history[-1]['stage_epoch'] != 8:
        raise ValueError('Expected the declared eight-epoch endpoint')
    evaluation = folder / (mode + '-development')
    cases = json.loads((evaluation / 'cases.json').read_text())
    if any(row['status'] != 'evaluated' for row in cases):
        raise ValueError('Execution includes an unfinished or invalid case')
    counts, verbs = Counter(), Counter()
    for case in cases:
        actions = [json.loads(line) for line in (evaluation / case['id'] / 'actions.jsonl').read_text().splitlines()]
        if len(actions) != case['steps']:
            raise ValueError('Action count does not match the case summary')
        counts['actions'] += len(actions)
        counts['constant_action_cases'] += int(bool(actions) and all(row['action'] == actions[0]['action'] for row in actions))
        for first, second in zip(actions, actions[1:]):
            if not first['result']['success']:
                counts['failed_actions_with_following_action'] += 1
                counts['same_action_after_failure'] += int(first['action'] == second['action'])
        verbs.update(row['action'].get('action', 'DIALOG') for row in actions)
    return dict(epochs=history, cases=cases, successes=sum(row['success'] for row in cases),
                evaluated=len(cases), actions=dict(counts), verbs=dict(verbs))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--history', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    first = json.loads((args.baseline / 'data.json').read_text())
    second = json.loads((args.history / 'data.json').read_text())
    if any(first[key] != second[key] for key in ('training_files', 'development_files')):
        raise ValueError('The two conditions do not use identical trajectories')
    conditions = {mode: {name: summarize(folder, mode) for name, folder in
                         [('reference', args.baseline), ('action_history', args.history)]}
                  for mode in ('cdm', 'gru')}
    paired = {}
    for mode, rows in conditions.items():
        reference = {row['id']: row for row in rows['reference']['cases']}
        candidate = {row['id']: row for row in rows['action_history']['cases']}
        shared = sorted(reference.keys() & candidate.keys())
        paired[mode] = dict(cases=shared, reference_successes=sum(reference[key]['success'] for key in shared),
                           history_successes=sum(candidate[key]['success'] for key in shared))
    result = dict(training_episodes=len(first['training_files']), development_episodes=len(first['development_files']),
        training_seed=301, endpoint='fixed eighth epoch, last.pt', action_history_capacity=8,
        scope='Completed demonstration imitation, independent Easy native execution. This is not the official 120-case benchmark.',
        feedback='Actual actions and public returned feedback; no expert proposal, reward or scorecard in the history input.',
        conditions=conditions, matched_cases=paired)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps({mode: {name: dict(successes=value['successes'], evaluated=value['evaluated'])
                            for name, value in rows.items()} for mode, rows in conditions.items()}))


if __name__ == '__main__':
    main()
