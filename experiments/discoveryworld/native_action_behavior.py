"""Summarize executed native actions, including the choice immediately after a failed action."""
import argparse
from collections import Counter
import json
from pathlib import Path


def lines(path):
    with path.open() as stream:
        return [json.loads(line) for line in stream]


def summarize(root):
    result = {}
    opposite = dict(north='south', south='north', east='west', west='east')
    for path in sorted(root.glob('*/condition.json')):
        folder = path.parent
        runs = {}
        for split in ('evaluation-seeded', 'development-greedy', 'development-sampled'):
            cases = json.loads((folder / split / 'cases.json').read_text())
            if len(cases) != (120 if split == 'evaluation-seeded' else 8):
                raise ValueError(f'Incomplete case list: {folder.name}/{split}')
            stats, verbs, accepted, pairs = Counter(), Counter(), Counter(), Counter()
            for case in cases:
                rows = lines(folder / split / case['id'] / 'actions.jsonl')
                if len(rows) != case['steps'] or case['status'] != 'evaluated':
                    raise ValueError(f'Incomplete case: {case["id"]}')
                acts = [r['action'] for r in rows]
                ok = [bool(r['result']['success']) for r in rows]
                stats.update(cases=1, actions=len(rows), accepted=sum(ok), successes=int(case['success']),
                             constant_cases=int(all(a == acts[0] for a in acts)), no_accepted_cases=int(not any(ok)))
                for i, (action, success) in enumerate(zip(acts, ok)):
                    verb = action.get('action', 'DIALOG')
                    verbs[verb] += 1
                    if success:
                        accepted[verb] += 1
                    if i + 1 == len(rows):
                        continue
                    after = acts[i+1]
                    prior = 'accepted' if success else 'failed'
                    pairs[prior + ('_repeat' if action == after else '_switch')] += 1
                    if verb == after.get('action') == 'MOVE_DIRECTION' and success and ok[i+1]:
                        pairs['consecutive_accepted_moves'] += 1
                        pairs['opposite_directions'] += int(opposite[action['arg1']] == after['arg1'])
            runs[split] = {**stats, 'verbs': verbs, 'accepted_verbs': accepted, 'adjacent_pairs': pairs}
        training, updates = lines(folder / 'training.jsonl'), lines(folder / 'updates.jsonl')
        wins = Counter(row['scenario'] for row in training if row['completed_successfully'])
        result[folder.name] = dict(training_successes_by_scenario=wins, updates=len(updates),
                                   nonzero_signed_update_reward=sum(abs(r['reward']) > 1e-8 for r in updates),
                                   execution=runs)
    if len(result) != 4:
        raise ValueError('Expected the four v0.18.0 native endpoints')
    return result


def handoffs(root):
    """Keep blocked reference cases in the denominator and report every evaluated suffix."""
    refs = sorted(root.glob('reference/*/trace.json'))
    if not refs:
        raise ValueError('Missing reference traces')
    references = [json.loads(p.read_text()) for p in refs]
    seeds = sorted(r['seed'] for r in references)
    out = {'scenario': 'Reactor Lab', 'difficulty': 'Normal', 'seeds': seeds,
           'scope': 'One action before specialist completion; finite independent continuation, no weight updates.',
           'handoff_public_hashes': {r['seed']: r['trace'][-1]['before'] for r in references
                                     if r['status'] == 'success'}, 'policies': {}}
    contexts = None
    names = ['uniform', 'autonomous-cdm', 'autonomous-gru', 'basic_operations-cdm', 'basic_operations-gru']
    if (root / 'autonomous-cdm-reset').is_dir():
        names.append('autonomous-cdm-reset')
    for name in names:
        rows = [json.loads((root / name / str(seed) / 'result.json').read_text()) for seed in seeds]
        tested = [r for r in rows if r['status'] == 'evaluated']
        keys = [(r['seed'], r['reference_steps'], r['progress_at_handoff'], r['reference_action'], r['candidates'])
                for r in tested]
        if contexts is not None and keys != contexts:
            raise ValueError('Handoff conditions do not match')
        contexts = keys
        out['policies'][name] = dict(attempted=len(rows), successful_references=len(tested),
            first_matches=sum(r['first_matches_reference'] for r in tested),
            successes=sum(r['success'] for r in tested),
            first_actions=Counter(json.dumps(r['actions'][0]['action'], sort_keys=True) for r in tested),
            constant_suffixes=sum(all(a['action'] == r['actions'][0]['action'] for a in r['actions']) for r in tested),
            cases=rows)
    if 'autonomous-cdm-reset' in out['policies']:
        a = out['policies']['autonomous-cdm']['cases']
        b = out['policies']['autonomous-cdm-reset']['cases']
        pairs = [(x, y) for x, y in zip(a, b) if x['status'] == y['status'] == 'evaluated']
        out['same_weight_reset'] = {'evaluated_pairs': len(pairs),
                                   'identical_action_sequences': sum(x['actions'] == y['actions'] for x, y in pairs),
                                   'identical_completions': sum(x['success'] == y['success'] for x, y in pairs)}
    return out


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs', type=Path, required=True)
    p.add_argument('--handoffs', type=Path)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    stats = {'released_behavior': summarize(a.runs)}
    if a.handoffs:
        stats['assisted_development'] = handoffs(a.handoffs)
    a.out.write_text(json.dumps(stats, indent=2, sort_keys=True) + '\n')
