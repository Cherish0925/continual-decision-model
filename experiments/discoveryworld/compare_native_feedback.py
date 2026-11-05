"""Compare fixed-data training with and without executed-action history."""
import argparse
from collections import Counter
from contextlib import redirect_stdout
import io
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


def training_coverage(data_file, data_root):
    """Use the actual supervision selector without computing neural features."""
    from .features import FeatureConfig
    from .train import load_episodes, prepare_episodes
    class LabelCache:
        config = FeatureConfig(mode='lexical', measurement_memory=True, object_relations=True)
        def encode(self, *args):
            return Path('not-used-for-label-counts')
    data = json.loads(data_file.read_text())
    episodes = load_episodes([data_root / path for path in data['training_files']])
    with redirect_stdout(io.StringIO()):
        prepared = prepare_episodes(episodes, LabelCache(), allow_failed_actions=True,
                                   imitate_successful_episodes=True)
    counts, themes = Counter(), {}
    for episode in prepared:
        local = Counter()
        rows = episode.episode.transitions
        for index, step in enumerate(episode.steps):
            label = step.expert_target >= 0
            failed = not rows[index]['action_result'].get('success', False)
            local['transitions'] += 1
            local['imitation_labels'] += int(label)
            local['failed_commands'] += int(failed)
            local['received_outcome_targets'] += int(step.executed_target >= 0)
            if index and not rows[index-1]['action_result'].get('success', False):
                local['states_after_failure'] += 1
                local['labels_after_failure'] += int(label)
                if label:
                    local['changed_action_labels_after_failure'] += int(rows[index].get('expert_action', rows[index]['action']) != rows[index-1]['action'])
        counts.update(local)
        themes.setdefault(episode.episode.metadata['scenario'], Counter()).update(local)
    return dict(episodes=len(prepared), counts=dict(counts), themes={k:dict(v) for k,v in themes.items()},
        scope='Immediate previous command failed. Valid teacher actions in successful episodes supply imitation; student actions do not.')


def inspect_decisions(checkpoint, evaluation, cache_path, *, device='cpu', encoder_path=None,
                      states_per_case=4, fixed_steps=32):
    """Teacher-force recorded actions; intervene before the same next decision.

    This measures scores at sampled recorded states. It neither solves an
    environment nor establishes that a feature is globally unused. Repeating a
    frozen input is a separate state-stability probe, not a native-world rollout.
    """
    import torch
    from cdm.digital import public_candidates
    from .evaluate import LearnedPolicy
    from .features import batch_features, canonical_action, load_features
    from .train import load_episodes, public_observation
    agent = LearnedPolicy(checkpoint, cache_path, device=device, encoder_path=encoder_path)
    cases = json.loads((evaluation / 'cases.json').read_text())
    if any(row['status'] != 'evaluated' for row in cases):
        raise ValueError('Use a completed native execution set')
    samples, fixed, counts = [], [], Counter()

    def scores(values, repeated):
        values = values[0].float()
        valid = torch.isfinite(values)
        prob = values.softmax(-1)
        alternatives = values.clone(); alternatives[repeated] = -torch.inf
        return dict(margin=float(values[repeated]-alternatives.max()),
                    probability=float(prob[repeated]), selected=int(values.argmax()),
                    repeated_selected=int(values.argmax()) == repeated,
                    entropy=float(-(prob[valid] * prob[valid].clamp_min(1e-30).log()).sum())), prob

    def changed(a, b, *, ignore_clock=False):
        if a is None or a.keys() != b.keys():
            return True
        for name in a:
            if ignore_clock and name in ('timing', 'measurement_age'):
                continue
            x, y = a[name], b[name]
            if x.shape != y.shape:
                return True
            if ignore_clock and name == 'numeric':
                x, y = x.clone(), y.clone(); x[:, 2] = 0; y[:, 2] = 0
            if not torch.equal(x, y):
                return True
        return False

    with torch.no_grad():
        for case in cases:
            episode = load_episodes([evaluation / case['id'] / 'trajectory.jsonl'])[0]
            failures = [i for i in range(1, len(episode.transitions))
                        if not episode.transitions[i-1]['action_result'].get('success', False)]
            chosen = set(failures[round(j*(len(failures)-1)/max(1, states_per_case-1))]
                         for j in range(min(states_per_case, len(failures))))
            agent.reset()
            previous_features = previous_candidates = None
            for index, row in enumerate(episode.transitions):
                observation = public_observation(episode, row)
                if agent.measurements is not None:
                    agent.measurements.observe_initial(observation)
                    observation = {**observation, 'measurement_records':agent.measurements.summaries()}
                    if agent.features.config.object_relations:
                        observation['measurement_dialog_owner'] = agent.measurements.dialog_owner
                observation = agent.memory.observe(observation)
                agent.previous_observation = observation
                observation = agent.action_history.augment(observation)
                candidates = public_candidates(observation)
                obs, actions = batch_features([load_features(str(agent.features.encode(observation, candidates)))], device)
                before = agent.state
                logits, after, _ = agent.model(obs, actions, before, mode=agent.mode, persistent=agent.persistent,
                    restart_workspace=agent.restart_workspace, predict_outcomes=False)
                keys = [canonical_action(c.action) for c in candidates]
                counts['replayed_decisions'] += 1
                counts['replay_matches_recorded_greedy'] += int(canonical_action(row['action']) == keys[int(logits[0].argmax())])
                if index and not episode.transitions[index-1]['action_result'].get('success', False):
                    repeated = canonical_action(episode.transitions[index-1]['action'])
                    repeat = canonical_action(row['action']) == repeated
                    counts['after_failure'] += 1
                    counts['repeated_after_failure'] += int(repeat)
                    for label, ignored in [('encoded_input_changed', False), ('changed_without_clock', True)]:
                        if changed(previous_features, obs, ignore_clock=ignored) or changed(previous_candidates, actions):
                            counts[label] += 1; counts[label + '_and_repeated'] += int(repeat)
                    if index in chosen and repeated in keys and len(keys) > 1:
                        target = keys.index(repeated)
                        base, probability = scores(logits, target)
                        variants = {}
                        no_history = {**observation, 'action_history':[]}
                        cleared, _ = batch_features([load_features(str(agent.features.encode(no_history, candidates)))], device)
                        for name, features, state in [('without_history', cleared, before),
                                                     ('reset_state', obs, agent.model.initial(1, device))]:
                            output, _, _ = agent.model(features, actions, state, mode=agent.mode,
                                persistent=agent.persistent, restart_workspace=agent.restart_workspace, predict_outcomes=False)
                            summary, other = scores(output, target)
                            variants[name] = {**summary, 'margin_change':summary['margin']-base['margin'],
                                              'probability_total_variation':float((probability-other).abs().sum()/2)}
                        event = agent.model.encode(obs)
                        stale, _ = agent.model.readout(event, actions, before, mode=agent.mode, predict_outcomes=False)
                        summary, other = scores(stale, target)
                        variants['without_current_state_update'] = {**summary, 'margin_change':summary['margin']-base['margin'],
                            'probability_total_variation':float((probability-other).abs().sum()/2)}
                        samples.append(dict(case=case['id'], step=index, repeated_action=episode.transitions[index-1]['action'],
                                            baseline=base, interventions=variants))
                        if index == min(chosen):
                            probe = after
                            first_change = None
                            initial_action = int(logits[0].argmax())
                            for tick in range(1, fixed_steps+1):
                                output, probe, _ = agent.model(obs, actions, probe, mode=agent.mode, predict_outcomes=False)
                                if int(output[0].argmax()) != initial_action:
                                    first_change = tick; break
                            fixed.append(dict(case=case['id'], step=index, first_change=first_change,
                                              limit=fixed_steps, censored=first_change is None))
                agent.state = after
                if getattr(agent.model.config, 'action_conditioned', False):
                    agent.previous_features, agent.previous_candidates = (obs, actions), candidates
                agent.observe(row)
                previous_features, previous_candidates = obs, actions
    agent.features.close()
    return dict(counts=dict(counts), samples=samples, fixed_input=fixed,
        fixed_input_definition='Reassimilate the identical encoded observation, clock and candidate set; no environment action or new feedback.',
        intervention_scope='Fixed recorded observations; removing history or resetting state can be outside the training distribution.',
        parameters=sum(p.numel() for p in agent.model.parameters()), mode=agent.mode,
        history_capacity=agent.features.config.action_history, features=agent.features.config.mode)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path)
    parser.add_argument('--history', type=Path)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--evaluation', type=Path)
    parser.add_argument('--feature-cache', type=Path)
    parser.add_argument('--encoder-path', type=Path)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--states-per-case', type=int, default=4)
    parser.add_argument('--fixed-steps', type=int, default=32)
    parser.add_argument('--coverage', type=Path)
    parser.add_argument('--data-root', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.coverage or args.checkpoint:
        import torch
        torch.set_num_threads(args.threads)
        if args.coverage:
            if args.data_root is None: parser.error('--coverage requires --data-root')
            result = training_coverage(args.coverage, args.data_root)
        else:
            if args.evaluation is None or args.feature_cache is None:
                parser.error('--checkpoint requires --evaluation and --feature-cache')
            result = inspect_decisions(args.checkpoint, args.evaluation, args.feature_cache,
                device=args.device, encoder_path=args.encoder_path, states_per_case=args.states_per_case, fixed_steps=args.fixed_steps)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
        print(json.dumps(result['counts']))
        return
    if args.baseline is None or args.history is None:
        parser.error('Choose --baseline with --history, --coverage, or --checkpoint')
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
