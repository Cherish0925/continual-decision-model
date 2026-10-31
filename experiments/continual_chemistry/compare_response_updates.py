"""Readout interventions on the same native, model-selected observation history.

Replay a saved trajectory with its original model. At each selected experiment,
compare the retained response correction with zero correction and one additional
learning update at the identical state. These predictions receive the same next
actual outcome. This measures prediction effects, not a changed acting policy.
"""
import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from cdm.continual_experiment import Activity
from experiments.continual_experiment.shared_tasks import load_shared_checkpoint
from .active_training import choose_activity
from .observations import observe, feedback


def prediction_effects(rows, tolerance=1e-9):
    """Descriptive paired loss effects; uncertainty is computed over streams."""
    differences = np.asarray([r['retained_nll'] - r['zero_nll'] for r in rows],
                             dtype=np.float64)
    improved = differences < -tolerance
    worsened = differences > tolerance
    unchanged = ~(improved | worsened)
    return {
        'predictions': len(rows),
        'improved': int(improved.sum()),
        'worsened': int(worsened.sum()),
        'unchanged': int(unchanged.sum()),
        'mean_improvement_nats': float(-differences[improved].mean()) if improved.any() else None,
        'mean_deterioration_nats': float(differences[worsened].mean()) if worsened.any() else None,
        'equality_tolerance': tolerance,
    }


def evaluate(checkpoint, root, output, device):
    torch.set_num_threads(1)
    shared, saved = load_shared_checkpoint(checkpoint, device)
    model = shared.tasks['chemistry'].eval()
    streams = json.loads((root / 'summary.json').read_text())
    rows, episode_rows = [], []
    started = perf_counter()
    with torch.no_grad():
        for stream in sorted(streams, key=lambda s: s['stream']):
            state = model.initial(1, device)
            for episode in sorted(stream['chemistry'], key=lambda r: r['episode_index']):
                previous = state
                state = model.initial(1, device)
                for key in ('h', 'z', 'response_slow'):
                    state[key] = previous[key]
                directory = root / stream['stream'] / f"chemistry{episode['episode_index']:03d}_seed{episode['seed']}"
                records = [json.loads(line) for line in (directory / 'experiments.jsonl').open()]
                public = records[0]['state']
                elapsed = records[0]['transition_start']
                obs, _ = observe(public, elapsed_actions=elapsed, device=device, candidate_recipes=[tuple(records[0]['component_counts'])])
                state = model.receive(state, obs)
                local = []
                for record in records:
                    leave = record.get('activity') == 'leave'
                    counts = (0,) * len(public['components']) if leave else tuple(record['component_counts'])
                    obs, _ = observe(public, elapsed_actions=elapsed, device=device, candidate_recipes=[counts])
                    state, _, _, _, _ = choose_activity(model, state, obs, sample=False)
                    out = model.readout(state, obs['candidates'], obs['action_mask'])
                    if not leave:
                        dropped = {**state, 'response_fast': state['response_fast'] * 0,
                                   'response_slow': state['response_slow'] * 0}
                        zero = model.readout(dropped, obs['candidates'], obs['action_mask'])
                        extra_state = model.internal(state, torch.tensor([Activity.LEARN], device=device), obs)
                        extra = model.readout(extra_state, obs['candidates'], obs['action_mask'])
                        y = record['rust_after']
                        row = dict(stream=stream['stream'], episode_index=episode['episode_index'],
                            experiment_index=record['index'], seed=episode['seed'],
                            retained_nll=-float(out['outcomes'][0, 0, y]),
                            zero_nll=-float(zero['outcomes'][0, 0, y]),
                            extra_update_nll=-float(extra['outcomes'][0, 0, y]),
                            replay_log_probability_error=abs(float(out['outcomes'][0, 0, y]) - record['model_outcome_logits'][y]))
                        rows.append(row); local.append(row)
                    action = torch.zeros(1, dtype=torch.long, device=device)
                    state = model.begin_action(state, out['features'], action)
                    state = model.feedback(state, feedback(record, device))
                    if not leave:
                        state = model.record_response(state, out, action,
                            torch.tensor([record['rust_before']], device=device),
                            torch.tensor([record['rust_after']], device=device))
                        public = record['next_state']
                        elapsed = record['transition_stop']
                    else:
                        elapsed += record['native_actions']
                    obs, _ = observe(public, elapsed_actions=elapsed, device=device, candidate_recipes=[counts])
                    state = model.receive(state, obs)
                episode_rows.append(dict(stream=stream['stream'], episode_index=episode['episode_index'],
                    n=len(local), **{key: sum(r[key] for r in local) / len(local)
                    for key in ('retained_nll', 'zero_nll', 'extra_update_nll')}))
            print(json.dumps({'stream': stream['stream'], 'transitions': len(rows)}), flush=True)
    names = sorted({row['stream'] for row in rows})
    totals = np.array([[sum(r[k] for r in rows if r['stream'] == name)
                       for k in ('retained_nll', 'zero_nll', 'extra_update_nll')] for name in names])
    counts = np.array([sum(r['stream'] == name for r in rows) for name in names])
    selected = np.random.default_rng(931).integers(len(names), size=(10000, len(names)))
    means = totals[selected].sum(1) / counts[selected].sum(1)[:, None]
    summary = dict(transitions=len(rows), streams=len(names), seconds=perf_counter() - started,
        mean_nll=dict(zip(('retained', 'zero', 'extra_update'), (totals.sum(0) / counts.sum()).tolist())),
        retained_minus_zero_95ci=np.quantile(means[:, 0] - means[:, 1], [.025, .975]).tolist(),
        extra_minus_retained_95ci=np.quantile(means[:, 2] - means[:, 0], [.025, .975]).tolist(),
        paired_prediction_effects=prediction_effects(rows),
        maximum_replay_log_probability_error=max(r['replay_log_probability_error'] for r in rows))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(dict(scope='same-state prediction interventions; post-result analysis on fixed native trajectories',
        source_checkpoint=str(checkpoint), summary=summary, episodes=episode_rows, predictions=rows), indent=2) + '\n')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True, type=Path)
    p.add_argument('--trajectories', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--device', default='cpu')
    args = p.parse_args()
    evaluate(args.checkpoint, args.trajectories, args.output, args.device)
