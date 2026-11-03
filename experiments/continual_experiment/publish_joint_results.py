"""Export the fixed joint-learning comparison, preserving task-stream pairing."""
import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

from .summarize_response_families import paired


SEEDS = (201, 202, 203, 204)


def read(path):
    return json.loads(path.read_text())


def statistics(rows):
    return dict(tasks=len(rows), completed=sum(int(r['completed']) for r in rows),
                operations=float(np.mean([r['native_actions'] for r in rows])))


def groups(rows):
    result = {'primary': [r for r in rows if r.get('group') != 'renamed_inputs']}
    for field in ('group', 'family', 'exposure'):
        for value in sorted({r[field] for r in rows if field in r}):
            result[f'{field}:{value}'] = [r for r in rows if r.get(field) == value]
    return result


def comparison(left, right):
    a, b = groups(left), groups(right)
    return {name: paired(rows, b[name]) for name, rows in a.items()
            if name in b and not name.startswith('family:')}


def rename_comparison(rows, manifest):
    sources = {s['name']: s['source_stream'] for s in manifest['streams'] if 'source_stream' in s}
    renamed = [{**r, 'stream': sources[r['stream']]} for r in rows if r['stream'] in sources]
    originals = [r for r in rows if r['stream'] in sources.values()]
    return dict(renamed=statistics(renamed), original=statistics(originals),
                paired=paired(renamed, originals),
                meaning='Same simulator laws and sequence; only consistent public ingredient names differ')


def write_csv(path, rows):
    with path.open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def native_rows(path, domain):
    result = []
    for stream in read(path):
        for row in stream[domain]:
            result.append(dict(stream=stream['stream'], task_index=row['episode_index'],
                native_actions=row['native_actions'], completed=int(row['completed_successfully']),
                exposure=row.get('exposure', row.get('boundary', 'unspecified')),
                seed=row['seed']))
    return result


def predictive_comparison(root, names, manifest):
    """Compare the two forecasts on exactly the same returned experiments."""
    stream_groups = {s['name']: s['group'] for s in manifest['streams']}
    result = {}
    for name in names:
        path = root / 'test' / name / 'predictions.jsonl'
        accumulators = {}
        for line in path.read_text().splitlines():
            row = json.loads(line)
            if 'relation_probabilities' not in row:
                continue
            group = stream_groups[row['stream']]
            keys = [group] + (['primary'] if group != 'renamed_inputs' else [])
            base, relation = row['probabilities'], row['relation_probabilities']
            y = row['after']
            values = [1, -math.log(max(base[y], 1e-30)), -math.log(max(relation[y], 1e-30)),
                      int(max(range(4), key=base.__getitem__) == y),
                      int(max(range(4), key=relation.__getitem__) == y)]
            for key in keys:
                if key not in accumulators:
                    accumulators[key] = np.zeros(5)
                accumulators[key] += values
        if accumulators:
            result[name] = {key: dict(predictions=int(v[0]), inherited_nll=v[1]/v[0],
                response_function_nll=v[2]/v[0], inherited_accuracy=v[3]/v[0],
                response_function_accuracy=v[4]/v[0]) for key, v in accumulators.items()}
    return dict(conditions=result,
        scope='Both forecasts scored on each policy\'s actually selected next experiments; histories differ across trained policies',
        units='nats per observed categorical outcome; zero probabilities clipped to 1e-30')


def export(root, output):
    output.mkdir(parents=True, exist_ok=True)
    plan, manifest = read(root / 'plan.json'), read(root / 'test_cohort.json')
    expected = {(s['name'], i) for s in manifest['streams'] for i in range(len(s['tasks']))}
    records, conditions, rows = {}, {}, []
    paths = {j['name']: root / 'test' / j['name'] / 'summary.json' for j in plan['jobs']}
    paths.update({f'parent{p}': root / 'test' / f'parent{p}' / 'summary.json' for p in (911, 912)})
    paths.update({f'information{s}': root / 'references' / f'information{s}' / 'summary.json' for s in SEEDS})
    for name, path in paths.items():
        data = read(path)
        items = data['records']
        if len(items) != len(expected) or {(r['stream'], r['task_index']) for r in items} != expected:
            raise ValueError(f'Incomplete or duplicated stream data: {name}')
        records[name] = items
        conditions[name] = {key: statistics(value) for key, value in groups(items).items()}
        for row in items:
            rows.append(dict(condition=name, **{k: row[k] for k in ('stream', 'task_index',
                'group', 'family', 'exposure', 'intervening_tasks', 'native_actions', 'completed',
                'returned_experiments', 'inference_seconds')}))
    comparisons = {}
    for index, seed in enumerate(SEEDS):
        left = f'joint{seed}'
        for right in (f'prediction_only{seed}', f'inherited{seed}', f'information{seed}',
                      f'parent{911 + index % 2}'):
            comparisons[f'{left}:{right}'] = comparison(records[left], records[right])
        comparisons[f'prediction_only{seed}:inherited{seed}'] = comparison(
            records[f'prediction_only{seed}'], records[f'inherited{seed}'])
    result = dict(conditions=conditions, comparisons=comparisons,
        renamed={name: rename_comparison(value, manifest) for name, value in records.items()},
        scope=manifest['scope'], repetitions=plan['repetitions'],
        interval='Paired bootstrap of complete streams, separately for each training repetition',
        primary='84 streams / 1008 task instances; consistently renamed replays excluded',
        timing='Evaluation overlaps training; CPU process interruptions were rerun on CUDA. Timings are not isolated latency measurements',
        selection='All fixed final endpoints, no selection by this test cohort')
    (output / 'joint_belief.json').write_text(json.dumps(result, indent=2) + '\n')
    (output / 'joint_belief_test_cohort.json').write_text(json.dumps(manifest, indent=2) + '\n')
    write_csv(output / 'joint_belief_instances.csv', rows)
    forecasts = predictive_comparison(root, [j['name'] for j in plan['jobs']], manifest)
    (output / 'joint_belief_predictions.json').write_text(json.dumps(forecasts, indent=2) + '\n')

    native, native_summary, native_export = {}, {}, []
    for job in plan['jobs']:
        name = job['name']
        directory = root / 'native' / name
        if not (directory / 'finished.json').exists():
            raise ValueError(f'Native comparison unfinished: {name}')
        native[name] = {d: native_rows(directory / 'summary.json', d) for d in ('physics', 'chemistry')}
    for parent in (911, 912):
        native[f'parent{parent}'] = {
            'physics': native_rows(root.parent / 'experimental_memory_20261101' / 'native' /
                                  f'initial{parent}' / 'summary.json', 'physics'),
            'chemistry': native_rows(root.parent / 'continual_belief_20261102' / 'parent_memory' /
                                    f'parent_memory{parent}' / 'summary.json', 'chemistry')}
    for name, domains in native.items():
        native_summary[name] = {}
        for domain, data in domains.items():
            if len(data) != 240 or len({(r['stream'], r['task_index']) for r in data}) != 240:
                raise ValueError(f'Native cohort differs: {name}/{domain}')
            native_summary[name][domain] = statistics(data)
            native_export.extend(dict(condition=name, domain=domain, **r) for r in data)
    native_comparisons = {}
    for index, seed in enumerate(SEEDS):
        for right in (f'parent{911 + index % 2}', f'prediction_only{seed}', f'inherited{seed}'):
            native_comparisons[f'joint{seed}:{right}'] = {
                d: paired(native[f'joint{seed}'][d], native[right][d]) for d in ('physics', 'chemistry')}
    native_result = dict(conditions=native_summary, comparisons=native_comparisons,
        scope='Original 20-stream native cohort and identical saved initial states; retention, not fresh generalization')
    (output / 'joint_belief_native.json').write_text(json.dumps(native_result, indent=2) + '\n')
    write_csv(output / 'joint_belief_native_instances.csv', native_export)

    training = []
    for job in plan['jobs']:
        for line in (root / 'train' / job['name'] / 'training.jsonl').read_text().splitlines():
            row = json.loads(line)
            training.append(dict(condition=job['name'], update=row['update'], seconds=row['seconds'],
                loss=row['loss'], physical_loss=row['physical_loss'], chemical_loss=row['chemical_loss'],
                response_nll=row['chemical_nll'],
                relation_nll=row['chemical_activity_learning'].get('relation_nll', 0),
                native_actions=row['chemical_activity_learning']['native_actions'],
                returned=row['chemical_activity_learning']['returned']))
    write_csv(output / 'joint_belief_training.csv', training)
    figure(result, native_result, output)
    return result


def figure(digital, native, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'svg.fonttype': 'none', 'pdf.fonttype': 42})
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), layout='constrained')
    colors = ['#246A83', '#CB6B42', '#60965D', '#8B6EA8']
    for ax, kind, title in zip(axes, ('digital', 'chemistry', 'physics'),
                              ('New digital task streams', 'Native chemistry: retention', 'Native physics: retention')):
        names = ('Joint action + response', 'Response gradients only', 'Inherited continuation', 'Original parent')
        for i, (label, color) in enumerate(zip(names, colors)):
            keys = [f'{("joint", "prediction_only", "inherited")[i]}{s}' for s in SEEDS] if i < 3 else ['parent911', 'parent912']
            vals = [digital['conditions'][k]['primary'] if kind == 'digital' else native['conditions'][k][kind] for k in keys]
            x = np.array([r['operations'] for r in vals])
            offsets = np.linspace(-.12, .12, len(x))
            ax.scatter(x, i + offsets, s=28, color=color, zorder=3)
            ax.plot([x.min(), x.max()], [i, i], color=color, alpha=.5)
        ax.set_yticks(range(4), names)
        ax.invert_yaxis()
        ax.set_xlabel('Mean operations (failures included)')
        ax.set_title(title, loc='left', fontsize=10, fontweight='bold')
        ax.grid(axis='x', alpha=.16)
    fig.get_layout_engine().set(w_pad=.12, h_pad=.1)
    for suffix in ('svg', 'pdf', 'png'):
        fig.savefig(output / f'joint_belief.{suffix}', dpi=180)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--output', type=Path, default=Path('experiments/continual_experiment/results'))
    args = p.parse_args()
    result = export(args.root, args.output)
    print(json.dumps({k: v['primary'] for k, v in result['conditions'].items()}, indent=2))


if __name__ == '__main__':
    main()
