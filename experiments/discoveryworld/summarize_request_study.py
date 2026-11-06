"""Summarize recorded shared-executor stages without counting repeats as new tasks."""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


def summarize(records, collections, study='shared_execution'):
    if study == 'language':
        from .language_study import CONDITIONS, SEEDS
        arms, seeds = CONDITIONS, SEEDS
    else:
        arms, seeds = ('self', 'demonstrations'), range(401, 409)
    planned_models = 2 * len(arms) * len(seeds)
    models = []
    outcomes = []
    aggregate = defaultdict(list)
    for mode in ('cdm', 'gru'):
        for arm in arms:
            for seed in seeds:
                name = f'{arm}_{mode}_s{seed}'
                matches = [p for p in records.rglob(name) if p.is_dir()]
                if not matches:
                    continue
                if len(matches) != 1:
                    raise ValueError(f'More than one copy of {name}; choose one record tree')
                job = matches[0]
                updates = read_rows(job/'interaction/updates.jsonl')
                initial = read_rows(job/'initial_imitation/training.jsonl')
                imitation = read_rows(job/'imitation/training.jsonl')
                episodes = read_rows(job/'interaction/training.jsonl')
                models.append(dict(model=name, initial_imitation_epochs=len(initial),
                    full_imitation_epochs=len(imitation), optimizer_updates=len(updates),
                    sampled_requests=updates[-1]['interactions'] if updates else 0,
                    finished_interaction_episodes=len(episodes),
                    interaction_training_complete=(job/'interaction.done.json').exists(),
                    interaction_training_successes=sum(x['completed_successfully'] for x in episodes),
                    native_actions_in_finished_episodes=sum(x['steps'] for x in episodes),
                    last_imitation=imitation[-1] if imitation else None,
                    complete=(job/'complete.json').exists()))
                for case_file in sorted(job.glob('*/cases.json')):
                    stage = case_file.parent.name
                    rows = json.loads(case_file.read_text())
                    for case in rows:
                        row = dict(model=name, arm=arm, mode=mode, initialization=seed, stage=stage)
                        # Public measurements only; no machine paths or provider metadata.
                        row.update({key: case[key] for key in ('id','theme','difficulty','seed','max_steps',
                            'status','steps','requests','success','progress','rollout_seconds',
                            'decision_seconds','environment_seconds') if key in case})
                        outcomes.append(row)
                        aggregate[(arm, mode, stage)].append(row)
    groups = []
    for (arm, mode, stage), rows in sorted(aggregate.items()):
        progress = [r['progress'] for r in rows if r.get('progress') is not None]
        actions = [r['steps'] for r in rows if r.get('steps') is not None]
        groups.append(dict(arm=arm, mode=mode, stage=stage,
            models=len({r['model'] for r in rows}), executions=len(rows),
            unique_instances=len({r['id'] for r in rows}),
            successes=sum(r.get('success',False) for r in rows), errors=sum(r['status']!='evaluated' for r in rows),
            recorded_progress=len(progress), mean_progress=sum(progress)/len(progress) if progress else None,
            recorded_native_cost=len(actions), mean_native_actions=sum(actions)/len(actions) if actions else None))
    data = {}
    collection_sizes = [('train',80), ('development',20)]
    if study != 'language':
        collection_sizes.append(('reference',20))
    for name, planned in collection_sizes:
        rows = [json.loads(p.read_text()) for p in (collections/name).glob('*/summary.json')]
        data[name] = dict(planned_instances=planned, latest_recorded_attempts=len(rows),
            successful_instances=sum(r['success'] for r in rows),
            policy_errors=sum(r['status']!='evaluated' for r in rows))
    completed = sum(m['complete'] for m in models)
    status = dict(as_of_utc=datetime.now(timezone.utc).isoformat(), status='complete' if completed==planned_models else 'running',
        study=study, planned_models=planned_models, models_with_recorded_updates=sum(
            bool(m['optimizer_updates'] or m['initial_imitation_epochs'] or m['full_imitation_epochs']) for m in models),
        completed_initial_imitation=sum(m['initial_imitation_epochs']==4 for m in models),
        completed_models=completed, models=models, collections=data,
        scope='Teacher completions are not student scores. Repeated model executions share instance identities. Counts may lag running processes.')
    snapshot=collections/'student_data/data-ready.json'
    if snapshot.exists():
        status['training_dataset']=json.loads(snapshot.read_text())
    results = dict(groups=groups, cases=outcomes,
        timing='Elapsed time was measured during concurrent training and evaluation; these runs do not isolate inference latency.',
        endpoint='Initial development scores use fixed four-epoch weights. Full imitation and interaction endpoints use their fixed final weights.')
    if study == 'language':
        status.pop('completed_initial_imitation')
        results['endpoint'] = 'Fixed 32-epoch supervised weights, then fixed 120-world interaction weights. No test score selects a checkpoint.'
    return status, results


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--records',type=Path,required=True)
    parser.add_argument('--collections',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--study',choices=('shared_execution','language'),default='shared_execution')
    args=parser.parse_args()
    status, results = summarize(args.records,args.collections,args.study)
    args.output.mkdir(parents=True,exist_ok=True)
    prefix = 'language_learning' if args.study == 'language' else 'common_execution'
    for name, value in ((prefix+'_status', status), (prefix, results)):
        (args.output/(name+'.json')).write_text(json.dumps(value,indent=2)+'\n')
    print(json.dumps(dict(models=status['models_with_recorded_updates'],groups=results['groups'])))


if __name__=='__main__':
    main()
