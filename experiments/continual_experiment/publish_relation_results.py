"""Build the public tables from completed response-function comparisons."""
import argparse
import csv
import json
from pathlib import Path
import shutil
import tempfile

from .summarize_response_families import summarize as digital_summary
from .evaluate_memory import summarize as native_summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--output', type=Path, default=Path(__file__).parent/'results')
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    manifest_path = args.root/'cohorts/test.json'
    manifest = json.loads(manifest_path.read_text())
    pairs = []
    for seed in (31,32):
        info = f'modulated{seed}_information'
        pairs += [(info, f'attentive_long{seed}_greedy'), (info, f'parent{seed+880}'),
                  (info, f'modulated{seed}_no_memory'), (info, f'long{seed}_information'),
                  (f'modulated{seed}_finite_plan',f'modulated{seed}_finite_marginal'),
                  (f'modulated{seed}_finite_plan',info),
                  (f'modulated{seed}_finite_plan',f'modulated{seed}_two_step'),
                  (f'modulated{seed}_two_step',f'modulated{seed}_marginal_plan'),
                  (f'cdm{seed}',f'parent{seed+880}')]
    with tempfile.TemporaryDirectory() as directory:
        stage=Path(directory)
        for area in ('test_refined','test_finite'):
            plan=json.loads((args.root/area/'plan.json').read_text())
            for job in plan['jobs']:
                source=args.root/area/job['name']/'summary.json'
                target=stage/job['name']; target.mkdir()
                shutil.copyfile(source,target/'summary.json')
        digital_summary(stage,manifest,pairs)
        shutil.copyfile(stage/'comparisons.json',args.output/'function_belief.json')
        shutil.copyfile(stage/'instances.csv',args.output/'function_belief_instances.csv')
    shutil.copyfile(manifest_path,args.output/'function_belief_test_cohort.json')

    native=args.root/'native_integrated'
    native_plan=dict(manifest='runs/continual_physics/experimental_memory_20261101/cohort/manifest.json',
        jobs=[],paired_conditions=[],
        comparison='Same saved native initial states; fixed integration and frozen original parents with twelve ledgers',
        core_updates=0, state_sharing='separated')
    for seed in (31,32):
        parent=f'parent{seed+880}'
        source=Path(f'runs/continual_physics/continual_belief_20261102/parent_memory/parent_memory{seed+880}/summary.json')
        target=native/parent; target.mkdir(exist_ok=True)
        shutil.copyfile(source,target/'summary.json')
        native_plan['jobs'] += [dict(name=f'cdm{seed}'),dict(name=parent)]
        native_plan['paired_conditions'].append((f'cdm{seed}',parent))
    (native/'plan.json').write_text(json.dumps(native_plan,indent=2)+'\n')
    native_summary(native,native_plan)
    shutil.copyfile(native/'comparisons.json',args.output/'function_belief_native.json')
    (args.output/'function_belief_native_instances.csv').write_text((native/'instances.csv').read_text())

    rows=[]
    for file in sorted((args.root/'train').glob('*/training.jsonl')):
        config=json.loads((file.parent/'configuration.json').read_text())
        for line in file.read_text().splitlines():
            row=json.loads(line)
            for condition,values in row.get('validation',{}).items():
                n,context=map(int,condition.split('_'))
                rows.append(dict(model=file.parent.name,update=row['update'],parameters=config['parameters'],
                    elapsed_seconds=row['seconds'],components=n,context=context,
                    objective=values[0],future_marginal_nll=values[1]))
    with (args.output/'function_belief_training.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n')
        writer.writeheader();writer.writerows(rows)


if __name__=='__main__':
    main()
