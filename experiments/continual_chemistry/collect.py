"""Collect native chemical experience and split by canonical recipe identity.

Recipe labels are extracted from evaluator-only simulator diagnostics *after*
execution. They determine data groups and never enter the acting reference,
observation encoder, experiment targets or prediction-guided policy.
"""
from __future__ import annotations
import argparse,ast
from collections import Counter
from concurrent.futures import ProcessPoolExecutor,as_completed
from functools import reduce
import json,math,multiprocessing,re
from pathlib import Path
from experiments.discoveryworld.collect_chemistry import collect


def recipe_group(log_path):
    text=Path(log_path).read_text()
    match=re.search(r'^Vector \(solution\): (\[[^\n]+\]).*?^All Keys:\s*(\{[^\n]+\}|\[[^\n]+\])',text,re.MULTILINE|re.DOTALL)
    if match is None:raise ValueError(f'No evaluator recipe diagnostic in {log_path}')
    values=ast.literal_eval(match.group(1))
    # Preserve printed set iteration order: the simulator constructs its
    # coefficient vector in that same order before printing the set.
    printed_names=match.group(2)
    names=ast.literal_eval('['+printed_names[1:-1]+']')
    if len(values)!=len(names):raise ValueError('Recipe diagnostic has mismatched dimensions')
    counts={name:int(value) for name,value in zip(names,values) if value}
    if any(counts[name]!=value for name,value in zip(names,values) if value):raise ValueError('Expected integer recipe counts')
    divisor=reduce(math.gcd,counts.values())
    return '|'.join(f'{name}:{counts[name]//divisor}' for name in sorted(counts))


def legacy_groups(root):
    manifest=json.loads((root/'manifest.json').read_text())
    return {recipe_group(root/row['directory']/'native.log') for row in manifest['episodes']}


def write_manifest(root,previously_explored):
    rows=[];sequence_groups={}
    for file in sorted(root.glob('*/seed*/summary.json')):
        row=json.loads(file.read_text());directory=file.parent.relative_to(root)
        trials=file.parent/'experiments.jsonl'
        # Preserve budget failures and their complete returned prefix. Interface
        # errors are recorded as such and do not silently become task failures.
        records=[json.loads(line) for line in trials.read_text().splitlines() if line]
        group=recipe_group(file.parent/'native.log') if records else None
        public_sequence=tuple((tuple(r['component_counts']),r['rust_before'],r['rust_after'],r['native_actions']) for r in records)
        if public_sequence not in sequence_groups:sequence_groups[public_sequence]=len(sequence_groups)
        rows.append({'directory':str(directory),'difficulty':row['difficulty'],'seed':row['seed'],
            'recipe_group':group,'public_sequence_group':sequence_groups[public_sequence],
            'nonterminating_experiments':sum(r['rust_after']>0 for r in records),
            'partial_improvements':sum(0<r['rust_after']<r['rust_before'] for r in records),'transitions':len(records),'native_actions':row['native_actions'],
            'seconds':row['seconds'],'error':row['error'],
            'task_completed':all(r['completed_successfully'] for r in row['task_metrics'])})
    new_groups=sorted({r['recipe_group'] for r in rows if r['recipe_group']} - previously_explored)
    # Whole recipe groups are reserved before any neural training on this data.
    heldout=set(new_groups[::4])
    occurrences=Counter()
    for row in rows:
        group=row['recipe_group'];occurrences[group]+=1
        if row['error'] or not group:
            row.update(split='excluded_interface_error',evaluation_regime=None)
        elif group in heldout:
            row.update(split='validation',evaluation_regime='untrained_recipe')
        elif occurrences[group]%5==0:
            row.update(split='validation',evaluation_regime='seen_recipe_new_instance')
        else:row.update(split='train',evaluation_regime='training')
    manifest={'scenario':'Combinatorial Chemistry','policy':'public_integer_recipe_reference',
        'is_learned_policy':False,'recipe_label_scope':'split and post-execution analysis only; never model input',
        'previously_explored_recipe_groups':sorted(previously_explored),'heldout_recipe_groups':sorted(heldout),
        'new_unique_recipe_groups':len(new_groups),'episodes':rows,
        'summary':{'episodes':len(rows),'transitions':sum(r['transitions'] for r in rows),
          'unique_recipe_groups':len({r['recipe_group'] for r in rows if r['recipe_group']}),
          'distinct_public_experiment_sequences':len(sequence_groups),
          'nonterminating_experiments':sum(r['nonterminating_experiments'] for r in rows),
          'partial_improvements':sum(r['partial_improvements'] for r in rows),
          'development_regimes':dict(Counter(r['evaluation_regime'] for r in rows if r['split']=='validation')),
          'splits':dict(Counter(r['split'] for r in rows)),
          'completed':sum(r['task_completed'] for r in rows),'errors':sum(bool(r['error']) for r in rows)}}
    (root/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return manifest


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--legacy-root',type=Path,default=Path('runs/discoveryworld/task_transfer/chemistry'))
    p.add_argument('--normal-seeds',type=int,nargs='*',default=[]);p.add_argument('--challenge-seeds',type=int,nargs='*',default=[])
    p.add_argument('--workers',type=int,default=8);p.add_argument('--max-actions',type=int,default=1000)
    p.add_argument('--summarize-only',action='store_true');args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    explored=legacy_groups(args.legacy_root)
    entry_root=Path('runs/continual_physics/chemistry_entry')
    for log in entry_root.glob('*/native.log'):
        if (log.parent/'summary.json').exists():
            explored.add(recipe_group(log))
    if not args.summarize_only:
        jobs=[]
        for difficulty,seeds in [('Normal',args.normal_seeds),('Challenge',args.challenge_seeds)]:
            root=args.output/difficulty.lower();root.mkdir(exist_ok=True)
            for seed in seeds:
                if (root/f'seed{seed}'/'summary.json').exists():continue
                jobs.append((seed,root,difficulty,'unassigned',args.max_actions,6))
        with ProcessPoolExecutor(args.workers,mp_context=multiprocessing.get_context('spawn')) as pool:
            for future in as_completed([pool.submit(collect,job) for job in jobs]):
                result=future.result();print(json.dumps({k:result[k] for k in ('seed','difficulty','experiments','native_actions','error','seconds')}),flush=True)
    manifest=write_manifest(args.output,explored);print(json.dumps(manifest['summary']),flush=True)

if __name__=='__main__':main()
