"""Freeze new chemical instance groups before sampling or model evaluation.

This evaluator-only program reads generator recipe identities solely to split
instances. Its output is never a model feature or a training outcome target.
The native sampler receives only difficulty, seed and its own behavior seed.
"""
from __future__ import annotations
import argparse,json,math,multiprocessing,os
from collections import Counter
from contextlib import redirect_stdout
from concurrent.futures import ProcessPoolExecutor
from functools import reduce
from pathlib import Path
from experiments.discoveryworld.collect_chemistry import integer_recipes
from experiments.discoveryworld.structured_observation import StructuredDiscoveryWorldEnv


def canonical(counts):
    divisor=reduce(math.gcd,counts.values())
    return '|'.join(f'{name}:{counts[name]//divisor}' for name in sorted(counts))



def proportion_structure(group):
    return ':'.join(str(value) for value in sorted(int(item.rsplit(':',1)[1]) for item in group.split('|')))


def structural_coverage(training,development,confirmation):
    splits={'training':training,'development':development,
            'confirmation_untrained_recipe':[row for row in confirmation if row['regime']=='untrained_recipe'],
            'confirmation_seen_recipe':[row for row in confirmation if row['regime']=='seen_recipe_new_instance']}
    result={}
    trained={proportion_structure(row['recipe_group']) for row in training}
    for name,rows in splits.items():
        structures=Counter(proportion_structure(row['recipe_group']) for row in rows)
        result[name]={'instances':len(rows),'distinct_named_compositions':len({row['recipe_group'] for row in rows}),
            'distinct_proportion_structures':len(structures),'structure_instance_counts':dict(sorted(structures.items())),
            'structures_absent_from_training':sorted(set(structures)-trained)}
    result['meaning']='Untrained named ingredient-ratio combinations; component-permutation-equivalent proportion structures may be present in training. The chemical response mechanism is unchanged.'
    return result

def identify(job):
    difficulty,seed,root=job
    directory=Path(root)/difficulty.lower()/f'seed{seed}';directory.mkdir(parents=True,exist_ok=True)
    with (directory/'generation.log').open('w') as log,redirect_stdout(log):
        env=StructuredDiscoveryWorldEnv(directory,1000);env.reset('Combinatorial Chemistry',difficulty,seed)
        group=canonical(env.api.world.taskScorer.tasks[0].scoringInfo['chemicalSolutionDict'])
    return {'difficulty':difficulty,'seed':seed,'recipe_group':group,'directory':f'{difficulty.lower()}/seed{seed}'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--previous',type=Path,default=Path('runs/continual_physics/chemistry_data/manifest.json'))
    p.add_argument('--output',type=Path,default=Path('runs/continual_physics/chemistry_exploration'))
    p.add_argument('--workers',type=int,default=12);args=p.parse_args()
    if os.environ.get('PYTHONHASHSEED')!='0':raise ValueError('Use PYTHONHASHSEED=0')
    args.output.mkdir(parents=True,exist_ok=True);old=json.loads(args.previous.read_text())
    explored={row['recipe_group'] for row in old['episodes']}|set(old['previously_explored_recipe_groups'])
    possible=set()
    for counts in integer_recipes(4,4):
        if sum(counts) in (3,4) and 2<=sum(x>0 for x in counts)<=3:
            possible.add(canonical({f'Substance {chr(65+i)}':x for i,x in enumerate(counts) if x}))
    new_groups=sorted(possible-explored);reserved=set(new_groups[::2])
    development=set(old['heldout_recipe_groups'])
    jobs=[('Normal',seed,str(args.output/'generation')) for seed in range(10000,10300)]+[
          ('Challenge',seed,str(args.output/'generation')) for seed in range(11000,11600)]
    with ProcessPoolExecutor(args.workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        identified=list(pool.map(identify,jobs,chunksize=4))
    training=[];confirmation=[]
    for difficulty in ('Normal','Challenge'):
        eligible=[row for row in identified if row['difficulty']==difficulty and row['recipe_group'] not in reserved|development]
        if len(eligible)<160:raise ValueError(f'Too few eligible {difficulty} instances')
        training.extend({**row,'split':'train','behavior_seed':row['seed']+100000} for row in eligible[:160])
    selected={(row['difficulty'],row['seed']) for row in training}
    for group in sorted(reserved):
        matching=[row for row in identified if row['recipe_group']==group]
        if len(matching)<5:raise ValueError(f'Too few new recipe instances: {group}')
        confirmation.extend({**row,'regime':'untrained_recipe'} for row in matching[:5])
    known_training={row['recipe_group'] for row in training}
    for difficulty,n in [('Normal',15),('Challenge',15)]:
        matching=[row for row in identified if row['difficulty']==difficulty and row['recipe_group'] in known_training
                  and (row['difficulty'],row['seed']) not in selected]
        confirmation.extend({**row,'regime':'seen_recipe_new_instance'} for row in matching[:n])
    plan={'scenario':'Combinatorial Chemistry','pythonhashseed':0,'max_native_actions':1000,
          'max_recipe_parts':6,'candidate_constraint':'primitive integer proportions; no identical returned recipe repeats',
          'behavior':'half total-parts-stratified random; half two-model predictive entropy plus disagreement',
          'development_recipe_groups':sorted(development),'confirmation_recipe_groups':sorted(reserved),
          'split_labels':'generator identity only; never input, training labels, or behavior',
          'training_episodes':training,'confirmation_episodes':confirmation,
          'training_count':len(training),'confirmation_count':len(confirmation),
          'structural_coverage':structural_coverage(training,[row for row in old['episodes'] if row['split']=='validation'],confirmation)}
    (args.output/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    print(json.dumps({key:plan[key] for key in ('training_count','confirmation_count','confirmation_recipe_groups')}),flush=True)

if __name__=='__main__':main()
