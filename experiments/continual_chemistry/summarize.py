"""Summarize fixed-endpoint chemical decisions and conditional predictions."""
from __future__ import annotations
import argparse,csv,json,math
from collections import Counter
from pathlib import Path
import numpy as np


def paired(left,right):
    lookup={row['directory']:row for row in right};groups={}
    for row in left:
        other=lookup[row['directory']]
        groups.setdefault(row['recipe_group'],[]).append([
            row['native_actions']-other['native_actions'],
            int(row['completed_successfully'])-int(other['completed_successfully']),
            row['seconds']-other['seconds']])
    totals=np.array([np.sum(group,axis=0) for group in groups.values()]);counts=np.array([len(group) for group in groups.values()])
    samples=np.random.default_rng(741).integers(len(groups),size=(10000,len(groups)))
    distribution=totals[samples].sum(1)/counts[samples].sum(1)[:,None]
    return {'episodes':len(left),'recipe_groups':len(groups),'bootstrap_unit':'whole_recipe_parameter_group',
        **{name:{'mean_difference':float(totals[:,index].sum()/counts.sum()),
                 'difference_95ci':np.quantile(distribution[:,index],[.025,.975]).tolist()}
           for index,name in enumerate(['native_actions','completion_fraction','seconds'])}}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('runs/continual_physics/chemistry_development'))
    p.add_argument('--output',type=Path,default=Path('experiments/continual_chemistry/results'))
    args=p.parse_args();data=json.loads((args.root/'summary.json').read_text());comparisons={}
    for seed in (741,742):
        for control in ('reference',f'initial_seed{seed}'):
            for regime in ('all','untrained_recipe','seen_recipe_new_instance'):
                rows=[row for row in data['episodes'] if regime=='all' or row['evaluation_regime']==regime]
                left=[row for row in rows if row['condition']==f'trained_seed{seed}'];right=[row for row in rows if row['condition']==control]
                comparisons[f'trained_seed{seed}:{control}:{regime}']=paired(left,right)
    predictions={file.stem:json.loads(file.read_text())['summary'] for file in sorted((args.root/'predictions').glob('*.json'))}
    frequency=json.loads((args.root/'predictive_frequency_reference.json').read_text())
    coverage={}
    for condition in ('initial_seed741','initial_seed742','trained_seed741','trained_seed742'):
        records=[]
        for file in (args.root/condition).glob('*/seed*/experiments.jsonl'):
            records.extend(row for line in file.read_text().splitlines() if (row:=json.loads(line)).get('component_counts'))
        coverage[condition]={'trials':len(records),
            'cost_floor_trials':sum(row['model_progress_cost_prediction'][1]<=math.log(2) for row in records),
            'total_parts':dict(sorted(Counter(sum(row['component_counts']) for row in records).items()))}
    compact={'deployment_coverage':coverage,'predictive_frequency_reference' :{'summary':frequency['summary'],'scope':frequency['scope']},'conditions':data['results'],'paired_comparisons':comparisons,'conditional_predictions':predictions,
        'interpretation_scope':{'task':'native Combinatorial Chemistry','observations':'structured public history',
          'learned_choice':'one-step outcome/cost ranking','cross_task':'shared initialized recurrent core with trained chemical interfaces',
          'computation':'one assimilation per returned event; autonomous activity learning not evaluated',
          'regimes':'untrained recipe parameters vs seen recipe new instances; same official chemical mechanism'}}
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/'development.json').write_text(json.dumps(compact,indent=2)+'\n')
    (args.output/'development.csv').write_text((args.root/'summary.csv').read_text())
    keys=['condition','difficulty','seed','evaluation_regime','recipe_group','completed_successfully','native_actions','experiments','returned_rust','partial_improvements','inference_seconds','execution_seconds','seconds','ending','error']
    with (args.output/'development_instances.csv').open('w') as file:
        writer=csv.DictWriter(file,fieldnames=keys);writer.writeheader()
        writer.writerows({key:row[key] for key in keys} for row in data['episodes'])
    print(json.dumps(data['results'],indent=2))

if __name__=='__main__':main()
