"""Export native reaction-policy results with recipe-group paired uncertainty."""
from __future__ import annotations
import argparse,csv,json
from pathlib import Path
from .summarize import paired


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--name',required=True)
    args=p.parse_args()
    data=json.loads((args.root/'summary.json').read_text())
    comparisons={}
    for control in ('reference','initial'):
        for regime in ('all','untrained_recipe','seen_recipe_new_instance'):
            rows=[r for r in data['episodes'] if regime=='all' or r['evaluation_regime']==regime]
            left=[r for r in rows if r['condition']=='trained'];right=[r for r in rows if r['condition']==control]
            if left and len(left)==len(right):comparisons[f'trained:{control}:{regime}']=paired(left,right)
    compact={'protocol':data['protocol'],'conditions':data['conditions'],'paired_comparisons':comparisons,
        'scope':{'environment':'Native Combinatorial Chemistry with public observations and actual instrument operations',
            'regimes':'Unseen named ingredient-ratio combinations versus seen-combination new instances; same chemical response mechanism',
            'selection':'Fixed complete-epoch endpoint; no native-evaluation checkpoint selection',
            'warmup':'Reaction and cost heads trained with the shared recurrent core frozen',
            'policy':'Predicted observable rust reduction per predicted operation; current-task returned history',
            'not_evaluated':'Autonomous internal-compute allocation or cross-task neural online learning'}}
    output=Path('experiments/continual_chemistry/results');output.mkdir(exist_ok=True)
    (output/f'{args.name}.json').write_text(json.dumps(compact,indent=2)+'\n')
    (output/f'{args.name}.csv').write_text((args.root/'summary.csv').read_text())
    keys=['condition','difficulty','seed','evaluation_regime','recipe_group','completed_successfully','native_actions',
          'experiments','returned_rust','partial_improvements','inference_seconds','execution_seconds','seconds','ending','error']
    with (output/f'{args.name}_instances.csv').open('w') as file:
        writer=csv.DictWriter(file,fieldnames=keys);writer.writeheader()
        writer.writerows({k:r[k] for k in keys} for r in data['episodes'])
    print(json.dumps({'conditions':data['conditions'],'paired_comparisons':comparisons},indent=2))


if __name__=='__main__':main()
