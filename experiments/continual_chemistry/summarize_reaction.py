"""Export native reaction-policy results with recipe-group paired uncertainty."""
from __future__ import annotations
import argparse,csv,json
from pathlib import Path
from .summarize import paired


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--name',required=True)
    p.add_argument('--phase',choices=['warmup','joint'],default='warmup')
    p.add_argument('--reference-root',type=Path,help='Reuse completed matched reference, initial and warmup records')
    args=p.parse_args()
    data=json.loads((args.root/'summary.json').read_text())
    if args.reference_root:
        controls=json.loads((args.reference_root/'summary.json').read_text())
        for key in ('conditions','episodes'):
            data[key].extend({**r,'condition':'warmup' if r['condition']=='trained' else r['condition']} for r in controls[key])
        data['protocol']['reused_controls']=str(args.reference_root/'summary.json')
    comparisons={}
    for control in ('reference','initial','warmup'):
        for regime in ('all','untrained_recipe','seen_recipe_new_instance'):
            rows=[r for r in data['episodes'] if regime=='all' or r['evaluation_regime']==regime]
            left=[r for r in rows if r['condition']=='trained'];right=[r for r in rows if r['condition']==control]
            if left and len(left)==len(right):comparisons[f'trained:{control}:{regime}']=paired(left,right)
    compact={'protocol':data['protocol'],'conditions':data['conditions'],'paired_comparisons':comparisons,
        'scope':{'environment':'Native Combinatorial Chemistry with public observations and actual instrument operations',
            'regimes':('Unseen named ingredient-ratio combinations versus seen-combination new instances; same chemical response mechanism'
                       if args.phase=='warmup' else 'Native-unsampled named ingredient-ratio groups versus sampled-group new instances; joint ingredient-renaming augmentation may cover isomorphic combinations'),
            'selection':'Fixed complete-epoch endpoint; no native-evaluation checkpoint selection',
            'training':('Reaction and cost heads trained with the shared recurrent core frozen' if args.phase=='warmup'
                        else 'Fixed two complete joint epochs updating the shared recurrent core and chemical readouts with consistent ingredient-renaming augmentation; joint-versus-warmup does not isolate core unfreezing'),
            'policy':'Predicted observable rust reduction per predicted operation; current-task returned history',
            'not_evaluated':'Autonomous internal-compute allocation or cross-task neural online learning'}}
    if args.reference_root:
        compact['scope']['initial_random_states']='The added joint development run shares scenario seeds and world/API/UUID random states with reused controls, but its per-object and global random-state snapshots were saved independently. Do not interpret this historical development comparison as sharing every random-state snapshot. The later 60-instance confirmation shares one saved initial state across all methods.'
    output=Path('experiments/continual_chemistry/results');output.mkdir(exist_ok=True)
    (output/f'{args.name}.json').write_text(json.dumps(compact,indent=2)+'\n')
    with (output/f'{args.name}.csv').open('w') as file:
        writer=csv.DictWriter(file,fieldnames=list(data['conditions'][0]));writer.writeheader();writer.writerows(data['conditions'])
    keys=['condition','difficulty','seed','evaluation_regime','recipe_group','completed_successfully','native_actions',
          'experiments','returned_rust','partial_improvements','inference_seconds','execution_seconds','seconds','ending','error']
    with (output/f'{args.name}_instances.csv').open('w') as file:
        writer=csv.DictWriter(file,fieldnames=keys);writer.writeheader()
        writer.writerows({k:r[k] for k in keys} for r in data['episodes'])
    print(json.dumps({'conditions':data['conditions'],'paired_comparisons':comparisons},indent=2))


if __name__=='__main__':main()
