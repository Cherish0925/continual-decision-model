"""Export the complete fixed chemistry confirmation, preserving actual group units."""
from __future__ import annotations
import argparse,csv,json,time
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('runs/continual_physics/chemistry_reaction/native_confirmation'))
    p.add_argument('--wait',action='store_true')
    args=p.parse_args()
    while args.wait and not (args.root/'completed.json').exists():time.sleep(10)
    completion=json.loads((args.root/'completed.json').read_text())
    if completion!={'conditions':6,'episodes':360}:raise ValueError('Incomplete frozen confirmation')
    data=json.loads((args.root/'summary.json').read_text())
    expected={(r['difficulty'],r['seed']) for r in data['protocol']['episodes']}
    for item in data['protocol']['conditions']:
        rows=[r for r in data['episodes'] if r['condition']==item['name']]
        if len(rows)!=60 or {(r['difficulty'],r['seed']) for r in rows}!=expected:raise ValueError(item['name'])
    compact={key:data[key] for key in ('protocol','conditions','comparisons')}
    compact['interpretation']={
        'regime_labels':{'untrained_recipe':'Native-unsampled named ingredient-ratio groups',
                         'seen_recipe_new_instance':'Native-sampled-group new instances'},
        'training_coverage':'Both regimes use represented ratio shapes and the same official response mechanism. Joint training applies consistent ingredient renaming and may therefore cover isomorphic versions of native-unsampled named groups.',
        'protocol_note':'The protocol retains its original pre-execution group labels; the preceding interpretation corrects their meaning under joint augmentation without changing instances or outcomes.',
        'warmup_vs_joint':'Joint training combines recurrent-core unfreezing and ingredient-renaming augmentation; this comparison does not isolate either change.',
        'same_weight_interventions':'Direct-history off removes the response readout history. Core-response off zeros its recurrent-state input. Both retain actual recurrent state updates, tools, and execution constraints.',
        'statistical_unit':'Whole named recipe parameter groups, not independent per-seed laws',
        'scope':'Task-local learned chemical prediction and decisions. Autonomous compute selection, cross-task retention, and runtime neural learning are not evaluated by this comparison.'}
    output=Path('experiments/continual_chemistry/results');output.mkdir(exist_ok=True)
    (output/'reaction_confirmation.json').write_text(json.dumps(compact,indent=2)+'\n')
    (output/'reaction_confirmation.csv').write_text((args.root/'summary.csv').read_text())
    keys=['condition','difficulty','seed','evaluation_regime','recipe_group','completed_successfully','native_actions',
          'experiments','returned_rust','partial_improvements','inference_seconds','execution_seconds','seconds','ending','error']
    with (output/'reaction_confirmation_instances.csv').open('w') as file:
        writer=csv.DictWriter(file,fieldnames=keys);writer.writeheader()
        writer.writerows({k:r[k] for k in keys} for r in data['episodes'])
    print(json.dumps([r for r in data['conditions'] if r['regime']=='all'],indent=2))


if __name__=='__main__':main()
