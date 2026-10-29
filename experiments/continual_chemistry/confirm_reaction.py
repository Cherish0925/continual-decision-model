"""Frozen native comparison on new ingredient-ratio combinations and instances."""
from __future__ import annotations
import argparse,csv,json,multiprocessing,os,time
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
from .evaluate import episode
from .summarize import paired

ROOT=Path('runs/continual_physics/chemistry_reaction')
CONDITIONS=[
    ('reference',None,{}),
    ('warmup753',ROOT/'warmup753/warmup.pt',{}),
    ('joint753',ROOT/'joint753/joint.pt',{}),
    ('joint754',ROOT/'joint754/joint.pt',{}),
    ('joint753_no_direct_history',ROOT/'joint753/joint.pt',{'disable_direct_history':True}),
    ('joint753_no_response_core',ROOT/'joint753/joint.pt',{'disable_response_core':True}),
]


def summarize(output,rows,plan):
    conditions=[]
    for name,_,_ in CONDITIONS:
        for regime in ('all','untrained_recipe','seen_recipe_new_instance'):
            values=[r for r in rows if r['condition']==name and (regime=='all' or r['evaluation_regime']==regime)]
            if not values:continue
            conditions.append({'condition':name,'regime':regime,'instances':len(values),
                'recipe_groups':len({r['recipe_group'] for r in values}),
                'completed':sum(r['completed_successfully'] for r in values),
                **{'mean_'+key:sum(r[key] for r in values)/len(values) for key in
                   ('native_actions','experiments','returned_rust','inference_seconds','execution_seconds','seconds')},
                'execution_errors':sum(bool(r['error']) for r in values)})
    comparisons={}
    for left,right in [('warmup753','reference'),('joint753','reference'),('joint754','reference'),
                       ('joint753','warmup753'),('joint753','joint753_no_direct_history'),
                       ('joint753','joint753_no_response_core')]:
        for regime in ('all','untrained_recipe','seen_recipe_new_instance'):
            subset=[r for r in rows if regime=='all' or r['evaluation_regime']==regime]
            l=[r for r in subset if r['condition']==left];r=[r for r in subset if r['condition']==right]
            if l and len(l)==len(r):comparisons[f'{left}:{right}:{regime}']=paired(l,r)
    summary={'protocol':plan,'conditions':conditions,'comparisons':comparisons,'episodes':rows}
    (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    with (output/'summary.csv').open('w') as file:
        writer=csv.DictWriter(file,fieldnames=list(conditions[0]));writer.writeheader();writer.writerows(conditions)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=ROOT/'native_confirmation')
    p.add_argument('--workers',type=int,default=8)
    args=p.parse_args()
    if os.environ.get('PYTHONHASHSEED')!='0':raise ValueError('Use PYTHONHASHSEED=0')
    manifest_path=Path('runs/continual_physics/chemistry_exploration/plan.json')
    manifest=json.loads(manifest_path.read_text())
    rows=[{**row,'evaluation_regime':row['regime']} for row in manifest['confirmation_episodes']]
    if len(rows)!=60:raise ValueError('Expected frozen 60-instance confirmation')
    args.output.mkdir(parents=True,exist_ok=True)
    plan={'manifest':str(manifest_path),'episodes':rows,'conditions':[
        {'name':name,'checkpoint':None if ckpt is None else str(ckpt),'intervention':flags} for name,ckpt,flags in CONDITIONS],
        'pythonhashseed':0,'max_native_actions':1000,'max_experiments':209,'primitive_candidates':True,
        'training_endpoints':'12 full warmup epochs and 2 joint epochs, fixed before native confirmation',
        'selection':'all specified conditions; no confirmation-based checkpoint or hyperparameter selection',
        'new_recipe_meaning':'new named ingredient-ratio combinations; ratio shapes and chemical mechanism already represented in training',
        'regimes':'30 untrained ingredient-ratio combinations and 30 training-combination new instances',
        'bootstrap_unit':'whole named ingredient-ratio combination',
        'core_intervention':'zero response readout core input; recurrent state still assimilates every returned observation',
        'scoring':'predicted observable rust reduction per predicted native operation; identical tools and initial RNG'}
    protocol=args.output/'protocol.json'
    if protocol.exists() and json.loads(protocol.read_text())!=plan:raise ValueError('Existing protocol differs')
    protocol.write_text(json.dumps(plan,indent=2)+'\n')
    results=[]
    with ProcessPoolExecutor(args.workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        for name,checkpoint,flags in CONDITIONS:
            if checkpoint:
                marker=checkpoint.parent/'completed.json'
                while not marker.exists():time.sleep(10)
                if not checkpoint.exists():raise FileNotFoundError(checkpoint)
            jobs=[]
            for row in rows:
                out=args.output/name/row['directory'];saved=out/'comparison.json'
                if saved.exists():results.append(json.loads(saved.read_text()));continue
                jobs.append((name,row,{'checkpoint':None if checkpoint is None else str(checkpoint),
                    'seed':row['seed'],'difficulty':row['difficulty'],'policy':'reference' if checkpoint is None else 'model',
                    'output':str(out),'max_actions':1000,'max_experiments':209,'threads':1,
                    'initial_rng_root':str(args.output/'initial_rng'),'primitive_candidates':True,**flags}))
            futures={pool.submit(episode,job):job for job in jobs}
            for future in as_completed(futures):
                job=futures[future];result=future.result();results.append(result)
                (Path(job[2]['output'])/'comparison.json').write_text(json.dumps(result,indent=2)+'\n')
                print(json.dumps({key:result[key] for key in ('condition','difficulty','seed','completed_successfully','native_actions','error')}),flush=True)
            summarize(args.output,results,plan)
    (args.output/'completed.json').write_text(json.dumps({'conditions':len(CONDITIONS),'episodes':len(results)},indent=2)+'\n')


if __name__=='__main__':main()
