"""Frozen native development comparison of learned conditional predictions."""
from __future__ import annotations
import argparse,csv,json,os,multiprocessing,time
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
from .native import run
from .collect import recipe_group


def episode(job):
    condition,row,options=job
    result=run(options)
    actual=recipe_group(Path(options['output'])/'native.log')
    if actual!=row['recipe_group']:
        raise RuntimeError(f"Native recipe changed for {row['directory']}: {actual} != frozen {row['recipe_group']}")
    return {**result,'condition':condition,'evaluation_regime':row['evaluation_regime'],
            'recipe_group':actual,'directory':row['directory']}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root',type=Path,default=Path('runs/continual_physics/chemistry_data'))
    p.add_argument('--training-root',type=Path,default=Path('runs/continual_physics/chemistry_training'))
    p.add_argument('--output',type=Path,default=Path('runs/continual_physics/chemistry_development'))
    p.add_argument('--workers',type=int,default=8);p.add_argument('--seeds',type=int,nargs='+',default=[741,742])
    args=p.parse_args()
    if os.environ.get('PYTHONHASHSEED')!='0':raise ValueError('Launch this fixed comparison with PYTHONHASHSEED=0')
    args.output.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((args.data_root/'manifest.json').read_text())
    rows=[row for row in manifest['episodes'] if row['split']=='validation']
    conditions=[('reference',None)]+[(f'{phase}_seed{seed}',args.training_root/f'seed{seed}'/f'{name}.pt')
        for phase,name in [('initial','initial'),('trained','last')] for seed in args.seeds]
    plan={'conditions':[name for name,_ in conditions],'episodes':rows,'max_native_actions':1000,
        'max_experiments':209,'pythonhashseed':0,'training_updates':500,'selection':'fixed endpoint, no development model selection',
        'repeated_recipe_rule':'returned identical integer recipe with outcome rust >= current rust is unavailable',
        'scoring':'expected immediate rust reduction / predicted actual action cost',
        'new_recipe_meaning':'recipe parameters absent from training; same official chemical mechanism',
        'recipe_check':'actual native diagnostics are checked after execution, never passed to policy'}
    (args.output/'protocol.json').write_text(json.dumps(plan,indent=2)+'\n')
    results=[]
    with ProcessPoolExecutor(args.workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        for condition,checkpoint in conditions:
            if checkpoint:
                marker=checkpoint.parent/('completed.json' if checkpoint.name=='last.pt' else 'initial.pt')
                while not marker.exists():time.sleep(10)
                if checkpoint.name=='last.pt':
                    completed=json.loads(marker.read_text())
                    if completed['updates']!=500:raise ValueError('This comparison requires fixed update 500')
            jobs=[]
            for row in rows:
                output=args.output/condition/row['directory']
                summary=output/'comparison.json'
                if summary.exists():results.append(json.loads(summary.read_text()));continue
                options={'checkpoint':None if checkpoint is None else str(checkpoint),'seed':row['seed'],'difficulty':row['difficulty'],
                    'policy':'reference' if checkpoint is None else 'model','output':str(output),'max_actions':1000,
                    'max_experiments':209,'threads':1,'initial_rng_root':str(args.output/'initial_rng')}
                jobs.append((condition,row,options))
            futures={pool.submit(episode,job):job for job in jobs}
            for future in as_completed(futures):
                job=futures[future]
                result=future.result();results.append(result)
                (Path(job[2]['output'])/'comparison.json').write_text(json.dumps(result,indent=2)+'\n')
                print(json.dumps({k:result[k] for k in ('condition','difficulty','seed','completed_successfully','native_actions','error')}),flush=True)
            (args.output/'partial_results.json').write_text(json.dumps(results,indent=2)+'\n')
    summaries=[]
    for condition,_ in conditions:
        for regime in ['all','untrained_recipe','seen_recipe_new_instance']:
            group=[r for r in results if r['condition']==condition and (regime=='all' or r['evaluation_regime']==regime)]
            summaries.append({'condition':condition,'regime':regime,'n':len(group),'completed':sum(r['completed_successfully'] for r in group),
                **{f'mean_{key}':sum(r[key] for r in group)/len(group) for key in
                   ['native_actions','experiments','returned_rust','inference_seconds','execution_seconds','seconds']},
                'execution_errors':sum(bool(r['error']) for r in group)})
    (args.output/'summary.json').write_text(json.dumps({'protocol':plan,'results':summaries,'episodes':results},indent=2)+'\n')
    with (args.output/'summary.csv').open('w') as file:
        writer=csv.DictWriter(file,fieldnames=list(summaries[0]));writer.writeheader();writer.writerows(summaries)
    print(json.dumps(summaries),flush=True)

if __name__=='__main__':main()
