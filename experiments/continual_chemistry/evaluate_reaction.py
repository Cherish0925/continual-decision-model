"""Fixed primitive-candidate native comparison of a reaction checkpoint."""
from __future__ import annotations
import argparse,csv,json,multiprocessing,os
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
from .evaluate import episode


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',required=True,type=Path);p.add_argument('--initial',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path);p.add_argument('--workers',type=int,default=8)
    p.add_argument('--manifest',type=Path,default=Path('runs/continual_physics/chemistry_data/manifest.json'))
    p.add_argument('--cohort',choices=['development','confirmation'],default='development')
    p.add_argument('--conditions',nargs='+',choices=['reference','initial','trained'],default=['reference','initial','trained'])
    args=p.parse_args()
    if os.environ.get('PYTHONHASHSEED')!='0':raise ValueError('Use PYTHONHASHSEED=0')
    manifest=json.loads(args.manifest.read_text())
    if args.cohort=='development':rows=[row for row in manifest['episodes'] if row['split']=='validation']
    else:rows=[{**row,'evaluation_regime':row['regime']} for row in manifest['confirmation_episodes']]
    args.output.mkdir(parents=True)
    conditions=[(name,path) for name,path in [('reference',None),('initial',args.initial),('trained',args.checkpoint)] if name in args.conditions]
    protocol={'manifest':str(args.manifest),'cohort':args.cohort,'episodes':rows,
        'conditions':{name:None if path is None else str(path) for name,path in conditions},
        'max_native_actions':1000,'max_experiments':209,'primitive_candidates':True,
        'selection':'preselected fixed checkpoints; no native-result checkpoint selection',
        'scoring':'predicted observable rust reduction per predicted native operation; same public tools and starting RNG'}
    (args.output/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    jobs=[]
    for name,checkpoint in conditions:
        for row in rows:
            jobs.append((name,row,{'checkpoint':None if checkpoint is None else str(checkpoint),
                'seed':row['seed'],'difficulty':row['difficulty'],'policy':'reference' if checkpoint is None else 'model',
                'output':str(args.output/name/row['directory']),'max_actions':1000,'max_experiments':209,'threads':1,
                'initial_rng_root':str(args.output/'initial_rng'),'primitive_candidates':True}))
    results=[]
    with ProcessPoolExecutor(args.workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        futures={pool.submit(episode,job):job for job in jobs}
        for future in as_completed(futures):
            job=futures[future];result=future.result();results.append(result)
            (Path(job[2]['output'])/'comparison.json').write_text(json.dumps(result,indent=2)+'\n')
            print(json.dumps({key:result[key] for key in ('condition','difficulty','seed','completed_successfully','native_actions','error')}),flush=True)
    summary=[]
    for condition,_ in conditions:
        for regime in ('all','untrained_recipe','seen_recipe_new_instance'):
            values=[row for row in results if row['condition']==condition and (regime=='all' or row['evaluation_regime']==regime)]
            summary.append({'condition':condition,'regime':regime,'instances':len(values),
                'completed':sum(row['completed_successfully'] for row in values),
                **{'mean_'+key:sum(row[key] for row in values)/len(values) for key in ('native_actions','experiments','returned_rust','inference_seconds','execution_seconds','seconds')},
                'execution_errors':sum(bool(row['error']) for row in values)})
    (args.output/'summary.json').write_text(json.dumps({'protocol':protocol,'conditions':summary,'episodes':results},indent=2)+'\n')
    with (args.output/'summary.csv').open('w') as file:
        writer=csv.DictWriter(file,fieldnames=list(summary[0]));writer.writeheader();writer.writerows(summary)
    print(json.dumps(summary),flush=True)

if __name__=='__main__':main()
