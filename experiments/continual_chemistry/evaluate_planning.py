"""Matched one- and two-experiment planning on the existing development cohort."""
from __future__ import annotations
import argparse,json,multiprocessing,os
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
from .evaluate import episode
from .summarize import paired


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=Path('runs/continual_physics/chemistry_planning/development'))
    p.add_argument('--workers',type=int,default=8)
    args=p.parse_args()
    if os.environ.get('PYTHONHASHSEED')!='0':raise ValueError('Use PYTHONHASHSEED=0')
    manifest=json.loads(Path('runs/continual_physics/chemistry_data/manifest.json').read_text())
    rows=[r for r in manifest['episodes'] if r['split']=='validation'];args.output.mkdir(parents=True,exist_ok=True)
    checkpoint='runs/continual_physics/chemistry_reaction/joint753/joint.pt'
    rng='runs/continual_physics/chemistry_reaction/joint753/native_development/initial_rng'
    plan={'episodes':rows,'conditions':['completion','two_step'],'checkpoint':checkpoint,'topk':6,
        'max_native_actions':1000,'max_experiments':209,'primitive_candidates':True,
        'initial_rng':rng,'selection':'Fixed two-epoch checkpoint, top K six and scoring definitions before the 30-instance comparison',
        'scope':'Old development cohort only; new 60-instance confirmation untouched',
        'control':'Identical learned model and completion/cost objective; extra conditional imagined experiment only in two_step',
        'progress_reference':'Previously completed joint753/native_development records, not re-executed'}
    (args.output/'protocol.json').write_text(json.dumps(plan,indent=2)+'\n')
    jobs=[]
    for mode in plan['conditions']:
        for row in rows:
            jobs.append((mode,row,{'checkpoint':checkpoint,'policy':'model','seed':row['seed'],'difficulty':row['difficulty'],
                'output':str(args.output/mode/row['directory']),'max_actions':1000,'max_experiments':209,'threads':1,
                'initial_rng_root':rng,'primitive_candidates':True,'planning_mode':mode,'planning_topk':6}))
    results=[]
    with ProcessPoolExecutor(args.workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        futures={pool.submit(episode,job):job for job in jobs}
        for future in as_completed(futures):
            job=futures[future];r=future.result();results.append(r)
            (Path(job[2]['output'])/'comparison.json').write_text(json.dumps(r,indent=2)+'\n')
            print(json.dumps({key:r[key] for key in ('condition','seed','completed_successfully','native_actions','inference_seconds','error')}),flush=True)
    old=json.loads(Path('runs/continual_physics/chemistry_reaction/joint753/native_development/summary.json').read_text())
    results.extend({**r,'condition':'progress'} for r in old['episodes'] if r['condition']=='trained')
    summaries=[]
    for mode in ('progress','completion','two_step'):
        for regime in ('all','untrained_recipe','seen_recipe_new_instance'):
            group=[r for r in results if r['condition']==mode and (regime=='all' or r['evaluation_regime']==regime)]
            summaries.append({'condition':mode,'regime':regime,'instances':len(group),'completed':sum(r['completed_successfully'] for r in group),
                **{'mean_'+k:sum(r[k] for r in group)/len(group) for k in ('native_actions','experiments','inference_seconds','execution_seconds','seconds')},
                'execution_errors':sum(bool(r['error']) for r in group)})
    comparisons={}
    for left,right in [('completion','progress'),('two_step','completion'),('two_step','progress')]:
        for regime in ('all','untrained_recipe','seen_recipe_new_instance'):
            group=[r for r in results if regime=='all' or r['evaluation_regime']==regime]
            comparisons[f'{left}:{right}:{regime}']=paired([r for r in group if r['condition']==left],[r for r in group if r['condition']==right])
    (args.output/'summary.json').write_text(json.dumps({'protocol':plan,'conditions':summaries,'comparisons':comparisons,'episodes':results},indent=2)+'\n')
    print(json.dumps(summaries,indent=2))


if __name__=='__main__':main()
