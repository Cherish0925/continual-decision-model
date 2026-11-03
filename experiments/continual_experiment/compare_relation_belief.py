"""Run matched development and independently fixed final task comparisons."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import random
import subprocess
import sys

from experiments.continual_chemistry.response_families import TRAIN_FAMILIES, sample_lifetime
from .summarize_response_families import summarize


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--stage',choices=('development','test'),default='development')
    p.add_argument('--parallel',type=int,default=4)
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--models',nargs='+',default=['functions','conditional','attentive'])
    p.add_argument('--series',default='')
    p.add_argument('--candidate',default='functions')
    p.add_argument('--rules',nargs='+',default=['greedy','information','two_step'])
    args=p.parse_args()
    suffix='_'+args.series if args.series else ''
    target=args.root/(args.stage+suffix); target.mkdir(exist_ok=True)
    cohorts=args.root/'cohorts'; cohorts.mkdir(exist_ok=True)
    final=cohorts/'test.json'
    if not final.exists():
        groups=['new_parameters']*24+['wider_parameters']*12+['heldout_combination']*12
        streams=[]
        for i,g in enumerate(groups):
            families=('interaction_inhibition',) if g=='heldout_combination' else TRAIN_FAMILIES
            streams.append(dict(name=f'{g}_{i:03d}',group=g,
                tasks=sample_lifetime(random.Random(914083+104729*i),families,wider=g=='wider_parameters')))
        final.write_text(json.dumps(dict(seed=914083,streams=streams,
            split='Fixed before development comparisons; independent streams, discrete targets may recur'),indent=2)+'\n')
    manifest=(Path(__file__).parent/'results/response_families_development.json'
              if args.stage=='development' else final)
    selection_path=cohorts/('selection'+suffix+'.json')
    if args.stage=='development':
        candidates=[(kind,rule) for kind in args.models
                    for rule in args.rules]
    else:
        selected=json.loads(selection_path.read_text())
        candidates=list(selected['rules'].items())
        candidates += [(args.candidate,'two_step'),(args.candidate,'marginal_plan')]
        candidates=list(dict.fromkeys(candidates))
    jobs=[]; pairs=[]
    for kind,rule in candidates:
        for seed in (31,32):
            name=f'{kind}{seed}_{rule}'
            command=['-m','experiments.continual_experiment.evaluate_relation_belief',
                '--checkpoint',str(args.root/'train'/f'{kind}{seed}'/'last.pt'),
                '--manifest',str(manifest),'--selection',rule,'--workers',str(args.workers),'--output',str(target/name)]
            jobs.append(dict(name=name,arguments=command))
    if args.stage=='test':
        for seed in (31,32):
            name=f'{args.candidate}{seed}_no_memory'
            jobs.append(dict(name=name,arguments=['-m','experiments.continual_experiment.evaluate_relation_belief',
                '--checkpoint',str(args.root/'train'/f'{args.candidate}{seed}'/'last.pt'),'--manifest',str(manifest),
                '--selection',selected['rules'][args.candidate],'--capacity','0','--workers',str(args.workers),
                '--output',str(target/name)]))
            base=f'{args.candidate}{seed}_{selected["rules"][args.candidate]}'
            pairs.extend((base,f'{other}{seed}_{rule}') for other,rule in selected['rules'].items()
                         if other != args.candidate)
            pairs.extend([(base,name),(f'{args.candidate}{seed}_two_step',f'{args.candidate}{seed}_marginal_plan')])
        for seed in (911,912):
            name=f'parent{seed}'
            jobs.append(dict(name=name,arguments=['-m','experiments.continual_experiment.evaluate_response_families',
                '--checkpoint',f'runs/continual_physics/experimental_memory_20261101/train/memory{seed}/initial.pt',
                '--manifest',str(manifest),'--memory-mode','evidence','--workers',str(args.workers),'--batch','1',
                '--output',str(target/name)]))
            pairs.append((f'{args.candidate}{seed-880}_{selected["rules"][args.candidate]}',name))
        jobs.append(dict(name='known_formula',arguments=['-m','experiments.continual_experiment.evaluate_family_reference',
            '--manifest',str(manifest),'--selection','information','--workers',str(args.workers),
            '--output',str(target/'known_formula')]))
    plan=dict(stage=args.stage,manifest=str(manifest),jobs=jobs,paired_conditions=pairs,
        selection='Best summed completion over both initializations, then lowest mean cost on development only',
        scope='Predictor and planning comparison; old CDM weights remain unchanged')
    (target/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    for kind in args.models:
        for seed in (31,32):
            if not (args.root/'train'/f'{kind}{seed}'/'finished.json').exists():
                raise RuntimeError(f'Training {kind}{seed} is still running; evaluate the fixed final checkpoint')
    def run(job):
        if (target/job['name']/'summary.json').exists(): return dict(name=job['name'],exit_code=0,reused=True)
        env={**os.environ,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1'}
        with (target/(job['name']+'.log')).open('w') as log:
            r=subprocess.run([sys.executable,*job['arguments']],env=env,stdout=log,stderr=subprocess.STDOUT)
        result=dict(name=job['name'],exit_code=r.returncode)
        print(json.dumps(result),flush=True)
        return result
    with ThreadPoolExecutor(args.parallel) as pool:
        done=[f.result() for f in as_completed([pool.submit(run,j) for j in jobs])]
    (target/'completed.json').write_text(json.dumps(done,indent=2)+'\n')
    if any(r['exit_code'] for r in done): raise SystemExit(1)
    data=summarize(target,json.loads(manifest.read_text()),pairs)
    if args.stage=='development':
        rules={}
        for kind in args.models:
            def score(rule):
                rows=[data['conditions'][f'{kind}{s}_{rule}']['all'] for s in (31,32)]
                return (-sum(r['completed'] for r in rows),sum(r['operations'] for r in rows))
            rules[kind]=min(args.rules,key=score)
        selection_path.write_text(json.dumps(dict(rules=rules,criterion=plan['selection']),indent=2)+'\n')
    print(json.dumps({k:v['all'] for k,v in data['conditions'].items()},indent=2))


if __name__=='__main__': main()
