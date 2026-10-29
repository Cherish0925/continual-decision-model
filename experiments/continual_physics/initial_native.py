"""Post-hoc initial-path comparison, separate from the frozen final-model study."""
from __future__ import annotations
import json,os,subprocess,sys,time
from pathlib import Path
from experiments.physical_belief.summarize_native import load_rows
from .native_campaign import make_jobs
from .summarize_native import comparison


def main():
    output=Path('runs/continual_physics/initial_native')
    primary=Path('runs/continual_physics/native_confirmation')
    reference=Path('runs/continual_physics/recurrent_reference_native')
    training=Path('runs/continual_physics/joint_20261029')
    output.mkdir(parents=True,exist_ok=True)
    jobs=[]
    candidates=make_jobs(primary,training,Path('experiments/continual_physics/native_confirmation.json'),6,delay_ticks=2)
    for seed in (231,232):
        job=next(j for j in candidates if j['name']==f'full_seed{seed}')
        name=f'initial_full_seed{seed}';command=list(job['command'])
        command[command.index('--output')+1]=str(output/name)
        command[command.index('--checkpoint')+1]=str(training/f'full_seed{seed}'/'initial.pt')
        jobs.append({'name':name,'seed':seed,'command':command})
    plan={'scope':'Post-hoc decomposition after observing fixed-final performance; original comparison unchanged',
          'interpretation':'Initial new runtime and new-module parameters jointly; not an isolated runtime-only intervention',
          'selection':'Both declared training initializations, unchanged weights and original 120-instance manifest',
          'jobs':[{**j,'command':['python',*j['command'][1:]]} for j in jobs]}
    (output/'evaluation_plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    for path,count in [(primary/'evaluation_completion.json',12),(reference/'evaluation_completion.json',2)]:
        while True:
            try:rows=json.loads(path.read_text()) if path.exists() else []
            except json.JSONDecodeError:rows=[]
            if len(rows)>=count:break
            time.sleep(10)
    processes=[]
    for job in jobs:
        if (output/job['name']).exists():raise FileExistsError(output/job['name'])
        log=(output/f"{job['name']}.log").open('w')
        process=subprocess.Popen(job['command'],stdout=log,stderr=subprocess.STDOUT,
            env={**os.environ,'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1'})
        processes.append((job,process,log))
        print(json.dumps({'started':job['name'],'pid':process.pid}),flush=True)
    complete=[]
    for job,process,log in processes:
        code=process.wait();log.close();complete.append({'name':job['name'],'exit_code':code})
        (output/'evaluation_completion.json').write_text(json.dumps(complete,indent=2)+'\n')
    if any(row['exit_code'] for row in complete):raise SystemExit(1)
    summary={'protocol':plan,'conditions':{},'comparisons':{}}
    for job in jobs:
        rows=load_rows(output/job['name'])
        if len(rows)!=120:raise ValueError(job['name'])
        summary['conditions'][job['name']]={'instances':120,'completed':sum(r['completed_successfully'] for r in rows),
            'mean_actions':sum(r['native_actions'] for r in rows)/120}
        parent=f"parent_seed{job['seed']-10}";final=f"full_seed{job['seed']}"
        summary['comparisons'][job['name']+':'+parent]=comparison(rows,load_rows(primary/parent))
        summary['comparisons'][final+':'+job['name']]=comparison(load_rows(primary/final),rows)
    (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')


if __name__=='__main__':main()
