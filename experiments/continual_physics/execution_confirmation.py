"""Fixed independent native comparison with execution-aware training endpoints."""
from __future__ import annotations
import csv,json,os,subprocess,sys,time
from pathlib import Path
from experiments.physical_belief.summarize_native import load_rows
from .summarize_native import comparison,paired_seconds

ROOT=Path('runs/continual_physics/execution_confirmation')
MANIFEST=Path('experiments/continual_physics/execution_confirmation.json')


def jobs():
    common=['--manifest',str(MANIFEST),'--initial-rng-root',str(ROOT/'initial_rng'),
            '--max-actions','400','--max-decisions','160','--workers','6']
    neural=[*common,'--threads','1','--device','cpu'];result=[]
    for seed,parent_seed in [(241,221),(242,222)]:
        parent=Path('runs/physical_belief')/('joint_20261029' if parent_seed==221 else 'joint_repeat_20261029')/f'carry_seed{parent_seed}'
        result.append({'name':f'parent_seed{parent_seed}','ready':None,
            'command':[sys.executable,'-m','experiments.physical_belief.evaluate_native',
                '--checkpoint',str(parent/'policy.pt'),'--belief-checkpoint',str(parent/'belief.pt'),
                '--belief-mode','carry','--state-mode','reset','--mixture-checkpoint',
                'runs/discoveryworld/calibration/mixture/shared_train_only_seed31/last.pt',*neural]})
        for mode in ('full','recurrent_reference'):
            root=Path('runs/continual_physics')/('native_execution_joint_20261030_point' if mode=='full' else 'native_execution_joint_20261030')
            condition=f'{mode}_seed{seed}';directory=root/condition
            variants=[('',[])]+([('_discard_physical',['--discard-physical-updates']),('_task_reset',['--state-mode','reset'])] if mode=='full' else [])
            for suffix,flags in variants:
                result.append({'name':condition+suffix,'training_condition':condition,'training_root':str(root),
                    'ready':str(directory/'finished.json'),'command':[sys.executable,'-m','experiments.continual_physics.evaluate_native',
                        '--checkpoint',str(directory/'last.pt'),'--feedback-delay-ticks','2',*neural,*flags]})
    result.append({'name':'classical_memory','ready':None,'command':[sys.executable,'-m','experiments.classical_calibration.evaluate_native',
        '--mode','forgetting','--samples','8','--iterations','3',*common]})
    for job in result:job['command'] += ['--output',str(ROOT/job['name'])]
    return result


def summarize(plan):
    expected={(s['name'],i,seed) for s in json.loads(MANIFEST.read_text())['streams'] for i,seed in enumerate(s['seeds'])}
    rows={j['name']:load_rows(ROOT/j['name']) for j in plan['jobs']}
    result={'protocol':plan,'conditions':{},'comparisons':{}}
    for name,values in rows.items():
        if len(values)!=120 or {(r['stream'],r['episode_index'],r['seed']) for r in values}!=expected:raise ValueError(name)
        result['conditions'][name]={'instances':120,'completed':sum(r['completed_successfully'] for r in values),
            'mean_actions':sum(r['native_actions'] for r in values)/120,'mean_seconds':sum(r['seconds'] for r in values)/120,
            'execution_errors':sum(bool(r.get('error')) for r in values)}
    for seed,parent in ((241,221),(242,222)):
        full=f'full_seed{seed}'
        for control in [f'parent_seed{parent}',f'recurrent_reference_seed{seed}',f'{full}_discard_physical',f'{full}_task_reset','classical_memory']:
            compared=comparison(rows[full],rows[control]);compared['timing']={k:paired_seconds(rows[full],rows[control],k) for k in
                ('episode_seconds','inference_seconds','execution_seconds','feedback_seconds')}
            result['comparisons'][full+':'+control]=compared
    (ROOT/'comparison_summary.json').write_text(json.dumps(result,indent=2)+'\n')
    keys=['stream','episode_index','seed','segment_role','position_in_segment','completed_successfully','native_actions','measurement_requests','test_requests','seconds']
    with (ROOT/'instances.csv').open('w') as file:
        writer=csv.DictWriter(file,fieldnames=['condition',*keys]);writer.writeheader()
        writer.writerows({'condition':name,**{k:r.get(k) for k in keys}} for name,values in rows.items() for r in values)


def main():
    ROOT.mkdir(parents=True,exist_ok=True);declared=jobs()
    plan={'manifest':str(MANIFEST),'jobs':[{**j,'command':['python',*j['command'][1:]]} for j in declared],
        'instances_per_condition':120,'native_executions':1320,'maximum_workers':18,'training_updates':200,
        'selection':'All fixed endpoints and declared interventions; no new-confirmation checkpoint selection',
        'discard_physical':'Discard only physical fast/slow corrections; retain numerical memory and cost adaptation',
        'task_reset':'Reset all model state at task boundaries, including h/z, numerical memory, both adapters and replay; normal within-task updates',
        'feedback_delay_ticks':2,'timing_scope':'Parents and classical policy are synchronous; native operation costs match, artificial internal delays do not',
        'new_instances':'No overlaps with earlier experiment JSON seeds; familiar five native relation families',
        'bootstrap_unit':'whole six-instance A2B2A2 stream','source':'execution-aware training with exact native probes/cost and open-interval point feedback'}
    saved=ROOT/'evaluation_plan.json'
    if saved.exists() and json.loads(saved.read_text())!=plan:raise ValueError('Existing declared plan differs')
    saved.write_text(json.dumps(plan,indent=2)+'\n')
    complete_path=ROOT/'evaluation_completion.json';completed=json.loads(complete_path.read_text()) if complete_path.exists() else []
    done={r['name'] for r in completed};pending=[j for j in declared if j['name'] not in done];running=[]
    while pending or running:
        for job in list(pending):
            if len(running)>=3:break
            if job['ready']:
                marker=Path(job['ready']);status=Path(job['training_root'])/'completed.json'
                try:training=json.loads(status.read_text()) if status.exists() else []
                except json.JSONDecodeError:training=[]
                failed=next((r for r in training if r['name']==job['training_condition'] and r['exit_code']),None)
                if failed:
                    completed.append({'name':job['name'],'exit_code':failed['exit_code'],'reason':'training_failed'});pending.remove(job);continue
                if not marker.exists():continue
                if json.loads(marker.read_text())['updates']!=200:raise ValueError('Requires fixed final update 200')
            if (ROOT/job['name']).exists():raise FileExistsError(ROOT/job['name'])
            log=(ROOT/f"{job['name']}.log").open('w')
            process=subprocess.Popen(job['command'],stdout=log,stderr=subprocess.STDOUT,
                env={**os.environ,'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','PYTHONHASHSEED':'0'})
            running.append((job,process,log,time.monotonic()));pending.remove(job)
            print(json.dumps({'started':job['name'],'pid':process.pid}),flush=True)
        for item in list(running):
            job,process,log,start=item;code=process.poll()
            if code is None:continue
            log.close();running.remove(item);completed.append({'name':job['name'],'exit_code':code,'seconds':time.monotonic()-start})
            print(json.dumps(completed[-1]),flush=True)
        complete_path.write_text(json.dumps(completed,indent=2)+'\n')
        if pending or running:time.sleep(5)
    if any(r['exit_code'] for r in completed):raise SystemExit(1)
    summarize(plan)


if __name__=='__main__':main()
