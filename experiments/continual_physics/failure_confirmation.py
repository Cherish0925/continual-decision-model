"""Fixed independent native comparison of failure-conditioned decisions."""
from __future__ import annotations
import json,os,subprocess,sys,time
from pathlib import Path

ROOT=Path('runs/continual_physics/failure_conditioning_confirmation')
MANIFEST=Path('experiments/continual_physics/failure_confirmation.json')


def main():
    ROOT.mkdir(parents=True,exist_ok=True)
    common=['--manifest',str(MANIFEST),'--initial-rng-root',str(ROOT/'initial_rng'),
            '--max-actions','400','--max-decisions','160','--workers','6','--threads','1','--device','cpu']
    parent='runs/physical_belief/joint_repeat_20261029/carry_seed222'
    base=[sys.executable,'-m','experiments.continual_physics.evaluate_native',
          '--checkpoint','runs/continual_physics/joint_20261029/task_reset_seed232/last.pt',
          '--state-mode','reset','--feedback-delay-ticks','2',*common]
    jobs=[{'name':'parent_seed222','command':[sys.executable,'-m','experiments.physical_belief.evaluate_native',
        '--checkpoint',parent+'/policy.pt','--belief-checkpoint',parent+'/belief.pt','--belief-mode','carry',
        '--state-mode','reset','--mixture-checkpoint','runs/discoveryworld/calibration/mixture/shared_train_only_seed31/last.pt',*common]},
        {'name':'task_reset_seed232','command':base},
        {'name':'task_reset_seed232_conditioned','command':[*base,'--condition-failure-weights']}]
    for job in jobs:job['command']=[*job['command'],'--output',str(ROOT/job['name'])]
    plan={'manifest':str(MANIFEST),'instances_per_condition':120,'native_executions':360,'jobs':jobs,
        'selection':'Fixed parent222 and final200 reset232; same weights for the conditioned intervention; no confirmation selection',
        'intervention':'Candidate relation-confidence features condition original Gaussian mixture on complete current-target failed-interval union',
        'unchanged':'Model parameters, physical prior saved for learning, tools, control, local probes, observations and primitive budget',
        'feedback_delay_ticks':2,'timing_scope':'Neural policies have matched two-tick delivery; the static parent has no internal delay loop; all native action costs are counted',
        'bootstrap_unit':'whole A2B2A2 six-instance stream','source_revision':'1a99bab'}
    (ROOT/'evaluation_plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    running=[]
    for job in jobs:
        if (ROOT/job['name']).exists():raise FileExistsError(job['name'])
        log=(ROOT/(job['name']+'.log')).open('w')
        process=subprocess.Popen(job['command'],stdout=log,stderr=subprocess.STDOUT,
            env={**os.environ,'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','PYTHONHASHSEED':'0'})
        running.append((job,process,log,time.monotonic()))
        print(json.dumps({'started':job['name']}),flush=True)
    complete=[]
    while running:
        for item in list(running):
            job,process,log,start=item
            code=process.poll()
            if code is None:continue
            log.close();running.remove(item)
            complete.append({'name':job['name'],'exit_code':code,'seconds':time.monotonic()-start})
            (ROOT/'evaluation_completion.json').write_text(json.dumps(complete,indent=2)+'\n')
            print(json.dumps(complete[-1]),flush=True)
        if running:time.sleep(2)
    if any(row['exit_code'] for row in complete):raise SystemExit(1)

if __name__=='__main__':main()
