"""Independent Bayesian active calibration in the unmodified native task."""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
from contextlib import redirect_stdout
import fcntl,json,multiprocessing
from pathlib import Path
from time import perf_counter
import numpy as np
import torch
from experiments.discoveryworld.active_native import NativeActiveEpisode
from experiments.discoveryworld.adapter import TrajectoryWriter
from experiments.discoveryworld.calibration import FEATURE_QUANTITIES
from experiments.discoveryworld.calibration_mixture import load_mixture_checkpoint
from experiments.discoveryworld.evaluate_active_stream import schedule_position,public_object_order
from experiments.discoveryworld.structured_observation import StructuredDiscoveryWorldEnv
from .model import BayesianDesigner,Ledger


def setting_outcomes(decision):
    if decision['action_index']<20:return []
    result=decision['result'];attempts=list(result.get('attempts',[]))
    setting=result.get('setting');request=result.get('request',{})
    if (setting is not None and setting.get('object_handle')==request.get('target_handle')
        and setting.get('reactor_handle')==request.get('reactor_handle')):attempts.append(setting)
    return list(dict.fromkeys((float(r['frequency_hz']),r['activated']) for r in attempts
                if r.get('activated') in (True,False) and r.get('frequency_hz') is not None))


def ledger_from_episode(episode,designer):
    x=np.tile(designer.x_mean,(4,1));var=np.tile(designer.x_scale**2,(4,1))
    seen=np.zeros((4,9),bool);counts=np.zeros((4,9),int)
    grouped={}
    for r in episode.operator.calibration.measurements.records:
        if r.kind!='instrument' or r.target_handle not in episode.objects or r.quantity not in FEATURE_QUANTITIES:continue
        key=(episode.objects.index(r.target_handle),FEATURE_QUANTITIES.index(r.quantity))
        grouped.setdefault(key,[]).append(r.value)
    for (o,q),readings in grouped.items():
        readings=np.asarray(readings)
        if q==2:
            # An isolated meter adds U(0,.05), rounded to .01. This approximate
            # interval posterior retains finite-sample uncertainty; the public
            # execution tool performs the same isolation as the neural policy.
            low=max(0.,float(readings.max())-.055);high=float(readings.min())+.005
            if low>high: # observed spread also exposes imperfect isolation
                low=max(0.,float(readings.min())-.055);high=float(readings.min())+.005
            x[o,q]=(low+high)/2;var[o,q]=max((high-low)**2/12,.005**2/12)
        else:
            x[o,q]=readings.mean();var[o,q]=.01**2/12
        seen[o,q]=True;counts[o,q]=len(readings)
    query=int(episode.query[0])
    failures=sorted({f for d in episode.decisions if d['query_index']==query for f,active in setting_outcomes(d) if not active})
    return Ledger(x,var,seen,episode.known_hz[0].numpy().copy(),episode.known[0].numpy().copy(),counts,query,failures)


def run_stream(job):
    options,stream=job;torch.set_num_threads(1)
    prior,_=load_mixture_checkpoint(options['mixture_checkpoint'])
    table=np.load(options['training_data'])
    marginal=[table['x'][...,q][table['reading_present'][...,q]].astype(float) for q in range(9)]
    designer=BayesianDesigner(prior,marginal,mode=options['mode'],samples=options['samples'],seed=stream['seeds'][0]+700000,iterations=options.get('iterations',3))
    root=Path(options['output'])/stream['name'];root.mkdir(parents=True)
    rows=[]
    for index,seed in enumerate(stream['seeds']):
        output=root/f'episode{index:03d}_seed{seed}';output.mkdir()
        started=perf_counter();inference=0.;execution=0.;error=None;ending='decision_budget'
        with (output/'native.log').open('w') as log,redirect_stdout(log):
            env=StructuredDiscoveryWorldEnv(output,options['max_actions']);env.reset('Reactor Lab','Normal',seed)
            rngroot=Path(options['initial_rng_root']);rngroot.mkdir(parents=True,exist_ok=True)
            snapshot=rngroot/f'seed{seed}.json.gz'
            with (rngroot/f'seed{seed}.lock').open('a') as lock:
                fcntl.flock(lock,fcntl.LOCK_EX)
                env.restore_initial_rng(snapshot) if snapshot.exists() else env.save_initial_rng(snapshot)
            (output/'initial_rng.json.gz').write_bytes(snapshot.read_bytes())
            with TrajectoryWriter(output,{**env.configuration,'policy':'bayesian_active_calibration',
                    'mode':options['mode'],'max_decisions':options['max_decisions'],
                    'inputs':'public instrument readings and actually returned device settings',
                    'scalar_executor':'shared NativeExperimentOperator including local probes',
                    'planning_rng_seed':stream['seeds'][0]+700000,'posterior_samples':options['samples'],
                    'moment_refit_iterations':options.get('iterations',3)}) as writer:
                episode=NativeActiveEpisode(env,writer)
                try:
                    episode.bootstrap()
                    for step in range(options['max_decisions']):
                        if env.ended:
                            ending='native_termination' if all(m['completed_successfully'] for m in env.metrics) else 'native_action_budget';break
                        begin=perf_counter();ledger=ledger_from_episode(episode,designer)
                        query=ledger.query
                        (action,mean,spread),diagnostics,posterior=designer.choose(ledger,episode.controls[query],float(episode.current_hz[0,query]))
                        inference+=perf_counter()-begin
                        begin=perf_counter();decision=episode.apply(action,mean_hz=mean,std_hz=spread);execution+=perf_counter()-begin
                        decision['classical_design']=diagnostics
                        designer.observe_cost(action,decision['native_actions'])
                    designer.remember(ledger_from_episode(episode,designer))
                except Exception as exc:
                    import traceback;traceback.print_exc();error=f'{type(exc).__name__}: {exc}';ending='error'
                summary={**episode.summary(),'seed':seed,'index':index,'stream':stream['name'],
                    **schedule_position(stream,index),'ending':ending,'error':error,'seconds':perf_counter()-started,
                    'inference_seconds':inference,'execution_seconds':execution,'mode':options['mode'],
                    'public_object_order':public_object_order(episode)}
                (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');rows.append(summary)
    (root/'summary.json').write_text(json.dumps(rows,indent=2)+'\n')
    return {'stream':stream['name'],'rows':[{k:v for k,v in r.items() if k!='decisions'} for r in rows]}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',required=True);p.add_argument('--mode',choices=('reset','forgetting'),default='reset')
    p.add_argument('--manifest',default='experiments/continual_physics/native_development.json')
    p.add_argument('--mixture-checkpoint',default='runs/discoveryworld/calibration/mixture/shared_train_only_seed31/last.pt')
    p.add_argument('--training-data',default='runs/discoveryworld/active_experiments/data_large/train.npz')
    p.add_argument('--initial-rng-root',default='runs/continual_physics/native_development/initial_rng')
    p.add_argument('--iterations',type=int,default=3);p.add_argument('--samples',type=int,default=8);p.add_argument('--workers',type=int,default=5)
    p.add_argument('--max-actions',type=int,default=400);p.add_argument('--max-decisions',type=int,default=160)
    args=p.parse_args();root=Path(args.output);root.mkdir(parents=True,exist_ok=True)
    streams=json.loads(Path(args.manifest).read_text())['streams']
    (root/'configuration.json').write_text(json.dumps({**vars(args),'planning_rng_seeds':{s['name']:s['seeds'][0]+700000 for s in streams}},indent=2)+'\n')
    rows=[]
    with ProcessPoolExecutor(args.workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        for future in as_completed([pool.submit(run_stream,(vars(args),s)) for s in streams]):
            rows.append(future.result());(root/'summary.json').write_text(json.dumps(rows,indent=2)+'\n')
            print(json.dumps({'stream':rows[-1]['stream'],'completed':sum(r['completed_successfully'] for r in rows[-1]['rows']),
                              'errors':sum(bool(r['error']) for r in rows[-1]['rows'])}),flush=True)
    if any(r['error'] for stream in rows for r in stream['rows']):raise SystemExit(1)

if __name__=='__main__':main()
