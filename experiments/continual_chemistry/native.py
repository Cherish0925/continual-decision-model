"""Execute the same continuing core on public chemical experiments.

New task heads are untrained. Reference mode executes the existing exploratory
recipe ordering while the shared core receives each actual result; model mode
uses the new action readout. Neither is presented as a trained chemistry result.
"""
from __future__ import annotations
import argparse,json
from contextlib import redirect_stdout
from pathlib import Path
from time import perf_counter
import torch
from cdm.continual_experiment import Activity
from experiments.discoveryworld.adapter import TrajectoryWriter
from experiments.discoveryworld.collect_chemistry import ChemistryExperiments
from experiments.discoveryworld.structured_observation import StructuredDiscoveryWorldEnv
from .observations import observe,feedback,model_from_core_checkpoint


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',default='runs/physical_belief/joint_20261029/carry_seed221/policy.pt')
    p.add_argument('--seed',type=int,required=True);p.add_argument('--difficulty',choices=('Normal','Challenge'),default='Normal')
    p.add_argument('--policy',choices=('reference','untrained_model'),default='reference')
    p.add_argument('--output',type=Path,required=True);p.add_argument('--max-actions',type=int,default=1000)
    p.add_argument('--max-experiments',type=int,default=60);p.add_argument('--threads',type=int,default=1)
    args=p.parse_args();torch.set_num_threads(args.threads);args.output.mkdir(parents=True)
    model=model_from_core_checkpoint(args.checkpoint);model.eval();state=model.initial(1,'cpu')
    initial_h=state['h'].clone();initial_z=state['z'].clone();started=perf_counter();error=None
    with (args.output/'native.log').open('w') as log,redirect_stdout(log),torch.no_grad():
        env=StructuredDiscoveryWorldEnv(args.output,args.max_actions);env.reset('Combinatorial Chemistry',args.difficulty,args.seed)
        with TrajectoryWriter(args.output,{**env.configuration,'policy':args.policy,'is_trained_chemistry_policy':False,
                    'input':'visible components, current rust and returned experimental history',
                    'core_source':'shared GRU and Transformer workspace; new chemistry input/output heads'}) as writer, \
             (args.output/'experiments.jsonl').open('w') as experiments:
            executor=ChemistryExperiments(env,writer)
            try:
                public=executor.prepare();obs,choices=observe(public,max_actions=args.max_actions,elapsed_actions=env.steps)
                state=model.receive(state,obs)
                for index in range(args.max_experiments+1):
                    out=model.readout(state,obs['candidates'],obs['action_mask'])
                    if public['rust_level']==0:
                        selected=len(choices)-1
                    elif index>=args.max_experiments:break
                    elif args.policy=='reference':selected=index
                    else:selected=int(out['logits'][0].argmax())
                    action=torch.tensor([selected]);state=model.begin_action(state,out['features'],action)
                    before_h=state['h'].clone();before_z=state['z'].clone()
                    if not any(choices[selected]):
                        start=env.steps;executor.leave()
                        result={'rust_before':0,'rust_after':0,'native_actions':env.steps-start}
                    else:
                        result=executor.experiment(choices[selected])
                        result['model_outcome_logits']=out['outcomes'][0,selected].tolist()
                        result['model_progress_cost_prediction']=out['predictions'][0,selected].tolist()
                    state=model.feedback(state,feedback(result))
                    public=executor.state();obs,choices=observe(public,max_actions=args.max_actions,elapsed_actions=env.steps)
                    state=model.receive(state,obs)
                    result['core_state_change']={'h':float((state['h']-before_h).norm()),'z':float((state['z']-before_z).norm())}
                    result['received_feedback_count']=int(state['memory_cursor'][0])
                    experiments.write(json.dumps(result)+'\n');experiments.flush()
                    if env.ended:break
            except (ValueError,RuntimeError) as exc:
                import traceback;traceback.print_exc();error=f'{type(exc).__name__}: {exc}'
        summary={'scenario':'Combinatorial Chemistry','difficulty':args.difficulty,'seed':args.seed,
            'policy':args.policy,'is_trained_chemistry_policy':False,'native_actions':env.steps,'experiments':len(executor.history),
            'returned_rust':executor.state()['rust_level'],'completed_successfully':all(r['completed_successfully'] for r in env.metrics),
            'feedback_events':int(state['memory_cursor'][0]),'observations':int(state['observation_count'][0]),
            'final_h_change':float((state['h']-initial_h).norm()),'final_z_change':float((state['z']-initial_z).norm()),
            'error':error,'seconds':perf_counter()-started}
        (args.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary))
    if error:raise SystemExit(1)

if __name__=='__main__':main()
