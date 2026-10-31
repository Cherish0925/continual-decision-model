"""Execute prediction-guided chemical experiments from public observations."""
from __future__ import annotations
import argparse,json,fcntl
from contextlib import redirect_stdout
from pathlib import Path
from time import perf_counter
import torch
from experiments.discoveryworld.adapter import TrajectoryWriter
from experiments.discoveryworld.collect_chemistry import ChemistryExperiments
from experiments.discoveryworld.structured_observation import StructuredDiscoveryWorldEnv
from .observations import observe,feedback,load_chemistry_policy,outcome_action_scores


class ActionBudgetReached(Exception):
    """A proposed multi-action experiment used the remaining native budget."""


class BoundedChemistryExperiments(ChemistryExperiments):
    def _action(self,*args,**kwargs):
        if self.env.steps>=self.env.max_steps:
            raise ActionBudgetReached
        return super()._action(*args,**kwargs)


def run(options, *, model=None, state=None, return_state=False):
    args=argparse.Namespace(**options);args.output=Path(args.output)
    torch.set_num_threads(args.threads);args.output.mkdir(parents=True)
    trained_heads=False
    if model is not None:
        model.eval()
        state=model.initial(1,'cpu') if state is None else state
        trained_heads=True
    elif args.policy!='reference':
        model,record=load_chemistry_policy(args.checkpoint);model.eval();state=model.initial(1,'cpu')
        if getattr(args,'disable_direct_history',False):model.direct_history=False
        if getattr(args,'disable_response_core',False):model.core_response=False
        trained_heads=record.get('kind') in {'chemistry_continual_v1','chemistry_reaction_v1'} and record.get('training',{}).get('updates',0)>0
    primitive_candidates=getattr(args,'primitive_candidates',False) or bool(model and hasattr(model,'response'))
    initial_h=None if state is None else state['h'].clone()
    initial_z=None if state is None else state['z'].clone()
    adaptive_response = bool(model is not None and getattr(model, 'response_adaptation', None) is not None)
    if adaptive_response:
        model.retain_response_adaptation = not getattr(args, 'discard_response_updates', False)
    internal_activities = 0
    started=perf_counter();error=None;ending='experiment_budget';inference_seconds=execution_seconds=0.
    with (args.output/'native.log').open('w') as log,redirect_stdout(log),torch.no_grad():
        env=StructuredDiscoveryWorldEnv(args.output,args.max_actions)
        env.reset('Combinatorial Chemistry',args.difficulty,args.seed)
        rng_root=Path(args.initial_rng_root);rng_root.mkdir(parents=True,exist_ok=True)
        snapshot=rng_root/f'{args.difficulty.lower()}_seed{args.seed}.json.gz'
        with snapshot.with_suffix('.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            rng=env.restore_initial_rng(snapshot) if snapshot.exists() else env.save_initial_rng(snapshot)
        (args.output/'initial_rng.json.gz').write_bytes(snapshot.read_bytes())
        metadata={**env.configuration,'policy':args.policy,'has_trained_chemistry_heads':trained_heads,
            'primitive_candidates':primitive_candidates,
            'direct_history':None if model is None else getattr(model,'direct_history',None),
            'response_reads_core':None if model is None else getattr(model,'core_response',None),
            'planning_mode':getattr(args,'planning_mode','progress'),
            'planning_topk':getattr(args,'planning_topk',6),
            'input':'visible components, current rust and returned experimental history',
            'initial_rng':rng,'repeated_recipe_rule':'exclude an identical returned recipe with rust endpoint >= current rust',
            'selection':('predicted '+getattr(args,'planning_mode','progress')+' per predicted native experiment action') if model else 'ascending bounded integer recipes'}
        with TrajectoryWriter(args.output,metadata) as writer, (args.output/'experiments.jsonl').open('w') as experiments:
            executor=BoundedChemistryExperiments(env,writer)
            try:
                start=perf_counter();public=executor.prepare();execution_seconds+=perf_counter()-start
                obs,choices=observe(public,max_actions=args.max_actions,elapsed_actions=env.steps)
                if primitive_candidates:
                    from .reaction import primitive_observation_mask
                    obs=primitive_observation_mask(obs)
                if model:
                    start=perf_counter();state=model.receive(state,obs);inference_seconds+=perf_counter()-start
                for index in range(args.max_experiments+1):
                    if env.ended:
                        ending='native_action_budget' if env.steps>=args.max_actions else 'native_termination';break
                    if public['rust_level']>0 and index>=args.max_experiments:break
                    start=perf_counter()
                    if not bool(obs['action_mask'].any()):ending='no_untried_recipe';break
                    if adaptive_response:
                        from .active_training import choose_activity
                        state, _, _, computations, _ = choose_activity(model, state, obs, sample=False)
                        internal_activities += int(computations.sum())
                    out=None if model is None else model.readout(state,obs['candidates'],obs['action_mask'])
                    planning=None
                    if public['rust_level']==0:selected=len(choices)-1
                    elif model is None:selected=int(obs['action_mask'][0].nonzero()[0])
                    elif getattr(args,'planning_mode','progress')!='progress':
                        from .foresight import choose_experiment,ForesightConfig
                        selected,planning=choose_experiment(model,state,public,obs,choices,out,
                            elapsed_actions=env.steps,max_actions=args.max_actions,mode=args.planning_mode,
                            config=ForesightConfig(topk=getattr(args,'planning_topk',6),
                                update_memory_evidence=not getattr(args,'freeze_forecast_memory',False)))
                    elif adaptive_response:selected=int(model.experiment_logits(out,obs)[0].argmax())
                    else:selected=int(outcome_action_scores(out,obs)[0].argmax())
                    retrieved = None
                    if getattr(args, 'direct_retrieval', False) and public['rust_level'] > 0:
                        from .episodic_retrieval import previous_success
                        retrieved = previous_success(state, obs['candidates'], obs['action_mask'])
                        if retrieved is not None:
                            selected = retrieved
                    if model:
                        action=torch.tensor([selected]);state=model.begin_action(state,out['features'],action)
                        before_h=state['h'].clone();before_z=state['z'].clone()
                    inference_seconds+=perf_counter()-start
                    start=perf_counter()
                    try:
                        if not any(choices[selected]):
                            start_actions=env.steps;executor.leave()
                            result={'rust_before':0,'rust_after':0,'native_actions':env.steps-start_actions,'activity':'leave'}
                        else:result=executor.experiment(choices[selected])
                    finally:execution_seconds+=perf_counter()-start
                    if model:
                        if getattr(args, 'direct_retrieval', False):
                            result['directly_retrieved_recipe'] = retrieved is not None
                        if planning is not None:result['planning']=planning
                        result['model_outcome_logits']=out['outcomes'][0,selected].tolist()
                        result['model_recipe_score']=float(outcome_action_scores(out,obs)[0,selected])
                        result['model_progress_cost_prediction']=out['predictions'][0,selected].tolist()
                        if 'memory_experts' in out:
                            memory = model.experimental_memory
                            result['experimental_memory'] = {
                                'weights': memory.weights(state, obs['candidates'][:, 0, 16:24].bool()).exp()[0].tolist(),
                                'expert_return_log_probabilities': out['memory_experts'][0, selected].tolist(),
                                'archived_tasks': int(state['archive_mask'][0].any(-1).sum())}
                        start=perf_counter();state=model.feedback(state,feedback(result))
                        if adaptive_response and any(choices[selected]):
                            state = model.record_response(state, out, torch.tensor([selected]),
                                torch.tensor([result['rust_before']]), torch.tensor([result['rust_after']]))
                    public=executor.state();obs,choices=observe(public,max_actions=args.max_actions,elapsed_actions=env.steps)
                    if primitive_candidates:obs=primitive_observation_mask(obs)
                    if model:
                        state=model.receive(state,obs);inference_seconds+=perf_counter()-start
                        result['core_state_change']={'h':float((state['h']-before_h).norm()),'z':float((state['z']-before_z).norm())}
                        result['received_feedback_count']=int(state['memory_cursor'][0])
                    experiments.write(json.dumps(result)+'\n');experiments.flush()
                    if env.ended:ending='native_termination';break
            except ActionBudgetReached:ending='native_action_budget'
            except (ValueError,RuntimeError) as exc:
                import traceback;traceback.print_exc();error=f'{type(exc).__name__}: {exc}';ending='execution_error'
        summary={'scenario':'Combinatorial Chemistry','difficulty':args.difficulty,'seed':args.seed,
            'policy':args.policy,'has_trained_chemistry_heads':trained_heads,'native_actions':env.steps,'experiments':len(executor.history),
            'returned_rust':executor.state()['rust_level'],'completed_successfully':all(r['completed_successfully'] for r in env.metrics),
            'partial_improvements':sum(0<r['rust_after']<r['rust_before'] for r in executor.history),
            'feedback_events':0 if state is None else int(state['memory_cursor'][0]),
            'observations':0 if state is None else int(state['observation_count'][0]),
            'final_h_change':None if state is None else float((state['h']-initial_h).norm()),
            'final_z_change':None if state is None else float((state['z']-initial_z).norm()),
            'inference_seconds':inference_seconds,'execution_seconds':execution_seconds,
            'planning_mode':getattr(args,'planning_mode','progress'),
            'internal_activities':internal_activities,
            'activity_counts':None if state is None else state['activity_counts'][0].tolist(),
            'response_adaptation_norm':None if not adaptive_response else
                float((state['response_fast'] + state['response_slow']).norm()),
            'ending':ending,'error':error,'seconds':perf_counter()-started}
        (args.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    return (summary,state) if return_state else summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',default='runs/physical_belief/joint_20261029/carry_seed221/policy.pt')
    p.add_argument('--seed',type=int,required=True);p.add_argument('--difficulty',choices=('Normal','Challenge'),default='Normal')
    p.add_argument('--policy',choices=('reference','untrained_model','model'),default='reference')
    p.add_argument('--output',type=Path,required=True);p.add_argument('--max-actions',type=int,default=1000)
    p.add_argument('--max-experiments',type=int,default=209);p.add_argument('--threads',type=int,default=1)
    p.add_argument('--initial-rng-root',default='runs/continual_physics/chemistry_initial_rng')
    p.add_argument('--primitive-candidates',action='store_true',help='Use the same unique proportions for reference and learned policies')
    p.add_argument('--disable-direct-history',action='store_true')
    p.add_argument('--disable-response-core',action='store_true')
    p.add_argument('--discard-response-updates', action='store_true')
    p.add_argument('--planning-mode',choices=['progress','completion','two_step'],default='progress')
    p.add_argument('--planning-topk',type=int,default=6)
    p.add_argument('--freeze-forecast-memory',action='store_true')
    args=p.parse_args();summary=run(vars(args));print(json.dumps(summary))
    if summary['error']:raise SystemExit(1)

if __name__=='__main__':main()
