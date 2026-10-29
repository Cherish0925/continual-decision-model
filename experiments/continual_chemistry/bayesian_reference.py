"""Finite chemical-response hypotheses with public-history Bayesian decisions.

This reference knows the official cosine-threshold response family, unlike the
learned model. It never reads an instance's hidden recipe or scenario seed.
"""
from __future__ import annotations
import argparse,csv,fcntl,json,multiprocessing,os
from concurrent.futures import ProcessPoolExecutor,as_completed
from contextlib import redirect_stdout
from pathlib import Path
from time import perf_counter
import numpy as np
from .recipes import primitive_recipes
from .native import BoundedChemistryExperiments,ActionBudgetReached
from .collect import recipe_group
from .summarize import paired
from experiments.discoveryworld.adapter import TrajectoryWriter
from experiments.discoveryworld.structured_observation import StructuredDiscoveryWorldEnv


class FiniteChemicalBelief:
    def __init__(self,components):
        self.recipes=list(primitive_recipes(components))
        counts=np.asarray(self.recipes,dtype=np.float64)
        self.unit=counts/np.linalg.norm(counts,axis=1,keepdims=True)
        cosine=self.unit@self.unit.T
        self.response=np.where(cosine>=.99,0,np.where(cosine>=.66,1,np.where(cosine>=.33,2,3)))
        self.cost=8+counts.sum(1)+(counts>0).sum(1)

    def posterior(self,history):
        compatible=np.ones(len(self.recipes),dtype=bool)
        for trial in history:
            q=np.asarray(trial['component_counts'],dtype=np.float64);q=q/np.linalg.norm(q)
            cosine=self.unit@q
            endpoint=np.where(cosine>=.99,0,np.where(cosine>=.66,1,np.where(cosine>=.33,2,3)))
            compatible &= np.minimum(trial['rust_before'],endpoint)==trial['rust_after']
        if not compatible.any():raise ValueError('The declared response family has no hypothesis consistent with public feedback')
        return compatible.astype(float)/compatible.sum()

    def available(self,history,current):
        mask=np.ones(len(self.recipes),dtype=bool)
        for trial in history:
            if trial['rust_after']>=current:
                q=np.asarray(trial['component_counts'],dtype=np.int64);q=q/np.gcd.reduce(q)
                for i,recipe in enumerate(self.recipes):
                    if np.array_equal(recipe,q):mask[i]=False
        return mask

    def choose(self,public,mode):
        history=public['history'];rust=public['rust_level'];weight=self.posterior(history)
        valid=self.available(history,rust);cost=self.cost-(2 if not history else 0)
        clean=(self.response==0).astype(float);pclean=weight@clean
        values=pclean/(cost+4*pclean)
        if mode=='two_step':
            observed=np.minimum(rust,self.response)
            for a in np.flatnonzero(valid):
                branches=[]
                for y in range(1,rust+1):
                    w=weight*(observed[:,a]==y);mass=w.sum()
                    if mass==0:continue
                    support=valid.copy();support[a]=False
                    probability=(w/mass)@clean
                    branches.append((mass,probability[support],self.cost[support]))
                # The second experiment can depend on the first public return.
                # Optimize expected completion / expected charged operations;
                # door opening contributes four operations upon completion.
                ratio=0.
                for _ in range(64):
                    numerator=pclean[a];denominator=cost[a]+4*pclean[a]
                    for mass,probability,branch_cost in branches:
                        if not len(probability):continue
                        best=int(np.argmax(probability-ratio*(branch_cost+4*probability)))
                        numerator+=mass*probability[best]
                        denominator+=mass*(branch_cost[best]+4*probability[best])
                    updated=numerator/denominator
                    if abs(updated-ratio)<1e-12:break
                    ratio=updated
                values[a]=updated
        values[~valid]=-np.inf
        chosen=int(np.argmax(values))
        if not np.isfinite(values[chosen]):raise ValueError('No untried public recipe remains')
        return self.recipes[chosen],{'compatible_hypotheses':int((weight>0).sum()),
            'predicted_first_clean_probability':float(pclean[chosen]),'predicted_recipe_operations':float(cost[chosen]),
            'completion_per_operation':float(values[chosen])}


def verify_records(root):
    count=cost_errors=0;episodes=0
    for path in sorted(root.glob('*/*/experiments.jsonl')):
        records=[json.loads(line) for line in path.open()];model=FiniteChemicalBelief(len(records[0]['component_counts']))
        model.posterior(records);episodes+=1
        for trial in records:
            q=trial['component_counts'];predicted=6+sum(q)+sum(x>0 for x in q)+2*(trial['index']>0)
            count+=1;cost_errors+=predicted!=trial['native_actions']
    return {'episodes':episodes,'actual_returns':count,'inconsistent_response_histories':0,'cost_prediction_errors':cost_errors}


def run(job):
    mode,row,root,rng_root=job;output=Path(root)/mode/row['directory'];output.mkdir(parents=True)
    started=perf_counter();solve_seconds=execution_seconds=0.;error=None;ending='experiment_budget'
    with (output/'native.log').open('w') as log,redirect_stdout(log):
        env=StructuredDiscoveryWorldEnv(output,1000);env.reset('Combinatorial Chemistry',row['difficulty'],row['seed'])
        snapshot=Path(rng_root)/f"{row['difficulty'].lower()}_seed{row['seed']}.json.gz"
        with snapshot.with_suffix('.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX);env.restore_initial_rng(snapshot)
        metadata={**env.configuration,'policy':f'finite_hypothesis_{mode}',
            'prior':'uniform over primitive ratios through six parts; known official cosine response family',
            'inputs':'public component count, rust and returned history; no hidden recipe or scenario seed'}
        with TrajectoryWriter(output,metadata) as writer,(output/'experiments.jsonl').open('w') as log_trials:
            executor=BoundedChemistryExperiments(env,writer)
            try:
                tick=perf_counter();public=executor.prepare();execution_seconds+=perf_counter()-tick
                model=FiniteChemicalBelief(len(public['components']))
                for _ in range(209):
                    if not public['rust_level']:
                        tick=perf_counter();executor.leave();execution_seconds+=perf_counter()-tick;ending='native_termination';break
                    tick=perf_counter();recipe,prediction=model.choose(public,mode);solve_seconds+=perf_counter()-tick
                    tick=perf_counter()
                    try:record=executor.experiment(recipe)
                    finally:execution_seconds+=perf_counter()-tick
                    record['bayesian_prediction']=prediction;log_trials.write(json.dumps(record)+'\n');log_trials.flush()
                    public=executor.state()
                    if env.ended:ending='native_action_budget';break
            except ActionBudgetReached:ending='native_action_budget'
            except (ValueError,RuntimeError) as exc:error=f'{type(exc).__name__}: {exc}';ending='execution_error'
        result={'condition':mode,'difficulty':row['difficulty'],'seed':row['seed'],'directory':row['directory'],
            'evaluation_regime':row['evaluation_regime'],'recipe_group':row['recipe_group'],
            'completed_successfully':all(r['completed_successfully'] for r in env.metrics),'native_actions':env.steps,
            'experiments':len(executor.history),'returned_rust':executor.state()['rust_level'],
            'inference_seconds':solve_seconds,'execution_seconds':execution_seconds,'seconds':perf_counter()-started,
            'ending':ending,'error':error}
        (output/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    if recipe_group(output/'native.log')!=row['recipe_group']:raise ValueError('Native evaluator group differs from frozen instance')
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=Path('runs/continual_physics/chemistry_bayesian/development'))
    p.add_argument('--workers',type=int,default=8);p.add_argument('--check-data',action='store_true')
    p.add_argument('--cohort',choices=['development','confirmation'],default='development')
    args=p.parse_args()
    if args.check_data:
        print(json.dumps(verify_records(Path('runs/continual_physics/chemistry_exploration')),indent=2));return
    if os.environ.get('PYTHONHASHSEED')!='0':raise ValueError('Use PYTHONHASHSEED=0')
    if args.cohort=='development':
        manifest=json.loads(Path('runs/continual_physics/chemistry_data/manifest.json').read_text())
        rows=[r for r in manifest['episodes'] if r['split']=='validation']
        rng_root='runs/continual_physics/chemistry_reaction/warmup753/native_development/initial_rng'
    else:
        manifest=json.loads(Path('runs/continual_physics/chemistry_exploration/plan.json').read_text())
        rows=[{**r,'evaluation_regime':r['regime']} for r in manifest['confirmation_episodes']]
        rng_root='runs/continual_physics/chemistry_reaction/native_confirmation/initial_rng'
    args.output.mkdir(parents=True,exist_ok=True)
    plan={'episodes':rows,'conditions':['one_step','two_step'],'max_native_operations':1000,'max_experiments':209,
          'hypotheses':'all 55/161 primitive integer proportions through six parts; uniform prior',
          'response':'cosine >= .99/.66/.33 implies intrinsic endpoint 0/1/2, otherwise 3; observed min(before, endpoint)',
          'selection':'one- or two-experiment completion probability divided by expected actual operations, including door exit',
          'cost':'6 + total parts + active components + 2 if previous experiment; four exit actions',
          'scope':'Known public reaction-family reference, not matched-prior learning',
          'cohort':args.cohort,'selection':'Fixed one/two-step methods, no parameter or endpoint selection on confirmation',
          'comparison_timing':'Additional references declared after the primary learned-model comparison' if args.cohort=='confirmation' else 'Development comparison',
          'initial_rng':rng_root}
    (args.output/'protocol.json').write_text(json.dumps(plan,indent=2)+'\n')
    jobs=[(mode,row,str(args.output),plan['initial_rng']) for mode in plan['conditions'] for row in rows]
    results=[]
    with ProcessPoolExecutor(args.workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        for future in as_completed([pool.submit(run,job) for job in jobs]):
            r=future.result();results.append(r);print(json.dumps({k:r[k] for k in ('condition','seed','native_actions','completed_successfully','error')}),flush=True)
    summaries=[]
    for mode in plan['conditions']:
        for regime in ('all','untrained_recipe','seen_recipe_new_instance'):
            group=[r for r in results if r['condition']==mode and (regime=='all' or r['evaluation_regime']==regime)]
            summaries.append({'condition':mode,'regime':regime,'instances':len(group),'completed':sum(r['completed_successfully'] for r in group),
                **{'mean_'+key:float(np.mean([r[key] for r in group])) for key in ('native_actions','experiments','inference_seconds','execution_seconds','seconds')},
                'execution_errors':sum(bool(r['error']) for r in group)})
    paired_comparisons={regime:paired([r for r in results if r['condition']=='two_step' and (regime=='all' or r['evaluation_regime']==regime)],
        [r for r in results if r['condition']=='one_step' and (regime=='all' or r['evaluation_regime']==regime)]) for regime in ('all','untrained_recipe','seen_recipe_new_instance')}
    (args.output/'summary.json').write_text(json.dumps({'protocol':plan,'conditions':summaries,'comparisons':paired_comparisons,'episodes':results},indent=2)+'\n')
    print(json.dumps(summaries,indent=2))


if __name__=='__main__':main()
