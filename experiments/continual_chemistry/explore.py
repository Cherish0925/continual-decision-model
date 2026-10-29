"""Collect varied native chemical histories with public feedback only."""
from __future__ import annotations
import argparse,json,math,multiprocessing,os,random
from collections import Counter
from concurrent.futures import ProcessPoolExecutor,as_completed
from contextlib import redirect_stdout
from pathlib import Path
from time import perf_counter
import torch
from experiments.discoveryworld.adapter import TrajectoryWriter
from experiments.discoveryworld.structured_observation import StructuredDiscoveryWorldEnv
from .collect import recipe_group
from .native import BoundedChemistryExperiments,ActionBudgetReached
from .observations import observe,feedback,load_chemistry_policy
from .recipes import primitive_recipes

_MODELS=None


def models():
    global _MODELS
    if _MODELS is None:
        torch.set_num_threads(1)
        _MODELS=[load_chemistry_policy(f'runs/continual_physics/chemistry_training/seed{seed}/last.pt')[0].eval() for seed in (741,742)]
    return _MODELS


def collect(options):
    # This worker receives no recipe identity or split-generating metadata.
    output=Path(options['output']);output.mkdir(parents=True)
    behavior_rng=random.Random(options['behavior_seed']);policies=models()
    states=[model.initial(1,'cpu') for model in policies]
    error=None;ending='native_action_budget';started=perf_counter()
    with (output/'native.log').open('w') as log,redirect_stdout(log),torch.no_grad():
        env=StructuredDiscoveryWorldEnv(output,options['max_native_actions'])
        env.reset('Combinatorial Chemistry',options['difficulty'],options['seed'])
        env.save_initial_rng(output/'initial_rng.json.gz')
        with TrajectoryWriter(output,{**env.configuration,'behavior':'mixed_random_and_predictive_uncertainty',
                'behavior_seed':options['behavior_seed'],'prediction_models':[741,742],
                'policy_inputs':'public components, current rust and actual returned experiments'}) as writer, \
             (output/'experiments.jsonl').open('w') as file:
            executor=BoundedChemistryExperiments(env,writer)
            try:
                public=executor.prepare();primitive=set(primitive_recipes(len(public['components']),6,include_leave=True))
                while not env.ended:
                    if public['rust_level']==0:
                        executor.leave();ending='native_termination';break
                    obs,choices=observe(public,max_actions=options['max_native_actions'],elapsed_actions=env.steps)
                    obs['action_mask'] &= torch.tensor([[counts in primitive for counts in choices]])
                    available=obs['action_mask'][0].nonzero().flatten().tolist()
                    if not available:ending='all_distinct_proportions_tried';break
                    states=[model.receive(state,obs) for model,state in zip(policies,states)]
                    outs=[model.readout(state,obs['candidates'],obs['action_mask']) for model,state in zip(policies,states)]
                    probabilities=torch.stack([out['outcomes'][0].softmax(-1) for out in outs])
                    mixture=probabilities.mean(0)
                    entropy=-(mixture*mixture.clamp_min(1e-12).log()).sum(-1)
                    individual=-(probabilities*probabilities.clamp_min(1e-12).log()).sum(-1).mean(0)
                    disagreement=(entropy-individual).clamp_min(0)
                    if behavior_rng.random()<.5:
                        sizes=sorted({sum(choices[index]) for index in available})
                        size=behavior_rng.choice(sizes)
                        selected=behavior_rng.choice([index for index in available if sum(choices[index])==size])
                        selection='stratified_random'
                    else:
                        utility=(entropy+disagreement).masked_fill(~obs['action_mask'][0],-torch.inf)
                        selected=int(utility.argmax());selection='predictive_uncertainty'
                    action=torch.tensor([selected])
                    states=[model.begin_action(state,out['features'],action) for model,state,out in zip(policies,states,outs)]
                    result=executor.experiment(choices[selected])
                    result['sampling']={'selection':selection,'entropy':float(entropy[selected]),
                                        'disagreement':float(disagreement[selected])}
                    file.write(json.dumps(result)+'\n');file.flush()
                    states=[model.feedback(state,feedback(result)) for model,state in zip(policies,states)]
                    public=executor.state()
            except ActionBudgetReached:ending='native_action_budget'
            except (RuntimeError,ValueError) as exc:
                import traceback;traceback.print_exc();error=f'{type(exc).__name__}: {exc}';ending='execution_error'
        summary={'difficulty':options['difficulty'],'seed':options['seed'],'behavior_seed':options['behavior_seed'],
            'native_actions':env.steps,'experiments':len(executor.history),
            'returned_rust':executor.state()['rust_level'],'completed_successfully':all(row['completed_successfully'] for row in env.metrics),
            'partial_improvements':sum(0<row['rust_after']<row['rust_before'] for row in executor.history),
            'ending':ending,'error':error,'seconds':perf_counter()-started}
        (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    return summary


def summarize(root,plan):
    episodes=[];sizes=Counter();pairs=Counter();behavior=Counter()
    for identity in plan['training_episodes']:
        directory=root/identity['directory'];summary=json.loads((directory/'summary.json').read_text())
        actual=recipe_group(directory/'native.log')
        if actual!=identity['recipe_group']:raise ValueError('Generated and observed recipe identities differ')
        records=[json.loads(line) for line in (directory/'experiments.jsonl').read_text().splitlines()]
        for row in records:
            sizes[sum(row['component_counts'])]+=1
            pairs[row['rust_before'],row['rust_after']]+=1
            behavior[row['sampling']['selection']]+=1
        episodes.append({**identity,'transitions':len(records),'native_actions':summary['native_actions'],
            'task_completed':summary['completed_successfully'],'partial_improvements':summary['partial_improvements'],
            'error':summary['error'],'seconds':summary['seconds']})
    manifest={'scenario':'Combinatorial Chemistry','policy':'mixed_random_and_predictive_uncertainty',
        'recipe_label_scope':'split and post-execution analysis only; never model input',
        'confirmation_recipe_groups':plan['confirmation_recipe_groups'],'episodes':episodes,
        'summary':{'episodes':len(episodes),'transitions':sum(row['transitions'] for row in episodes),
             'native_actions':sum(row['native_actions'] for row in episodes),
             'sum_episode_seconds':sum(row['seconds'] for row in episodes),
             'completed':sum(row['task_completed'] for row in episodes),
             'partial_improvements':sum(row['partial_improvements'] for row in episodes),
             'errors':sum(bool(row['error']) for row in episodes),
             'total_parts_counts':dict(sorted(sizes.items())),
             'rust_transition_counts':{f'{a}->{b}':n for (a,b),n in sorted(pairs.items())},
             'behavior_choices':dict(behavior)}}
    (root/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return manifest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=Path('runs/continual_physics/chemistry_exploration'))
    p.add_argument('--workers',type=int,default=12);args=p.parse_args()
    if os.environ.get('PYTHONHASHSEED')!='0':raise ValueError('Use PYTHONHASHSEED=0')
    plan=json.loads((args.output/'plan.json').read_text());jobs=[]
    for identity in plan['training_episodes']:
        output=args.output/identity['directory']
        if (output/'summary.json').exists():continue
        jobs.append({key:identity[key] for key in ('difficulty','seed','behavior_seed')}|
                    {'output':str(output),'max_native_actions':plan['max_native_actions']})
    with ProcessPoolExecutor(args.workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        for future in as_completed([pool.submit(collect,options) for options in jobs]):
            print(json.dumps(future.result()),flush=True)
    manifest=summarize(args.output,plan);print(json.dumps(manifest['summary']),flush=True)

if __name__=='__main__':main()
