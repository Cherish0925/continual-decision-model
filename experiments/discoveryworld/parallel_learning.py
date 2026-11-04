"""Batched on-policy learning with independent native simulator processes."""
from contextlib import redirect_stdout
from dataclasses import asdict
import json
import multiprocessing as mp
from pathlib import Path
import random
from time import perf_counter
import traceback

import numpy as np
import torch
import torch.nn.functional as F

from cdm.digital import PublicObjectMemory, public_candidates
from .adapter import TrajectoryWriter
from .features import batch_features, load_features
from .structured_observation import StructuredDiscoveryWorldEnv
from .train import create_measurement_log, with_measurement_records


def _environment(pipe):
    torch.set_num_threads(1)
    env = writer = log = None
    started = 0.
    try:
        while True:
            command, payload = pipe.recv()
            if command == 'close':
                return
            if command == 'reset':
                if writer:
                    writer.close()
                    log.close()
                case, folder, settings = payload
                theme, difficulty, seed, limit = case
                random.seed(settings['environment_rng'])
                np.random.seed(settings['environment_rng'] % (2**32))
                folder = Path(folder)
                folder.mkdir(parents=True, exist_ok=True)
                log = (folder / 'simulator.log').open('w')
                started = perf_counter()
                with redirect_stdout(log):
                    env = StructuredDiscoveryWorldEnv(folder, limit)
                    observation = env.reset(theme, difficulty, seed)
                    env.save_initial_rng(folder / 'initial_rng.json.gz')
                    configuration = dict(env.configuration, policy='batched_on_policy_actor_critic',
                        action_descriptions=env.actions, teleport_locations=env.teleports,
                        initial_task_metrics=env.metrics, **settings)
                    writer = TrajectoryWriter(folder, configuration)
                pipe.send({'observation': observation})
            else:
                action, behavior = payload
                with redirect_stdout(log):
                    transition = env.step(action)
                    writer.append({**transition, **behavior})
                reply = {key: transition[key] for key in
                         ('action', 'action_result', 'index', 'training_reward', 'terminated', 'truncated')}
                reply['observation'] = env.policy_input()
                reply['done'] = env.ended
                if env.ended:
                    summary = dict(scenario=case[0], difficulty=case[1], seed=case[2], steps=env.steps,
                        completed_successfully=all(m['completed_successfully'] for m in env.metrics),
                        task_metrics=env.metrics, elapsed_seconds=perf_counter()-started)
                    (folder / 'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
                    reply['summary'] = summary
                    writer.close(); writer = None
                    log.close(); log = None
                pipe.send(reply)
    except EOFError:
        return  # The training process has closed its end of the connection.
    except (Exception, SystemExit):
        pipe.send({'error': traceback.format_exc()})
    finally:
        if writer:
            writer.close()
        if log:
            log.close()
        pipe.close()


def _receive(worker):
    value = worker['pipe'].recv()
    if 'error' in value:
        raise RuntimeError(value['error'])
    return value


def train_parallel(model, critic, optimizer, cache, source, schedule, args, arguments, source_hash):
    """Every sampled action receives its own on-policy return; no teacher is used."""
    from .finetune import save_checkpoint
    ctx = mp.get_context('spawn')
    device = next(model.parameters()).device
    count = min(args.workers, len(schedule))
    state = model.initial(count, device)
    workers, episodes = [], []
    next_case = updates = interactions = 0
    model.train(); critic.train()
    kwargs = dict(mode=args.mode, persistent=not args.restart_state,
                  restart_workspace=args.restart_workspace, predict_outcomes=False)
    settings = dict(features=asdict(cache.config), mode=args.mode,
                    refinements=model.config.refinements, supervision='returned_training_reward')

    def assign(indices):
        nonlocal next_case, state
        assigned = []
        for index in indices:
            worker = workers[index]
            if next_case == len(schedule):
                worker['active'] = False
                continue
            job = next_case
            next_case += 1
            case = schedule[job]
            folder = args.output / 'episodes' / f'episode{job:04d}_seed{case[2]}'
            worker['pipe'].send(('reset', (case, str(folder),
                {**settings, 'environment_rng': args.random_seed * 100000 + job})))
            worker.update(active=True, done=False, job=job, reward=0., updates=0,
                          memory=PublicObjectMemory(), records=None)
            assigned.append(index)
        for index in assigned:
            worker = workers[index]
            worker['observation'] = _receive(worker)['observation']
            worker['records'] = create_measurement_log(worker['observation'], cache.config)
        if assigned:
            indices = torch.tensor(assigned, device=device)
            fresh = model.initial(len(assigned), device)
            state = {k: v.index_copy(0, indices, fresh[k]) for k, v in state.items()}

    def inputs(indices):
        features, candidates = [], []
        for index in indices:
            worker = workers[index]
            observation = with_measurement_records(worker['memory'].observe(worker['observation']),
                worker['records'], object_relations=cache.config.object_relations)
            choices = public_candidates(observation)
            features.append(load_features(str(cache.encode(observation, choices))))
            candidates.append(choices)
        return candidates, batch_features(features, device)

    started = perf_counter()
    try:
        for _ in range(count):
            parent, child = ctx.Pipe()
            process = ctx.Process(target=_environment, args=(child,))
            process.start(); child.close()
            workers.append(dict(pipe=parent, process=process, active=False))
        assign(range(count))
        with (args.output / 'training.jsonl').open('x') as log, (args.output / 'updates.jsonl').open('x') as update_log:
            while any(w['active'] for w in workers):
                optimizer.zero_grad(set_to_none=True)
                log_probs, values, entropies, rewards, terminals, masks = [], [], [], [], [], []
                participating = [i for i,w in enumerate(workers) if w['active']]
                for _ in range(args.sequence_length):
                    active = [i for i in participating if not workers[i]['done']]
                    if not active:
                        break
                    indices = torch.tensor(active, device=device)
                    choices, (obs, actions) = inputs(active)
                    selected_state = {k:v.index_select(0,indices) for k,v in state.items()}
                    logits, next_state, _ = model(obs, actions, selected_state, **kwargs)
                    distribution = torch.distributions.Categorical(logits=logits)
                    selected = distribution.sample()
                    logp = distribution.log_prob(selected)
                    value = critic(model, next_state, args.mode)
                    state = {k:v.index_copy(0,indices,next_state[k]) for k,v in state.items()}
                    zero = value.new_zeros(count)
                    log_probs.append(zero.index_copy(0,indices,logp))
                    values.append(zero.index_copy(0,indices,value))
                    entropies.append(zero.index_copy(0,indices,distribution.entropy()))
                    reward = torch.zeros(count,device=device)
                    terminal = torch.zeros(count,dtype=torch.bool,device=device)
                    mask = torch.zeros(count,dtype=torch.bool,device=device); mask[indices]=True
                    for position,index in enumerate(active):
                        workers[index]['pipe'].send(('step', (dict(choices[position][int(selected[position])].action),
                            {'behavior_log_probability':float(logp[position].detach()),
                             'critic_value':float(value[position].detach()),'policy_update':updates})))
                    for index in active:
                        worker=workers[index]; reply=_receive(worker)
                        worker['observation']=reply['observation']
                        worker['done']=reply['done']
                        worker['reward']+=float(reply['training_reward'])
                        if worker['records'] is not None:
                            worker['records'].observe(reply['action'],reply['action_result'],reply['observation'],sequence=reply['index'])
                        reward[index]=reply['training_reward']; terminal[index]=reply['terminated']
                        if reply['done']:
                            worker['summary']=reply['summary']
                    interactions+=len(active)
                    rewards.append(reward); terminals.append(terminal); masks.append(mask)
                indices=torch.tensor(participating,device=device)
                with torch.no_grad():
                    _,(obs,actions)=inputs(participating)
                    _,next_state,_=model(obs,actions,{k:v.index_select(0,indices) for k,v in state.items()},**kwargs)
                    bootstrap=critic(model,next_state,args.mode)
                    target=bootstrap.new_zeros(count).index_copy(0,indices,bootstrap)
                    returns=[]
                    for reward,terminal,mask in zip(reversed(rewards),reversed(terminals),reversed(masks)):
                        target=torch.where(mask,reward+args.gamma*target*(~terminal),target)
                        returns.append(target)
                    returns=torch.stack(list(reversed(returns)))
                valid=torch.stack(masks)
                predicted=torch.stack(values)
                policy_loss=-(torch.stack(log_probs)*(returns-predicted.detach()))[valid].mean()
                value_loss=F.mse_loss(predicted[valid],returns[valid])
                entropy=torch.stack(entropies)[valid].mean()
                loss=policy_loss+args.value_weight*value_loss-args.entropy_weight*entropy
                loss.backward()
                norm=torch.nn.utils.clip_grad_norm_([*model.parameters(),*critic.parameters()],1.)
                optimizer.step()
                state={k:v.detach() for k,v in state.items()}
                updates+=1
                row=dict(update=updates,interactions=interactions,active_environments=len(participating),
                    reward=float(torch.stack(rewards).sum()),policy_loss=float(policy_loss.detach()),
                    value_loss=float(value_loss.detach()),entropy=float(entropy.detach()),
                    gradient_norm=float(norm),elapsed_seconds=perf_counter()-started)
                update_log.write(json.dumps(row)+'\n');update_log.flush()
                completed=[]
                for index in participating:
                    worker=workers[index];worker['updates']+=1
                    if worker['done']:
                        summary={**worker['summary'],'updates':worker['updates'],'training_reward':worker['reward'],
                            'schedule_index':worker['job'],'protocol':'official_step_based',
                            'training_phase':'batched_on_policy_actor_critic','mode':args.mode}
                        episodes.append(summary);log.write(json.dumps(summary)+'\n');log.flush()
                        print(json.dumps(summary),flush=True);completed.append(index)
                if completed or updates%25==0:
                    save_checkpoint(model,critic,optimizer,source,args.output,episodes,arguments,source_hash)
                assign(completed)
            save_checkpoint(model,critic,optimizer,source,args.output,episodes,arguments,source_hash)
    finally:
        for worker in workers:
            if worker['process'].is_alive():
                try:
                    worker['pipe'].send(('close',None))
                except (BrokenPipeError, EOFError):
                    pass  # Preserve the simulator exception already received.
        for worker in workers:
            worker['process'].join(timeout=30)
            worker['pipe'].close()
    return dict(episodes=len(episodes),interactions=interactions,updates=updates,
                elapsed_seconds=perf_counter()-started)
