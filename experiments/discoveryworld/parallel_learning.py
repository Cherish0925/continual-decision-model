"""Batched on-policy learning with independent native simulator processes."""
from dataclasses import asdict
import json
from pathlib import Path
from time import perf_counter
import subprocess
import sys

import torch
import torch.nn.functional as F

from cdm.digital import PublicActionHistory, PublicObjectMemory, public_candidates
from .features import batch_features, load_features
from .native_worker import JsonConnection
from .train import (create_measurement_log, with_measurement_records, next_numeric_target,
                    outcome_target, received_outcome_loss)


def _receive(worker):
    value = worker['pipe'].recv()
    if 'error' in value:
        raise RuntimeError(value['error'])
    return value


def train_parallel(model, critic, optimizer, cache, source, schedule, args, arguments, source_hash):
    """Every sampled action receives its own on-policy return; no teacher is used."""
    from .finetune import save_checkpoint
    device = next(model.parameters()).device
    count = min(args.workers, len(schedule))
    state = model.initial(count, device)
    workers, episodes = [], []
    next_case = updates = interactions = 0
    model.train(); critic.train()
    requests = cache.config.action_interface == 'object_requests_v1'
    kwargs = dict(mode=args.mode, persistent=not args.restart_state,
                  restart_workspace=args.restart_workspace, predict_outcomes=bool(args.outcome_weight))
    settings = dict(features=asdict(cache.config), mode=args.mode,
                    refinements=model.config.refinements, supervision='returned_training_reward',
                    max_request_actions=getattr(args,'max_request_actions',12))

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
                          memory=PublicObjectMemory(), records=None,
                          action_history=PublicActionHistory(cache.config.action_history))
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
            if requests:
                worker['memory'].objects.update((o['uuid'],o) for o in worker['observation'].get('remembered_objects',[]))
            observation = with_measurement_records(worker['memory'].observe(worker['observation']),
                worker['records'], object_relations=cache.config.object_relations)
            observation = worker['action_history'].augment(observation)
            worker['feature_observation'] = observation
            if requests:
                from .request_actions import request_candidates
                choices = request_candidates(observation)
            else:
                choices = public_candidates(observation)
            features.append(load_features(str(cache.encode(observation, choices))))
            candidates.append(choices)
        return candidates, batch_features(features, device)

    started = perf_counter()
    try:
        for _ in range(count):
            process = subprocess.Popen(
                [sys.executable, '-u', '-m', 'experiments.discoveryworld.native_worker'],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
            workers.append(dict(pipe=JsonConnection(process.stdout, process.stdin),
                                process=process, active=False))
        assign(range(count))
        with (args.output / 'training.jsonl').open('x') as log, (args.output / 'updates.jsonl').open('x') as update_log:
            while any(w['active'] for w in workers):
                optimizer.zero_grad(set_to_none=True)
                log_probs, values, entropies, rewards, terminals, masks = [], [], [], [], [], []
                discounts = []
                prediction_losses = []
                participating = [i for i,w in enumerate(workers) if w['active']]
                for _ in range(args.sequence_length):
                    active = [i for i in participating if not workers[i]['done']]
                    if not active:
                        break
                    indices = torch.tensor(active, device=device)
                    choices, (obs, actions) = inputs(active)
                    selected_state = {k:v.index_select(0,indices) for k,v in state.items()}
                    if requests:
                        from .request_learning import sample_requests, request_return
                        commands,next_state,logp,entropy,predicted,forecast = sample_requests(
                            model,[workers[i]['feature_observation'] for i in active],choices,obs,actions,
                            selected_state,cache,**kwargs)
                    else:
                        logits, next_state, predicted = model(obs, actions, selected_state, **kwargs)
                        distribution = torch.distributions.Categorical(logits=logits)
                        selected = distribution.sample()
                        logp,entropy = distribution.log_prob(selected),distribution.entropy()
                        commands = [dict(rows[int(i)].action) for rows,i in zip(choices,selected)]
                    value = critic(model, next_state, args.mode)
                    if not requests:
                        next_state, forecast = model.commit_action(obs, actions, next_state, selected, mode=args.mode)
                    state = {k:v.index_copy(0,indices,next_state[k]) for k,v in state.items()}
                    zero = value.new_zeros(count)
                    log_probs.append(zero.index_copy(0,indices,logp))
                    values.append(zero.index_copy(0,indices,value))
                    entropies.append(zero.index_copy(0,indices,entropy))
                    reward = torch.zeros(count,device=device)
                    terminal = torch.zeros(count,dtype=torch.bool,device=device)
                    discount = torch.full((count,),args.gamma,device=device)
                    mask = torch.zeros(count,dtype=torch.bool,device=device); mask[indices]=True
                    for position,index in enumerate(active):
                        workers[index]['pipe'].send(('step', (commands[position],
                            {'behavior_log_probability':float(logp[position].detach()),
                             'critic_value':float(value[position].detach()),'policy_update':updates})))
                    for position, index in enumerate(active):
                        worker=workers[index]; reply=_receive(worker)
                        prediction_loss = logp.new_zeros(())
                        if args.outcome_weight:
                            target = torch.tensor([outcome_target({**reply, 'next_observation':reply['observation']})], device=device)
                            prediction_loss = args.outcome_weight * received_outcome_loss(
                                (predicted[position] if requests else predicted[position, selected[position]])[None], target).mean()
                        if forecast is not None:
                            target = torch.tensor(next_numeric_target(reply['observation'], worker['memory']), device=device)
                            prediction_loss = prediction_loss + args.transition_weight * F.smooth_l1_loss(forecast[position], target)
                        prediction_losses.append(prediction_loss)
                        worker['action_history'].append_transition(
                            dict(action=reply['action'], action_result=reply['action_result'], next_observation=reply['observation']),
                            worker['feature_observation'])
                        worker['observation']=reply['observation']
                        worker['done']=reply['done']
                        worker['reward']+=float(reply['training_reward'])
                        if worker['records'] is not None:
                            worker['records'].observe(reply['action'],reply['action_result'],reply['observation'],sequence=reply['index'])
                        reward[index]=reply['training_reward']; terminal[index]=reply['terminated']
                        if requests:
                            reward[index],discount[index] = request_return(
                                reply['potential_before'],reply['potential_after'],success=reply['completed_successfully'],
                                done=reply['done'],native_actions=reply['executor']['native_actions'],
                                gamma=args.gamma,action_cost=args.native_action_cost)
                            terminal[index] = reply['done']
                        if reply['done']:
                            worker['summary']=reply['summary']
                    interactions+=len(active)
                    rewards.append(reward); terminals.append(terminal); masks.append(mask); discounts.append(discount)
                indices=torch.tensor(participating,device=device)
                with torch.no_grad():
                    _,(obs,actions)=inputs(participating)
                    _,next_state,_=model(obs,actions,{k:v.index_select(0,indices) for k,v in state.items()},**kwargs)
                    bootstrap=critic(model,next_state,args.mode)
                    target=bootstrap.new_zeros(count).index_copy(0,indices,bootstrap)
                    returns=[]
                    for reward,terminal,mask,discount in zip(reversed(rewards),reversed(terminals),reversed(masks),reversed(discounts)):
                        target=torch.where(mask,reward+discount*target*(~terminal),target)
                        returns.append(target)
                    returns=torch.stack(list(reversed(returns)))
                valid=torch.stack(masks)
                predicted=torch.stack(values)
                policy_loss=-(torch.stack(log_probs)*(returns-predicted.detach()))[valid].mean()
                value_loss=F.mse_loss(predicted[valid],returns[valid])
                entropy=torch.stack(entropies)[valid].mean()
                prediction_loss=torch.stack(prediction_losses).mean()
                loss=policy_loss+args.value_weight*value_loss-args.entropy_weight*entropy+prediction_loss
                loss.backward()
                norm=torch.nn.utils.clip_grad_norm_([*model.parameters(),*critic.parameters()],1.)
                optimizer.step()
                state={k:v.detach() for k,v in state.items()}
                updates+=1
                row=dict(update=updates,interactions=interactions,active_environments=len(participating),
                    reward=float(torch.stack(rewards).sum()),policy_loss=float(policy_loss.detach()),
                    value_loss=float(value_loss.detach()),entropy=float(entropy.detach()),
                    prediction_loss=float(prediction_loss.detach()),
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
            if worker['process'].poll() is None:
                try:
                    worker['pipe'].send(('close',None))
                except (BrokenPipeError, EOFError):
                    pass  # Preserve the simulator exception already received.
        for worker in workers:
            # Drain an outstanding observation as the worker exits. Without
            # this, an error in another world can leave a writer blocked on a
            # full stdout pipe while the trainer waits for that writer.
            try:
                worker['process'].communicate(timeout=30)
            except subprocess.TimeoutExpired:
                worker['process'].terminate()
                worker['process'].communicate()
    return dict(episodes=len(episodes),interactions=interactions,updates=updates,
                elapsed_seconds=perf_counter()-started)
