"""Can the released native policies finish from the last step of a successful public experiment?"""
import argparse
from contextlib import redirect_stdout
from hashlib import sha256
import json
import os
import random
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch

from cdm.digital import PublicObjectMemory, public_candidates
from .evaluate import LearnedPolicy
from .features import canonical_action
from .full_suite import seeded_object_initialization
from .recovery import RecoveryExpert, RecoveryError
from .structured_observation import StructuredDiscoveryWorldEnv
from .train import batch_features, load_features


class TracedPolicy(LearnedPolicy):
    """Read out a reference action's rank without changing the policy logits."""
    probe_action = None
    probe = None

    @torch.no_grad()
    def receive_prefix(self, observation, carry_state=True):
        """The action head is stateless. Skip scoring candidates while the expert is in control."""
        if self.measurements is not None:
            self.measurements.observe_initial(observation)
            observation = {**observation, 'measurement_records': self.measurements.summaries()}
            if self.features.config.object_relations:
                observation['measurement_dialog_owner'] = self.measurements.dialog_owner
        observation = self.memory.observe(observation)
        if not carry_state:
            # Preserve the public object and measurement histories while leaving
            # the GRU and workspace at their initial values for the handoff.
            return
        # Candidate features are separate from observation tokens. Use one real candidate
        # to obtain the same observation encoding without evaluating an unused action menu.
        candidate = public_candidates(observation)[0]
        path = self.features.encode(observation, [candidate])
        obs, _ = batch_features([load_features(str(path))], self.device)
        event, self.state = self.model.assimilate(obs, self.state, persistent=self.persistent,
                                                  restart_workspace=self.restart_workspace)
        if self.mode == 'cdm':
            count = self.model.config.refinements if self.refinements is None else self.refinements
            self.state = self.model.refine_state(event, self.state, count)

    def action_logits(self, logits, observation, candidates, features):
        if self.probe_action is not None:
            keys = [canonical_action(c.action) for c in candidates]
            self.probe = None
            if self.probe_action in keys:
                scores = logits[0].detach().float()
                target = keys.index(self.probe_action)
                self.probe = {'rank': 1 + int((scores > scores[target]).sum()),
                              'ties': int((scores == scores[target]).sum()),
                              'probability': float(scores.softmax(0)[target])}
            self.probe_action = None
        return logits


def public_hash(obs):
    fields = [obs['ui'], obs['actions'], obs['teleport_locations']]
    return sha256(json.dumps(fields, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def completed(env):
    return all(row['completed_successfully'] for row in env.metrics)


def prepare(seed, root, object_seed):
    out = root / 'reference' / str(seed)
    out.mkdir(parents=True, exist_ok=True)
    trace, status, reason = [], 'unfinished', None
    with (out / 'simulator.log').open('w') as log, redirect_stdout(log), seeded_object_initialization(object_seed):
        np.random.seed((object_seed + seed) % 2**32)
        env = StructuredDiscoveryWorldEnv(out, 1000)
        obs = env.reset('Reactor Lab', 'Normal', seed)
        try:
            expert = RecoveryExpert(obs)
            while not env.ended:
                action = expert.recommend(obs)
                item = {'before': public_hash(obs), 'action': dict(action)}
                transition = env.step(action)
                obs = env.policy_input()
                item['returned'] = transition['action_result']
                trace.append(item)
                expert.observe(transition['action'], obs, transition['action_result'])
            status = 'success' if completed(env) else 'unfinished'
        except RecoveryError as exc:
            status, reason = 'reference_blocked', str(exc)
    data = dict(seed=seed, object_seed=object_seed, steps=len(trace), status=status, reason=reason, trace=trace)
    (out / 'trace.json').write_text(json.dumps(data, separators=(',', ':')) + '\n')
    return status


def run(seed, root, label, policy, suffix):
    reference = json.loads((root / 'reference' / str(seed) / 'trace.json').read_text())
    result = {'seed': seed, 'policy': label, 'reference_status': reference['status'],
              'reference_steps': reference['steps']}
    out = root / label / str(seed)
    out.mkdir(parents=True, exist_ok=True)
    if reference['status'] != 'success':
        result['status'] = 'no_successful_reference'
    else:
        prefix = reference['trace'][:-1]
        terminal = reference['trace'][-1]
        if policy is not None:
            policy.reset()
        memory, rng = PublicObjectMemory(), random.Random(1949 + seed)
        with (out / 'simulator.log').open('w') as log, redirect_stdout(log), seeded_object_initialization(reference['object_seed']):
            np.random.seed((reference['object_seed'] + seed) % 2**32)
            env = StructuredDiscoveryWorldEnv(out, 1000)
            obs = env.reset('Reactor Lab', 'Normal', seed)
            for i, row in enumerate(prefix):
                if public_hash(obs) != row['before']:
                    raise ValueError(f'Public prefix differs on seed {seed}, step {i}')
                memory.observe(obs)
                if policy is not None:
                    policy.receive_prefix(obs, carry_state=label != 'autonomous-cdm-reset')
                transition = env.step(row['action'])
                if transition['action_result'] != row['returned']:
                    raise ValueError(f'Returned result differs on seed {seed}, step {i}')
                if policy is not None:
                    policy.observe(transition)
                obs = env.policy_input()
            if public_hash(obs) != terminal['before'] or env.ended:
                raise ValueError(f'Final handoff state differs on seed {seed}')
            options = public_candidates(memory.observe(obs))
            keys = {canonical_action(c.action) for c in options}
            target = canonical_action(terminal['action'])
            result.update(prefix_steps=len(prefix), candidates=len(options), reference_action=terminal['action'],
                          reference_action_in_menu=target in keys,
                          progress_at_handoff=sum(r['progress'] for r in env.metrics) / len(env.metrics))
            if policy is not None:
                policy.probe_action = target
            actions = []
            while not env.ended and len(actions) < suffix:
                if policy is None:
                    choices = options if not actions else public_candidates(memory.observe(obs))
                    chosen = rng.choice(choices).action
                else:
                    chosen = policy(obs)
                transition = env.step(chosen)
                if policy is not None:
                    policy.observe(transition)
                actions.append({'action': chosen, 'accepted': transition['action_result']['success']})
                obs = env.policy_input()
            result.update(status='evaluated', actions=actions, first_matches_reference=
                          canonical_action(actions[0]['action']) == target, success=completed(env),
                          final_progress=sum(r['progress'] for r in env.metrics) / len(env.metrics))
            if policy is not None:
                result['reference_readout'] = policy.probe
    (out / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    return result['status']


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=['prepare', 'run'])
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--seeds', type=int, nargs='+', required=True)
    p.add_argument('--object-seed', type=int, default=73105)
    p.add_argument('--suffix', type=int, default=32)
    p.add_argument('--models', type=Path, help='Extracted native_actor directory from v0.18.0')
    p.add_argument('--label', choices=('autonomous-cdm', 'autonomous-cdm-reset', 'autonomous-gru', 'basic_operations-cdm',
                                      'basic_operations-gru', 'uniform'), default='uniform')
    a = p.parse_args()
    # Each native world starts in a fresh process. Earlier worlds in the same
    # interpreter can change a later initialization even with identical seeds.
    # Preserve the strict public-history hashes rather than accepting those mismatches.
    if len(a.seeds) > 1 or os.environ.get('PYTHONHASHSEED') != '0':
        for seed in a.seeds:
            cmd = [sys.executable, '-m', 'experiments.discoveryworld.native_handoff', a.stage,
                   '--root', str(a.root), '--seeds', str(seed), '--object-seed', str(a.object_seed),
                   '--suffix', str(a.suffix), '--label', a.label]
            if a.models is not None:
                cmd.extend(('--models', str(a.models)))
            subprocess.run(cmd, check=True, env={**os.environ, 'PYTHONHASHSEED': '0'})
        return
    torch.set_num_threads(1)
    torch.manual_seed(a.object_seed + a.seeds[0])
    if a.suffix < 1:
        p.error('Require at least one independent action')
    agent = None
    if a.stage == 'run' and a.label != 'uniform':
        if a.models is None:
            p.error('Learned policies require --models')
        checkpoint = a.label.removesuffix('-reset')
        agent = TracedPolicy(a.models / checkpoint / 'last.pt', a.root / 'features',
                              device='cpu', mode=checkpoint.rsplit('-', 1)[1])
    for seed in a.seeds:
        status = (prepare(seed, a.root, a.object_seed) if a.stage == 'prepare'
                  else run(seed, a.root, a.label, agent, a.suffix))
        print(seed, status, flush=True)


if __name__ == '__main__':
    main()
