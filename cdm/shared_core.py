"""Task interfaces that train and use one continuing neural core.

Task adapters and numeric memories retain their own units. h/z are the single
continuing state; changing the active interface does not make a second copy.
"""
from __future__ import annotations

import torch
from torch import nn


class SharedCorePolicies(nn.Module):
    def __init__(self, policies: dict[str, nn.Module]):
        super().__init__()
        if not policies:
            raise ValueError('At least one task interface is required')
        first = next(iter(policies.values())).core
        reference = first.state_dict()
        for name, policy in policies.items():
            other = policy.core.state_dict()
            if policy.core.config != first.config or other.keys() != reference.keys():
                raise ValueError(f'{name} has an incompatible core configuration')
            if any(not torch.equal(other[key], reference[key]) for key in reference):
                raise ValueError(f'{name} does not start from the same core weights')
        self.core = first
        for policy in policies.values():
            policy.base.core = self.core
        self.tasks = nn.ModuleDict(policies)

    def initial(self, batch, device):
        state = {}
        for name, policy in self.tasks.items():
            task = policy.initial(batch, device)
            if not state:
                state.update(h=task['h'], z=task['z'])
            state.update({name + ':' + key: value for key, value in task.items() if key not in {'h', 'z'}})
        return state

    def task_state(self, state, task):
        prefix = task + ':'
        return {'h': state['h'], 'z': state['z'],
                **{key.removeprefix(prefix): value for key, value in state.items() if key.startswith(prefix)}}

    @staticmethod
    def merge(state, task, revised):
        return {**state, 'h': revised['h'], 'z': revised['z'],
                **{task + ':' + key: value for key, value in revised.items() if key not in {'h', 'z'}}}

    def receive(self, task, state, observation, selected=None):
        return self.merge(state, task, self.tasks[task].receive(self.task_state(state, task), observation, selected))

    def readout(self, task, state, observation):
        return self.tasks[task].readout(self.task_state(state, task), observation['candidates'], observation['action_mask'])

    def begin_action(self, task, state, features, actions, selected=None, **kwargs):
        revised = self.tasks[task].begin_action(self.task_state(state, task), features, actions, selected, **kwargs)
        return self.merge(state, task, revised)

    def feedback(self, task, state, returned, selected=None, **kwargs):
        revised = self.tasks[task].feedback(self.task_state(state, task), returned, selected, **kwargs)
        return self.merge(state, task, revised)

    def internal(self, task, state, activity, observation):
        revised = self.tasks[task].internal(self.task_state(state, task), activity, observation)
        return self.merge(state, task, revised)

    def forward(self, task, observation, state):
        state = self.receive(task, state, observation)
        return {**self.readout(task, state, observation), 'state': state}

    def parameter_groups(self, *, core_lr=3e-5, interface_lr=1e-3):
        core_ids = {id(p) for p in self.core.parameters()}
        shared, task_parameters, seen = [], [], set()
        for parameter in self.parameters():
            if not parameter.requires_grad or id(parameter) in seen:
                continue
            seen.add(id(parameter))
            (shared if id(parameter) in core_ids else task_parameters).append(parameter)
        return [{'params': shared, 'lr': core_lr, 'name': 'shared_core'},
                {'params': task_parameters, 'lr': interface_lr, 'name': 'task_interfaces'}]

    def checkpoint(self):
        tasks = {}
        for name, policy in self.tasks.items():
            weights = {key: value for key, value in policy.state_dict().items() if not key.startswith('base.core.')}
            configuration = policy.configuration()
            configuration.pop('core_config', None)
            flags = {key: getattr(policy, key) for key in ('retain_adaptation', 'retain_physical_adaptation',
                     'direct_history', 'core_response', 'memory_action_credit') if hasattr(policy, key)}
            tasks[name] = {'configuration': configuration, 'model': weights, 'runtime': flags}
        return {'kind': 'shared_experiment_core_v1', 'core_config': dict(vars(self.core.config)),
                'core': self.core.state_dict(), 'tasks': tasks}
