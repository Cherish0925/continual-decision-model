"""Table tasks executed through the native tools' primitive-action semantics.

The geometry is reconstructed from the same training worlds. Navigation,
carried tools, radiation isolation, device placement, additive tuning and local
probing have persistent execution state. The actor receives returned evidence,
settings and costs; it cannot inspect the geometry or private resonance arrays.
"""
from __future__ import annotations

import numpy as np
import torch

from experiments.discoveryworld.train_active_core import measurement_event, test_event
from .delayed import ExactPhysicalTasks
from .scalar_execution import ScalarExecution

EXECUTION_FIELDS = ('native_pose', 'native_instruments', 'native_crystal_xy',
                    'native_container_xy', 'native_isolated', 'native_bound',
                    'native_steps', 'current_hz')


class NativeToolExecution:
    fields = EXECUTION_FIELDS

    def __init__(self, geometry_path, device, *, action_budget=400, action_cost=.005):
        with np.load(geometry_path) as source:
            self.geometry = {k: torch.as_tensor(v, device=device) for k, v in source.items()
                             if k != 'scenario_seed'}
        self.device, self.action_budget, self.action_cost = torch.device(device), action_budget, action_cost
        self.scalar = ScalarExecution(device)
        self.directions = torch.tensor(((0, -1), (1, 0), (0, 1), (-1, 0)), device=device)

    def object_coordinates(self, table, key):
        values = self.geometry[key][table.indices]
        return values.gather(1, table.permutation[:, :, None].expand(-1, -1, 2)).long()

    def initial(self, table):
        crystals = self.object_coordinates(table, 'crystal_xy')
        batch, device = table.batch, table.device
        # Normal Reactor Lab bootstrap reads all four devices after visiting the
        # three public stations. Its final read is reactor four, facing north.
        reactor_four = self.geometry['reactor_xy'][table.indices].max(1).values.long()
        pose = self.geometry['approach_pose'][table.indices, reactor_four[:, 0], reactor_four[:, 1]].long()
        return {'native_pose': pose, 'native_instruments': torch.zeros(batch, 5, device=device, dtype=torch.bool),
                'native_crystal_xy': crystals.clone(), 'native_container_xy': crystals.clone(),
                'native_isolated': torch.full((batch,), -1, device=device, dtype=torch.long),
                'native_bound': table.initial_support.clone(),
                'native_steps': torch.full((batch,), 15, device=device, dtype=torch.long),
                'current_hz': torch.where(table.known, table.known_hz,
                                         torch.full_like(table.known_hz, 5000.))}

    @torch.no_grad()
    def __call__(self, table, action, predictions):
        batch, device = table.batch, table.device
        rows = torch.arange(batch, device=device)
        before = table.native_steps.clone()
        is_measure = action < 20
        obj = torch.where(is_measure, action // 5, table.query)
        group = action.remainder(5)
        test = ~is_measure
        target_query = table.query.clone()
        table.steps += 1

        def pay(selected, amount=1):
            amount = torch.as_tensor(amount, device=device).expand(batch)
            available = (self.action_budget - table.native_steps).clamp_min(0)
            paid = torch.minimum(available, amount) * selected
            table.native_steps += paid
            return selected & (available >= amount)

        def approach(coordinate, selected, held=None):
            facing = table.native_pose[:, :2] + self.directions[table.native_pose[:, 2]]
            reachable = ((coordinate == table.native_pose[:, :2]).all(-1)
                         | (coordinate == facing).all(-1))
            if held is not None:
                reachable |= held
            move = pay(selected & ~reachable)
            index = table.indices[move]
            position = coordinate[move]
            table.native_pose[move] = self.geometry['approach_pose'][index, position[:, 0], position[:, 1]].long()
            return selected & (reachable | move)

        # Restore an isolated crystal before moving on to another operation.
        radiation = is_measure & (group == 2)
        same_isolated = radiation & (table.native_isolated == obj)
        restore = (table.native_isolated >= 0) & ~same_isolated
        isolated_obj = table.native_isolated.clamp_min(0)
        container = table.native_container_xy[rows, isolated_obj]
        reached = approach(container, restore)
        restored = pay(reached)
        table.native_crystal_xy[rows[restored], isolated_obj[restored]] = container[restored]
        table.native_isolated[restored] = -1
        ready = ~restore | restored

        # Instruments remain carried for the whole task, just as in native runs.
        instrument_xy = self.geometry['instrument_xy'][table.indices, group].long()
        held = table.native_instruments[rows, group]
        need_instrument = is_measure & ready & ~held
        reached = approach(instrument_xy, need_instrument)
        picked = pay(reached)
        table.native_instruments[rows[picked], group[picked]] = True
        ready &= ~is_measure | held | picked
        position = table.native_crystal_xy[rows, obj]
        isolate = radiation & ready & (table.native_isolated != obj)
        reached = approach(position, isolate)
        picked = pay(reached)
        # The route depends only on the source world's public terrain. It is
        # prepared using the exact navigation algorithm and all teleports count.
        route_cost = self.geometry['isolation_actions'][table.indices].long()
        moved = pay(picked, route_cost)
        table.native_pose[moved] = self.geometry['isolation_pose'][table.indices[moved]].long()
        table.native_isolated[moved] = obj[moved]
        ready &= ~isolate | moved
        ordinary_measure = is_measure & ~radiation & ready
        reached = approach(position, ordinary_measure)
        measure_ready = ready & (radiation | reached) & is_measure
        returned_measurement = pay(measure_ready)

        # Bind an unplaced target to its designated device, then start tuning.
        reactor_xy = self.object_coordinates(table, 'reactor_xy')[rows, obj]
        bound = table.native_bound[rows, obj]
        place = test & ready & ~bound
        reached = approach(position, place)
        picked = pay(reached)
        reached = approach(reactor_xy, picked)
        placed = pay(reached)
        table.native_bound[rows[placed], obj[placed]] = True
        table.native_crystal_xy[rows[placed], obj[placed]] = reactor_xy[placed]
        table.native_container_xy[rows[placed], obj[placed]] = reactor_xy[placed]
        ready &= ~test | bound | placed
        reached = approach(reactor_xy, test & ready)
        test_ready = test & ready & reached

        success = torch.zeros(batch, device=device, dtype=torch.bool)
        frequencies = table.known_hz.new_zeros(batch, 64)
        activations = torch.zeros(batch, 64, device=device, dtype=torch.bool)
        mask = torch.zeros_like(activations)
        if test_ready.any():
            selected = test_ready.nonzero().flatten()
            hypothesis = action[selected] - 20
            target = table.targets[selected, target_query[selected]]
            frequency = predictions['means'][selected, hypothesis]
            std = predictions['std'][selected, hypothesis]
            final_query = (~table.known).sum(-1)[selected] == 1
            result = self.scalar(table.current_hz[selected, target_query[selected]], frequency,
                                 std, target, self.action_budget - table.native_steps[selected],
                                 terminate_on_success=final_query)
            table.native_steps[selected] += result['actions']
            success[selected] = result['success']
            table.current_hz[selected, target_query[selected]] = result['current_hz']
            length = result['frequencies'].shape[-1]
            if length > frequencies.shape[-1]:
                raise RuntimeError('Native scalar request exceeded its expected feedback extent')
            frequencies[selected, :length] = result['frequencies']
            activations[selected, :length] = result['activated']
            mask[selected, :length] = result['mask']
            error = (table.current_hz[selected, target_query[selected]] - target).abs()
            table.best_error[selected, target_query[selected]] = torch.minimum(
                table.best_error[selected, target_query[selected]], error.to(table.best_error))
        cost = table.native_steps - before
        missing = (is_measure & ~returned_measurement).nonzero().flatten()
        if len(missing):
            zeros = table.values.new_zeros(len(missing), 9)
            table.event[missing] = measurement_event(obj[missing], group[missing], zeros, zeros.bool(),
                table.query[missing], table.steps[missing], table.max_steps, table.mixture, cost=cost[missing])
        measured = returned_measurement.nonzero().flatten()
        if len(measured):
            o, g = obj[measured], group[measured]
            sample = table.group_counts[measured, o, g]
            returned = table.samples[measured, o, g, sample]
            present = table.sample_present[measured, o, g, sample]
            previous, old = table.counts[measured, o], table.values[measured, o]
            new = (old * previous + returned * present) / (previous + present).clamp_min(1)
            new[:, 2] = torch.where(present[:, 2], torch.where(previous[:, 2] > 0,
                torch.minimum(old[:, 2], returned[:, 2]), returned[:, 2]), old[:, 2])
            table.values[measured, o] = new
            table.counts[measured, o] += present.long()
            table.group_counts[measured, o, g] += 1
            table.event[measured] = measurement_event(o, g, returned, present, table.query[measured],
                table.steps[measured], table.max_steps, table.mixture, cost=cost[measured])
            table.acquisitions[measured] += 1
        selected = test.nonzero().flatten()
        if len(selected):
            table.event[selected] = test_event(target_query[selected], action[selected] - 20, success[selected],
                table.steps[selected], table.max_steps, device, cost=cost[selected])
            table.tests[selected] += 1
        completed = success.nonzero().flatten()
        table.known_hz[completed, target_query[completed]] = table.current_hz[completed, target_query[completed]]
        table.known[completed, target_query[completed]] = True
        table.successes[completed] += 1
        remaining = table.target_mask & ~table.known
        table.query[completed] = remaining.float().argmax(-1)[completed]
        table.event[completed, 32:36] = 0
        table.event[completed, 32 + table.query[completed]] = 1
        reward = success.float() - cost * self.action_cost
        done = ~remaining.any(-1) | (table.native_steps >= self.action_budget) | (table.steps >= table.max_steps)
        table.alive &= ~done
        done |= ~table.observation()['action_mask'].any(-1)
        table.alive &= ~done
        return {'reward': reward, 'done': done, 'success': success, 'cost': cost,
                'frequencies': frequencies, 'activated': activations, 'mask': mask,
                'query': target_query}


class NativeExecutionTasks(ExactPhysicalTasks):
    def __init__(self, pool, plan, mixture, *, truth_path, geometry_path, max_steps=160,
                 action_budget=400, action_cost=.005):
        self.total_bootstrap_actions = 0
        self.execution = NativeToolExecution(geometry_path, pool.device,
                                            action_budget=action_budget, action_cost=action_cost)
        super().__init__(pool, plan, mixture, truth_path=truth_path, max_steps=max_steps)

    @torch.no_grad()
    def _begin_tasks(self, rows, initial=False):
        super()._begin_tasks(rows, initial=initial)
        self.total_bootstrap_actions += 15 * len(rows)
        state = self.execution.initial(self.table)
        for key, value in state.items():
            if initial:
                setattr(self.table, key, value)
            else:
                getattr(self.table, key)[rows] = value[rows]

    @torch.no_grad()
    def next_tasks(self, task_done, *, restart_lifetimes=True):
        costs = self.table.native_steps[task_done].tolist()
        begin = len(self.task_rows)
        super().next_tasks(task_done, restart_lifetimes=restart_lifetimes)
        for row, cost in zip(self.task_rows[begin:], costs):
            row['native_actions'] = cost


from experiments.discoveryworld.train_active_lifetime import _TASK_FIELDS
from .delayed import DelayedExperiments


class NativeDelayedExperiments(DelayedExperiments):
    """Delay the complete native-tool receipt, including every executed probe."""

    def __init__(self, environment, *, seed, max_delay=4):
        super().__init__(environment, seed=seed, max_delay=max_delay)
        self.fields = (*_TASK_FIELDS, *EXECUTION_FIELDS)
        self.snapshot = {key: getattr(environment.table, key).clone() for key in self.fields}
        self.pending_cost = torch.zeros(self.batch, device=self.device, dtype=torch.long)
        self.pending_frequencies = torch.zeros(self.batch, 64, device=self.device, dtype=torch.float64)
        self.pending_activated = torch.zeros(self.batch, 64, device=self.device, dtype=torch.bool)
        self.pending_mask = torch.zeros_like(self.pending_activated)
        self.failed_settings = torch.zeros(self.batch, 4, environment.execution.action_budget,
                                          dtype=torch.float64, device=self.device)
        self.total_native_actions = 0

    @torch.no_grad()
    def dispatch(self, actions, predictions, selected):
        if (selected & self.pending).any():
            raise ValueError('Cannot execute a second request before its first result returns')
        rows = selected.nonzero().flatten()
        if not len(rows):
            return
        from copy import copy
        table = copy(self.env.table)
        table.batch = len(rows)
        for key in self.fields:
            setattr(table, key, getattr(self.env.table, key)[rows].clone())
        predictions = {key: value[rows].detach() for key, value in predictions.items()}
        result = self.env.execution(table, actions[rows], predictions)
        for key in self.fields:
            self.snapshot[key][rows] = getattr(table, key)
        latency = 1 + ((rows * 1103515245 + self.dispatched[rows] * 12345 + self.seed) % self.max_delay)
        self.due[rows] = self.tick + latency
        self.pending_latency[rows] = latency
        self.pending_reward[rows], self.pending_done[rows] = result['reward'], result['done']
        self.pending_success[rows], self.pending_cost[rows] = result['success'], result['cost']
        self.pending_frequencies[rows] = result['frequencies']
        self.pending_activated[rows] = result['activated']
        self.pending_mask[rows] = result['mask']
        self.pending_query[rows] = result['query']
        self.pending[rows] = True
        self.dispatched[rows] += 1

    @torch.no_grad()
    def advance(self):
        self.tick += 1
        arrived = self.pending & (self.due <= self.tick)
        for key in self.fields:
            getattr(self.env.table, key)[arrived] = self.snapshot[key][arrived]
        reward = self.pending_reward * arrived
        self.env.task_rewards += reward
        done = self.pending_done & arrived
        rows = torch.arange(self.batch, device=self.device)
        lengths = self.env.plan.lengths[rows, self.env.lifetime]
        lifetime_done = done & (self.env.task + 1 == lengths)
        self.pending &= ~arrived
        self.returned += arrived.long()
        mask = self.pending_mask & arrived[:, None]
        failed = mask & ~self.pending_activated
        indices, attempts = failed.nonzero(as_tuple=True)
        count_before = failed.long().cumsum(-1) - 1
        query = self.pending_query[indices]
        position = self.failed_counts[indices, query] + count_before[indices, attempts]
        if len(position) and position.max() >= self.failed_settings.shape[-1]:
            raise RuntimeError('Executed settings exceeded the primitive-action budget')
        self.failed_settings[indices, query, position] = self.pending_frequencies[indices, attempts]
        self.failed_counts[rows, self.pending_query] += failed.sum(-1)
        cost = self.pending_cost * arrived
        self.total_native_actions += int(cost.sum())
        feedback = {'intervals': torch.stack((self.pending_frequencies - 2., self.pending_frequencies + 2.), -1),
                    'success': self.pending_activated.clone(), 'mask': mask}
        targets = torch.stack((self.pending_success.float(), self.pending_cost.float().log1p()), -1)
        return {'arrived': arrived, 'reward': reward, 'task_done': done, 'lifetime_done': lifetime_done,
                'targets': targets, 'physical_feedback': feedback, 'native_actions': cost}
