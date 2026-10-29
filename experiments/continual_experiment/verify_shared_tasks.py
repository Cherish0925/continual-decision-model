"""One real two-task gradient update, checkpoint roundtrip and state continuation.

This checks a training entry point, not improvement in task performance. The
physical example executes one measurement in a training world; the chemical
example uses a recorded training experiment. No hidden law labels are read.
"""
from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from torch.nn import functional as F

from cdm.continual_physics import load_continual_physics_checkpoint
from cdm.shared_core import SharedCorePolicies
from experiments.continual_chemistry.observations import observe, feedback
from experiments.continual_chemistry.reaction import load_reaction_checkpoint, primitive_observation_mask
from experiments.continual_chemistry.objectives import scaled_feedback_loss
from experiments.continual_physics.evaluate_native import native_observation
from experiments.discoveryworld.active_native import NativeActiveEpisode
from experiments.discoveryworld.adapter import TrajectoryWriter
from experiments.discoveryworld.evaluate_active_stream import feedback_targets
from experiments.discoveryworld.structured_observation import StructuredDiscoveryWorldEnv
from .shared_tasks import save_shared_checkpoint, load_shared_checkpoint


def measured_example(model, output, training_path):
    with np.load(training_path) as data:
        seed = int(data['scenario_seed'][0])
    folder = output / 'physical_measurement'
    folder.mkdir()
    with (folder / 'native.log').open('w') as log, redirect_stdout(log), torch.no_grad():
        env = StructuredDiscoveryWorldEnv(folder, 60)
        env.reset('Reactor Lab', 'Normal', seed)
        with TrajectoryWriter(folder, {'purpose': 'shared-core training-path verification'}) as writer:
            episode = NativeActiveEpisode(env, writer)
            episode.bootstrap()
            state = model.initial(1, 'cpu')
            event = torch.zeros(1, 36)
            event[0, 32 + int(episode.query[0])] = 1
            obs, _ = native_observation(model, state, episode, event, 0, 160, 'cpu', task_boundary=True)
            state = model.receive(state, obs)
            out = model.readout(state, obs['candidates'], obs['action_mask'])
            # Select an actual available measurement. This is one public return,
            # not a counterfactual label or an uncompleted task score.
            action = int(out['logits'][0, :20].argmax())
            decision = episode.apply(action)
            returned = feedback_targets(decision, 'cpu')
    return obs, action, returned, {'seed': seed, 'action': action,
        'returned_targets': returned.tolist(), 'native_operations': int(round(float(returned[0, 1].expm1())))}


def compare_outputs(expected, actual, keys):
    differences = {}
    for key in keys:
        a, b = expected[key], actual[key]
        torch.testing.assert_close(a, b, rtol=0, atol=0)
        differences[key] = float(torch.where(torch.isfinite(a), (a-b).abs(), 0.).max().detach())
    return differences


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--physics', type=Path, required=True)
    parser.add_argument('--chemistry', type=Path, required=True)
    parser.add_argument('--chemical-data', type=Path, default=Path('runs/continual_physics/chemistry_exploration'))
    parser.add_argument('--physical-data', type=Path, default=Path('runs/discoveryworld/active_experiments/data_large/train.npz'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(811)
    args.output.mkdir(parents=True, exist_ok=False)
    start = perf_counter()
    physics, _ = load_continual_physics_checkpoint(args.physics)
    chemistry, chemical_saved = load_reaction_checkpoint(args.chemistry)
    physics.eval(); chemistry.eval()
    physical_obs, action, returned, measurement = measured_example(physics, args.output, args.physical_data)
    manifest = json.loads((args.chemical_data/'manifest.json').read_text())
    first = next(row for row in manifest['episodes'] if row['split'] == 'train')
    path = args.chemical_data/first['directory']/'experiments.jsonl'
    with path.open() as source:
        record = json.loads(next(source))
    chemical_obs, choices = observe(record['state'], elapsed_actions=record['transition_start'])
    chemical_obs = primitive_observation_mask(chemical_obs)
    chemical_action = choices.index(tuple(record['component_counts']))
    observations = {'physics': physical_obs, 'chemistry': chemical_obs}
    originals = {}
    for name, policy in [('physics', physics), ('chemistry', chemistry)]:
        obs = observations[name]
        state = policy.receive(policy.initial(1, 'cpu'), obs)
        original = policy.readout(state, obs['candidates'], obs['action_mask'])
        # Keep the same differentiable attention path on both sides; no_grad
        # can select a different PyTorch inference kernel with roundoff changes.
        originals[name] = {key: value.detach() for key, value in original.items()}
    model = SharedCorePolicies({'physics': physics, 'chemistry': chemistry})
    assert model.core is model.tasks['physics'].core is model.tasks['chemistry'].core
    groups = model.parameter_groups(core_lr=3e-5, interface_lr=1e-3)
    identities = [id(p) for group in groups for p in group['params']]
    assert len(identities) == len(set(identities))
    equivalent, outputs = {}, {}
    for name in observations:
        state = model.receive(name, model.initial(1, 'cpu'), observations[name])
        outputs[name] = model.readout(name, state, observations[name])
        keys = ['logits', 'outcomes', 'predictions']
        if name == 'physics': keys += ['physical_means', 'physical_std', 'physical_settings']
        equivalent[name] = compare_outputs(originals[name], outputs[name], keys)

    physical_loss = F.smooth_l1_loss(outputs['physics']['predictions'][:, action], returned)
    chemical_out = outputs['chemistry']
    chemical_loss = (-chemical_out['outcomes'][0, chemical_action, record['rust_after']]
        + scaled_feedback_loss(chemical_out['predictions'][:, chemical_action], feedback(record),
                               chemical_saved['training']['feedback_statistics']))
    parameters = [*model.core.gru.parameters(), *model.core.block.parameters()]
    gradients = {}
    for name, loss in [('physics', physical_loss), ('chemistry', chemical_loss)]:
        values = torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True)
        norm = torch.stack([g.square().sum() for g in values if g is not None]).sum().sqrt()
        assert torch.isfinite(norm) and norm > 0
        gradients[name] = float(norm)
    before = model.core.gru.weight_ih_l0.detach().clone()
    optimizer = torch.optim.AdamW(groups)
    optimizer.zero_grad(set_to_none=True)
    ((physical_loss + chemical_loss) / 2).backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
    optimizer.step()
    change = float((before-model.core.gru.weight_ih_l0.detach()).norm())
    assert change > 0
    save_shared_checkpoint(args.output/'shared.pt', model,
        training={'updates': 1, 'purpose': 'real two-task gradient and continuation check',
                  'physical_loss': 'actual measurement progress and log native-operation cost',
                  'chemical_loss': 'censored returned-rust likelihood and train-scaled native cost'})
    restored, saved = load_shared_checkpoint(args.output/'shared.pt')
    model.eval(); restored.eval()
    assert restored.core is restored.tasks['physics'].core is restored.tasks['chemistry'].core
    assert all(not any(key.startswith('base.core.') for key in task['model']) for task in saved['tasks'].values())
    assert len(saved['core']) == len(model.core.state_dict())
    reloaded = {}
    with torch.no_grad():
        for name, obs in observations.items():
            a = model(name, obs, model.initial(1, 'cpu'))
            b = restored(name, obs, restored.initial(1, 'cpu'))
            reloaded[name] = compare_outputs(a, b, ['logits', 'outcomes', 'predictions'])
        state = model.initial(1, 'cpu')
        state = model.receive('physics', state, physical_obs)
        held = {key: value.clone() for key, value in state.items() if key.startswith('physics:')}
        h = state['h'].clone()
        state = model.receive('chemistry', state, chemical_obs)
        assert not torch.equal(state['h'], h)
        assert state['h'] is model.task_state(state, 'physics')['h']
        assert state['z'] is model.task_state(state, 'chemistry')['z']
        for key, value in held.items():
            torch.testing.assert_close(state[key], value, rtol=0, atol=0)
    summary = {'purpose': 'shared-core implementation and one real-data update; no task-performance claim',
        'core': {'width': model.core.config.width, 'slots': model.core.config.slots,
                 'tensors_stored_once': len(saved['core'])},
        'initial_output_max_difference': equivalent, 'reloaded_output_max_difference': reloaded,
        'core_gradient_norm_by_task': gradients, 'updated_gru_weight_norm': change,
        'optimizer_parameters': len(identities), 'optimizer_unique_parameters': len(set(identities)),
        'task_head_tensors': {name: len(row['model']) for name, row in saved['tasks'].items()},
        'loss': {'physics': float(physical_loss.detach()), 'chemistry': float(chemical_loss.detach())},
        'physical_measurement': measurement,
        'chemical_training_record': str(Path(first['directory'])/'experiments.jsonl'),
        'shared_state_continues': True, 'task_numeric_memories_preserved': True,
        'seconds': perf_counter()-start}
    (args.output/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__': main()
