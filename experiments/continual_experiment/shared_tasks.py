"""Load the existing physical and chemical interfaces over one neural core."""
from __future__ import annotations

import torch

from cdm.active_experiment import ActiveExperimentConfig, ActiveExperimentPolicy
from cdm.continual_experiment import ContinualExperimentConfig
from cdm.continual_physics import ContinualPhysicsConfig, ContinualPhysicsPolicy, load_continual_physics_checkpoint
from cdm.core import CoreConfig, ContinualCore
from cdm.experimental_belief import ExperimentalBelief
from cdm.conditional_response import ConditionalResponseConfig
from cdm.adaptive_response import ResponseAdaptationConfig
from cdm.recurrent_feedback import RecurrentFeedbackConfig, configure_recurrent_feedback
from cdm.shared_core import SharedCorePolicies
from experiments.continual_chemistry.reaction import ReactionExperimentPolicy, load_reaction_checkpoint


def from_task_checkpoints(physical_path, chemistry_path, device='cpu', *, belief_checkpoint=None,
                          initialization_seed=811):
    # A released active policy may need the continuing/physical interfaces
    # initialized. Fix that initialization separately from the inherited core.
    torch.manual_seed(initialization_seed)
    physical, _ = load_continual_physics_checkpoint(physical_path, device, belief_checkpoint=belief_checkpoint)
    chemistry, _ = load_reaction_checkpoint(chemistry_path, device)
    return SharedCorePolicies({'physics': physical, 'chemistry': chemistry})


def save_shared_checkpoint(path, model, *, training=None):
    torch.save({**model.checkpoint(), 'training': training or {}}, path)


def load_shared_checkpoint(path, device='cpu'):
    saved = torch.load(path, map_location='cpu', weights_only=True)
    if saved['kind'] != 'shared_experiment_core_v1':
        raise ValueError('Expected a shared-core checkpoint')
    core = ContinualCore(CoreConfig(**saved['core_config']), saved['core']['action_values'])
    core.load_state_dict(saved['core'])
    tasks = {}
    for name, entry in saved['tasks'].items():
        config = entry['configuration']
        base = ActiveExperimentPolicy(ActiveExperimentConfig(**config['active_config']), core)
        continual = ContinualExperimentConfig(**config['continual_config'])
        if name == 'physics':
            c = config['belief_config']
            belief = ExperimentalBelief(c['normalization'], memory_slots=c['memory_slots'], noise_floor_hz=c['noise_floor_hz'])
            policy = ContinualPhysicsPolicy(base, belief, continual, ContinualPhysicsConfig(**config['physics_config']))
        elif name == 'chemistry':
            policy = ReactionExperimentPolicy(base, continual, ConditionalResponseConfig(**config['response_config']),
                ResponseAdaptationConfig(**config['response_adaptation_config'])
                if 'response_adaptation_config' in config else None)
        else:
            raise ValueError(f'Unknown task interface {name}')
        if 'recurrent_feedback_config' in config:
            configure_recurrent_feedback(policy, RecurrentFeedbackConfig(**config['recurrent_feedback_config']))
        weights = {**entry['model'], **{'base.core.' + key: value for key, value in saved['core'].items()}}
        policy.load_state_dict(weights, strict=True)
        for key, value in entry['runtime'].items():
            setattr(policy, key, value)
        tasks[name] = policy
    return SharedCorePolicies(tasks).to(device), saved
