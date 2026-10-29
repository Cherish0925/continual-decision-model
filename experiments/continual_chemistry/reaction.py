"""History-conditioned chemical responses inside the continuing experiment model."""
from __future__ import annotations

from dataclasses import asdict
import torch

from cdm.active_experiment import ActiveExperimentPolicy, ActiveExperimentConfig
from cdm.continual_experiment import ContinualExperimentPolicy, ContinualExperimentConfig, select_rows
from cdm.conditional_response import ConditionalResponse, ConditionalResponseConfig
from cdm.core import ContinualCore, CoreConfig
from .observations import model_from_core_checkpoint
from .recipes import primitive_recipes


class ReactionExperimentPolicy(ContinualExperimentPolicy):
    def __init__(self, base, config=ContinualExperimentConfig(), response_config=ConditionalResponseConfig()):
        super().__init__(base, config)
        self.response = ConditionalResponse(2 * self.core.config.width, response_config)
        self.direct_history = True
        self.core_response = True

    def initial(self, batch, device):
        state = super().initial(batch, device)
        c = self.response.config
        state.update(reaction_history=torch.zeros(batch, c.history_capacity, 20, device=device),
                     reaction_mask=torch.zeros(batch, c.history_capacity, dtype=torch.bool, device=device))
        return state

    def receive(self, state, observation, selected=None):
        revised = super().receive(state, observation, selected)
        rows = torch.ones(len(state['h']), dtype=torch.bool, device=state['h'].device) if selected is None else selected
        # All already returned experiments are visible. The first evidence row
        # describes the current apparatus, so it is not a response observation.
        history, mask = observation['evidence'][:, 1:], observation['evidence_mask'][:, 1:]
        if history.shape[1] > self.response.config.history_capacity:
            raise ValueError('Public history exceeds response capacity; increase the saved configuration')
        h = torch.zeros_like(state['reaction_history']); m = torch.zeros_like(state['reaction_mask'])
        h[:, :history.shape[1]] = history; m[:, :mask.shape[1]] = mask
        update = {**revised, 'reaction_history': h, 'reaction_mask': m}
        return select_rows(revised, update, rows)

    def readout(self, state, candidates, action_mask):
        out = super().readout(state, candidates, action_mask)
        proportions = candidates[..., 8:16]
        component_mask = candidates[:, 0, 16:24].bool()
        current = (candidates[..., 25] * 3).round().long()
        # Trim only padding, never observations. This matters for CPU inference
        # when the apparatus has a long but still sparse public ledger.
        valid = state['reaction_mask']
        used = max(1, int(valid.any(0).nonzero()[-1]) + 1) if valid.any() else 1
        history = state['reaction_history'][:, :used]
        history_mask = valid[:, :used] if self.direct_history else torch.zeros_like(valid[:, :used])
        history_q = history[..., :8] / history[..., :8].sum(-1, keepdim=True).clamp_min(1e-8)
        features = self.core.state_features(state['h'], state['z'])
        if not self.core_response:
            features = features * 0
        prediction = self.response(proportions, component_mask, current, history_q,
                                   (history[..., 16] * 3).round().long(),
                                   (history[..., 17] * 3).round().long(), history_mask, features)
        return {**out, 'outcomes': prediction['log_probabilities'],
                'latent_response_logits': prediction['latent_logits']}

    def configuration(self):
        return {**super().configuration(), 'response_config': asdict(self.response.config)}


def reaction_from_core_checkpoint(path, *, device='cpu', initialization_seed=750, response_config=ConditionalResponseConfig()):
    parent = model_from_core_checkpoint(path, device='cpu', initialization_seed=initialization_seed)
    model = ReactionExperimentPolicy(parent.base, parent.continual_config, response_config)
    model.load_state_dict(parent.state_dict(), strict=False)
    return model.to(device)


def load_reaction_checkpoint(path, device='cpu'):
    saved = torch.load(path, map_location='cpu', weights_only=True)
    core = ContinualCore(CoreConfig(**saved['core_config']), saved['model']['base.core.action_values'])
    base = ActiveExperimentPolicy(ActiveExperimentConfig(**saved['active_config']), core)
    model = ReactionExperimentPolicy(base, ContinualExperimentConfig(**saved['continual_config']),
                                    ConditionalResponseConfig(**saved['response_config']))
    model.load_state_dict(saved['model'])
    return model.to(device), saved


@torch.no_grad()
def initialize_response_prior(model, records):
    """Train-only conditional frequencies, with the unobserved tail distributed.

    This exactly initializes the *visible* probabilities to the conditional
    frequency reference. Splitting a censored tail uniformly claims no latent
    distinction unsupported by its corresponding observations.
    """
    k = model.response.config.levels
    counts = torch.ones_like(model.response.prior)
    for row in records:
        counts[len(row['component_counts']), row['rust_before'], row['rust_after']] += 1
    for before in range(k):
        visible = counts[:, before, :before + 1]
        visible = visible / visible.sum(-1, keepdim=True)
        counts[:, before, :before] = visible[:, :before]
        counts[:, before, before:] = visible[:, before:] / (k - before)
    model.response.prior.copy_(counts.log())


def primitive_observation_mask(observation):
    """Share exactly the proposal family used by the new native collection."""
    mask = observation['action_mask'].clone()
    for b in range(len(mask)):
        n = int(observation['candidates'][b, 0, 16:24].sum())
        allowed = set(primitive_recipes(n, include_leave=True))
        for a, features in enumerate(observation['candidates'][b]):
            counts = tuple((features[:n] * 6).round().int().tolist())
            mask[b, a] &= counts in allowed
        history = observation['evidence'][b, 1:]
        returned = observation['evidence_mask'][b, 1:]
        q = history[..., :8] / history[..., :8].sum(-1, keepdim=True).clamp_min(1e-8)
        candidate = observation['candidates'][b]
        same = (candidate[:, None, 8:16] - q[None]).abs().amax(-1) < 1e-6
        cannot_improve = history[:, 17][None] >= candidate[:, None, 25]
        mask[b] &= ~(same & cannot_improve & returned[None]).any(-1)
    return {**observation, 'action_mask': mask}


def permute_observation_components(observation, permutation):
    """Consistently rename ingredient identities across one observed history.

    The caller retains one permutation throughout a trajectory prefix/window.
    Candidate indices and returned labels stay attached to their same experiment.
    """
    result = dict(observation)
    n = len(permutation)
    for key, offsets in [('event', (0, 8)), ('evidence', (0, 8)), ('candidates', (0, 8, 16))]:
        source = observation[key]
        value = source.clone()
        for offset in offsets:
            value[..., offset:offset + n] = source[..., offset + permutation]
        result[key] = value
    return result
