"""History-conditioned chemical responses inside the continuing experiment model."""
from __future__ import annotations

from dataclasses import asdict
import torch

from cdm.active_experiment import ActiveExperimentPolicy, ActiveExperimentConfig
from cdm.continual_experiment import Activity, ContinualExperimentPolicy, ContinualExperimentConfig, select_rows
from cdm.conditional_response import ConditionalResponse, ConditionalResponseConfig, observable_log_probabilities
from cdm.adaptive_response import AdaptiveResponse, ResponseAdaptationConfig
from cdm.response_replay import ResponseReplay, ResponseReplayConfig, recent_support
from cdm.experimental_memory import ExperimentalMemory, ExperimentalMemoryConfig
from cdm.core import ContinualCore, CoreConfig
from .observations import model_from_core_checkpoint
from .recipes import primitive_recipes


class ReactionExperimentPolicy(ContinualExperimentPolicy):
    def __init__(self, base, config=ContinualExperimentConfig(), response_config=ConditionalResponseConfig(),
                 adaptation_config=None, replay_config=None, memory_config=None):
        super().__init__(base, config)
        self.response = ConditionalResponse(2 * self.core.config.width, response_config)
        self.direct_history = True
        self.core_response = True
        self.response_adaptation = (None if adaptation_config is None else
            AdaptiveResponse(response_config.width, response_config.levels, adaptation_config))
        self.retain_response_adaptation = True
        self.response_replay = None if replay_config is None else ResponseReplay(response_config.levels, replay_config)
        self.experimental_memory = None if memory_config is None else ExperimentalMemory(memory_config)
        if adaptation_config is not None:
            self.experiment_residual = torch.nn.Linear(config.adapter_width + 5, 1)
            torch.nn.init.zeros_(self.experiment_residual.weight)
            torch.nn.init.zeros_(self.experiment_residual.bias)

    def initial(self, batch, device):
        state = super().initial(batch, device)
        c = self.response.config
        state.update(reaction_history=torch.zeros(batch, c.history_capacity, 20, device=device),
                     reaction_mask=torch.zeros(batch, c.history_capacity, dtype=torch.bool, device=device))
        if self.response_adaptation is not None:
            state.update(self.response_adaptation.initial(batch, device))
        if self.experimental_memory is not None:
            state.update(self.experimental_memory.initial(batch, device))
        return state

    def next_apparatus(self, state, selected, lifetime_boundary):
        if self.experimental_memory is not None:
            state = self.experimental_memory.archive(state, selected & ~lifetime_boundary)
        fresh = self.initial(len(selected), selected.device)
        continued = {**fresh, 'h': state['h'], 'z': state['z']}
        if 'response_slow' in state:
            continued['response_slow'] = state['response_slow']
        for key in ('archive_history', 'archive_mask', 'archive_components', 'archive_cursor'):
            if key in state:
                continued[key] = state[key]
        return select_rows(select_rows(state, continued, selected), fresh, lifetime_boundary)

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
        extra = {}
        if self.response_adaptation is not None:
            features = self.response_adaptation.features(prediction['hidden'])
            if self.response_replay is None:
                logits = self.response_adaptation.predict(state, prediction['latent_logits'], features)
            else:
                support = recent_support(self.replay_context(state, candidates), self.response_adaptation.config.memory_size)
                n = support['x'].shape[1]
                if self.response_replay.config.mode == 'stored':
                    size = self.response_adaptation.config.memory_size
                    index = (torch.arange(n, device=features.device)[None] +
                             (state['response_cursor'] - size).clamp_min(0)[:, None]) % size
                    support['x'] = state['response_x'].gather(1, index[..., None].expand(-1, -1, features.shape[-1]))
                    support['base'] = state['response_base'].gather(1, index[..., None].expand(-1, -1, self.response.config.levels))
                    support['valid'] = state['response_valid'].gather(1, index)
                logits = self.response_replay(prediction['latent_logits'], features, proportions,
                    support['base'], support['x'], support['inputs'], support['before'], support['after'], support['valid'])
            prediction['log_probabilities'] = observable_log_probabilities(logits, current)
            extra = {'response_features': features, 'response_base_logits': prediction['latent_logits']}
            prediction['latent_logits'] = logits
        if self.experimental_memory is not None:
            old = self.archived_predictions(state, proportions, component_mask, current)
            experts = torch.cat((prediction['latent_logits'].log_softmax(-1)[:, :, None], old), 2)
            mixed = self.experimental_memory.combine(state, component_mask, experts)
            prediction['log_probabilities'] = observable_log_probabilities(mixed, current)
            extra['memory_experts'] = observable_log_probabilities(experts, current[:, :, None])
        return {**out, **extra, 'outcomes': prediction['log_probabilities'],
                'latent_response_logits': prediction['latent_logits']}

    def archived_predictions(self, state, proportions, components, current):
        b, a = proportions.shape[:2]
        k = self.experimental_memory.config.slots
        valid = state['archive_mask']
        used = max(1, int(valid.any((0, 1)).nonzero()[-1]) + 1) if valid.any() else 1
        history = state['archive_history'][:, :, :used].flatten(0, 1)
        mask = valid[:, :, :used].flatten(0, 1)
        q = (history[..., :8] / history[..., :8].sum(-1, keepdim=True).clamp_min(1e-8))
        before, after = (history[..., 16] * 3).round().long(), (history[..., 17] * 3).round().long()
        f = self.core.state_features(state['h'], state['z'])[:, None].expand(-1, k, -1).flatten(0, 1)
        c = state['archive_components'].flatten(0, 1)
        query = proportions[:, None].expand(-1, k, -1, -1).flatten(0, 1)
        rust = current[:, None].expand(-1, k, -1).flatten(0, 1)
        # The conditional predictor is shared with the current task. Each old
        # ledger is re-encoded under its present weights, not old hidden features.
        prediction = self.response(query, c, rust, q, before, after, mask, f)
        logits = prediction['latent_logits']
        if self.response_replay is not None:
            support = self.response(q, c, before, q, before, after, mask, f)
            logits = self.response_replay(logits, self.response_adaptation.features(prediction['hidden']), query,
                support['latent_logits'], self.response_adaptation.features(support['hidden']), q, before, after, mask)
        return logits.log_softmax(-1).reshape(b, k, a, -1).transpose(1, 2)

    def replay_context(self, state, candidates):
        valid = state['reaction_mask']
        used = max(1, int(valid.any(0).nonzero()[-1]) + 1) if valid.any() else 1
        history = state['reaction_history'][:, :used]
        proportions = history[..., :8] / history[..., :8].sum(-1, keepdim=True).clamp_min(1e-8)
        before, after = (history[..., 16] * 3).round().long(), (history[..., 17] * 3).round().long()
        f = self.core.state_features(state['h'], state['z'])
        prediction = self.response(proportions, candidates[:, 0, 16:24].bool(), before,
            proportions, before, after, valid[:, :used], f)
        return dict(base=prediction['latent_logits'], x=self.response_adaptation.features(prediction['hidden']),
                    inputs=proportions, before=before, after=after, valid=valid[:, :used])

    def experiment_logits(self, out, observation):
        from .observations import outcome_action_scores
        score = outcome_action_scores(out, observation)
        probability = out['outcomes'].softmax(-1).detach()
        residual = self.experiment_residual(torch.cat((out['features'], probability,
            out['predictions'][..., 1:].detach()), -1)).squeeze(-1)
        # Start with the inherited progress/cost ranking. The learned residual
        # is trained from subsequent task return, including nonterminal tests.
        return (4 * score.clamp_min(1e-8).log().detach() + residual).masked_fill(
            ~observation['action_mask'], -torch.inf)

    def record_response(self, state, out, actions, before, after):
        row = torch.arange(len(actions), device=actions.device)
        state = self.response_adaptation.record(state, out['response_features'][row, actions],
            out['response_base_logits'][row, actions], before, after)
        if self.experimental_memory is not None:
            state = self.experimental_memory.record(state, out['memory_experts'][row, actions], after)
        return state

    def internal(self, state, activity, observation):
        revised = super().internal(state, activity, observation)
        if self.response_adaptation is not None and self.response_replay is None:
            for kind, name in ((Activity.LEARN, 'fast'), (Activity.CONSOLIDATE, 'slow')):
                selected = activity == kind
                if selected.any():
                    revised = self.response_adaptation.update(revised, name, selected,
                        retain=self.retain_response_adaptation)
        return revised

    def configuration(self):
        result = {**super().configuration(), 'response_config': asdict(self.response.config)}
        if self.response_adaptation is not None:
            result['response_adaptation_config'] = asdict(self.response_adaptation.config)
        if self.response_replay is not None:
            result['response_replay_config'] = asdict(self.response_replay.config)
        if self.experimental_memory is not None:
            result['experimental_memory_config'] = asdict(self.experimental_memory.config)
        return result


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
                                    ConditionalResponseConfig(**saved['response_config']),
                                    ResponseAdaptationConfig(**saved['response_adaptation_config'])
                                    if 'response_adaptation_config' in saved else None,
                                    ResponseReplayConfig(**saved['response_replay_config'])
                                    if 'response_replay_config' in saved else None,
                                    ExperimentalMemoryConfig(**saved['experimental_memory_config'])
                                    if 'experimental_memory_config' in saved else None)
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
