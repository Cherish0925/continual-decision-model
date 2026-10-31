"""Train one core on interleaved physical decisions and chemical returns.

Each gradient window contains physical actions, returned chemical experiments,
then further physical actions. The continuing condition carries h/z through
both switches. The matched separated condition uses the same weights, losses
and budgets with a separate h/z for each task interface. Chemistry labels are
only recorded, actually executed experiments; they do not supply a policy reward
for unexecuted alternatives. Physics uses exact native execution and costs.
"""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import random
from time import perf_counter

import torch
from torch.nn import functional as F

from cdm.continual_experiment import detach_state
from cdm.shared_core import SharedCorePolicies
from experiments.continual_chemistry.observations import feedback
from experiments.continual_chemistry.objectives import scaled_feedback_loss
from experiments.continual_chemistry.reaction import load_reaction_checkpoint, permute_observation_components
from experiments.continual_chemistry.warmup_reaction import batch_observations, query_observation
from experiments.continual_physics.train import (
    initialize_experiment, observe, reference_receive, training_parser, train_window,
)
from .shared_tasks import save_shared_checkpoint, load_shared_checkpoint


class ChemicalStreams:
    """Successive recorded episodes; a lane never sees a future return."""

    def __init__(self, root, batch, seed):
        manifest = json.loads((root / 'manifest.json').read_text())
        self.entries = [row for row in manifest['episodes'] if row['split'] == 'train']
        self.episodes = [[json.loads(line) for line in
                          (root / row['directory'] / 'experiments.jsonl').read_text().splitlines()]
                         for row in self.entries]
        self.rng = random.Random(seed)
        self.order = []
        self.episode = [0] * batch
        self.position = [0] * batch
        self.permutations = [[] for _ in range(batch)]
        self.new = [True] * batch
        self.returned = self.ended = 0
        for lane in range(batch):
            self._next(lane)

    def _next(self, lane):
        if not self.order:
            self.order = list(range(len(self.episodes)))
            self.rng.shuffle(self.order)
        self.episode[lane] = self.order.pop()
        self.position[lane] = 0
        n = len(self.episodes[self.episode[lane]][0]['component_counts'])
        self.permutations[lane] = self.rng.sample(range(n), n)
        self.new[lane] = True

    def next(self, device):
        records = [self.episodes[e][p] for e, p in zip(self.episode, self.position)]
        observations = [permute_observation_components(query_observation(row), torch.tensor(permutation))
                        for row, permutation in zip(records, self.permutations)]
        observation = {k: v.to(device) for k, v in batch_observations(observations).items()}
        boundary = torch.tensor(self.new, dtype=torch.bool, device=device)
        targets = torch.tensor([row['rust_after'] for row in records], device=device)
        returned = torch.cat([feedback(row, device) for row in records])
        self.new = [False] * len(records)
        for i in range(len(records)):
            self.position[i] += 1
            if self.position[i] == len(self.episodes[self.episode[i]]):
                self.ended += 1
                self._next(i)
        self.returned += len(records)
        return observation, boundary, targets, returned


def chemical_window(model, state, streams, count, statistics):
    losses, nlls = [], []
    for _ in range(count):
        obs, boundary, targets, returned = streams.next(state['h'].device)
        # A new apparatus has a new ledger and adaptation buffer. In the
        # continuing condition its h/z still follow the preceding computation.
        revised = model.reset_rows(state, boundary)
        state = {**revised, 'h': state['h'], 'z': state['z']}
        state = model.receive(state, obs)
        out = model.readout(state, obs['candidates'], obs['action_mask'])
        nll = F.nll_loss(out['outcomes'][:, 0], targets)
        losses.append(nll + scaled_feedback_loss(out['predictions'][:, 0], returned, statistics))
        nlls.append(nll)
        action = torch.zeros(len(targets), dtype=torch.long, device=targets.device)
        state = model.begin_action(state, out['features'], action)
        state = model.feedback(state, returned)
    return torch.stack(losses).mean(), state, float(torch.stack(nlls).mean().detach())


def main():
    parser = training_parser()
    parser.description = __doc__
    parser.add_argument('--chemistry-checkpoint', type=Path, required=True)
    parser.add_argument('--chemical-data', type=Path, required=True)
    parser.add_argument('--chemical-steps', type=int, default=4)
    parser.add_argument('--chemical-lr', type=float, default=1e-4)
    parser.add_argument('--chemical-weight', type=float, default=1.)
    parser.add_argument('--state-sharing', choices=('continuing', 'separated'), default='continuing')
    parser.add_argument('--chemical-mode', choices=('recorded', 'active'), default='recorded')
    parser.add_argument('--chemical-activity', choices=('learned', 'fixed'), default='learned')
    parser.add_argument('--shared-initial', type=Path)
    parser.add_argument('--response-replay-checkpoint', type=Path)
    parser.add_argument('--response-replay-control', action='store_true',
                        help='Keep the fitted temperature/bias but omit within-task response updates')
    parser.add_argument('--experimental-memory', choices=('learned', 'disabled', 'evidence', 'recent'))
    parser.add_argument('--chemical-schedule', choices=('paired', 'mixed'), default='paired')
    parser.set_defaults(batch=16, external_per_stream=16, inherit_parent_interface=True,
                        condition_failure_weights=True, cache_observations=True)
    args = parser.parse_args()
    if args.external_per_stream % 2 or args.external_per_stream < 2:
        parser.error('Split an even number of physical requests around each chemical window')
    if args.execution_geometry is None or args.environment_truth is None:
        parser.error('Shared training requires actual native physical execution')
    args.output.mkdir(parents=True, exist_ok=False)
    physical, _, plan, environment, delayed = initialize_experiment(args)
    if args.shared_initial is not None:
        model, inherited = load_shared_checkpoint(args.shared_initial, args.device)
        physical, chemical = model.tasks['physics'], model.tasks['chemistry']
        chemical_saved = {'training': {'feedback_statistics': inherited['training']['chemical_feedback_statistics']}}
    else:
        chemical, chemical_saved = load_reaction_checkpoint(args.chemistry_checkpoint, args.device)
    if args.chemical_mode == 'active' and args.shared_initial is None:
        from experiments.continual_chemistry.active_training import enable_response_learning
        chemical = enable_response_learning(chemical, args.chemical_activity)
    if args.response_replay_checkpoint is not None:
        from cdm.response_replay import ResponseReplay, ResponseReplayConfig
        update = torch.load(args.response_replay_checkpoint, map_location=args.device, weights_only=True)
        config = dict(update['config'])
        if args.response_replay_control:
            config['mode'] = 'calibration'
        chemical.response_replay = ResponseReplay(config=ResponseReplayConfig(**config)).to(args.device)
        chemical.response_replay.load_state_dict(update['model'])
    if args.experimental_memory is not None:
        from cdm.experimental_memory import ExperimentalMemory, ExperimentalMemoryConfig
        chemical.experimental_memory = ExperimentalMemory(ExperimentalMemoryConfig(mode=args.experimental_memory)).to(args.device)
    model = SharedCorePolicies({'physics': physical, 'chemistry': chemical}).to(args.device).train()
    reference = deepcopy(physical.base).eval().requires_grad_(False)
    reference.core.gru.flatten_parameters()
    if args.chemical_mode == 'active':
        from experiments.continual_chemistry.active_environment import ActiveChemicalEnvironment
        from experiments.continual_chemistry.active_training import active_chemical_window
        streams = ActiveChemicalEnvironment(args.chemical_data, args.batch, args.seed, args.device,
                                             schedule=args.chemical_schedule)
    else:
        streams = ChemicalStreams(args.chemical_data, args.batch, args.seed)
    statistics = chemical_saved['training']['feedback_statistics']
    # Keep the effective rates of each inherited task. A shared parameter is
    # listed exactly once; chemistry cannot create another optimizer for it.
    groups, seen = [], set()
    for module, prefix, rate in [(model.core, '', args.core_lr),
                                  (physical, 'base.', args.head_lr),
                                  (physical, 'belief.', args.belief_lr),
                                  (physical, '', args.new_lr),
                                  (chemical, '', args.chemical_lr)]:
        parameters = [p for name, p in module.named_parameters()
                      if name.startswith(prefix) and p.requires_grad and id(p) not in seen]
        seen.update(id(p) for p in parameters)
        if parameters:
            groups.append({'params': parameters, 'lr': rate})
    optimizer = torch.optim.AdamW(groups, weight_decay=.001)
    parameters = [p for group in groups for p in group['params']]
    metadata = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    metadata.update(
        method='shared-core physical decisions and chemical conditional-response learning',
        selection='fixed final update; no evaluation checkpoint selection',
        sequence='physical requests / chemical returns / physical requests within one gradient window',
        physics_schedule=plan.description(),
        physical_training='native action reward, actor-critic, physical prediction and inherited-policy anchor',
        chemical_training='proper observed-response likelihood and native cost, recorded training actions only',
        chemical_feedback_statistics=statistics,
        chemical_training_episodes=[r['directory'] for r in streams.entries],
        chemical_activity='one observation update; no learned chemical activity scheduler in this experiment',
        timing='chemical replay pauses the simulated physical scheduler; no asynchronous-world timing claim',
        recurrence='continuing across physical and chemical tasks' if args.state_sharing == 'continuing'
                   else 'one persistent h/z per interface, with one shared trainable core',
        truncation='detach only between optimizer windows; backpropagate across both domain switches',
        inherited_interface='original physical candidates/masks; zero-initialized physical action residual; pending WAIT')
    if args.chemical_mode == 'active':
        metadata.update(chemical_training='selected chemical experiments, received-response likelihood and native-cost actor-critic',
            chemical_activity=args.chemical_activity,
            chemical_environment='training-split recipes, exact cosine/persistent-rust transition and native operation cost',
            chemical_sequence='A A B B A A; current evidence/fast weights reset, h/z and slow response weights continue',
            chemical_update='censored categorical gradient; subsequent received predictions train through chosen updates',
            chemical_counterfactuals='No response labels for unexecuted alternatives',
            timing='physical scheduler pauses during chemical segments; no independent-clock claim')
    if args.response_replay_checkpoint is not None:
        metadata.update(chemical_update='four current-history functional response updates, separate from cost LEARN',
            chemical_update_control='same fitted calibration without response fitting' if args.response_replay_control else None,
            chemical_update_initial=str(args.response_replay_checkpoint),
            chemical_update_schedule='fixed before each response readout; no learned response-compute allocation')
    if args.experimental_memory is not None:
        metadata.update(chemical_memory=chemical.experimental_memory.config.__dict__,
            chemical_sequence=args.chemical_schedule,
            chemical_memory_evidence='prequential log likelihood of returned observations; no environment task identities',
            chemical_memory_write='archive actual returned ledgers at task boundaries; no learned write selection',
            chemical_memory_read='fresh-task expert competes with archived predictors; future response and action losses')
    (args.output / 'config.json').write_text(json.dumps(metadata, indent=2) + '\n')
    save_shared_checkpoint(args.output / 'initial.pt', model, training={**metadata, 'updates': 0})
    physical_state, chemical_state = physical.initial(args.batch, args.device), chemical.initial(args.batch, args.device)
    obs, _ = observe(physical, environment, physical_state, delayed=delayed)
    physical_state = physical.receive(physical_state, obs)
    reference_state = reference_receive(reference, reference.initial(args.batch, args.device), obs,
                                        torch.ones(args.batch, device=args.device, dtype=torch.bool))
    started = perf_counter()
    with (args.output / 'training.jsonl').open('w') as log, (args.output / 'tasks.jsonl').open('w') as tasks:
        for update in range(1, args.updates + 1):
            physical_state, chemical_state = detach_state(physical_state), detach_state(chemical_state)
            progress = min(1., (update - 1) / max(1., args.updates * args.activity_entropy_anneal_fraction))
            entropy = args.activity_entropy_initial + progress * (args.activity_entropy_final - args.activity_entropy_initial)
            options = dict(external_per_stream=args.external_per_stream // 2, mode=args.mode,
                           compute_cost=args.compute_cost, latency_cost=args.latency_cost,
                           physical_weight=args.physical_weight, activity_entropy_weight=entropy,
                           action_entropy_weight=args.action_entropy_weight, reference=reference,
                           anchor_weight=args.anchor_weight, cache_observations=True)
            optimizer.zero_grad(set_to_none=True)
            first, physical_state, reference_state, before = train_window(
                physical, delayed, physical_state, reference_state=reference_state, **options)
            if args.state_sharing == 'continuing':
                chemical_state = {**chemical_state, 'h': physical_state['h'], 'z': physical_state['z']}
            chemical_metrics = {}
            if args.chemical_mode == 'active':
                chemical_loss, chemical_state, chemical_metrics = active_chemical_window(
                    chemical, chemical_state, streams, args.chemical_steps, statistics)
                nll = chemical_metrics['response_nll']
            else:
                chemical_loss, chemical_state, nll = chemical_window(
                    chemical, chemical_state, streams, args.chemical_steps, statistics)
            if args.state_sharing == 'continuing':
                physical_state = {**physical_state, 'h': chemical_state['h'], 'z': chemical_state['z']}
            second, physical_state, reference_state, after = train_window(
                physical, delayed, physical_state, reference_state=reference_state, **options)
            loss = (first + second) / 2 + args.chemical_weight * chemical_loss
            loss.backward()
            gradient = torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
            optimizer.step()
            row = {'update': update, 'loss': float(loss.detach()),
                   'physical_loss': float(((first + second) / 2).detach()),
                   'chemical_loss': float(chemical_loss.detach()), 'chemical_nll': nll,
                   'chemical_activity_learning': chemical_metrics,
                   'gradient_norm': float(gradient), 'physical_before': before, 'physical_after': after,
                   'physical_requests': int(delayed.dispatched.sum()), 'chemical_returns': streams.returned,
                   'chemical_episodes_ended': streams.ended,
                   'native_actions': delayed.total_native_actions + environment.total_bootstrap_actions,
                   'seconds': perf_counter() - started}
            if torch.device(args.device).type == 'cuda':
                row['peak_allocated_gib'] = torch.cuda.max_memory_allocated(args.device) / 2**30
            log.write(json.dumps(row) + '\n'); log.flush()
            for record in environment.task_rows:
                tasks.write(json.dumps(record) + '\n')
            tasks.flush(); environment.task_rows.clear()
            if args.chemical_mode == 'active':
                with (args.output / 'chemical_tasks.jsonl').open('a') as chemical_tasks:
                    for record in streams.task_rows:
                        chemical_tasks.write(json.dumps(record) + '\n')
                streams.task_rows.clear()
            if update == 1 or update % 10 == 0:
                print(json.dumps({k: v for k, v in row.items() if not isinstance(v, dict)}), flush=True)
            if update % args.save_every == 0 or update == args.updates:
                save_shared_checkpoint(args.output / 'last.pt', model, training={**metadata, 'updates': update})
    (args.output / 'finished.json').write_text(json.dumps({
        'updates': args.updates, 'physical_requests': int(delayed.dispatched.sum()),
        'chemical_returns': streams.returned, 'seconds': perf_counter() - started}, indent=2) + '\n')


if __name__ == '__main__':
    main()
