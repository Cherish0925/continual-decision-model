"""Batched recurrent context, iterative workspace, and computational memory.

The environment supplies observations and elapsed time. Each ``refine`` call is
one shared Transformer computation; the caller advances the environment between
calls when studying computation latency. All persistent model state is explicit.
"""

from dataclasses import dataclass, fields, replace
from typing import NamedTuple

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class CDMConfig:
    observation_dim: int
    action_dim: int
    goal_dim: int
    environment_dim: int
    hidden_dim: int = 128
    slots: int = 4
    memory_size: int = 8
    heads: int = 4
    expansion: int = 2
    dropout: float = 0.0

    def __post_init__(self):
        dimensions = (
            self.observation_dim, self.action_dim, self.goal_dim,
            self.environment_dim, self.hidden_dim, self.slots,
            self.memory_size, self.heads, self.expansion,
        )
        if min(dimensions) < 1 or self.hidden_dim % self.heads:
            raise ValueError("Dimensions must be positive and hidden_dim divisible by heads.")


@dataclass(frozen=True)
class CDMState:
    context: Tensor
    event_index: Tensor
    slots: Tensor
    goal: Tensor
    environment: Tensor
    workspace_context: Tensor
    workspace_goal: Tensor
    workspace_environment: Tensor
    workspace_version: Tensor
    workspace_event_index: Tensor
    depth: Tensor
    memory: Tensor
    memory_context: Tensor
    memory_goal: Tensor
    memory_environment: Tensor
    memory_version: Tensor
    memory_event_index: Tensor
    memory_depth: Tensor
    memory_valid: Tensor
    memory_cursor: Tensor
    model_version: Tensor

    def detach(self):
        """End a training segment while retaining its numerical state."""
        return replace(self, **{f.name: getattr(self, f.name).detach() for f in fields(self)})

    def to(self, device):
        return replace(self, **{f.name: getattr(self, f.name).to(device) for f in fields(self)})


class CDMOutput(NamedTuple):
    action_logits: Tensor
    activity_logits: Tensor
    value: Tensor


class MemoryRead(NamedTuple):
    # Candidate zero is the current workspace; candidates 1..M are stored records.
    weights: Tensor
    gates: Tensor  # [batch, slots, 3]: reuse, revise, initialize
    valid: Tensor


class ContinualDecisionModel(nn.Module):
    """Small neural CDM suitable for CPU experiments and GPU batch training.

    Activity indices are OBSERVE=0, COMPUTE=1, ACT=2. The activity head supplies
    logits; an external environment loop executes the selected operation.
    Memory factors are learned representations of the supplied goal and
    environment descriptors. They are not assumed to be causal dependencies.
    """

    OBSERVE, COMPUTE, ACT = 0, 1, 2

    def __init__(self, config: CDMConfig):
        super().__init__()
        self.config = config
        c = config
        d = c.hidden_dim
        input_dim = c.observation_dim + c.goal_dim + c.environment_dim + c.action_dim + 1
        self.input_encoder = nn.Sequential(nn.Linear(input_dim, d), nn.LayerNorm(d), nn.SiLU())
        self.context_cell = nn.GRUCell(d, d)
        self.initial_slots = nn.Parameter(torch.randn(c.slots, d) * 0.02)
        self.context_to_slots = nn.Linear(d, d)
        self.refinement = nn.TransformerEncoderLayer(
            d_model=d, nhead=c.heads, dim_feedforward=c.expansion * d,
            dropout=c.dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.slot_norm = nn.LayerNorm(d)
        self.goal_factor = nn.Linear(c.goal_dim, d, bias=False)
        self.environment_factor = nn.Linear(c.environment_dim, d, bias=False)
        self.context_query = nn.Linear(d, d, bias=False)
        self.context_key = nn.Linear(d, d, bias=False)
        self.depth_score = nn.Linear(1, 1, bias=False)
        self.revision = nn.Sequential(nn.Linear(4 * d, d), nn.SiLU(), nn.Linear(d, d))
        self.reuse_gate = nn.Linear(2 * d, 3)
        self.readout_body = nn.Sequential(nn.Linear(2 * d, d), nn.SiLU(), nn.LayerNorm(d))
        self.action_head = nn.Linear(d, c.action_dim)
        self.activity_head = nn.Linear(d, 3)
        self.value_head = nn.Linear(d, 1)

    def initialize(self, batch_size: int, device=None, dtype=None, model_version: int = 0):
        c = self.config
        parameter = self.initial_slots
        device = parameter.device if device is None else device
        dtype = parameter.dtype if dtype is None else dtype
        zeros = lambda *shape: torch.zeros(*shape, device=device, dtype=dtype)
        context = zeros(batch_size, c.hidden_dim)
        goal = zeros(batch_size, c.goal_dim)
        environment = zeros(batch_size, c.environment_dim)
        version = torch.full((batch_size,), model_version, device=device, dtype=torch.long)
        return CDMState(
            context=context,
            event_index=torch.zeros(batch_size, device=device, dtype=torch.long),
            slots=self.initial_slots.to(device=device, dtype=dtype).unsqueeze(0).expand(batch_size, -1, -1),
            goal=goal, environment=environment,
            workspace_context=context, workspace_goal=goal, workspace_environment=environment,
            workspace_version=version.clone(),
            workspace_event_index=torch.zeros(batch_size, device=device, dtype=torch.long),
            depth=zeros(batch_size),
            memory=zeros(batch_size, c.memory_size, c.slots, c.hidden_dim),
            memory_context=zeros(batch_size, c.memory_size, c.hidden_dim),
            memory_goal=zeros(batch_size, c.memory_size, c.goal_dim),
            memory_environment=zeros(batch_size, c.memory_size, c.environment_dim),
            memory_version=torch.full((batch_size, c.memory_size), -1, device=device, dtype=torch.long),
            memory_event_index=torch.zeros(batch_size, c.memory_size, device=device, dtype=torch.long),
            memory_depth=zeros(batch_size, c.memory_size),
            memory_valid=torch.zeros(batch_size, c.memory_size, device=device, dtype=torch.bool),
            memory_cursor=torch.zeros(batch_size, device=device, dtype=torch.long),
            model_version=version,
        )

    def assimilate(self, state: CDMState, observations: Tensor, goals: Tensor,
                   environments: Tensor, previous_action: Tensor | None = None,
                   elapsed: Tensor | None = None):
        """Update context without waiting for the persistent workspace to refine.

        ``previous_action`` is a float action vector (for example, one-hot).
        ``elapsed`` has shape [batch, 1] and uses the environment's time unit.
        """
        b = observations.shape[0]
        if previous_action is None:
            previous_action = observations.new_zeros(b, self.config.action_dim)
        if elapsed is None:
            elapsed = observations.new_ones(b, 1)
        inputs = torch.cat((observations, goals, environments, previous_action, elapsed), dim=-1)
        context = self.context_cell(self.input_encoder(inputs), state.context)
        return replace(state, context=context, event_index=state.event_index + 1,
                       goal=goals, environment=environments)

    def reset_workspace(self, state: CDMState):
        """Initialize a workspace from current context; retain context and memory."""
        slots = self.initial_slots.unsqueeze(0) + self.context_to_slots(state.context).unsqueeze(1)
        return replace(
            state, slots=slots, workspace_context=state.context,
            workspace_goal=state.goal, workspace_environment=state.environment,
            workspace_version=state.model_version.clone(), workspace_event_index=state.event_index,
            depth=torch.zeros_like(state.depth),
        )

    def set_model_version(self, state: CDMState, version: int):
        """Mark a parameter update; incompatible records are excluded at reading."""
        return replace(state, model_version=torch.full_like(state.model_version, version))

    def read_memory(self, state: CDMState):
        """Read current and stored computation under the latest goal and context.

        Each record receives a learned sum of goal, environment, context, and
        depth scores. A slotwise gate chooses reuse, contextual revision, or
        initialization. Reading changes computation state, not observation data.
        Depth after mixing is a weighted history statistic, not an exact count
        of the operations required to reproduce a mixed state.
        Record selection weights are shared across slots; the reuse/revise/
        initialize gates vary by slot. Event indices identify the observation
        prefix available when a state was generated, not its causal dependencies.
        """
        records = torch.cat((state.slots.unsqueeze(1), state.memory), dim=1)
        contexts = torch.cat((state.workspace_context.unsqueeze(1), state.memory_context), dim=1)
        goals = torch.cat((state.workspace_goal.unsqueeze(1), state.memory_goal), dim=1)
        environments = torch.cat((state.workspace_environment.unsqueeze(1), state.memory_environment), dim=1)
        depths = torch.cat((state.depth.unsqueeze(1), state.memory_depth), dim=1)
        versions = torch.cat((state.workspace_version.unsqueeze(1), state.memory_version), dim=1)
        valid = torch.cat((torch.ones_like(state.depth, dtype=torch.bool).unsqueeze(1), state.memory_valid), dim=1)
        valid = valid & (versions == state.model_version.unsqueeze(1))
        q_goal = self.goal_factor(state.goal)
        k_goal = self.goal_factor(goals)
        q_env = self.environment_factor(state.environment)
        k_env = self.environment_factor(environments)
        q_context = self.context_query(state.context)
        k_context = self.context_key(contexts)
        scores = ((q_goal.unsqueeze(1) * k_goal).sum(-1)
                  + (q_env.unsqueeze(1) * k_env).sum(-1)
                  + (q_context.unsqueeze(1) * k_context).sum(-1))
        scores = scores / (3 * self.config.hidden_dim) ** 0.5
        scores = scores + self.depth_score(torch.log1p(depths).unsqueeze(-1)).squeeze(-1)
        weights = torch.softmax(scores.masked_fill(~valid, -1e4), dim=1) * valid
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-8)
        retrieved = torch.einsum("bm,bmsd->bsd", weights, records)
        old_context = torch.einsum("bm,bmd->bd", weights, contexts)
        old_goal = torch.einsum("bm,bmd->bd", weights, k_goal)
        old_environment = torch.einsum("bm,bmd->bd", weights, k_env)
        s = self.config.slots
        expand = lambda x: x.unsqueeze(1).expand(-1, s, -1)
        revision_inputs = torch.cat((retrieved, expand(state.context - old_context),
                                    expand(q_goal - old_goal), expand(q_env - old_environment)), dim=-1)
        revised = retrieved + self.revision(revision_inputs)
        initial = self.initial_slots.unsqueeze(0) + expand(self.context_to_slots(state.context))
        gates = torch.softmax(self.reuse_gate(torch.cat((retrieved, expand(state.context)), dim=-1)), dim=-1)
        has_record = valid.any(dim=1)
        initial_only = torch.zeros_like(gates)
        initial_only[..., 2] = 1
        gates = torch.where(has_record[:, None, None], gates, initial_only)
        slots = gates[..., :1] * retrieved + gates[..., 1:2] * revised + gates[..., 2:] * initial
        depth = (weights * depths).sum(-1) * (1 - gates[..., 2].mean(-1))
        updated = replace(
            state, slots=slots, workspace_context=state.context,
            workspace_goal=state.goal, workspace_environment=state.environment,
            workspace_version=state.model_version.clone(), workspace_event_index=state.event_index,
            depth=depth,
        )
        return updated, MemoryRead(weights, gates, valid)

    def refine(self, state: CDMState, steps: int = 1):
        """Apply the same Transformer block repeatedly, preserving its workspace."""
        if steps < 0:
            raise ValueError("steps must be nonnegative")
        if steps == 0:
            return state
        slots = state.slots
        conditioning = self.context_to_slots(state.context).unsqueeze(1)
        for _ in range(steps):
            slots = self.refinement(slots + conditioning)
        return replace(
            state, slots=slots, depth=state.depth + steps,
            workspace_context=state.context, workspace_goal=state.goal,
            workspace_environment=state.environment,
            workspace_version=state.model_version.clone(),
            workspace_event_index=state.event_index,
        )

    def readout(self, state: CDMState):
        """Produce actions directly from fresh context and the current workspace."""
        pooled = self.slot_norm(state.slots).mean(dim=1)
        features = self.readout_body(torch.cat((state.context, pooled), dim=-1))
        return CDMOutput(self.action_head(features), self.activity_head(features), self.value_head(features).squeeze(-1))

    def write_memory(self, state: CDMState, mask: Tensor | None = None):
        """Append a record to each selected batch item's fixed-size ring.

        ``mask`` is an optional boolean tensor of shape [batch]. It lets an
        asynchronous environment write records at different times per stream.
        """
        def insert(memory, value):
            index = state.memory_cursor.view(-1, *([1] * (memory.ndim - 1)))
            index = index.expand(-1, 1, *memory.shape[2:])
            updated = memory.scatter(1, index, value.unsqueeze(1))
            if mask is not None:
                active = mask.view(-1, *([1] * (memory.ndim - 1)))
                updated = torch.where(active, updated, memory)
            return updated

        return replace(
            state, memory=insert(state.memory, state.slots),
            memory_context=insert(state.memory_context, state.workspace_context),
            memory_goal=insert(state.memory_goal, state.workspace_goal),
            memory_environment=insert(state.memory_environment, state.workspace_environment),
            memory_version=insert(state.memory_version, state.workspace_version),
            memory_event_index=insert(state.memory_event_index, state.workspace_event_index),
            memory_depth=insert(state.memory_depth, state.depth),
            memory_valid=insert(state.memory_valid, torch.ones_like(state.depth, dtype=torch.bool)),
            memory_cursor=(state.memory_cursor + (1 if mask is None else mask.long())) % self.config.memory_size,
        )

    def forward(self, observations: Tensor, goals: Tensor, environments: Tensor,
                state: CDMState | None = None, steps: int = 1, retrieve: bool = True,
                write: bool = True, previous_action: Tensor | None = None,
                elapsed: Tensor | None = None):
        """Convenience step; explicit methods permit interleaved environment events."""
        if state is None:
            state = self.initialize(observations.shape[0], observations.device, observations.dtype)
        state = self.assimilate(state, observations, goals, environments, previous_action, elapsed)
        if retrieve:
            state, _ = self.read_memory(state)
        state = self.refine(state, steps)
        output = self.readout(state)
        if write:
            state = self.write_memory(state)
        return output, state
