"""Public DiscoveryWorld inputs and a task head on the existing continual core.

Object UUIDs are routing handles, never continuous model features. Candidate
generation uses documented argument domains and the received observation only;
it does not know which accessible objects support a successful interaction.
Frozen image/text features enter through trainable projections. The original
GRU, workspace, state normalization and checkpoint tensors remain intact.
"""
from __future__ import annotations

from copy import deepcopy
from collections import deque
from dataclasses import dataclass, replace
import math
from os import PathLike
import re

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from cdm.core import ContinualCore, CoreConfig
from cdm.multimodal import EncodedEvent, MultimodalCore

NUMERIC_DIM = 32
CANDIDATE_NUMERIC_DIM = 8
_DIRECTIONS = ("north", "east", "south", "west")
_UUID = re.compile(r"\[?\buuid\s*[:=]?\s*['\"]?\d+['\"]?\]?", re.IGNORECASE)
_NUMBER = re.compile(r"(?<![\w.])[-+]?(?:\d{1,3}(?:,\d{3})+(?:\.\d*)?|\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?")


def _text(value) -> str:
    return _UUID.sub("", str(value)).strip()


def _signed_log(value: float) -> float:
    return math.copysign(math.log1p(abs(value)), value)


def observed_numbers(text: str) -> list[float]:
    """Received numerical values, excluding explicit UUID annotations.

    Exact strings/units remain in observation text and the trajectory. These
    numbers supply a complementary numeric representation, not a replacement
    for those records or a claim that every number is a scientific measurement.
    """
    return [value for match in _NUMBER.finditer(_text(text))
            if math.isfinite(value := float(match.group().replace(",", "")))]


def _objects(observation: dict) -> dict:
    ui = observation["ui"]
    result = {obj["uuid"]: {**obj, "inventory": False, "accessible": False,
                            "direction": "previously observed"}
              for obj in observation.get("remembered_objects", [])}
    nearby = ui.get("nearbyObjects", {}).get("objects", {})
    groups = nearby.items() if isinstance(nearby, dict) else [("nearby", nearby)]
    for direction, objects in groups:
        for obj in objects:
            if "uuid" in obj:
                result[obj["uuid"]] = {**obj, "direction": direction, "inventory": False,
                                       "accessible": False}
    for key, inventory in (("accessibleEnvironmentObjects", False), ("inventoryObjects", True)):
        for obj in ui.get(key, []):
            result[obj["uuid"]] = {**result.get(obj["uuid"], {}), **obj,
                                   "inventory": inventory, "accessible": True}
    return result


class PublicObjectMemory:
    """Episode-owned registry of received object handles, shared by all policies.

    Past visibility permits later navigation to an object; it does not imply
    current access or knowledge of unobserved changes. Reset by creating a new
    registry at each episode boundary. Raw observations remain untouched.
    """

    def __init__(self):
        self.objects = {}

    def observe(self, observation: dict) -> dict:
        current = _objects({key: value for key, value in observation.items() if key != "remembered_objects"})
        remembered = list(self.objects.values())
        self.objects.update(current)
        return {**observation, "remembered_objects": remembered}


class PublicActionHistory:
    """Bind executed actions to their received feedback, without expert labels.

    Names, dialogue text and locations come from the observation available when
    the action was selected. Object handles are only used to resolve those names.
    A record enters the next decision, never the decision that produced it.
    """

    def __init__(self, capacity: int = 0):
        if capacity < 0:
            raise ValueError("Action history capacity must be nonnegative")
        self.rows = deque(maxlen=capacity)

    def append_transition(self, transition: dict, observation: dict | None = None):
        if self.rows.maxlen == 0:
            return
        before = observation if observation is not None else transition["observation"]
        action = transition["action"]
        objects = _objects(before)
        if "chosen_dialog_option_int" in action:
            options = before["ui"].get("dialog_box", {}).get("dialogOptions", {})
            option = action["chosen_dialog_option_int"]
            label = options.get(str(option), options.get(option, "unknown reply")) if isinstance(options, dict) else options[option - 1]
            description = "DIALOG: " + _text(label)
        else:
            parts = [str(action.get("action", "unknown action"))]
            for key in ("arg1", "arg2"):
                if key not in action:
                    continue
                value = action[key]
                if value in objects:
                    obj = objects[value]
                    parts.append(f"{key} {_text(obj.get('name', 'object'))}: {_text(obj.get('description', ''))}")
                elif isinstance(value, str):
                    parts.append(f"{key} {_text(value)}")
                else:
                    parts.append(f"{key} previously observed object")
            description = "; ".join(parts)
        result = transition["action_result"]
        after = transition["next_observation"]["ui"]
        location = lambda ui: str({key: value for key, value in ui.get("agentLocation", {}).items()
                                   if key in ("x", "y", "faceDirection")})
        feedback = " ".join(_text(value) for value in (
            result.get("message", ""), after.get("lastActionMessage", ""), after.get("extended_action_message", "")) if value)
        self.rows.append(f"Executed {description}. Accepted {bool(result.get('success', False))}. "
                         f"From {location(before['ui'])} to {location(after)}. Received {feedback}")

    def augment(self, observation: dict) -> dict:
        return {**observation, "action_history": [f"{len(self.rows)-index} actions ago: {row}"
                for index, row in enumerate(self.rows)]}


def _object_text(obj: dict) -> str:
    place = "inventory" if obj.get("inventory") else (
        "accessible" if obj.get("accessible") else "nearby")
    direction = obj.get("direction", "")
    return _text(f"{obj.get('name', 'object')}: {obj.get('description', '')}; {place} {direction}")


def observation_text(observation: dict) -> str:
    """Task, object descriptions, dialogue and received messages without UUIDs.

    Object lists are sorted by content, since the simulator sorts some lists
    by UUID. Scorecard/training fields and arbitrary extra keys are not read.
    """
    ui = observation["ui"]
    parts = []
    for task in ui.get("taskProgress", []):
        parts.append(_text(task.get("taskName", "") + ": " + task.get("description", task.get("taskDescription", ""))))
    location = ui.get("agentLocation", {})
    parts.append("Facing " + str(location.get("faceDirection", "unknown")))
    parts.extend(sorted(_object_text(obj) for obj in _objects(observation).values()))
    dialog = ui.get("dialog_box", {})
    parts.append("Dialog: " + _text(dialog.get("dialogIn", "")))
    options = dialog.get("dialogOptions", {})
    parts.extend("Reply: " + _text(option) for option in (options.values() if isinstance(options, dict) else options))
    parts.append("Last result: " + _text(ui.get("lastActionMessage", "")))
    parts.append("Received message: " + _text(ui.get("extended_action_message", "")))
    if 'execution' in observation:
        parts.append('Remaining native operations: ' + str(observation['execution']['remaining_native_actions']))
        for event in observation.get('execution_events', []):
            action = event['action']
            objects = _objects(observation)
            names = [_object_text(objects[v]) if v in objects else str(v)
                     for k,v in action.items() if k in ('arg1','arg2')]
            parts.append('Executed ' + str(action.get('action','dialogue')) + ' ' + ' -> '.join(names)
                         + ': ' + _text(event.get('feedback',{}).get('lastActionMessage','')))
    for post in ui.get("discoveryFeed", {}).get("posts", []):
        parts.append(_text(post.get("content", "")))
    return "\n".join(parts)


def observation_numeric(observation: dict) -> list[float]:
    """32 values: location/time/counts, facing, and five received numbers.

    Each received number has signed log-magnitude, base-10 mantissa, signed
    log-exponent, and a presence bit. Coordinates and counts use signed log.
    Reading selection is deterministic; all original readings remain in text
    and raw trajectories, including those beyond the five numeric slots.
    """
    ui = observation["ui"]
    location, dialog = ui.get("agentLocation", {}), ui.get("dialog_box", {})
    objects = _objects(observation)
    values = [float(location.get("x", 0)), float(location.get("y", 0)),
              float(ui.get("world_steps", 0)), float(len(ui.get("inventoryObjects", []))),
              float(len(ui.get("accessibleEnvironmentObjects", []))), float(len(objects)),
              float(len(dialog.get("dialogOptions", {})))]
    result = [_signed_log(v) for v in values] + [float(dialog.get("is_in_dialog", False))]
    result += [float(location.get("faceDirection") == direction) for direction in _DIRECTIONS]
    messages = "\n".join(str(ui.get(key, "")) for key in ("lastActionMessage", "extended_action_message"))
    messages += "\n" + str(dialog.get("dialogIn", ""))
    numbers = observed_numbers(messages)[:5]
    for value in numbers:
        exponent = math.floor(math.log10(abs(value))) if value else 0
        mantissa = math.copysign(10.0 ** (math.log10(abs(value)) - exponent), value) if value else 0.0
        result += [_signed_log(value), mantissa / 10, _signed_log(exponent), 1.0]
    result += [0.0] * (NUMERIC_DIM - len(result))
    return result


@dataclass(frozen=True)
class NativeCandidate:
    action: dict
    text: str
    numeric: tuple[float, ...]
    object_arguments: tuple[int | str | None, int | str | None] = (None, None)


def public_candidates(observation: dict) -> list[NativeCandidate]:
    """Enumerate native actions from public argument domains, without a teacher.

    Accessible objects do not expose usability/container attributes. We retain
    uncertain interactions for the model to learn from their actual outcomes.
    No object type/name or scenario-specific whitelist determines valid actions.
    """
    ui, objects = observation["ui"], _objects(observation)
    candidates = []

    def add(action: dict, text: str, first=None, second=None):
        a, b = objects.get(first, {}), objects.get(second, {})
        numbers = observed_numbers(text)
        numeric = (float(a.get("inventory", False)), float(a.get("accessible", False)),
                   float(b.get("inventory", False)), float(b.get("accessible", False)),
                   _signed_log(float(a.get("distance", 0))), _signed_log(float(b.get("distance", 0))),
                   _signed_log(numbers[0]) if numbers else 0.0, float(bool(numbers)))
        candidates.append(NativeCandidate(action, _text(text), numeric, (first, second)))

    dialog = ui.get("dialog_box", {})
    # The native UI retains an exiting-dialog message for one observation after
    # the dialog ends. Empty options already permit normal native actions.
    if dialog.get("is_in_dialog", False) and dialog.get("dialogOptions"):
        options = dialog.get("dialogOptions", {})
        pairs = options.items() if isinstance(options, dict) else enumerate(options, 1)
        for index, text in pairs:
            add({"chosen_dialog_option_int": int(index)}, "DIALOG: " + _text(text))
    else:
        accessible = [key for key, obj in objects.items() if obj["accessible"]]
        inventory = [key for key in accessible if objects[key]["inventory"]]
        environment = [key for key in accessible if not objects[key]["inventory"]]
        for operation, schema in sorted(observation.get("actions", {}).items()):
            args = schema.get("args", [])
            if not args:
                add({"action": operation}, operation + ": " + _text(schema.get("desc", "")))
            elif operation in ("MOVE_DIRECTION", "ROTATE_DIRECTION"):
                directions = ui.get("agentLocation", {}).get("directions_you_can_move", _DIRECTIONS) \
                    if operation == "MOVE_DIRECTION" else _DIRECTIONS
                for direction in directions:
                    add({"action": operation, "arg1": direction}, operation + ": " + direction)
            elif operation == "TELEPORT_TO_LOCATION":
                for location in observation.get("teleport_locations", {}):
                    add({"action": operation, "arg1": location}, operation + ": " + str(location))
            elif operation == "DISCOVERY_FEED_GET_POST_BY_ID":
                feed = ui.get("discoveryFeed", {})
                for posts in feed.values():
                    if isinstance(posts, list):
                        for post in posts:
                            if isinstance(post, dict) and "postID" in post:
                                add({"action": operation, "arg1": post["postID"]},
                                    operation + ": " + _text(post.get("title", post.get("content", ""))))
            elif len(args) == 1:
                domain = list(objects) if operation == "TELEPORT_TO_OBJECT" else (
                    inventory if operation == "DROP" else environment if operation == "PICKUP" else accessible)
                for first in domain:
                    add({"action": operation, "arg1": first},
                        operation + ": " + _object_text(objects[first]), first)
            elif len(args) == 2:
                for first in (inventory if operation == "PUT" else accessible):
                    for second in accessible:
                        if first != second:
                            add({"action": operation, "arg1": first, "arg2": second},
                                operation + ": " + _object_text(objects[first]) + " -> " + _object_text(objects[second]),
                                first, second)
    # Semantic ordering is stable under arbitrary object handle renaming.
    # Handles route acts and establish equality relations, never magnitudes.
    return sorted(candidates, key=lambda candidate: (candidate.text, candidate.numeric))


@dataclass(frozen=True)
class DigitalConfig:
    width: int = 512
    text_dim: int = 768
    image_dim: int = 768
    numeric_dim: int = NUMERIC_DIM
    candidate_numeric_dim: int = CANDIDATE_NUMERIC_DIM
    outcome_dim: int = 4
    refinements: int = 1
    object_relations: bool = False
    candidate_read: bool = False
    action_conditioned: bool = False


class DigitalPolicy(nn.Module):
    """Dynamic native-action head; source process heads are retained unchanged.

    ``outcome[..., :]`` is success logit, progress delta, termination logit,
    numeric-result-presence logit (default four outputs). These are supervised
    only from returned transitions, not candidate-generator hints.
    """

    def __init__(self, config: DigitalConfig, core: ContinualCore | MultimodalCore):
        super().__init__()
        source = core if isinstance(core, MultimodalCore) else None
        self.core = core.core if source is not None else core
        self.config = c = config
        if c.width != self.core.config.width:
            raise ValueError("digital width must match the source core")
        if c.refinements < 0:
            raise ValueError("refinements must be nonnegative")
        w = c.width
        self.text = nn.Linear(c.text_dim, w)
        self.image = nn.Linear(c.image_dim, w)
        self.numeric = nn.Sequential(nn.Linear(c.numeric_dim, w), nn.GELU(), nn.Linear(w, w))
        self.time = nn.Linear(3, w)
        self.modality = nn.Parameter(0.02 * torch.randn(4, w))
        self.token_norm = nn.LayerNorm(w)
        self.ingest = source.ingest if source is not None else nn.MultiheadAttention(w, self.core.config.heads, batch_first=True)
        self.query_norm = source.query_norm if source is not None else nn.LayerNorm(w)
        self.summary_norm = source.summary_norm if source is not None else nn.LayerNorm(w)
        self.input_bridge = source.input_bridge if source is not None else (
            nn.Identity() if self.core.config.event_dim == w else nn.Linear(w, self.core.config.event_dim))
        self.candidate_text = nn.Linear(c.text_dim, w)
        self.candidate_numeric = nn.Linear(c.candidate_numeric_dim, w)
        self.candidate_norm = nn.LayerNorm(w)
        self.state_query = nn.Linear(2 * w, w)
        self.score = nn.Sequential(nn.GELU(), nn.Linear(w, w), nn.GELU(), nn.Linear(w, 1))
        self.outcome = nn.Sequential(nn.Linear(3 * w, w), nn.GELU(), nn.Linear(w, c.outcome_dim))
        if c.object_relations:
            self._init_object_relations()
        if c.candidate_read:
            self._init_candidate_read()
        if c.action_conditioned:
            self._init_action_conditioned()

    def _init_candidate_read(self):
        w = self.config.width
        self.candidate_attention = nn.MultiheadAttention(w, self.core.config.heads, batch_first=True)
        self.candidate_read_output = nn.Linear(w, w, bias=False)
        nn.init.zeros_(self.candidate_read_output.weight)

    def _init_action_conditioned(self):
        w, d = self.config.width, self.config.numeric_dim
        hidden = min(w, 256)
        self.transition_state = nn.Linear(2 * w, hidden)
        self.transition_action = nn.Linear(w, hidden, bias=False)
        self.transition_numeric = nn.Linear(hidden, d)
        self.transition_hidden = nn.Linear(hidden, w, bias=False)
        self.innovation = nn.Linear(d, w, bias=False)
        self.prediction_read = nn.Linear(d + self.config.outcome_dim, w, bias=False)
        # An explicit migration preserves the loaded policy before learning.
        for module in (self.transition_hidden, self.innovation, self.prediction_read):
            nn.init.zeros_(module.weight)

    @classmethod
    def with_action_connections(cls, source: DigitalPolicy, *, candidate_read=None,
                                action_conditioned=None) -> DigitalPolicy:
        model = deepcopy(source)
        groups = {
            "candidate_read": ("candidate_attention", "candidate_read_output"),
            "action_conditioned": ("transition_state", "transition_action", "transition_numeric",
                                   "transition_hidden", "innovation", "prediction_read"),
        }
        for key, enabled in (("candidate_read", candidate_read), ("action_conditioned", action_conditioned)):
            if enabled is None or enabled == getattr(source.config, key):
                continue
            model.config = replace(model.config, **{key: enabled})
            if enabled:
                getattr(model, "_init_" + key)()
            else:
                for name in groups[key]:
                    delattr(model, name)
        model.to(device=source.candidate_text.weight.device, dtype=source.candidate_text.weight.dtype)
        return model.train(source.training)

    _relation_modules = ("measurement_key", "measurement_value", "measurement_relation_bias", "measurement_output")

    def _init_object_relations(self):
        w = self.config.width
        options = {"device": self.candidate_text.weight.device, "dtype": self.candidate_text.weight.dtype}
        self.measurement_key = nn.Linear(w, w, bias=False, **options)
        self.measurement_value = nn.Linear(w, w, bias=False, **options)
        self.measurement_relation_bias = nn.Linear(8, 1, bias=False, **options)
        self.measurement_output = nn.Linear(w, w, bias=False, **options)
        nn.init.zeros_(self.measurement_output.weight)

    @classmethod
    def with_object_relations(cls, source: DigitalPolicy, *, enabled: bool = True) -> DigitalPolicy:
        """Copy a strictly loaded policy and change only this explicit extension.

        Every existing parameter/buffer is copied, including all source-core
        tensors. Enabling zero-initializes the new output projection; copying an
        already enabled policy preserves its learned extension. Disabling drops
        only the four named extension modules, for an explicit ablation. No
        missing/unexpected checkpoint keys are silently accepted.
        """
        model = deepcopy(source)
        if enabled != source.config.object_relations:
            model.config = replace(source.config, object_relations=enabled)
            if enabled:
                model._init_object_relations()
            else:
                for name in cls._relation_modules:
                    delattr(model, name)
            model.train(source.training)
        return model

    def _measurement_read(self, query: Tensor, tokens: Tensor, present: Tensor, relations: Tensor) -> Tensor:
        """Candidate-specific record attention, without [B,C,M,width] copies."""
        batch, candidates, width = query.shape
        if relations.shape != (batch, candidates, tokens.shape[1], 8):
            raise ValueError("measurement_relations must be [batch, candidates, measurement rows, 8]")
        keys = self.measurement_key(tokens)
        values = self.measurement_value(tokens)
        # The null row has zero key, value and relation bias. In empty/all-masked
        # episodes its probability is one, giving an exactly zero residual.
        keys = torch.cat((keys, keys.new_zeros(batch, 1, width)), 1)
        values = torch.cat((values, values.new_zeros(batch, 1, width)), 1)
        bias = self.measurement_relation_bias(relations.to(query.dtype)).squeeze(-1)
        bias = torch.cat((bias, bias.new_zeros(batch, candidates, 1)), -1)
        present = torch.cat((present.bool(), torch.ones(batch, 1, dtype=torch.bool, device=present.device)), 1)
        weights = (torch.einsum("bcw,bmw->bcm", query, keys) / math.sqrt(width) + bias)
        weights = weights.masked_fill(~present[:, None], -torch.inf).softmax(-1)
        return self.measurement_output(torch.einsum("bcm,bmw->bcw", weights, values))

    @classmethod
    def from_lifetime_checkpoint(cls, checkpoint: dict | str | PathLike, config: DigitalConfig):
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True) \
            if isinstance(checkpoint, (str, PathLike)) else checkpoint
        core = ContinualCore(CoreConfig(**saved["config"]), saved["model"]["action_values"])
        core.load_state_dict(saved["model"], strict=True)
        return cls(config, core)

    def initial(self, batch: int, device) -> dict[str, Tensor]:
        state = self.core.initial(batch, device)
        if self.config.action_conditioned:
            state.update(pending_delta=torch.zeros(batch, self.config.width, device=device),
                         expected_numeric=torch.zeros(batch, self.config.numeric_dim, device=device),
                         observed_numeric=torch.zeros(batch, self.config.numeric_dim, device=device),
                         pending_action=torch.zeros(batch, dtype=torch.bool, device=device))
        return state

    def encode(self, features: dict[str, Tensor]) -> EncodedEvent:
        numeric = features["numeric"]
        n, device = numeric.shape[0], numeric.device
        dtype = self.numeric[0].weight.dtype
        tokens = [self.numeric(numeric.to(dtype)).unsqueeze(1) + self.modality[0]]
        masks = [torch.ones(n, 1, dtype=torch.bool, device=device)]
        if "timing" in features:
            tokens.append(self.time(torch.log1p(features["timing"].to(dtype)))[:, None] + self.modality[1])
            masks.append(torch.ones(n, 1, dtype=torch.bool, device=device))
        for name, projection, modality in (("text", self.text, 2), ("image", self.image, 3)):
            if features.get(name) is None:
                continue
            value = features[name].detach()
            value = value.unsqueeze(1) if value.ndim == 2 else value
            tokens.append(projection(value.to(dtype)) + self.modality[modality])
            mask = features.get(name + "_present")
            if mask is None:
                mask = torch.ones(value.shape[:2], dtype=torch.bool, device=device)
            elif mask.ndim == 1:
                mask = mask[:, None].expand(-1, value.shape[1])
            masks.append(mask.bool())
        if features.get("measurement_numeric") is not None:
            numbers = features["measurement_numeric"].to(dtype)
            if numbers.shape[-1] != self.core.config.record_dim:
                raise ValueError("Measurement record columns must match the source core record schema")
            age = features["measurement_age"].to(dtype).unsqueeze(-1)
            record = self.core.record(torch.cat((numbers, age), -1))
            label = self.text(features["measurement_text"].detach().to(dtype))
            tokens.append(record + label + self.modality[0])
            masks.append(features["measurement_present"].bool())
        present = torch.cat(masks, 1)
        return EncodedEvent(self.token_norm(torch.cat(tokens, 1)).masked_fill(~present[..., None], 0), present, numeric)

    def forward(self, observation_features: dict[str, Tensor] | EncodedEvent,
                candidate_features: dict[str, Tensor], state: dict[str, Tensor], *,
                persistent: bool = True, refinements: int | Tensor | None = None,
                mode: str = "cdm", records: Tensor | None = None,
                restart_workspace: bool = False, predict_outcomes: bool = True,
                outcome_indices: Tensor | None = None):
        """Return logits [B,C], continued {h,z,A,B}, and optional outcomes.

        Outcomes default to [B,C,outcome_dim]. Long outcome_indices [B] select
        one candidate per row before the outcome MLP, returning [B,outcome_dim].
        predict_outcomes=False returns None. The action-conditioned variant
        still computes predictions used by its decision readout. Return options
        never change logits or continuing state.

        Candidate text is [B,C,text_dim]; numeric is [B,C,candidate_numeric_dim].
        Optional ``present`` masks padding. Observations contain numeric [B,N]
        and optional frozen text/image tokens, masks and timing [B,3]. ``gru``
        removes workspace use/refinement for a separately trained baseline.
        ``persistent=False`` resets recurrent/workspace state for an intervention.
        ``restart_workspace=True`` resets only slots, retaining the GRU state
        and instance adapters. The caller's public-object memory is unaffected.
        With object_relations, candidates also supply measurement_relations
        [B,C,M,8], aligned with observation measurement rows. A pre-encoded
        EncodedEvent must place those M rows last, as encode() does; its present
        mask supplies their validity. No object handle enters a numeric layer.
        """
        if mode not in ("cdm", "gru"):
            raise ValueError("mode must be 'cdm' or 'gru'")
        if self.config.object_relations and not isinstance(observation_features, EncodedEvent):
            measurements = observation_features.get("measurement_numeric")
            count = measurements.shape[1] if measurements is not None else 0
            if candidate_features["measurement_relations"].shape[2] != count:
                raise ValueError("measurement_relations must be aligned with observation measurement rows")
        event, state = self.assimilate(observation_features, state, persistent=persistent,
                                       restart_workspace=restart_workspace)
        if mode == "cdm":
            state = self.refine_state(event, state, self.config.refinements if refinements is None else refinements,
                                      records=records)
        logits, outcome = self.readout(event, candidate_features, state, mode=mode,
                                       predict_outcomes=predict_outcomes, outcome_indices=outcome_indices)
        return logits, state, outcome

    def assimilate(self, observation_features, state, *, persistent=True, restart_workspace=False):
        """Receive one external observation, independently of internal compute.

        Additional reasoning uses refine_state on this event; it must not feed
        the same external observation through the GRU for a second time.
        """
        event = observation_features if isinstance(observation_features, EncodedEvent) else self.encode(observation_features)
        batch = event.tokens.shape[0]
        if not persistent or restart_workspace:
            state = {**state, "h": state["h"] if persistent else torch.zeros_like(state["h"]),
                     "z": self.core.slots.unsqueeze(0).expand(batch, -1, -1)}
        prior = state["h"]
        if self.config.action_conditioned:
            if event.numeric is None:
                raise ValueError("Action-conditioned updates require the received numeric observation")
            pending = state["pending_action"] & persistent
            prior = prior + state["pending_delta"] * pending[:, None]
        summary = self.ingest(self.query_norm(prior)[:, None], event.tokens, event.tokens,
                              key_padding_mask=~event.present, need_weights=False)[0][:, 0]
        if self.config.action_conditioned:
            residual = event.numeric.to(prior.dtype) - state["expected_numeric"]
            summary = summary + self.innovation(residual) * pending[:, None]
            state = {**state, "observed_numeric": event.numeric.to(prior.dtype),
                     "pending_action": torch.zeros_like(pending)}
        h = self.core.fast(prior, self.input_bridge(self.summary_norm(summary)))
        return event, {**state, "h": h}

    def refine_state(self, event, state, refinements, *, records=None):
        """Continue the workspace without another observation or world tick."""
        h, z = state["h"], state["z"]
        counts = torch.full((h.shape[0],), refinements, dtype=torch.long, device=h.device) \
            if isinstance(refinements, int) else refinements.to(device=h.device, dtype=torch.long)
        if counts.shape != (h.shape[0],) or bool((counts < 0).any()):
            raise ValueError("refinements must be nonnegative with one value per batch row")
        if records is None:
            records = torch.empty(h.shape[0], 0, self.core.config.record_dim, device=h.device, dtype=h.dtype)
        z = self.core.refine(z, h, records, counts, event.tokens, event.present)
        return {**state, "z": z}

    def _candidate_state(self, event, candidate_features, state, mode):
        h, z = state["h"], state["z"]
        features = self.core.state_features(h, z) if mode == "cdm" else \
            torch.cat((self.core.norm_h(h), torch.zeros_like(h)), -1)
        candidate = self.candidate_norm(self.candidate_text(candidate_features["text"].detach().to(h.dtype))
                                        + self.candidate_numeric(candidate_features["numeric"].to(h.dtype)))
        if self.config.object_relations:
            relations = candidate_features["measurement_relations"]
            count = relations.shape[2]
            if count > event.tokens.shape[1]:
                raise ValueError("EncodedEvent has fewer tokens than the declared measurement rows")
            measurement_tokens = event.tokens[:, -count:] if count else event.tokens[:, :0]
            measurement_present = event.present[:, -count:] if count else event.present[:, :0]
            query = self.state_query(features)[:, None] + candidate
            candidate = candidate + self._measurement_read(query, measurement_tokens, measurement_present, relations)
        if self.config.candidate_read:
            # Each action can read a different part of the received observation
            # and workspace. GRU comparisons do not read workspace slots.
            context = event.tokens
            present = event.present
            if mode == "cdm":
                context = torch.cat((context, self.core.norm_z(z)), 1)
                present = torch.cat((present, torch.ones(z.shape[:2], dtype=torch.bool, device=z.device)), 1)
            query = self.state_query(features)[:, None] + candidate
            read = self.candidate_attention(query, context, context, key_padding_mask=~present,
                                            need_weights=False)[0]
            candidate = candidate + self.candidate_read_output(read)
        return features, candidate

    def _transition(self, features, candidate, state):
        hidden = F.gelu(self.transition_state(features)[:, None] + self.transition_action(candidate))
        change = self.transition_numeric(hidden)
        return hidden, state["observed_numeric"][:, None] + change, change

    def commit_action(self, observation_features, candidate_features, state, indices, *, mode="cdm"):
        """Predict after the command actually executed, before reading its result.

        An expert target or a proposed action is not a state transition. Negative
        indices mark commands outside the recorded menu and clear the pending
        transition. Public observations will still be assimilated normally.
        """
        if not self.config.action_conditioned:
            return state, None
        valid = indices >= 0
        rows = torch.arange(indices.shape[0], device=indices.device)
        chosen = {key: value[rows, indices.clamp_min(0)][:, None] for key, value in candidate_features.items()}
        event = observation_features if isinstance(observation_features, EncodedEvent) else self.encode(observation_features)
        features, candidate = self._candidate_state(event, chosen, state, mode)
        hidden, prediction, _ = self._transition(features, candidate, state)
        pending = self.transition_hidden(hidden[:, 0]).to(state["h"].dtype)
        prediction = prediction[:, 0].to(state["h"].dtype)
        return {**state, "pending_delta": pending * valid[:, None], "expected_numeric": prediction,
                "pending_action": valid}, prediction

    def readout(self, event, candidate_features, state, *, mode="cdm", predict_outcomes=True, outcome_indices=None):
        """Score public actions using the current state without changing it."""
        features, candidate = self._candidate_state(event, candidate_features, state, mode)
        batch = features.shape[0]
        decision = self.state_query(features)[:, None] + candidate
        all_outcomes = None
        if self.config.action_conditioned:
            _, _, change = self._transition(features, candidate, state)
            # Factor the first affine layer instead of copying state for every
            # candidate. This is the same retained outcome head and parameters.
            first, activation, last = self.outcome
            width = self.config.width
            hidden = F.linear(features, first.weight[:, :2 * width], first.bias)[:, None]
            hidden = hidden + F.linear(candidate, first.weight[:, 2 * width:])
            all_outcomes = last(activation(hidden))
            decision = decision + self.prediction_read(torch.cat((change, all_outcomes), -1))
        logits = self.score(decision).squeeze(-1)
        if "present" in candidate_features:
            logits = logits.masked_fill(~candidate_features["present"].bool(), -torch.inf)
        outcome = None
        if predict_outcomes:
            if all_outcomes is not None:
                outcome = all_outcomes if outcome_indices is None else all_outcomes[
                    torch.arange(batch, device=features.device), outcome_indices]
            elif outcome_indices is None:
                outcome_input = torch.cat((features[:, None].expand(-1, candidate.shape[1], -1), candidate), -1)
            else:
                if outcome_indices.shape != (batch,):
                    raise ValueError("outcome_indices must have shape [batch]")
                selected = candidate.gather(1, outcome_indices[:, None, None].expand(-1, 1, candidate.shape[-1]))[:, 0]
                outcome_input = torch.cat((features, selected), -1)
            if all_outcomes is None:
                outcome = self.outcome(outcome_input)
        return logits, outcome
