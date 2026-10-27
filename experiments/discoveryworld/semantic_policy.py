"""A frozen local SemanticExpert selecting native actions by text generation.

The policy consumes public observations and its actual feedback. It has no
reference controller, candidate selector, fitted scientific rule or CDM state.
"""
from __future__ import annotations

import argparse
import base64
from copy import deepcopy
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
import re
from time import perf_counter

import torch

from cdm.digital import PublicObjectMemory
from cdm.semantic import SEMANTIC_ID, SEMANTIC_REVISION, SemanticExpert
from .adapter import DiscoveryWorldEnv, TrajectoryWriter
from .measurements import MeasurementLog


SYSTEM_PROMPT = """You are conducting the scientific task described in the public observation.
Choose exactly one next action using the documented native action interface.
Return only one JSON object, with no prose or Markdown. Follow the current
output_protocol supplied with the observation.
Object UUIDs are exact routing handles, not physical quantities. Previously
observed objects may have changed or be inaccessible. Read actual outcomes;
an action proposal is not evidence that an action succeeded. Measurement values
and units are retained with their observed conditions. These conditions need
not describe every physical influence. Use observations to investigate the
task and test your conclusions. No further action will occur until you reply.
"""


def action_protocol(observation: dict) -> dict:
    """The native JSON protocol for the currently displayed public UI mode.

    Ordinary syntax follows DiscoveryWorldAPI.additionalActionDescriptionString;
    dialogue syntax follows UserInterface._parseDialogJSON. This describes the
    interface and never validates, repairs or selects a generated action.
    """
    dialog = observation["ui"].get("dialog_box", {})
    options = dialog.get("dialogOptions", {})
    if dialog.get("is_in_dialog") and options:
        numbers = [int(key) for key in options] if isinstance(options, dict) else list(range(1, len(options) + 1))
        return {"mode": "dialogue", "instruction": "Select one number from the currently displayed dialogue options.",
                "schema": {"type": "object", "required": ["chosen_dialog_option_int"],
                           "properties": {"chosen_dialog_option_int": {"type": "integer", "enum": numbers}},
                           "additionalProperties": False}}
    return {"mode": "ordinary_action",
            "instruction": "There are no selectable dialogue options now. Return an ordinary action object. "
                           "The action field is required and must exactly match a name in actions. "
                           "Include arg1 and arg2 only as required by that action's documented args. "
                           "Object arguments use integer UUIDs from observations; directions, teleport names "
                           "and other arguments use the values described in actions. Examples below show "
                           "syntax only: 5 and 12 are example UUIDs, not objects selected for this task.",
            "schema": {"type": "object", "required": ["action"],
                       "properties": {"action": {"type": "string", "enum": list(observation.get("actions", {}))},
                                      "arg1": {"type": ["integer", "string"]},
                                      "arg2": {"type": ["integer", "string"]}},
                       "additionalProperties": False},
            "syntax_examples": [{"action": "USE", "arg1": 5, "arg2": 12},
                                {"action": "ROTATE_DIRECTION", "arg1": "north"}]}

MEASUREMENT_COLUMNS = ("kind", "instrument_handle", "instrument_name", "target_handle", "target_name",
                       "quantity", "unit", "conditions_id", "contained_objects")
NOTE_FIELDS = ("memory", "running_hypotheses", "explanation")


def _public_observation(observation: dict) -> dict:
    """Keep the native UI, without arbitrary annotations or scorecard fields."""
    ui = observation["ui"]
    keys = ("agentLocation", "world_steps", "inventoryObjects", "accessibleEnvironmentObjects",
            "nearbyObjects", "dialog_box", "lastActionMessage", "extended_action_message", "discoveryFeed")
    public = {key: deepcopy(ui[key]) for key in keys if key in ui}
    public["taskProgress"] = [{key: task[key] for key in ("taskName", "description", "taskDescription")
                               if key in task} for task in ui.get("taskProgress", [])]
    return {"ui": public, "actions": deepcopy(observation.get("actions", {})),
            "teleport_locations": deepcopy(observation.get("teleport_locations", {}))}


class SemanticPolicyError(RuntimeError):
    """A recorded model/budget/JSON failure, with no replacement action."""


class SemanticPolicy:
    def __init__(self, model_path: str = SEMANTIC_ID, *, revision: str = SEMANTIC_REVISION,
                 device: str = "cpu", dtype: str = "auto", max_input_tokens: int = 32768,
                 max_new_tokens: int = 128, temperature: float = 0., history_steps: int = 32,
                 thinking: bool = False, top_p: float | None = None, top_k: int | None = None,
                 research_notebook: bool = False, constrained_actions: bool = False,
                 vision: bool = True, image_max_pixels: int = 262144,
                 trace_path: str | Path | None = None, expert=None, processor=None):
        if min(max_input_tokens, max_new_tokens, image_max_pixels) <= 0 or history_steps < 0 or temperature < 0:
            raise ValueError("Token/pixel budgets must be positive; history and temperature nonnegative")
        if (top_p is not None and not 0 < top_p <= 1) or (top_k is not None and top_k < 0):
            raise ValueError("top_p must be in (0, 1] and top_k must be nonnegative")
        if constrained_actions and (thinking or research_notebook):
            raise ValueError("Native-candidate decoding emits an action only; thinking and notebook use free generation")
        self.configuration = dict(model_path=str(model_path), revision=revision, device=device, dtype=dtype,
                                  max_input_tokens=max_input_tokens, max_new_tokens=max_new_tokens,
                                  temperature=temperature, top_p=top_p, top_k=top_k, history_steps=history_steps,
                                  vision=vision, image_max_pixels=image_max_pixels,
                                  enable_thinking=thinking, research_notebook=research_notebook,
                                  constrained_actions=constrained_actions, local_files_only=True)
        self.device = torch.device(device)
        if processor is None:
            from transformers import AutoProcessor
            processor = AutoProcessor.from_pretrained(model_path, revision=revision, local_files_only=True)
        if expert is None:
            expert = SemanticExpert.from_pretrained(model_path, revision=revision, dtype=dtype,
                                                     local_files_only=True)
        self.processor, self.expert = processor, expert.to(self.device).eval()
        self.max_input_tokens, self.max_new_tokens = max_input_tokens, max_new_tokens
        self.temperature, self.history_steps = temperature, history_steps
        self.top_p, self.top_k = top_p, top_k
        self.thinking = thinking
        self.research_notebook = research_notebook
        self.constrained_actions = constrained_actions
        self.vision, self.image_max_pixels = vision, image_max_pixels
        self.reset(trace_path=trace_path)

    def reset(self, *, trace_path: str | Path | None = None):
        self.objects, self.measurements = PublicObjectMemory(), MeasurementLog()
        self.history, self.decisions = [], []
        self.model_authored_notes, self._pending_notes = None, None
        self.notebook_updates = []
        self.trace_path = None if trace_path is None else Path(trace_path)
        if self.trace_path is not None:
            self.trace_path.parent.mkdir(parents=True, exist_ok=True)
            self.trace_path.touch(exist_ok=False)

    def observe(self, transition: dict):
        """Only executed actions and received public results enter episode memory."""
        before = _public_observation(transition["observation"])
        after = _public_observation(transition["next_observation"])
        result = {key: deepcopy(transition["action_result"][key])
                  for key in ("success", "errors", "message", "successMessage")
                  if key in transition["action_result"]}
        self.measurements.observe_initial(before)
        self.measurements.observe(transition["action"], result, after, sequence=transition["index"])
        self.objects.observe(after)
        ui = after["ui"]
        history_action = deepcopy(transition["action"])
        if self.research_notebook:
            # The complete returned dictionary stays in the native trajectory.
            # Carry authored text once, in its explicitly labelled notebook.
            history_action = {key: value for key, value in history_action.items() if key not in NOTE_FIELDS}
        self.history.append({"sequence": transition["index"], "action": history_action,
                             "result": result, "lastActionMessage": ui.get("lastActionMessage", ""),
                             "dialog_box": deepcopy(ui.get("dialog_box", {}))})
        pending, self._pending_notes = self._pending_notes, None
        if self.research_notebook and pending is not None and pending["action"] == transition["action"]:
            update = {"source": "model_generated_before_action", "decision": pending["decision"],
                      "based_on_world_step": pending["world_step"], "content": pending["content"],
                      "missing_fields": [key for key in NOTE_FIELDS if key not in pending["content"]],
                      "received_feedback": {"sequence": transition["index"], "action_result": deepcopy(result)}}
            self.model_authored_notes = deepcopy(update)
            self.notebook_updates.append(update)
            if self.trace_path is not None:
                with self.trace_path.with_suffix(".notes.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(update, ensure_ascii=False) + "\n")

    @staticmethod
    def _payload(public, sections, omitted):
        # Input views preserve all values, units and bindings. The full original
        # records, source text and object descriptions remain in the disk ledger.
        conditions, condition_ids = [], {}
        payload = {**public, **deepcopy(sections), "omitted": dict(omitted)}
        participants = set()
        for row in payload["measurements"]:
            participants.update((row["instrument_handle"], row["target_handle"], *row["contained_objects"]))
        for name in ("measurements", "measurement_summaries"):
            for row in payload[name]:
                condition = row.pop("conditions")
                key = json.dumps(condition, sort_keys=True)
                if key not in condition_ids:
                    condition_ids[key] = len(conditions)
                    # Public view membership, distance and position are exact.
                    # Retain historical descriptions for every actual participant
                    # (including contained objects); other descriptions remain in
                    # the current UI/object registry and original on-disk ledger.
                    objects = condition["inventory"] + condition["accessible"]
                    objects += [obj for group in condition["nearby"].values() for obj in group]
                    snapshots = {obj["uuid"]: obj for obj in objects if obj["uuid"] in participants}
                    conditions.append({"agent_location": condition["agent_location"],
                                       "inventory": [obj["uuid"] for obj in condition["inventory"]],
                                       "accessible": [obj["uuid"] for obj in condition["accessible"]],
                                       "nearby": {direction: [[obj["uuid"], obj.get("distance")] for obj in group]
                                                  for direction, group in condition["nearby"].items()},
                                       "participant_objects": list(snapshots.values())})
                row["conditions_id"] = condition_ids[key]
        payload["measurement_conditions"] = conditions
        # Share only identical metadata. Every received sample retains its exact
        # value and source sequence; repeated radiation is never averaged away.
        groups = {}
        for row in payload["measurements"]:
            values = [row[column] for column in MEASUREMENT_COLUMNS]
            key = json.dumps(values)
            if key not in groups:
                groups[key] = values + [[]]
            groups[key][-1].append([row["sequence"], row["world_step"], row["value"]])
        payload["measurement_columns"] = (*MEASUREMENT_COLUMNS, "samples")
        payload["sample_columns"] = ("sequence", "world_step", "value")
        payload["measurements"] = list(groups.values())
        return payload

    def _prepare(self, observation):
        public = _public_observation(observation)
        self.measurements.observe_initial(public)
        self.objects.observe(public)
        sections = {"actual_history": deepcopy(self.history[-self.history_steps:]) if self.history_steps else [],
                    "measurements": [asdict(record) for record in self.measurements.records],
                    "measurement_summaries": [asdict(record) for record in self.measurements.summaries()],
                    "previously_observed_objects": [{key: obj[key] for key in ("uuid", "name", "description")
                                                      if key in obj} for obj in self.objects.objects.values()]}
        omitted = {name: 0 for name in sections}
        omitted["actual_history"] = len(self.history) - len(sections["actual_history"])
        image = observation.get("vision", {}).get("base64_no_grid") if self.vision else None
        while True:
            payload = self._payload(public, sections, omitted)
            payload["output_protocol"] = action_protocol(public)
            if self.research_notebook:
                payload["model_authored_notes"] = deepcopy(self.model_authored_notes)
                protocol = payload["output_protocol"]
                protocol["instruction"] += (
                    " In this same JSON object, include memory, running_hypotheses and explanation. "
                    "Rewrite the compact notes you want to retain for your next decision. "
                    "model_authored_notes are your earlier claims, not verified measurements; revise them "
                    "using received feedback. The action you now propose has not executed yet.")
                descriptions = {
                    "memory": "Information and plans to retain from the observations and results already received.",
                    "running_hypotheses": "Current scientific hypotheses, supporting observations and unresolved tests.",
                    "explanation": "Why you choose this next action based on the current evidence."}
                protocol["schema"]["required"].extend(NOTE_FIELDS)
                protocol["schema"]["properties"].update({key: {"type": ["string", "array", "object"],
                                                               "description": descriptions[key]} for key in NOTE_FIELDS})
            content = [{"type": "text", "text": json.dumps(payload, ensure_ascii=False, separators=(",", ":"))}]
            if image is not None:
                content.append({"type": "image", "image": image})
            messages = [{"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
                        {"role": "user", "content": content}]
            # All message content uses the processor's typed multimodal format,
            # including the system message. Never truncate through JSON/images.
            inputs = self.processor.apply_chat_template(
                messages, add_generation_prompt=True, enable_thinking=self.thinking, tokenize=True,
                return_dict=True, return_tensors="pt", processor_kwargs={"images_kwargs": {"size": {
                    "shortest_edge": min(65536, self.image_max_pixels), "longest_edge": self.image_max_pixels}}})
            tokens = inputs["input_ids"].shape[-1]
            if tokens <= self.max_input_tokens:
                return messages, inputs, omitted
            # Remove whole oldest entries, in this documented order. Memory and
            # raw trajectory remain intact; only this call's view is bounded.
            for name in ("actual_history", "measurement_summaries", "previously_observed_objects", "measurements"):
                if sections[name]:
                    count = max(1, len(sections[name]) // 2)
                    sections[name] = sections[name][count:]
                    omitted[name] += count
                    break
            else:
                raise SemanticPolicyError(f"Current public observation/action schema requires {tokens} tokens; "
                                          f"input budget is {self.max_input_tokens}")

    def _trace_messages(self, messages):
        logged = deepcopy(messages)
        for message in logged:
            for item in message["content"]:
                if item["type"] != "image":
                    continue
                header, encoded = item.pop("image").split(",", 1)
                if header != "data:image/png;base64":
                    raise ValueError("Expected an inline public PNG observation")
                payload = base64.b64decode(encoded, validate=True)
                digest = sha256(payload).hexdigest()
                item.update(sha256=digest, path=f"semantic_images/{digest}.png")
                if self.trace_path is not None:
                    path = self.trace_path.parent / item["path"]
                    path.parent.mkdir(exist_ok=True)
                    if not path.exists():
                        path.write_bytes(payload)
        return logged

    @torch.no_grad()
    def __call__(self, observation: dict) -> dict:
        start = perf_counter()
        self._pending_notes = None
        decision = {"decision": len(self.decisions), "world_step": observation["ui"].get("world_steps"),
                    "configuration": self.configuration, "generate_calls": 0}
        try:
            messages, inputs, omitted = self._prepare(observation)
            decision.update(messages=self._trace_messages(messages), omitted=omitted,
                            input_ids=inputs["input_ids"][0].tolist(), input_tokens=inputs["input_ids"].shape[-1])
            inputs = {key: value.to(self.device) if isinstance(value, torch.Tensor) else value
                      for key, value in inputs.items()}
            prepared = perf_counter()
            kwargs = {"max_new_tokens": self.max_new_tokens, "do_sample": self.temperature > 0}
            if self.temperature > 0:
                kwargs["temperature"] = self.temperature
            if self.top_p is not None:
                kwargs["top_p"] = self.top_p
            if self.top_k is not None:
                kwargs["top_k"] = self.top_k
            constraint = None
            if self.constrained_actions:
                from cdm.digital import public_candidates
                from .native_decoding import NativeActionTrie
                public = self.objects.observe(_public_observation(observation))
                eos_ids = self.expert.backbone.generation_config.eos_token_id
                constraint = NativeActionTrie(public_candidates(public), self.processor.tokenizer,
                                              input_tokens=inputs["input_ids"].shape[-1], eos_ids=eos_ids,
                                              max_new_tokens=self.max_new_tokens)
                kwargs["prefix_allowed_tokens_fn"] = constraint
                decision["native_candidates"] = {"count": len(constraint.actions),
                                                 "maximum_tokens": constraint.maximum_tokens,
                                                 "source": "public_candidates_shared_with_cdm"}
            decision["generate_calls"] = 1
            output = self.expert.generate(**inputs, **kwargs)
            generated = output[0, inputs["input_ids"].shape[-1]:].tolist()
            text = self.processor.decode(generated, skip_special_tokens=True)
            raw_text = self.processor.decode(generated, skip_special_tokens=False)
            generation_config = getattr(self.expert.backbone, "generation_config", None)
            eos = getattr(generation_config, "eos_token_id", None)
            if eos is None:
                eos = getattr(getattr(self.processor, "tokenizer", None), "eos_token_id", None)
            eos = [eos] if isinstance(eos, int) else (eos or [])
            ended = bool(generated and generated[-1] in eos)
            limit_reached = len(generated) >= self.max_new_tokens and not ended
            decision.update(raw_output=text, raw_output_with_special_tokens=raw_text,
                            output_ids=generated, output_tokens=len(generated),
                            stop_reason="eos" if ended else "length" if limit_reached else "other",
                            output_limit_reached=limit_reached, thinking_complete=None,
                            preparation_seconds=prepared - start, generation_seconds=perf_counter() - prepared)
            if self.thinking:
                reasoning, boundary, final = text.partition("</think>")
                decision.update(reasoning_output=reasoning, thinking_complete=bool(boundary),
                                final_output=final if boundary else None)
                if not boundary:
                    raise SemanticPolicyError("Generation stopped before the </think> boundary; no final action")
                text = final
            else:
                decision["final_output"] = text
            # Accept a complete Markdown JSON wrapper, never a JSON fragment
            # searched out of prose or reasoning. Multiple blocks leave invalid
            # JSON in the body and are rejected by the same strict decoder.
            fenced = re.fullmatch(r"```(?:json)?[ \t]*\r?\n(.*?)\r?\n[ \t]*```", text.strip(), re.DOTALL)
            action = json.loads(fenced.group(1) if fenced else text)
            if not isinstance(action, dict):
                raise ValueError("Native action must be a JSON object")
            if constraint is not None and not constraint.contains(action):
                raise SemanticPolicyError("Generated action is outside the supplied native candidates")
            decision["action"] = deepcopy(action)
            if self.research_notebook:
                self._pending_notes = {"action": deepcopy(action), "decision": decision["decision"],
                                       "world_step": decision["world_step"],
                                       "content": {key: deepcopy(action[key]) for key in NOTE_FIELDS if key in action}}
            return action  # Native validity and failure are decided by the environment.
        except Exception as error:
            decision["error"] = {"type": type(error).__name__, "message": str(error)}
            raise SemanticPolicyError(str(error)) from error
        finally:
            decision["decision_seconds"] = perf_counter() - start
            self.decisions.append(decision)
            if self.trace_path is not None:
                with self.trace_path.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(decision, ensure_ascii=False) + "\n")


def create(config: dict):
    """Factory for collect --policy experiments.discoveryworld.semantic_policy:create."""
    return SemanticPolicy(**config)


def evaluate_episode(agent: SemanticPolicy, output: Path, scenario: str, difficulty: str,
                     seed: int, max_steps: int) -> dict:
    if (output / "trajectory.jsonl").exists():
        raise ValueError(f"Choose a new output directory: {output}")
    agent.reset(trace_path=output / "decisions.jsonl")
    env = DiscoveryWorldEnv(output, max_steps)
    observation = env.reset(scenario, difficulty, seed)
    metadata = {**env.configuration, "policy": "frozen_semantic_native_generation",
                "policy_config": agent.configuration, "action_descriptions": env.actions,
                "teleport_locations": env.teleports, "initial_task_metrics": env.metrics,
                "source_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
                "prompt_sha256": sha256(SYSTEM_PROMPT.encode()).hexdigest(),
                "action_protocol": "public_ui_mode_v1"}
    error, start = None, perf_counter()
    with TrajectoryWriter(output, metadata) as writer:
        try:
            while not env.ended:
                action = agent(observation)
                transition = env.step(action)
                writer.append(transition)
                agent.observe(transition)
                observation = env.policy_input()
        except Exception as failure:
            error = {"type": type(failure).__name__, "message": str(failure)}
    summary = {"scenario": scenario, "difficulty": difficulty, "seed": seed, "steps": env.steps,
               "policy": metadata["policy"], "task_metrics": env.metrics,
               "completed": all(row["completed"] for row in env.metrics),
               "completed_successfully": all(row["completed_successfully"] for row in env.metrics),
               "progress": sum(row["progress"] for row in env.metrics) / len(env.metrics),
               "truncated": env.steps >= max_steps and not all(row["completed"] for row in env.metrics),
               "error": error, "model_calls": sum(row["generate_calls"] for row in agent.decisions),
               "elapsed_seconds": perf_counter() - start, "protocol": "official_step_based",
               "discovery_knowledge_score": None}
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (output / "measurements.json").write_text(json.dumps([asdict(row) for row in agent.measurements.records], indent=2) + "\n")
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default=SEMANTIC_ID)
    parser.add_argument("--revision", default=SEMANTIC_REVISION)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", choices=("auto", "float32", "bfloat16", "float16"), default="auto")
    parser.add_argument("--max-input-tokens", type=int, default=32768)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.)
    parser.add_argument("--top-p", type=float)
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--thinking", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--research-notebook", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--constrained-actions", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--history-steps", type=int, default=32)
    parser.add_argument("--vision", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--image-max-pixels", type=int, default=262144)
    parser.add_argument("--scenario", default="Reactor Lab")
    parser.add_argument("--difficulty", default="Normal")
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--max-steps", type=int, default=150)
    parser.add_argument("--random-seed", type=int, default=0)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if len(set(args.seeds)) != len(args.seeds) or args.threads <= 0 or args.max_steps <= 0:
        parser.error("Use unique seeds and positive thread/step counts")
    if (args.output / "summary.json").exists() or any((args.output / f"seed{s}").exists() for s in args.seeds):
        parser.error("Choose a new output directory")
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.random_seed)
    policy_keys = ("model_path", "revision", "device", "dtype", "max_input_tokens", "max_new_tokens",
                   "temperature", "top_p", "top_k", "thinking", "research_notebook", "constrained_actions",
                   "history_steps", "vision", "image_max_pixels")
    agent = SemanticPolicy(**{key: getattr(args, key) for key in policy_keys})
    results = []
    for seed in args.seeds:
        result = evaluate_episode(agent, args.output / f"seed{seed}", args.scenario, args.difficulty, seed, args.max_steps)
        results.append(result)
        print(json.dumps(result), flush=True)
    (args.output / "summary.json").write_text(json.dumps({"policy_config": agent.configuration,
        "random_seed": args.random_seed, "episodes": results}, indent=2) + "\n")


if __name__ == "__main__":
    main()
