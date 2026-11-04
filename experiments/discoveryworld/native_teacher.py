"""Batched local language teacher for public, native scientific interactions.

The same prompt and weights serve every theme. The teacher receives no task
score, generator state, seed, solution, or reference controller. Its compact
notebook is an authored hypothesis record; actual observations remain in the
trajectory. Teacher results are reported separately from learned CDM results.
"""
from __future__ import annotations

from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import re
from time import perf_counter
from urllib.request import Request, urlopen

import torch

from cdm.digital import PublicObjectMemory, public_candidates
from .semantic_policy import _public_observation, action_protocol
from .measurements import MeasurementLog


SYSTEM = """You operate a scientific laboratory through its native action API.
Read the task and public observations, investigate the unknown relationship,
and use actual experimental evidence to achieve the task. Select a useful next
action: observations can distinguish explanations, and negative results should
revise your plan. Track measurements with their objects and units. Tasks may
require waiting for a physical change or returning to an earlier observation.

Return one JSON object containing the native action fields described in
output_protocol and a memory string. Rewrite memory as your compact notebook:
observed facts, current hypotheses, and the next unresolved experimental issue.
Keep it within 180 words. The notebook is your own earlier interpretation;
correct it when actual feedback disagrees. Execute one action per reply.
Use exact observed object identifiers and displayed dialogue options. Returned
errors are observations too: revise an unsuccessful operation. Only actual
actions advance the world. Objects listed as nearby can be out of reach:
ordinary object interactions require inventory or accessible environment objects.
Move, turn or teleport before interacting with a more distant object. Read the
UI's lastActionMessage for the actual outcome; native argument-conversion
warnings can also occur for a valid navigation command. Record confirmed facts
separately from intended actions: the action you select has not happened yet.
No task-specific solution is supplied.
"""


class NativeTeacher:
    def __init__(self, model_path, *, device="cuda:0", max_input_tokens=16384,
                 max_new_tokens=1024, dequantize=False, reasoning_effort="low",
                 model_source="unsloth/gpt-oss-20b", attention="flex_attention",
                 endpoint=None, served_model="teacher", temperature=.7, candidate_actions=True,
                 api_style="chat"):
        from transformers import AutoModelForCausalLM, AutoTokenizer, Mxfp4Config
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, local_files_only=True, padding_side="left")
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        options = {"quantization_config": Mxfp4Config(dequantize=True)} if dequantize else {}
        self.endpoint, self.served_model, self.api_style = endpoint, served_model, api_style
        self.model = None if endpoint else AutoModelForCausalLM.from_pretrained(
                model_path, local_files_only=True, dtype=torch.bfloat16,
                device_map=device, attn_implementation=attention,
                **options).eval().requires_grad_(False)
        model_type = json.loads((Path(model_path) / 'config.json').read_text())['model_type']
        self.model_type = model_type
        self.device = device
        self.max_input_tokens, self.max_new_tokens = max_input_tokens, max_new_tokens
        self.reasoning_effort = reasoning_effort
        self.temperature, self.candidate_actions = temperature, candidate_actions
        self.configuration = dict(model_path=str(model_path), max_input_tokens=max_input_tokens,
            max_new_tokens=max_new_tokens, reasoning_effort=reasoning_effort,
            dequantize=dequantize, observation="public_text", temperature=temperature,
            top_p=.95 if temperature else 1., action_interface="observed_native_action_domains" if candidate_actions else "native_json",
            model_type=model_type, model_source=model_source,
            attention=attention if not endpoint else "server configuration",
            inference_backend="transformers" if not endpoint else f"local {api_style} server",
            prompt_version="native_public_domains_v7")

    @staticmethod
    def initial():
        return dict(objects=PublicObjectMemory(), measurements=MeasurementLog(), history=[], notebook="")

    def prompt(self, observation, state):
        public = _public_observation(observation)
        remembered = state["objects"].observe(public)
        state['measurements'].observe_initial(public)
        protocol = action_protocol(public)
        protocol["schema"]["properties"]["memory"] = {"type": "string"}
        protocol["schema"]["required"].append("memory")
        payload = dict(public, output_protocol=protocol,
            previously_observed_objects=[{k: obj[k] for k in ("uuid", "name") if k in obj}
                                         for obj in state["objects"].objects.values()],
            actual_history=state["history"][-8:], notebook=state["notebook"])
        measurements = state['measurements'].records
        payload['received_measurements'] = [{key: getattr(record, key) for key in
            ('sequence', 'instrument_handle', 'instrument_name', 'target_handle', 'target_name',
             'quantity', 'value', 'unit', 'text', 'contained_objects')} for record in measurements[-48:]]
        payload['earlier_measurements_not_shown'] = max(0, len(measurements) - 48)
        system = SYSTEM
        if self.candidate_actions:
            candidates = public_candidates(remembered)
            state['candidates'] = [candidate.action for candidate in candidates]
            # Describe argument domains once per operation. Enumerating every
            # object pair makes crowded rooms require a quadratic-size prompt.
            # Use the actual API, including operations on observed objects that
            # are currently out of reach; the environment returns that failure.
            groups = {}
            object_definition = {'enum': list(state['objects'].objects)}
            dialog = public['ui'].get('dialog_box', {})
            if dialog.get('is_in_dialog') and dialog.get('dialogOptions'):
                options = dialog['dialogOptions']
                numbers = list(map(int, options)) if isinstance(options, dict) else list(range(1, len(options) + 1))
                groups['dialogue'] = {'chosen_dialog_option_int': numbers}
            else:
                objects = list(state['objects'].objects)
                for operation, description in public.get('actions', {}).items():
                    arguments = description.get('args', [])
                    if operation in ('MOVE_DIRECTION', 'ROTATE_DIRECTION'):
                        domains = {'arg1': ['north', 'east', 'south', 'west']}
                    elif operation == 'TELEPORT_TO_LOCATION':
                        domains = {'arg1': list(public.get('teleport_locations', {}))}
                    elif operation == 'DISCOVERY_FEED_GET_POST_BY_ID':
                        posts = public['ui'].get('discoveryFeed', {})
                        domains = {'arg1': list(dict.fromkeys(post['postID']
                            for rows in posts.values() if isinstance(rows, list)
                            for post in rows if isinstance(post, dict) and 'postID' in post))}
                    else:
                        domains = {argument: {'$ref': '#/$defs/observed_object'} for argument in arguments} if objects else \
                                  {argument: [] for argument in arguments}
                    if all(domains.values()):
                        groups[operation] = {'action': [operation], **domains}
            alternatives = []
            for operation, group in groups.items():
                properties = {key: values if isinstance(values, dict) else {'enum': values}
                              for key, values in group.items()}
                properties['memory'] = {'type': 'string', 'maxLength': 1000}
                alternatives.append({'type': 'object', 'properties': properties,
                    'required': list(properties), 'additionalProperties': False})
            schema = {'anyOf': alternatives}
            if state['objects'].objects:
                schema['$defs'] = {'observed_object': object_definition}
            payload['output_protocol'] = dict(schema=schema,
                instruction='Choose a native operation and observed arguments from these public domains. '
                            'Preserve the intended instrument and target handles. If an object is out of reach, '
                            'first move or teleport to it. For two-object operations use different objects. '
                            'Return a compact memory, at most 1000 characters.')
        messages = [dict(role="system", content=system),
                    dict(role="user", content=json.dumps(payload, ensure_ascii=False))]
        state['messages'] = messages
        state['output_schema'] = payload['output_protocol']['schema']
        if self.endpoint and self.api_style == 'chat':
            # The server applies the checkpoint's template and parses its
            # reasoning channels; the client consumes only final content.
            text = json.dumps(messages, ensure_ascii=False)
        else:
            text = self.tokenizer.apply_chat_template(messages, tokenize=False,
                add_generation_prompt=True, reasoning_effort=self.reasoning_effort)
        if self.tokenizer is not None:
            count = len(self.tokenizer.encode(text, add_special_tokens=False))
            if count > self.max_input_tokens:
                raise ValueError(f"Public input needs {count} tokens, exceeding {self.max_input_tokens}")
        return text

    @staticmethod
    def parse(raw):
        # Harmony separates analysis and final messages. Only the final message
        # is an action proposal; never take a JSON fragment from private analysis.
        final = list(re.finditer(r"<\|channel\|>final\b", raw))
        if final:
            raw = raw[final[-1].end():]
            if "<|message|>" in raw:
                raw = raw.split("<|message|>", 1)[1]
            else:
                raw = raw.strip().removeprefix("<|constrain|>").removeprefix("json").strip()
        elif "<|channel|>analysis" in raw or raw.startswith("analysis"):
            raise ValueError("Generation ended without a final action message")
        for token in ("<|return|>", "<|end|>", "<|im_end|>", "<|fim_suffix|>"):
            raw = raw.replace(token, "")
        raw = re.sub(r"^```(?:json)?\s*([\s\S]*?)\s*```$", r"\1", raw.strip())
        result = json.loads(raw.strip())
        if not isinstance(result, dict):
            raise ValueError("Expected one native-action JSON object")
        action = {key: result[key] for key in ("action", "arg1", "arg2", "chosen_dialog_option_int")
                  if key in result and result[key] is not None}
        if 'action_index' in result:
            action = {'action_index': result['action_index']}
        if not action:
            raise ValueError("No native action in final message")
        memory = result.get("memory", "")
        if not isinstance(memory, str):
            memory = json.dumps(memory, ensure_ascii=False)
        return action, memory

    def chat(self, state):
        body = dict(model=self.served_model, messages=state['messages'],
                    temperature=self.temperature, top_p=.95 if self.temperature else 1.,
                    max_tokens=self.max_new_tokens,
                    response_format={'type': 'json_schema', 'json_schema': {
                        'name': 'native_action', 'strict': True, 'schema': state['output_schema']}})
        if self.model_type == 'gpt_oss':
            body['reasoning_effort'] = self.reasoning_effort
        elif self.model_type == 'gemma4':
            body['chat_template_kwargs'] = {'enable_thinking': self.reasoning_effort != 'none'}
        request = Request(self.endpoint.rstrip('/') + '/chat/completions',
                          json.dumps(body).encode(), {'Content-Type': 'application/json'})
        with urlopen(request, timeout=600) as stream:
            response = json.load(stream)
        choice = response['choices'][0]
        return (choice['message'].get('content') or '',
                response.get('usage', {}).get('completion_tokens', 0),
                {'finish_reason': choice['finish_reason'],
                 'reasoning': choice['message'].get('reasoning', choice['message'].get('reasoning_content'))})

    @torch.inference_mode()
    def batch(self, observations, states):
        prompts = [self.prompt(obs, state) for obs, state in zip(observations, states)]
        start = perf_counter()
        extra = [{} for _ in prompts]
        if self.endpoint and self.api_style == 'chat':
            if len(states) == 1:
                replies = [self.chat(states[0])]
            else:
                with ThreadPoolExecutor(max_workers=len(states)) as pool:
                    replies = list(pool.map(self.chat, states))
            outputs, lengths, extra = zip(*replies)
        elif self.endpoint:
            body = dict(model=self.served_model, prompt=prompts, temperature=self.temperature,
                        top_p=.95 if self.temperature else 1.,
                        max_tokens=self.max_new_tokens, skip_special_tokens=False,
                        add_special_tokens=False)
            request = Request(self.endpoint.rstrip('/') + '/completions',
                              json.dumps(body).encode(), {'Content-Type': 'application/json'})
            with urlopen(request, timeout=600) as stream:
                response = json.load(stream)
            choices = sorted(response['choices'], key=lambda choice: choice['index'])
            if len(choices) != len(prompts):
                raise ValueError('Completion server returned a different number of actions')
            outputs = [choice['text'] for choice in choices]
            lengths = [len(self.tokenizer.encode(text, add_special_tokens=False)) for text in outputs]
        else:
            inputs = self.tokenizer(prompts, return_tensors="pt", padding=True,
                                    add_special_tokens=False).to(self.device)
            sampling = dict(temperature=self.temperature, top_p=.95) if self.temperature else {}
            generated = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens,
                do_sample=bool(self.temperature), pad_token_id=self.tokenizer.pad_token_id, **sampling)
            outputs, lengths = [], []
            for output in generated[:, inputs["input_ids"].shape[1]:]:
                tokens = output.tolist()
                while tokens and tokens[-1] == self.tokenizer.pad_token_id:
                    tokens.pop()
                outputs.append(self.tokenizer.decode(tokens, skip_special_tokens=False))
                lengths.append(len(tokens))
        seconds = perf_counter() - start
        result = []
        for raw, length, state, details in zip(outputs, lengths, states, extra):
            row = dict(raw_output=raw, generated_tokens=length, batch_seconds=seconds,
                       batch_size=len(prompts), model_source=self.configuration['model_source'],
                       prompt_version=self.configuration['prompt_version'], **details)
            if "error" in row:
                result.append(row)
                continue
            try:
                row["action"], row["memory"] = self.parse(raw)
                if self.candidate_actions:
                    index = state['candidates'].index(row['action']) if row['action'] in state['candidates'] else None
                    row['selected_candidate'] = index
                    row['candidate_count'] = len(state['candidates'])
            except (ValueError, TypeError) as error:
                row["error"] = str(error)
            result.append(row)
        return result

    @staticmethod
    def observe(state, transition, generated):
        state['measurements'].observe(transition['action'], transition['action_result'],
                                      transition['next_observation'], sequence=transition.get('index'))
        state["notebook"] = generated["memory"]
        state["history"].append(dict(action=deepcopy(transition["action"]),
            result=deepcopy(transition["action_result"]),
            feedback={key: transition["next_observation"]["ui"].get(key, '')
                      for key in ('lastActionMessage', 'extended_action_message')},
            world_step=transition["next_observation"]["ui"].get("world_steps")))
