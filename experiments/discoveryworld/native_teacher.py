"""Batched local language teacher for public, native scientific interactions.

The same prompt and weights serve every theme. The teacher receives no task
score, generator state, seed, solution, or reference controller. Its compact
notebook is an authored hypothesis record; actual observations remain in the
trajectory. Teacher results are reported separately from learned CDM results.
"""
from __future__ import annotations

from copy import deepcopy
import json
import re
from time import perf_counter

import torch

from cdm.digital import PublicObjectMemory
from .semantic_policy import _public_observation, action_protocol


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
actions advance the world. No task-specific solution is supplied.
"""


class NativeTeacher:
    def __init__(self, model_path, *, device="cuda:0", max_input_tokens=16384,
                 max_new_tokens=1024, dequantize=False, reasoning_effort="low",
                 model_source="unsloth/gpt-oss-20b", attention="flex_attention"):
        from transformers import AutoModelForCausalLM, AutoTokenizer, Mxfp4Config
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, local_files_only=True, padding_side="left")
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        options = {"quantization_config": Mxfp4Config(dequantize=True)} if dequantize else {}
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, local_files_only=True, dtype=torch.bfloat16,
            device_map=device, attn_implementation=attention,
            **options).eval().requires_grad_(False)
        self.device = device
        self.max_input_tokens, self.max_new_tokens = max_input_tokens, max_new_tokens
        self.reasoning_effort = reasoning_effort
        self.configuration = dict(model_path=str(model_path), max_input_tokens=max_input_tokens,
            max_new_tokens=max_new_tokens, reasoning_effort=reasoning_effort,
            dequantize=dequantize, observation="public_text", decoding="greedy",
            model_type=self.model.config.model_type, model_source=model_source,
            attention=attention)

    @staticmethod
    def initial():
        return dict(objects=PublicObjectMemory(), history=[], notebook="")

    def prompt(self, observation, state):
        public = _public_observation(observation)
        state["objects"].observe(public)
        protocol = action_protocol(public)
        protocol["schema"]["properties"]["memory"] = {"type": "string"}
        protocol["schema"]["required"].append("memory")
        payload = dict(public, output_protocol=protocol,
            previously_observed_objects=[{k: obj[k] for k in ("uuid", "name") if k in obj}
                                         for obj in state["objects"].objects.values()],
            actual_history=state["history"][-8:], notebook=state["notebook"])
        messages = [dict(role="system", content=SYSTEM),
                    dict(role="user", content=json.dumps(payload, ensure_ascii=False))]
        text = self.tokenizer.apply_chat_template(messages, tokenize=False,
            add_generation_prompt=True, reasoning_effort=self.reasoning_effort)
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
        result = json.loads(raw.strip())
        if not isinstance(result, dict):
            raise ValueError("Expected one native-action JSON object")
        action = {key: result[key] for key in ("action", "arg1", "arg2", "chosen_dialog_option_int")
                  if key in result and result[key] is not None}
        if not action:
            raise ValueError("No native action in final message")
        memory = result.get("memory", "")
        if not isinstance(memory, str):
            memory = json.dumps(memory, ensure_ascii=False)
        return action, memory

    @torch.inference_mode()
    def batch(self, observations, states):
        prompts = [self.prompt(obs, state) for obs, state in zip(observations, states)]
        inputs = self.tokenizer(prompts, return_tensors="pt", padding=True,
                                add_special_tokens=False).to(self.device)
        start = perf_counter()
        generated = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens,
            do_sample=False, pad_token_id=self.tokenizer.pad_token_id)
        seconds = perf_counter() - start
        result = []
        for output in generated[:, inputs["input_ids"].shape[1]:]:
            tokens = output.tolist()
            # Padding after a completed message is not part of the completion.
            while tokens and tokens[-1] == self.tokenizer.pad_token_id:
                tokens.pop()
            raw = self.tokenizer.decode(tokens, skip_special_tokens=False)
            row = dict(raw_output=raw, generated_tokens=len(tokens), batch_seconds=seconds,
                       batch_size=len(prompts))
            try:
                row["action"], row["memory"] = self.parse(raw)
            except (ValueError, TypeError) as error:
                row["error"] = str(error)
            result.append(row)
        return result

    @staticmethod
    def observe(state, transition, generated):
        state["notebook"] = generated["memory"]
        state["history"].append(dict(action=deepcopy(transition["action"]),
            result=deepcopy(transition["action_result"]),
            world_step=transition["next_observation"]["ui"].get("world_steps")))
