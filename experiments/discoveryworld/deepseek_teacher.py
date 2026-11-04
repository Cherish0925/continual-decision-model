"""Hosted Flash teacher using the same public observations and native action domains."""
import json
import os
from urllib.request import Request, urlopen

from .native_teacher import NativeTeacher


class DeepSeekFlashTeacher(NativeTeacher):
    endpoint = "https://api.deepseek.com"
    served_model = "deepseek-flash"
    api_style = "chat"
    candidate_actions = True
    tokenizer = None  # The server reports the input count for this tokenizer.

    def __init__(self, *, max_input_tokens, max_new_tokens, reasoning_effort, temperature):
        from jsonschema import Draft202012Validator
        self.validator = Draft202012Validator
        self._key = os.environ.get("DEEPSEEK_API_KEY")
        if not self._key:
            raise ValueError("Set DEEPSEEK_API_KEY to collect hosted demonstrations.")
        if min(max_input_tokens, max_new_tokens) <= 0:
            raise ValueError("Token limits must be positive.")
        self.max_input_tokens, self.max_new_tokens = max_input_tokens, max_new_tokens
        self.reasoning_effort, self.temperature = reasoning_effort, temperature
        self.configuration = dict(model_source=self.served_model, endpoint=self.endpoint,
            inference_backend="DeepSeek API", prompt_version="native_public_domains_v7",
            observation="public_text", action_interface="observed_native_action_domains",
            max_input_tokens=max_input_tokens, max_new_tokens=max_new_tokens,
            reasoning_effort=reasoning_effort, temperature=temperature if reasoning_effort == "none" else None)

    def chat(self, state):
        thinking = self.reasoning_effort != "none"
        body = dict(model=self.served_model, messages=state["messages"],
                    response_format={"type": "json_object"}, max_tokens=self.max_new_tokens,
                    thinking={"type": "enabled" if thinking else "disabled"})
        if thinking:
            body["reasoning_effort"] = self.reasoning_effort
        else:
            body["temperature"] = self.temperature
        request = Request(self.endpoint + "/chat/completions", json.dumps(body).encode(),
                          {"Content-Type": "application/json", "Authorization": "Bearer " + self._key})
        with urlopen(request, timeout=600) as stream:
            response = json.load(stream)
        choice, usage = response["choices"][0], response["usage"]
        raw = choice["message"].get("content") or ""
        details = dict(provider_model=response["model"], provider_created=response.get("created"),
                       system_fingerprint=response.get("system_fingerprint"),
                       usage=usage, finish_reason=choice["finish_reason"])
        if usage["prompt_tokens"] > self.max_input_tokens:
            details["error"] = "Provider-reported input exceeds the study token budget"
        elif choice["finish_reason"] != "stop":
            details["error"] = "Incomplete action message: " + str(choice["finish_reason"])
        else:
            try:
                proposal = json.loads(raw)
                if not self.validator(state["output_schema"]).is_valid(proposal):
                    details["error"] = "Proposal is outside the current public action domains"
            except (ValueError, TypeError):
                details["error"] = "Expected one JSON action object"
        return raw, usage["completion_tokens"], details
