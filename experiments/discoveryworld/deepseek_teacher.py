"""Hosted Flash teacher using the same public observations and native action domains."""
import json
import os
from time import sleep
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from .native_teacher import NativeTeacher


class DeepSeekFlashTeacher(NativeTeacher):
    endpoint = "https://api.deepseek.com"
    served_model = "deepseek-flash"
    api_style = "chat"
    candidate_actions = True
    tokenizer = None  # The server reports the input count for this tokenizer.

    def __init__(self, *, max_input_tokens, max_new_tokens, reasoning_effort, temperature,
                 provider="deepseek", model=None, action_corrections=0):
        from jsonschema import Draft202012Validator
        self.validator = Draft202012Validator
        self.provider = provider
        if provider == "bailian":
            self.endpoint = os.environ.get("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1").rstrip("/")
            self.served_model = model or "deepseek-v4.1-flash-1017"
            credential = "DASHSCOPE_API_KEY"
            backend = "Alibaba Cloud Bailian API"
        elif provider == "deepseek":
            self.served_model = model or "deepseek-flash"
            credential = "DEEPSEEK_API_KEY"
            backend = "DeepSeek API"
        else:
            raise ValueError("Choose deepseek or bailian as the hosted provider")
        self._key = os.environ.get(credential)
        if not self._key:
            raise ValueError(f"Set {credential} to collect hosted demonstrations.")
        if min(max_input_tokens, max_new_tokens) <= 0:
            raise ValueError("Token limits must be positive.")
        self.max_input_tokens, self.max_new_tokens = max_input_tokens, max_new_tokens
        self.reasoning_effort, self.temperature = reasoning_effort, temperature
        if action_corrections < 0:
            raise ValueError("Action corrections must be nonnegative")
        self.action_corrections = action_corrections
        self.configuration = dict(model_source=self.served_model, endpoint=self.endpoint,
            inference_backend=backend, prompt_version="native_public_domains_v7",
            observation="public_text", action_interface="observed_native_action_domains",
            max_input_tokens=max_input_tokens, max_new_tokens=max_new_tokens,
            reasoning_effort=reasoning_effort, temperature=temperature if reasoning_effort == "none" else None)
        if action_corrections:
            self.configuration["action_corrections"] = action_corrections

    def chat(self, state):
        messages = list(state["messages"])
        calls = []
        for correction in range(self.action_corrections + 1):
            raw, tokens, details = self._chat_once({**state, "messages": messages})
            calls.append(dict(raw_output=raw, **details))
            if correction == self.action_corrections or details.get("error") not in (
                    "Proposal is outside the current public action domains", "Expected one JSON action object"):
                break
            # Return the public interface error to the teacher. The simulator
            # has not acted, and no successful action or task answer is supplied.
            messages.extend([dict(role="assistant", content=raw), dict(role="user", content=
                "No action was executed. The reply does not match the output schema. "
                "Return one JSON object with exactly the required fields, observed arguments, "
                "and a memory of at most 1000 characters. Do not add other fields. "
                "During dialogue use the displayed dialogue option instead of an action.")])
        if len(calls) > 1:
            details = {**details, "calls": calls, "action_corrections": len(calls)-1,
                "usage": {key: sum(call["usage"].get(key, 0) for call in calls)
                          for key in ("prompt_tokens", "completion_tokens", "total_tokens")}}
            tokens = details["usage"]["completion_tokens"]
        return raw, tokens, details

    def _chat_once(self, state):
        thinking = self.reasoning_effort != "none"
        body = dict(model=self.served_model, messages=state["messages"],
                    response_format={"type": "json_object"}, max_tokens=self.max_new_tokens)
        if self.provider == "bailian":
            body["enable_thinking"] = thinking
        else:
            body["thinking"] = {"type": "enabled" if thinking else "disabled"}
        if thinking:
            body["reasoning_effort"] = self.reasoning_effort
        else:
            body["temperature"] = self.temperature
        request = Request(self.endpoint + "/chat/completions", json.dumps(body).encode(),
                          {"Content-Type": "application/json", "Authorization": "Bearer " + self._key})
        for attempt in range(6):
            try:
                with urlopen(request, timeout=600) as stream:
                    response = json.load(stream)
                break
            except HTTPError as error:
                if error.code not in (429, 502, 503, 504) or attempt == 5:
                    raise
                delay = error.headers.get("Retry-After", "")
                sleep(float(delay) if delay.replace(".", "", 1).isdigit() else min(2 ** attempt, 30))
        choice, usage = response["choices"][0], response["usage"]
        raw = choice["message"].get("content") or ""
        details = dict(provider_model=response["model"], provider_created=response.get("created"),
                       system_fingerprint=response.get("system_fingerprint"),
                       usage=usage, finish_reason=choice["finish_reason"], transport_retries=attempt)
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
