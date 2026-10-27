"""Decode exactly one action from the same public candidates used by CDM.

The model chooses tokens. The trie supplies syntax and argument domains, never
task answers, action values or a replacement action. Unsuccessful interactions
with accessible objects remain possible and are evaluated by the simulator.
"""
from __future__ import annotations

import json


class NativeActionTrie:
    def __init__(self, candidates, tokenizer, *, input_tokens: int, eos_ids, max_new_tokens: int):
        self.root = {}
        self.input_tokens = input_tokens
        self.eos_ids = [eos_ids] if isinstance(eos_ids, int) else list(eos_ids or [])
        if not self.eos_ids:
            raise ValueError("Constrained native actions require the model's EOS token")
        self.actions = {}
        self.maximum_tokens = 0
        for candidate in candidates:
            action = candidate.action
            text = json.dumps(action, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
            ids = tokenizer.encode(text, add_special_tokens=False)
            if tokenizer.decode(ids, skip_special_tokens=True) != text:
                raise ValueError("Tokenizer must exactly represent native action JSON")
            self.maximum_tokens = max(self.maximum_tokens, len(ids) + 1)
            node = self.root
            for token in ids:
                node = node.setdefault(token, {})
            node[None] = True
            self.actions[text] = action
        if not self.actions:
            raise ValueError("The public interface supplied no candidate actions")
        if max_new_tokens < self.maximum_tokens:
            raise ValueError(f"All native candidates require a budget of at least {self.maximum_tokens} tokens")

    def __call__(self, batch_id, input_ids):
        node = self.root
        for token in input_ids[self.input_tokens:].tolist():
            node = node[token]
        return [token for token in node if token is not None] + (self.eos_ids if None in node else [])

    def contains(self, action):
        return json.dumps(action, sort_keys=True, separators=(",", ":"), ensure_ascii=False) in self.actions
