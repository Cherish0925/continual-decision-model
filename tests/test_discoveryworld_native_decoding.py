"""Native-action syntax does not invent arguments or choose successful actions."""
import json
from types import SimpleNamespace
import unittest

import torch

from experiments.discoveryworld.native_decoding import NativeActionTrie


class CharacterCodec:
    def encode(self, text, **kwargs):
        return list(map(ord, text))

    def decode(self, ids, **kwargs):
        return ''.join(map(chr, ids))


class NativeDecodingTests(unittest.TestCase):
    def test_only_complete_public_actions_can_terminate(self):
        actions = [{'action': 'USE', 'arg1': 12, 'arg2': 34},
                   {'action': 'USE', 'arg1': 56, 'arg2': 78},
                   {'action': 'MOVE_DIRECTION', 'arg1': 'east'}]
        trie = NativeActionTrie([SimpleNamespace(action=a) for a in actions], CharacterCodec(),
                                input_tokens=2, eos_ids=1, max_new_tokens=100)
        for action in actions:
            prefix = [99, 99]
            for token in CharacterCodec().encode(json.dumps(action, sort_keys=True, separators=(',', ':'))):
                allowed = trie(0, torch.tensor(prefix))
                self.assertNotIn(1, allowed)
                self.assertIn(token, allowed)
                prefix.append(token)
            self.assertEqual(trie(0, torch.tensor(prefix)), [1])
        self.assertFalse(trie.contains({'action': 'USE', 'arg1': 12, 'arg2': 78}))
        self.assertFalse(trie.contains({'action': 'MOVE_DIRECTION', 'arg1': 'north'}))

    def test_dialogue_uses_real_option_ids_and_budget_does_not_remove_long_candidates(self):
        candidates = [SimpleNamespace(action={'chosen_dialog_option_int': i}) for i in [2, 9]]
        trie = NativeActionTrie(candidates, CharacterCodec(), input_tokens=0, eos_ids=[1, 3], max_new_tokens=80)
        self.assertTrue(trie.contains({'chosen_dialog_option_int': 9}))
        self.assertFalse(trie.contains({'chosen_dialog_option_int': 1}))
        with self.assertRaises(ValueError):
            NativeActionTrie(candidates, CharacterCodec(), input_tokens=0, eos_ids=1, max_new_tokens=4)


if __name__ == '__main__':
    unittest.main()
