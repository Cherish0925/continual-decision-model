"""Public object equality links, unchanged input tensors and legacy cache compatibility."""
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import re
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from experiments.discoveryworld.features import (FeatureCache, FeatureConfig, _digest,
                                                  batch_features, load_features)
from experiments.discoveryworld.measurements import MeasurementLog


def candidate(action, arguments):
    # The separate policy change adds this tuple to NativeCandidate. These are
    # object arguments only, not arbitrary numeric dialogue/direction arguments.
    return SimpleNamespace(action=action, object_arguments=arguments,
                           text=action.get('action', 'DIALOG'), numeric=(0.,) * 8)


class ObjectRelationFeatureTests(unittest.TestCase):
    def setUp(self):
        path = Path(__file__).parent / 'fixtures/discoveryworld_measurements.json'
        rows = {case['label']: case['transition'] for case in json.loads(path.read_text())['cases']}
        log = MeasurementLog()
        for label in ('density', 'reactor1', 'reactor3'):
            log.append_transition(rows[label])
        self.observation = deepcopy(rows['reactor3']['next_observation'])
        # Deliberately reverse the supplied view: links must follow the same
        # chronological sort as the encoded numeric/text measurement tokens.
        self.observation['measurement_records'] = [asdict(row) for row in reversed(log.summaries())]
        self.observation['measurement_dialog_owner'] = log.dialog_owner
        self.candidates = [candidate({'action': 'USE', 'arg1': 18629, 'arg2': 63031}, (18629, 63031)),
                           candidate({'action': 'TALK', 'arg1': 10581}, (10581, None)),
                           candidate({'chosen_dialog_option_int': 5}, (None, None))]

    def test_real_binding_follows_measurement_token_order_without_changing_labels(self):
        with TemporaryDirectory() as folder:
            cache = FeatureCache(folder, FeatureConfig(mode='lexical', measurement_memory=True,
                                                       object_relations=True))
            with patch.object(cache, 'texts', wraps=cache.texts) as texts:
                data = load_features(str(cache.encode(self.observation, self.candidates)))
            relations = data['candidates']['measurement_relations']
            self.assertEqual(relations.shape, (3, 3, 8))
            self.assertEqual(relations[0, 0].tolist(), [1, 0, 0, 0, 1, 0, 0, 1])
            self.assertEqual(relations[0, 1].tolist(), [0] * 8)
            self.assertEqual(relations[0, 2].tolist(), [0, 0, 0, 0, 0, 1, 1, 0])
            self.assertEqual(relations[1, 2].tolist(), [1, 1, 0, 0, 0, 0, 1, 0])
            self.assertEqual(relations[2, :, :6].count_nonzero(), 0)
            labels = texts.call_args_list[-1].args[0]
            self.assertTrue(all('observed contents' not in label for label in labels))
            # A past or remembered location is never a current containment edge.
            hidden = deepcopy(self.observation)
            for key in ('inventoryObjects', 'accessibleEnvironmentObjects'):
                hidden['ui'][key] = [o for o in hidden['ui'][key] if o['uuid'] != 63031]
            for direction, group in hidden['ui']['nearbyObjects']['objects'].items():
                hidden['ui']['nearbyObjects']['objects'][direction] = [o for o in group if o['uuid'] != 63031]
            hidden['remembered_objects'] = [{'uuid': 63031, 'name': 'quantum crystal 4',
                                            'description': '(in reactor [uuid: 10581])'}]
            other = load_features(str(cache.encode(hidden, self.candidates)))
            self.assertEqual(other['candidates']['measurement_relations'][..., 7].count_nonzero(), 0)

    def test_same_raw_observation_preserves_every_existing_tensor_when_enabled(self):
        with TemporaryDirectory() as folder:
            old = FeatureCache(folder, FeatureConfig(mode='lexical', measurement_memory=True))
            new = FeatureCache(folder, FeatureConfig(mode='lexical', measurement_memory=True,
                                                     object_relations=True))
            old_path = old.encode(self.observation, self.candidates)
            new_path = new.encode(self.observation, self.candidates)
            self.assertNotEqual(old_path, new_path)
            before, after = load_features(str(old_path)), load_features(str(new_path))
            self.assertEqual(set(before['observation']), set(after['observation']))
            self.assertEqual(set(after['candidates']), set(before['candidates']) | {'measurement_relations'})
            for section in ('observation', 'candidates'):
                for key, value in before[section].items():
                    torch.testing.assert_close(value, after[section][key], rtol=0, atol=0)

    def test_uuid_relabelling_and_candidate_permutation_only_change_routing(self):
        def rename(value, key=None):
            if isinstance(value, dict):
                return {k: rename(v, k) for k, v in value.items()}
            if isinstance(value, (tuple, list)):
                return [rename(v, key) for v in value]
            if key in ('uuid', 'arg1', 'arg2', 'instrument_handle', 'target_handle',
                       'contained_objects', 'measurement_dialog_owner') and isinstance(value, int):
                return value + 700000
            if isinstance(value, str):
                return re.sub(r'\[uuid: (\d+)\]', lambda m: f'[uuid: {int(m.group(1)) + 700000}]', value)
            return value

        with TemporaryDirectory() as folder:
            cache = FeatureCache(folder, FeatureConfig(mode='lexical', measurement_memory=True,
                                                       object_relations=True))
            original = load_features(str(cache.encode(self.observation, self.candidates)))
            order = [2, 0, 1]
            candidates = [candidate(rename(self.candidates[i].action),
                                    tuple(x + 700000 if x is not None else None
                                          for x in self.candidates[i].object_arguments)) for i in order]
            renamed = load_features(str(cache.encode(rename(self.observation), candidates)))
            for key, value in original['observation'].items():
                torch.testing.assert_close(value, renamed['observation'][key], rtol=0, atol=0)
            for key, value in original['candidates'].items():
                torch.testing.assert_close(value[order], renamed['candidates'][key], rtol=0, atol=0)

    def test_masked_control_changes_only_relation_values_and_cache_identity(self):
        with self.assertRaisesRegex(ValueError, 'mask_object_relations requires object_relations'):
            FeatureConfig(measurement_memory=True, mask_object_relations=True)
        with TemporaryDirectory() as folder:
            config = FeatureConfig(mode='lexical', measurement_memory=True, object_relations=True)
            linked = FeatureCache(folder, config)
            masked = FeatureCache(folder, FeatureConfig(**{**asdict(config), 'mask_object_relations': True}))
            previous_identity = asdict(config)
            previous_identity.pop('mask_object_relations')
            self.assertEqual(linked.directory, Path(folder) / _digest(previous_identity)[:20])
            self.assertNotIn('mask_object_relations', json.loads((linked.directory / 'metadata.json').read_text()))
            self.assertEqual(linked.tower_directory, masked.tower_directory)
            linked_path = linked.encode(self.observation, self.candidates)
            masked_path = masked.encode(self.observation, self.candidates)
            self.assertNotEqual(linked_path, masked_path)
            normal, control = load_features(str(linked_path)), load_features(str(masked_path))
            for section in ('observation', 'candidates'):
                self.assertEqual(set(normal[section]), set(control[section]))
                for key, value in normal[section].items():
                    if key == 'measurement_relations':
                        self.assertGreater(value.count_nonzero(), 0)
                        self.assertEqual(value.shape, control[section][key].shape)
                        self.assertEqual(control[section][key].count_nonzero(), 0)
                    else:
                        torch.testing.assert_close(value, control[section][key], rtol=0, atol=0)

    def test_padding_handles_empty_records_and_missing_legacy_relations(self):
        with TemporaryDirectory() as folder:
            enabled = FeatureCache(folder, FeatureConfig(mode='lexical', measurement_memory=True,
                                                         object_relations=True))
            legacy = FeatureCache(folder, FeatureConfig(mode='lexical', measurement_memory=True))
            one, empty = deepcopy(self.observation), deepcopy(self.observation)
            one['measurement_records'] = one['measurement_records'][:1]
            empty['measurement_records'] = []
            rich = load_features(str(enabled.encode(self.observation, self.candidates)))
            single = load_features(str(enabled.encode(one, self.candidates[:1])))
            absent = load_features(str(enabled.encode(empty, self.candidates[:2])))
            old = load_features(str(legacy.encode(self.observation, self.candidates)))
            self.assertEqual(absent['candidates']['measurement_relations'].shape, (2, 0, 8))
            obs, actions = batch_features([rich, single, absent, old], 'cpu')
            relation = actions['measurement_relations']
            self.assertEqual(relation.shape, (4, 3, 3, 8))
            torch.testing.assert_close(relation[0], rich['candidates']['measurement_relations'])
            self.assertEqual(relation[1, 1:].count_nonzero(), 0)
            self.assertEqual(relation[1, :, 1:].count_nonzero(), 0)
            self.assertEqual(relation[2:].count_nonzero(), 0)
            self.assertEqual(obs['measurement_present'].tolist(),
                             [[True] * 3, [True, False, False], [False] * 3, [True] * 3])
            _, no_records = batch_features([absent], 'cpu')
            self.assertEqual(no_records['measurement_relations'].shape, (1, 2, 0, 8))
            _, legacy_only = batch_features([old], 'cpu')
            self.assertNotIn('measurement_relations', legacy_only)

    def test_off_mode_preserves_previous_cache_identity_and_ignores_relation_inputs(self):
        with self.assertRaisesRegex(ValueError, 'requires measurement_memory'):
            FeatureConfig(object_relations=True)
        with TemporaryDirectory() as folder:
            for memory in (False, True):
                config = FeatureConfig(mode='lexical', measurement_memory=memory)
                cache = FeatureCache(folder, config)
                previous_identity = asdict(config)
                previous_identity.pop('object_relations')
                previous_identity.pop('mask_object_relations')
                if not memory:
                    previous_identity.pop('measurement_memory')
                self.assertEqual(cache.directory, Path(folder) / _digest(previous_identity)[:20])
                self.assertNotIn('object_relations', json.loads((cache.directory / 'metadata.json').read_text()))
                path = cache.encode(self.observation, self.candidates)
                changed = deepcopy(self.observation)
                changed['measurement_dialog_owner'] = None
                for row in changed['measurement_records']:
                    row['contained_objects'] = ()
                with patch.object(cache, 'texts', side_effect=AssertionError('Legacy cache missed')):
                    self.assertEqual(path, cache.encode(changed, self.candidates))
                self.assertNotIn('measurement_relations', load_features(str(path))['candidates'])
            linked = FeatureCache(folder, FeatureConfig(mode='lexical', measurement_memory=True,
                                                        object_relations=True))
            self.assertEqual(linked.tower_directory, cache.tower_directory)
            changed = deepcopy(self.observation)
            changed['measurement_dialog_owner'] = None
            self.assertNotEqual(linked.encode(self.observation, self.candidates),
                                linked.encode(changed, self.candidates))


if __name__ == '__main__':
    unittest.main()
