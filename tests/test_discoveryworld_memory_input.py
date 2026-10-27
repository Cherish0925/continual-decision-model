"""Recorded measurements reach the reused core without changing legacy inputs."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

import torch

from cdm.core import ContinualCore, CoreConfig
from cdm.digital import DigitalConfig, DigitalPolicy, public_candidates
from experiments.discoveryworld.features import FeatureCache, FeatureConfig, batch_features, load_features
from experiments.discoveryworld.measurements import MeasurementLog


class MeasurementInputTests(unittest.TestCase):
    def test_real_history_changes_actions_and_trains_existing_record_reader(self):
        torch.set_num_threads(1)
        torch.manual_seed(4)
        fixture = Path(__file__).parent / 'fixtures/discoveryworld_measurements.json'
        cases = {case['label']: case['transition'] for case in json.loads(fixture.read_text())['cases']}
        log = MeasurementLog()
        for name in ('radiation_a1', 'radiation_a2', 'radiation_a3'):
            log.append_transition(cases[name])
        observation = deepcopy(cases['radiation_a3']['next_observation'])
        observation['actions'] = {'ROTATE_DIRECTION': {'args': ['arg1']}}
        observation['teleport_locations'] = {}
        observation['measurement_records'] = log.summaries()
        candidates = public_candidates(observation)
        empty = {key: value for key, value in observation.items() if key != 'measurement_records'}
        with tempfile.TemporaryDirectory() as folder:
            old = FeatureCache(folder, FeatureConfig(mode='lexical'))
            enabled = FeatureCache(folder, FeatureConfig(mode='lexical', measurement_memory=True))
            self.assertEqual(old.encode(observation, candidates), old.encode(empty, candidates))
            self.assertEqual(old.tower_directory, enabled.tower_directory)
            rich = load_features(str(enabled.encode(observation, candidates)))
            plain = load_features(str(enabled.encode(empty, candidates)))
            obs, actions = batch_features([rich, plain], 'cpu')
            self.assertEqual(obs['measurement_present'].tolist(), [[True], [False]])
            core = ContinualCore(CoreConfig(event_dim=16, record_dim=8, actions=1, width=32,
                                           slots=2, heads=4, records=8, rank=2, hidden=32,
                                           action_width=16), torch.zeros(1, 1))
            model = DigitalPolicy(DigitalConfig(width=32, text_dim=256, image_dim=256), core).eval()
            keys = set(model.state_dict())
            logits, _, _ = model(obs, actions, model.initial(2, 'cpu'))
            self.assertFalse(torch.allclose(logits[0], logits[1]))
            logits[0, 0].backward()
            self.assertGreater(sum(float(p.grad.abs().sum()) for p in core.record.parameters() if p.grad is not None), 0.)
            single_obs, single_actions = batch_features([plain], 'cpu')
            single, _, _ = model(single_obs, single_actions, model.initial(1, 'cpu'))
            torch.testing.assert_close(logits[1].detach(), single[0].detach(), rtol=1e-5, atol=1e-6)
            self.assertEqual(set(model.state_dict()), keys)


if __name__ == '__main__':
    unittest.main()
