"""Shared weights must not leak state or answers between scientific instances."""
from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from cdm.core import ContinualCore, CoreConfig
from experiments.discoveryworld import full_suite
from experiments.discoveryworld.collect_suite import training_cases
from experiments.discoveryworld.native_teacher import NativeTeacher
from experiments.discoveryworld.train import create_model


class FullSuiteTests(unittest.TestCase):
    def test_one_model_resets_between_instances_and_training_is_disjoint(self):
        class Policy:
            state = 0
            def reset(self):
                self.state = 0
        made = []
        def create(config):
            made.append(Policy())
            return made[-1]
        cfg = dict(policy="fixture:create", config={})
        full_suite._POLICIES.clear()
        try:
            with patch.object(full_suite, "import_module", return_value=SimpleNamespace(create=create)):
                first = full_suite.reusable_policy(cfg)
                first.state = 123
                second = full_suite.reusable_policy(cfg)
                self.assertIs(first, second)
                self.assertEqual(second.state, 0)
                self.assertEqual(len(made), 1)
        finally:
            full_suite._POLICIES.clear()
        official = full_suite.official_cases()
        training = training_cases([1000, 1001, 1002, 1003])
        identity = lambda rows: {(r['theme'], r['difficulty'], r['seed']) for r in rows}
        self.assertEqual(len(official), 120)
        self.assertEqual(len(training), 96)
        self.assertFalse(identity(official) & identity(training))

    def test_shared_scientific_core_preserves_original_configuration_and_tensors(self):
        torch.set_num_threads(1)
        config = CoreConfig(event_dim=7, record_dim=8, actions=3, width=24,
                            slots=3, heads=4, hidden=24, action_width=12)
        core = ContinualCore(config, torch.tensor([-1., 0., 1.]))
        with TemporaryDirectory() as directory:
            path = Path(directory) / 'source.pt'
            torch.save(dict(kind='shared_experiment_core_v1', core_config=asdict(config),
                            core=core.state_dict()), path)
            args = SimpleNamespace(width=128, refinements=1, object_relations=False,
                                   initial_checkpoint=path, measurement_memory=True)
            model = create_model(args, 12, 10)
            self.assertEqual(model.config.width, 24)
            self.assertEqual(model.core.config, config)
            for name, value in core.state_dict().items():
                torch.testing.assert_close(model.core.state_dict()[name], value)

    def test_teacher_action_comes_only_from_complete_final_message(self):
        response = ('<|channel|>analysis<|message|>{"action":"READ","arg1":99}'
                    '<|end|><|start|>assistant<|channel|>final<|message|>'
                    '{"action":"READ","arg1":12,"arg2":null,"memory":"unmeasured"}<|return|>')
        action, memory = NativeTeacher.parse(response)
        self.assertEqual(action, dict(action='READ', arg1=12))
        self.assertEqual(memory, 'unmeasured')
        formatted = response.replace('<|channel|>final<|message|>',
                                     '<|channel|>final <|constrain|>json<|message|>')
        self.assertEqual(NativeTeacher.parse(formatted), (action, memory))
        with self.assertRaises(ValueError):
            NativeTeacher.parse('<|channel|>analysis<|message|>{"action":"READ","arg1":99}')


if __name__ == '__main__':
    unittest.main()
