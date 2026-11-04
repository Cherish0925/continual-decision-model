"""Shared weights must not leak state or answers between scientific instances."""
from dataclasses import asdict
from importlib.util import find_spec
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import subprocess
import sys
import unittest
from unittest.mock import patch

import torch

from cdm.core import ContinualCore, CoreConfig
from experiments.discoveryworld import full_suite
from experiments.discoveryworld.collect_suite import training_cases, collect
from experiments.discoveryworld.native_teacher import NativeTeacher
from experiments.discoveryworld.train import create_model


class FullSuiteTests(unittest.TestCase):
    @unittest.skipUnless(find_spec('discoveryworld'), 'Install the pinned native simulator')
    def test_lost_simulator_is_recorded_and_other_native_cases_continue(self):
        class Teacher:
            endpoint = 'test'
            configuration = {'prompt_version': 'test'}
            def initial(self): return {}
            def observe(self, *args): pass
            def batch(self, observations, states):
                return [dict(action={'action': 'ROTATE_DIRECTION', 'arg1': 'north'}, memory='')]
        start = subprocess.Popen
        launched = []
        def launch(command, **kwargs):
            if not launched:
                command = [sys.executable, '-c', 'raise SystemExit(9)']
            process = start(command, **kwargs)
            launched.append(process)
            return process
        with TemporaryDirectory() as folder:
            args = SimpleNamespace(output=Path(folder), seeds=[1000], difficulties=['Easy'],
                themes=full_suite.THEMES[:2], max_steps=2, small_skills=False, resume=False,
                workers=1, max_new_tokens=100, max_input_tokens=1000, temperature=0,
                reasoning_effort='none', model='test', device='cpu', dequantize=False,
                attention=None, endpoint='test', served_model='test', model_source='test',
                api_style='chat')
            with patch('experiments.discoveryworld.native_teacher.NativeTeacher', return_value=Teacher()), \
                 patch('experiments.discoveryworld.collect_suite.subprocess.Popen', side_effect=launch):
                rows = collect(args)
            self.assertEqual([r['status'] for r in rows], ['environment_error', 'evaluated'])
            self.assertEqual([r['steps'] for r in rows], [0, 2])
            trace = Path(folder)/rows[1]['id']/'trajectory.jsonl'
            self.assertEqual(len([json.loads(line) for line in trace.read_text().splitlines()]), 2)
            self.assertTrue(all(p.poll() is not None for p in launched))

    def test_basic_skill_demonstrations_exclude_scientific_solutions(self):
        cases = training_cases([1000, 1001], small_skills=True)
        self.assertEqual(len(cases), 20)
        self.assertTrue(all(row['theme'].startswith('Small Skills:') for row in cases))
        self.assertTrue(all(row['difficulty'] == 'Normal' and row['max_steps'] == 100 for row in cases))
        self.assertFalse(set(full_suite.THEMES) & {row['theme'] for row in cases})

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

    def test_teacher_keeps_observed_argument_handles_when_they_leave_reach(self):
        teacher = NativeTeacher.__new__(NativeTeacher)
        teacher.candidate_actions, teacher.endpoint, teacher.api_style = True, 'local', 'chat'
        teacher.max_input_tokens = 10000
        teacher.tokenizer = SimpleNamespace(encode=lambda text, **kwargs: text.split())
        state = teacher.initial()
        state['objects'].observe({'ui': {'accessibleEnvironmentObjects': [
            {'uuid': 12, 'name': 'instrument'}, {'uuid': 34, 'name': 'sample'}]}})
        # Only the sample is in reach now. A two-object action is absent from
        # the student's candidate set, but the teacher's intended instrument
        # must not be silently replaced by the sample in structured decoding.
        observation = {'ui': {'accessibleEnvironmentObjects': [{'uuid': 34, 'name': 'sample'}]},
                       'actions': {'USE': {'args': ['arg1', 'arg2']},
                                   'TELEPORT_TO_OBJECT': {'args': ['arg1']}}}
        teacher.prompt(observation, state)
        use = next(row for row in state['output_schema']['anyOf']
                   if row['properties'].get('action', {}).get('enum') == ['USE'])
        self.assertEqual(use['properties']['arg1'], {'$ref': '#/$defs/observed_object'})
        self.assertEqual(use['properties']['arg2'], {'$ref': '#/$defs/observed_object'})
        self.assertEqual(set(state['output_schema']['$defs']['observed_object']['enum']), {12, 34})
        self.assertFalse(any(action.get('action') == 'USE' for action in state['candidates']))


if __name__ == '__main__':
    unittest.main()
