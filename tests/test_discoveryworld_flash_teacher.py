"""The hosted teacher sees the common public interface; only a finished legal proposal acts."""
import io
import json
import os
import unittest
from importlib.util import find_spec
from unittest.mock import patch

from experiments.discoveryworld.deepseek_teacher import DeepSeekFlashTeacher


@unittest.skipUnless(find_spec("jsonschema"), "Install the hosted-teacher extra")
class HostedTeacherTests(unittest.TestCase):
    def test_only_final_public_actions_reach_the_simulator(self):
        obs = {"ui": {"taskProgress": [{"taskName": "Experiment", "description": "Find the sample",
                                         "score": "offline-score-sentinel"}]},
               "actions": {"WAIT": {"args": []}}, "teleport_locations": {}, "seed": "offline-seed-sentinel"}
        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "test-only"}):
            teacher = DeepSeekFlashTeacher(max_input_tokens=4096, max_new_tokens=512,
                                          reasoning_effort="low", temperature=.7)
        def propose(value, finish, tokens=350):
            def reply(request, **kwargs):
                self.assertEqual(request.full_url, "https://api.deepseek.com/chat/completions")
                body = json.loads(request.data)
                self.assertEqual(body["model"], "deepseek-flash")
                self.assertEqual(body["response_format"], {"type": "json_object"})
                self.assertNotIn("offline-score-sentinel", str(body))
                self.assertNotIn("offline-seed-sentinel", str(body))
                response = {"model": "deepseek-flash", "usage": {"prompt_tokens": tokens, "completion_tokens": 23},
                            "choices": [{"message": {"content": json.dumps(value), "reasoning_content": "not-an-action"},
                                         "finish_reason": finish}]}
                return io.BytesIO(json.dumps(response).encode())
            with patch("experiments.discoveryworld.deepseek_teacher.urlopen", side_effect=reply):
                return teacher.batch([obs], [teacher.initial()])[0]
        valid = {"action": "WAIT", "memory": "Awaiting a measurement."}
        row = propose(valid, "stop")
        self.assertEqual(row["action"], {"action": "WAIT"})
        self.assertEqual(row["usage"]["prompt_tokens"], 350)
        self.assertNotIn("not-an-action", str(row))
        for bad in [propose({"action": "DOES_NOT_EXIST", "memory": ""}, "stop"),
                    propose(valid, "length"), propose(valid, "stop", tokens=4097)]:
            self.assertIn("error", bad)
            self.assertNotIn("action", bad)
