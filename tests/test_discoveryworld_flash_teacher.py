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
    def test_public_schema_feedback_preserves_all_calls_and_only_executes_corrected_action(self):
        obs = {"ui": {}, "actions": {"WAIT": {"args": []}}, "teleport_locations": {}}
        with patch.dict(os.environ, {"DASHSCOPE_API_KEY": "bailian-test-only"}):
            teacher = DeepSeekFlashTeacher(provider="bailian", max_input_tokens=4096,
                max_new_tokens=1024, reasoning_effort="high", temperature=.7, action_corrections=2)
        calls = []
        def reply(request, **kwargs):
            body = json.loads(request.data)
            calls.append(body)
            proposal = {"action": "WAIT", "memory": ""}
            if len(calls) == 1:
                proposal["extra"] = True
            else:
                self.assertIn("No action was executed", body["messages"][-1]["content"])
            return io.BytesIO(json.dumps({"model": body["model"],
                "usage": {"prompt_tokens": 200, "completion_tokens": 20, "total_tokens": 220},
                "choices": [{"message": {"content": json.dumps(proposal)}, "finish_reason": "stop"}]}).encode())
        with patch("experiments.discoveryworld.deepseek_teacher.urlopen", side_effect=reply):
            row = teacher.batch([obs], [teacher.initial()])[0]
        self.assertEqual(row["action"], {"action": "WAIT"})
        self.assertNotIn("error", row)
        self.assertEqual(row["usage"]["total_tokens"], 440)
        self.assertEqual(row["generated_tokens"], 40)
        self.assertEqual(len(row["calls"]), 2)
        self.assertIn("error", row["calls"][0])

    def test_bailian_uses_its_own_credentials_and_thinking_protocol(self):
        obs = {"ui": {}, "actions": {"WAIT": {"args": []}}, "teleport_locations": {}}
        with patch.dict(os.environ, {"DASHSCOPE_API_KEY": "bailian-test-only",
                                     "DEEPSEEK_API_KEY": "unused-test-only",
                                     "DASHSCOPE_BASE_URL": "https://provider.example/compatible-mode/v1/"}):
            teacher = DeepSeekFlashTeacher(provider="bailian", max_input_tokens=4096,
                                          max_new_tokens=1024, reasoning_effort="high", temperature=.7)
        def reply(request, **kwargs):
            self.assertEqual(request.full_url, "https://provider.example/compatible-mode/v1/chat/completions")
            self.assertEqual(request.get_header("Authorization"), "Bearer bailian-test-only")
            body = json.loads(request.data)
            self.assertEqual(body["model"], "deepseek-v4.1-flash-1017")
            self.assertTrue(body["enable_thinking"])
            self.assertNotIn("thinking", body)
            response = {"model": body["model"], "usage": {"prompt_tokens": 200, "completion_tokens": 20},
                        "choices": [{"message": {"content": json.dumps({"action": "WAIT", "memory": ""})},
                                     "finish_reason": "stop"}]}
            return io.BytesIO(json.dumps(response).encode())
        with patch("experiments.discoveryworld.deepseek_teacher.urlopen", side_effect=reply):
            row = teacher.batch([obs], [teacher.initial()])[0]
        self.assertEqual(row["action"], {"action": "WAIT"})
        self.assertEqual(teacher.configuration["inference_backend"], "Alibaba Cloud Bailian API")

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
