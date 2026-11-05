"""Native environment with identical object-request execution for all planners."""
import json

from .full_suite import seeded_object_initialization
from .request_executor import ObjectRequestExecutor
from .structured_observation import StructuredDiscoveryWorldEnv


class RequestEnvironment:
    def __init__(self, output, max_steps=1000, *, max_request_actions=12):
        self.native = StructuredDiscoveryWorldEnv(output, max_steps)
        self.max_request_actions = max_request_actions
        self.record_native = True

    def __getattr__(self, key):
        return getattr(self.native, key)

    @property
    def decision_steps(self):
        return self.executor.requests

    @property
    def episode_id(self):
        return self.native.episode_id

    @episode_id.setter
    def episode_id(self, value):
        self.native.episode_id = value

    def reset(self, scenario, difficulty, seed):
        # Seed construction itself: post-reset RNG restoration alone cannot
        # recover object UUIDs or attributes sampled during world creation.
        with seeded_object_initialization(seed):
            self.native.reset(scenario, difficulty, seed)
        # The upstream action description mentions agents only, but its TALK
        # implementation opens any object's native dialog/control interface.
        # Give every planner the same accurate API documentation; no device,
        # scientific answer or control choice is identified here.
        self.native.actions['TALK']['desc'] = 'Open the dialogue or control menu of an object (arg1), including an interactive device or agent.'
        self.native.configuration.update(action_interface=ObjectRequestExecutor.interface,
            max_request_actions=self.max_request_actions, object_rng_seed=seed)
        self.executor = ObjectRequestExecutor(self.native, max_request_actions=self.max_request_actions)
        return self.policy_input()

    def policy_input(self):
        return self.executor.policy_input()

    def step(self, request):
        def record(transition):
            if self.record_native:
                with (self.native.output / 'native-actions.jsonl').open('a') as stream:
                    stream.write(json.dumps(transition, ensure_ascii=False)+'\n')
        return self.executor.execute(request, on_native_transition=record)
