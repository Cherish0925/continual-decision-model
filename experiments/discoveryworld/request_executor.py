"""Execute a declared native operation using public access information.

The planner owns the operation and every argument. This executor can approach
those objects and acquire a required operand. It cannot choose another sample,
instrument, recipe, dialogue answer or scientific setting. Every prerequisite
is a real native action and consumes the common episode budget.
"""
from copy import deepcopy
from time import perf_counter

from cdm.digital import PublicObjectMemory, _objects


class ObjectRequestExecutor:
    interface = 'object_requests_v1'

    def __init__(self, environment, *, max_request_actions=12):
        if max_request_actions < 1:
            raise ValueError('A request needs a positive native action limit')
        self.environment = environment
        self.max_request_actions = max_request_actions
        self.objects = PublicObjectMemory()
        self.requests = 0
        self.events = []
        self.failure = None
        self._observe()

    def _observe(self):
        observation = self.environment.policy_input()
        self.objects.observe(observation)
        self.observation = observation
        return observation

    def policy_input(self):
        observation = deepcopy(self.observation)
        observation['remembered_objects'] = list(deepcopy(self.objects.objects).values())
        observation['execution'] = dict(interface=self.interface,
            max_request_actions=self.max_request_actions,
            remaining_native_actions=self.environment.max_steps-self.environment.steps)
        if self.events:
            observation['execution_events'] = deepcopy(self.events)
        if self.failure:
            observation['ui']['lastActionMessage'] = self.failure
        return observation

    def _accessible(self):
        return {h for h,o in _objects(self.policy_input()).items() if o['accessible']}

    def _inventory(self):
        return {o['uuid'] for o in self.observation['ui'].get('inventoryObjects', [])}

    def execute(self, request, *, on_native_transition=None):
        if self.environment.ended:
            raise RuntimeError('The episode has ended')
        request = deepcopy(request)
        before = self.policy_input()
        started = perf_counter()
        native = []
        requested_result = None
        failure = None

        def perform(action):
            if self.environment.ended or len(native) >= self.max_request_actions:
                return None
            transition = self.environment.step(action)
            native.append(transition)
            self._observe()
            if on_native_transition is not None:
                on_native_transition(transition)
            return transition

        def reach(handle):
            if handle in self._accessible():
                return True
            result = perform(dict(action='TELEPORT_TO_OBJECT', arg1=handle))
            return result is not None and handle in self._accessible()

        def hold(handle):
            if handle in self._inventory():
                return True
            if not reach(handle):
                return False
            result = perform(dict(action='PICKUP', arg1=handle))
            return result is not None and handle in self._inventory()

        operation = request.get('action')
        described = self.observation.get('actions', {}).get(operation, {})
        args = described.get('args', [])
        # Direction, location, feed and dialogue domains have their native
        # meanings. Only object-addressed interactions need access preparation.
        object_operation = operation is not None and operation not in {
            'MOVE_DIRECTION', 'ROTATE_DIRECTION', 'TELEPORT_TO_LOCATION',
            'TELEPORT_TO_OBJECT', 'DISCOVERY_FEED_GET_POST_BY_ID'}
        if object_operation and args:
            handles = [request.get(key) for key in args]
            if any(h not in self.objects.objects for h in handles):
                failure = 'The request refers to an object that has not been observed.'
            elif len(handles) == 1:
                accessible = hold(handles[0]) if operation == 'DROP' else reach(handles[0])
                if not accessible:
                    failure = 'The requested object could not be reached or acquired.'
            elif len(handles) == 2:
                first, second = handles
                if operation in ('PUT', 'GIVE'):
                    ready = hold(first) and reach(second)
                elif first in self._accessible() and second in self._accessible():
                    ready = True
                else:
                    ready = hold(first) and reach(second)
                    # A fixed instrument may not be portable. Transport the
                    # declared other operand; neither argument is substituted.
                    if not ready and not self.environment.ended:
                        ready = hold(second) and reach(first)
                if not ready:
                    failure = 'The two requested objects could not be made accessible together.'
        if failure is None:
            requested_result = perform(request)
        if requested_result is None and failure is None:
            failure = 'The native action budget ended before the requested operation.'

        # A rejected request still needs a transition for the planner. It costs
        # one native feed observation when no prerequisite advanced the world.
        if not native and not self.environment.ended:
            perform(dict(action='DISCOVERY_FEED_GET_UPDATES'))
        self.events = [dict(action=deepcopy(t['action']), result=deepcopy(t['action_result']),
                            feedback={k:t['next_observation']['ui'].get(k,'') for k in
                                      ('lastActionMessage','extended_action_message')},
                            world_step=t['next_observation']['ui'].get('world_steps')) for t in native]
        self.failure = failure
        result = deepcopy(requested_result['action_result']) if requested_result is not None else dict(
            success=False, errors=[failure], executor_message=failure)
        last = native[-1]
        after = self.policy_input()
        row = dict(episode_id=last['episode_id'], index=self.requests, observation=before,
            action=request, action_result=result, next_observation=after,
            training_reward=sum(t['training_reward'] for t in native),
            task_metrics=deepcopy(last['task_metrics']), terminated=last['terminated'], truncated=last['truncated'],
            timing=dict(last['timing']), executor=dict(interface=self.interface,
                native_actions=len(native), requested_operation_executed=requested_result is not None,
                fulfilled=bool(requested_result and result.get('success',False)),
                seconds=perf_counter()-started, events=deepcopy(self.events)))
        row['timing']['decision_seconds'] = native[0]['timing']['decision_seconds']
        self.requests += 1
        return row
