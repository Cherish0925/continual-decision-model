"""Language planner for the same object requests used by recurrent policies."""
from copy import deepcopy
import json

from .deepseek_teacher import DeepSeekFlashTeacher
from .semantic_policy import _public_observation
from .request_actions import request_candidates


REQUEST_SYSTEM = """Investigate the scientific goal using actual experimental evidence.
Choose one explicit operation and its exact arguments. A shared executor can
approach a previously observed object, acquire a declared operand, and carry
out your operation. All these native actions consume the displayed budget.
The executor does not choose instruments, samples, measurements, recipes,
scientific settings, dialogue answers, or what to investigate next. Those are
your decisions. An unsuccessful request returns its actual feedback: revise
your plan. Each native action advances world time. Reading discovery feed
updates also lets time advance while waiting for a physical change.

Return one JSON object matching output_protocol, including a memory string
of at most 1000 characters. Keep confirmed measurements, hypotheses and the
next unresolved issue in memory. Correct hypotheses when evidence disagrees.
Use only observed object handles and displayed options. For a two-object
operation the arguments must be different. No task-specific solution is given.
"""


class RequestTeacher(DeepSeekFlashTeacher):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.configuration.update(prompt_version='object_requests_controls_v4',
                                  action_interface='object_requests_v1')

    @staticmethod
    def public_proposal(proposal, schema):
        return (DeepSeekFlashTeacher.public_proposal(proposal, schema)
                and not ('arg2' in proposal and proposal.get('arg1') == proposal['arg2']))

    def prompt(self, observation, state):
        public = _public_observation(observation)
        state['objects'].observe(observation)
        state['measurements'].observe_initial(public)
        objects = deepcopy(observation.get('remembered_objects', []))
        handles = [obj['uuid'] for obj in objects]
        handle_set = set(handles)
        alternatives = []
        for choice in request_candidates(observation):
            properties = {key: {'enum':[value]} for key,value in choice.action.items()}
            op = choice.action.get('action')
            args = public.get('actions',{}).get(op,{}).get('args',[])
            prefix = choice.action
            if args:
                domains = request_candidates(observation, prefix)
                first = args[0]
                values = [c.action[first] for c in domains]
                # Object registries can contain hundreds of visible handles.
                # Repeating that same domain for every operation adds context
                # without adding information or changing eligible requests.
                properties[first] = ({'$ref':'#/$defs/observed_object'}
                    if set(values) == handle_set else {'enum':values})
                if len(args) > 1:
                    properties[args[1]] = {'$ref':'#/$defs/observed_object'}
            properties['memory'] = {'type':'string','maxLength':1000}
            alternatives.append(dict(type='object', properties=properties,
                required=list(properties), additionalProperties=False))
        schema = {'anyOf':alternatives, '$defs':{'observed_object':
            {'enum':handles}}}
        payload = dict(public, remembered_objects=objects,
            execution=observation['execution'], execution_events=observation.get('execution_events',[]),
            actual_history=state['history'][-8:], notebook=state['notebook'],
            output_protocol={'schema':schema})
        # The stored conditions include a full public snapshot for replay.
        # Passing it once per scalar reading duplicates the entire scene.
        payload['received_measurements'] = [{key:getattr(r,key) for key in
            ('sequence','instrument_handle','instrument_name','target_handle','target_name',
             'quantity','value','unit','text','contained_objects')}
            for r in state['measurements'].records[-48:]]
        payload['earlier_measurements_not_shown'] = max(0,len(state['measurements'].records)-48)
        state['output_schema'] = schema
        state['candidates'] = []  # Domains validate requests; no quadratic pair list.
        state['messages'] = [dict(role='system',content=REQUEST_SYSTEM),
            dict(role='user',content=json.dumps(payload, ensure_ascii=False))]
        return json.dumps(state['messages'], ensure_ascii=False)
