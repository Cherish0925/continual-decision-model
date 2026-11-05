"""Shared execution, argument learning and real event boundaries."""
from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import torch

from cdm.core import ContinualCore, CoreConfig
from cdm.digital import DigitalPolicy, DigitalConfig
from experiments.discoveryworld.features import FeatureCache, FeatureConfig, batch_features, load_features
from experiments.discoveryworld.request_actions import request_stages
from experiments.discoveryworld.request_executor import ObjectRequestExecutor
from experiments.discoveryworld.request_teacher import RequestTeacher
from experiments.discoveryworld.request_learning import score_requests, request_return
from experiments.discoveryworld.train import Episode, prepare_episodes, run_epoch


class Laboratory:
    def __init__(self):
        self.steps,self.max_steps,self.ended=0,20,False
        self.inventory=[];self.visible=[]
        self.objects=[dict(uuid=11,name='instrument'),dict(uuid=22,name='sample')]
        self.actions=[]
    def policy_input(self):
        return dict(ui=dict(world_steps=self.steps,inventoryObjects=deepcopy(self.inventory),
            accessibleEnvironmentObjects=deepcopy(self.visible),
            nearbyObjects={'objects':{'east':deepcopy(self.objects)}},lastActionMessage='received'),vision={},
            actions={'USE':{'args':['arg1','arg2']},'WAIT':{'args':[]}},teleport_locations={})
    def step(self, action):
        before=self.policy_input();self.actions.append(action);self.steps+=1
        if action['action']=='TELEPORT_TO_OBJECT':
            self.visible=[deepcopy(o) for o in self.objects if o['uuid']==action['arg1']]
        elif action['action']=='PICKUP':
            self.inventory.extend(deepcopy(self.visible))
        return dict(episode_id='fixture',index=self.steps-1,observation=before,action=action,
            action_result={'success':True},next_observation=self.policy_input(),training_reward=.1,
            task_metrics=[dict(completed_successfully=False)],terminated=False,truncated=False,
            timing={'decision_seconds':.01})


class RequestTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
    def test_transport_preserves_request_and_counts_every_action(self):
        env=Laboratory();executor=ObjectRequestExecutor(env)
        request={'action':'USE','arg1':11,'arg2':22}
        row=executor.execute(request)
        self.assertEqual([a['action'] for a in env.actions],['TELEPORT_TO_OBJECT','PICKUP','TELEPORT_TO_OBJECT','USE'])
        self.assertEqual(env.actions[-1],request)
        self.assertEqual(row['executor']['native_actions'],4)
        self.assertEqual(len(row['next_observation']['execution_events']),4)
        self.assertEqual(row['next_observation'],executor.policy_input())

    def test_teacher_batch_uses_public_request_domains(self):
        teacher=object.__new__(RequestTeacher)
        teacher.configuration={'model_source':'fixture','prompt_version':'object_requests_v1'}
        teacher.max_input_tokens=32768;teacher.max_new_tokens=64;teacher.reasoning_effort='none'
        teacher.temperature=0.;teacher.action_corrections=0
        output=json.dumps(dict(action='USE',arg1=11,arg2=22,memory='Measure the sample.'))
        teacher.chat=lambda state:(output,12,{})
        obs=ObjectRequestExecutor(Laboratory()).policy_input()
        state=teacher.initial();result=teacher.batch([obs],[state])[0]
        self.assertNotIn('error',result)
        self.assertTrue(teacher.public_proposal(json.loads(output),state['output_schema']))
        self.assertFalse(teacher.public_proposal(dict(action='USE',arg1=11,arg2=11,memory=''),state['output_schema']))

    def test_terminal_progress_cannot_reward_a_failed_completion(self):
        first,discount=request_return(.2,.8,success=False,done=False,native_actions=3,gamma=1.,action_cost=.01)
        failure,_=request_return(.8,1.,success=False,done=True,native_actions=2,gamma=1.,action_cost=.01)
        success,_=request_return(.8,1.,success=True,done=True,native_actions=2,gamma=1.,action_cost=.01)
        self.assertAlmostEqual(first+failure,-.2-.05)
        self.assertAlmostEqual(first+success,1.-.2-.05)
        _,discount=request_return(0.,0.,success=False,done=False,native_actions=3,gamma=.9,action_cost=0.)
        self.assertAlmostEqual(discount,.9**3)

    def test_three_argument_scores_receive_one_external_event(self):
        env=Laboratory();executor=ObjectRequestExecutor(env)
        request={'action':'USE','arg1':11,'arg2':22};row=executor.execute(request)
        self.assertEqual(len(request_stages(row['observation'],request)),3)
        with TemporaryDirectory() as directory:
            root=Path(directory)
            episode=Episode(root/'trajectory.jsonl',dict(action_interface='object_requests_v1',
                action_descriptions=env.policy_input()['actions'],teleport_locations={}),[row])
            cache=FeatureCache(root/'cache',FeatureConfig(mode='lexical',lexical_dim=16,action_interface='object_requests_v1'))
            prepared=prepare_episodes([episode],cache,allow_failed_actions=True)
            model=DigitalPolicy(DigitalConfig(width=16,text_dim=16,image_dim=16,outcome_dim=14),
                ContinualCore(CoreConfig(width=16,heads=4,slots=4,event_dim=16,record_dim=8,actions=2),torch.tensor([[-1.],[1.]])))
            step=prepared[0].steps[0]
            obs,actions=batch_features([load_features(step.feature_path)],'cpu')
            with patch.object(model,'assimilate',wraps=model.assimilate) as ingest:
                _,state,_,loss,_=score_requests(model,[step],obs,actions,model.initial(1,'cpu'),
                    mode='cdm',persistent=True,restart_workspace=False)
                self.assertEqual(ingest.call_count,1)
                loss.mean().backward()
            self.assertTrue(any(p.grad is not None and p.grad.abs().sum()>0 for p in model.core.parameters()))
            result=run_epoch(model,prepared,batch_size=1,sequence_length=2,device='cpu',
                optimizer=torch.optim.AdamW(model.parameters(),lr=1e-3))
            self.assertEqual(result['transitions'],1)
            self.assertTrue(torch.isfinite(torch.tensor(result['loss'])))


if __name__=='__main__':
    unittest.main()
