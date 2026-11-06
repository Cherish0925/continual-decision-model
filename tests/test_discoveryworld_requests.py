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
from experiments.discoveryworld.request_learning import score_requests, request_return, prepare_request_window
from experiments.discoveryworld.train import Episode, PreparedStep, PreparedEpisode, prepare_episodes, run_epoch


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
        properties = next(x['properties'] for x in state['output_schema']['anyOf']
                          if x['properties'].get('action', {}).get('enum') == ['USE'])
        self.assertEqual(properties['arg1'], {'$ref':'#/$defs/observed_object'})

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

            # Preparing a whole window cannot add another state assimilation,
            # expose a future observation, or disconnect a decision gradient.
            baseline=deepcopy(model)
            packed=deepcopy(model)
            before=run_epoch(baseline,prepared,batch_size=1,sequence_length=2,device='cpu',
                optimizer=torch.optim.SGD(baseline.parameters(),lr=.01),immediate_weight=0.)
            after=run_epoch(packed,prepared,batch_size=1,sequence_length=2,device='cpu',
                optimizer=torch.optim.SGD(packed.parameters(),lr=.01),immediate_weight=0.,window_text=True)
            self.assertAlmostEqual(before['loss'],after['loss'],places=6)
            for a,b in zip(baseline.parameters(),packed.parameters()):
                torch.testing.assert_close(a,b)

    def test_text_window_is_causal_and_preserves_sequence_gradients(self):
        from transformers import BertConfig
        cfg=BertConfig(vocab_size=32,hidden_size=16,num_hidden_layers=1,num_attention_heads=4,
                       intermediate_size=32,max_position_embeddings=32,
                       hidden_dropout_prob=0.,attention_probs_dropout_prob=0.)
        torch.manual_seed(20)
        with TemporaryDirectory() as directory:
            root=Path(directory)
            steps=[]
            for i in range(4):
                text=torch.randint(1,32,(3,16));text[:,7:]=0
                candidates=torch.randint(1,32,(3,1,16));candidates[...,5:]=0
                path=root/f'{i}.pt'
                torch.save(dict(observation=dict(numeric=torch.randn(32),timing=torch.zeros(3),text=text),
                                candidates=dict(text=candidates,numeric=torch.randn(3,8))),path)
                steps.append(PreparedStep(str(path),i%3,i%3,[0.]*14,request_stages=[(str(path),i%3)]))
            prepared=[PreparedEpisode(Episode(root/'a',{},[]),steps[:3]),
                      PreparedEpisode(Episode(root/'b',{},[]),steps[:2])]
            model=DigitalPolicy(DigitalConfig(width=16,text_dim=16,image_dim=16,outcome_dim=14,
                text_backbone=cfg.to_dict(),train_text_backbone=True,candidate_read=True),
                ContinualCore(CoreConfig(width=16,heads=4,slots=4,event_dim=16,record_dim=8,actions=1),torch.zeros(1,1)))
            baseline,packed=deepcopy(model),deepcopy(model)
            kwargs=dict(batch_size=2,sequence_length=3,device='cpu',immediate_weight=0.,outcome_weight=.1)
            before=run_epoch(baseline,prepared,optimizer=torch.optim.SGD(baseline.parameters(),lr=.01),**kwargs)
            after=run_epoch(packed,prepared,optimizer=torch.optim.SGD(packed.parameters(),lr=.01),window_text=True,**kwargs)
            self.assertAlmostEqual(before['loss'],after['loss'],places=6)
            self.assertEqual(before['transitions'],5)
            for a,b in zip(baseline.parameters(),packed.parameters()):
                torch.testing.assert_close(a,b,atol=2e-6,rtol=1e-4)
            model.eval()
            first=prepare_request_window(model,[[steps[0]],[steps[1]]],'cpu')[0]
            changed=prepare_request_window(model,[[steps[0]],[steps[3]]],'cpu')[0]
            def score(frame):
                obs,actions,stages=frame
                return score_requests(model,[steps[0]],obs,actions,model.initial(1,'cpu'),
                    mode='cdm',persistent=True,restart_workspace=False,prepared_stages=stages)[0]
            torch.testing.assert_close(score(first),score(changed))
            # Same forward trajectory and update count, different temporal
            # gradient span: the short condition retains state values.
            full,short=deepcopy(model),deepcopy(model)
            full_optimizer=torch.optim.SGD(full.parameters(),lr=0.)
            short_optimizer=torch.optim.SGD(short.parameters(),lr=0.)
            with patch.object(full_optimizer,'step',wraps=full_optimizer.step) as f, \
                 patch.object(short_optimizer,'step',wraps=short_optimizer.step) as s:
                a=run_epoch(full,prepared,optimizer=full_optimizer,window_text=True,**kwargs)
                b=run_epoch(short,prepared,optimizer=short_optimizer,window_text=True,state_gradient_length=1,**kwargs)
                self.assertEqual(f.call_count,s.call_count)
                self.assertAlmostEqual(a['loss'],b['loss'],places=6)
            self.assertTrue(any(not torch.allclose(a.grad,b.grad) for a,b in zip(full.parameters(),short.parameters())
                                if a.grad is not None and b.grad is not None))


if __name__=='__main__':
    unittest.main()
