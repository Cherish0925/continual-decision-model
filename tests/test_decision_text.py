"""Language parameters must receive decision gradients and survive deployment."""
from dataclasses import asdict
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import torch
from transformers import BertConfig, BertTokenizerFast

from cdm.core import ContinualCore, CoreConfig
from cdm.digital import DigitalConfig, DigitalPolicy
from cdm.text_encoder import DecisionTextEncoder
from experiments.discoveryworld.features import FeatureCache, FeatureConfig, batch_features, load_features
from experiments.discoveryworld.request_actions import request_candidates
from experiments.discoveryworld.train import load_checkpoint


class DecisionTextTests(unittest.TestCase):
    def test_decision_gradient_freezing_and_checkpoint_execution(self):
        torch.set_num_threads(1)
        with TemporaryDirectory() as folder:
            root = Path(folder)
            words = ['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]', 'use', 'read', 'sample', 'instrument', 'measure', 'temperature', ':', '.', 'the']
            (root/'vocab.txt').write_text('\n'.join(words))
            BertTokenizerFast(vocab_file=str(root/'vocab.txt')).save_pretrained(root)
            cfg = BertConfig(vocab_size=len(words), hidden_size=16, num_hidden_layers=1,
                             num_attention_heads=4, intermediate_size=32, max_position_embeddings=32)
            cfg.save_pretrained(root)
            features = FeatureConfig(mode='tokens', encoder_id='test', encoder_revision='test', text_context=16)
            cache = FeatureCache(root/'features', features, encoder_path=root)
            observation = dict(ui=dict(taskProgress=[dict(taskName='measure', description='temperature')],
                accessibleEnvironmentObjects=[dict(uuid=1,name='instrument'),dict(uuid=2,name='sample')]),
                actions={'USE':{'args':['arg1','arg2'],'desc':'measure the sample'},'READ':{'args':['arg1'],'desc':'read the instrument'}})
            choices = request_candidates(observation)
            obs, actions = batch_features([load_features(str(cache.encode(observation, choices)))], 'cpu')
            self.assertEqual(obs['text'].dtype, torch.long)
            self.assertEqual(actions['text'].dtype, torch.long)
            for trainable in (False, True):
                torch.manual_seed(4)
                core = ContinualCore(CoreConfig(width=16,heads=4,slots=4,event_dim=16,record_dim=8,actions=1),torch.zeros(1,1))
                model = DigitalPolicy(DigitalConfig(width=16,text_dim=16,image_dim=16,outcome_dim=14,
                    text_backbone=cfg.to_dict(),train_text_backbone=trainable,candidate_read=True,
                    text_batch_size=8,text_checkpointing=False),core).train()
                logits, _, _ = model(obs, actions, model.initial(1,'cpu'), predict_outcomes=False)
                torch.nn.functional.cross_entropy(logits, torch.tensor([1])).backward()
                encoder_grad = sum(float(p.grad.abs().sum()) for p in model.text_encoder.parameters() if p.grad is not None)
                self.assertEqual(encoder_grad > 0, trainable)
                self.assertGreater(sum(float(p.grad.abs().sum()) for p in model.core.parameters() if p.grad is not None),0)
                model.eval()
                expected = model(obs, actions, model.initial(1,'cpu'),predict_outcomes=False)[0]
                path=root/'checkpoint.pt'
                torch.save(dict(model=model.state_dict(),core_config=asdict(core.config),
                    digital_config=asdict(model.config),features=asdict(features)),path)
                restored,_=load_checkpoint(path)
                self.assertEqual(restored.text_encoder.batch_size,8)
                self.assertFalse(restored.text_encoder.backbone.is_gradient_checkpointing)
                restored.eval()
                actual=restored(obs,actions,restored.initial(1,'cpu'),predict_outcomes=False)[0]
                torch.testing.assert_close(actual,expected,rtol=0,atol=0)
            # The tail survives chunking beyond the encoder window.
            ids = cache.texts(['sample ' * 40 + 'temperature'])[0]
            self.assertGreater(len(ids),1)
            self.assertIn(words.index('temperature'),ids[-1].tolist())

    def test_execution_batches_preserve_text_values_and_gradients(self):
        torch.set_num_threads(1)
        cfg=BertConfig(vocab_size=32,hidden_size=16,num_hidden_layers=2,
                       num_attention_heads=4,intermediate_size=32,max_position_embeddings=32,
                       hidden_dropout_prob=0.,attention_probs_dropout_prob=0.)
        torch.manual_seed(5)
        ids=torch.randint(1,32,(2,3,2,16))
        ids[...,9:]=0
        ids[0,1]=ids[0,0]
        ids[1,2,1]=0
        reference=DecisionTextEncoder(cfg.to_dict(),True,batch_size=1,checkpointing=True)
        batched=DecisionTextEncoder(cfg.to_dict(),True,batch_size=8,checkpointing=False)
        batched.load_state_dict(reference.state_dict())
        expected,actual=reference(ids),batched(ids)
        torch.testing.assert_close(actual,expected)
        expected[...,0].sum().backward()
        actual[...,0].sum().backward()
        for old,new in zip(reference.parameters(),batched.parameters()):
            if old.grad is None:
                self.assertIsNone(new.grad)
            else:
                torch.testing.assert_close(new.grad,old.grad,rtol=1e-4,atol=1e-6)

    def test_window_encoding_preserves_each_view_and_its_gradients(self):
        torch.set_num_threads(1)
        cfg=BertConfig(vocab_size=32,hidden_size=16,num_hidden_layers=2,
                       num_attention_heads=4,intermediate_size=32,max_position_embeddings=64,
                       hidden_dropout_prob=0.,attention_probs_dropout_prob=0.)
        torch.manual_seed(17)
        inputs=[torch.randint(1,32,(2,3,48)),torch.randint(1,32,(2,4,2,48)),
                torch.zeros(2,1,48,dtype=torch.long)]
        inputs[0][...,8:]=0
        inputs[1][...,35:]=0
        inputs[1][0,1]=inputs[1][0,0]
        inputs[1][1,3]=0
        for trainable in (False,True):
            old=DecisionTextEncoder(cfg.to_dict(),trainable,batch_size=2)
            new=deepcopy(old)
            expected=[old(ids) for ids in inputs]
            actual=new.forward_many(inputs)
            for a,b in zip(actual,expected):
                torch.testing.assert_close(a,b,rtol=1e-5,atol=1e-6)
            if trainable:
                sum(x[...,0].sum() for x in expected).backward()
                sum(x[...,0].sum() for x in actual).backward()
                for a,b in zip(old.parameters(),new.parameters()):
                    if a.grad is not None:
                        torch.testing.assert_close(a.grad,b.grad,rtol=1e-4,atol=2e-6)
            changed=[x.clone() for x in inputs]
            changed[1][0,0,0,:5]=1
            torch.testing.assert_close(new.forward_many(changed)[0],actual[0])


if __name__=='__main__':
    unittest.main()
