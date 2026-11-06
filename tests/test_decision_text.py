"""Language parameters must receive decision gradients and survive deployment."""
from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import torch
from transformers import BertConfig, BertTokenizerFast

from cdm.core import ContinualCore, CoreConfig
from cdm.digital import DigitalConfig, DigitalPolicy
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
                    text_backbone=cfg.to_dict(),train_text_backbone=trainable,candidate_read=True),core).train()
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
                restored.eval()
                actual=restored(obs,actions,restored.initial(1,'cpu'),predict_outcomes=False)[0]
                torch.testing.assert_close(actual,expected,rtol=0,atol=0)
            # The tail survives chunking beyond the encoder window.
            ids = cache.texts(['sample ' * 40 + 'temperature'])[0]
            self.assertGreater(len(ids),1)
            self.assertIn(words.index('temperature'),ids[-1].tolist())


if __name__=='__main__':
    unittest.main()
