"""Image/text integration using real, small Siglip towers without downloading weights."""
import unittest
from importlib.util import find_spec

import torch

HAS_TRANSFORMERS = find_spec("transformers") is not None
if HAS_TRANSFORMERS:
    from transformers import SiglipConfig, SiglipModel

from cdm.core import CoreConfig
from cdm.multimodal import EncodedEvent, MultimodalCore, MultimodalEncoder


@unittest.skipUnless(HAS_TRANSFORMERS, "Install .[multimodal] for pretrained-encoder integration tests")
class MultimodalTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        torch.set_num_threads(1)
        vision = dict(hidden_size=32, intermediate_size=64, num_hidden_layers=1,
                      num_attention_heads=4, image_size=16, patch_size=4)
        text = dict(hidden_size=32, intermediate_size=64, num_hidden_layers=1,
                    num_attention_heads=4, vocab_size=64, max_position_embeddings=8)
        encoder = MultimodalEncoder(SiglipModel(SiglipConfig(text_config=text, vision_config=vision)),
                                    numeric_dim=6, width=32, image_grid=2)
        config = CoreConfig(event_dim=32, record_dim=5, actions=3, width=32, slots=4,
                            heads=4, records=3, hidden=32, action_width=16)
        self.model = MultimodalCore(config, torch.tensor([-1., 0., 1.]), encoder).train()
        self.numeric = torch.randn(2, 6)
        self.timing = torch.tensor([[0.1, 0.0, 0.1], [0.2, 0.1, 0.3]])
        self.pixels = torch.randn(2, 3, 16, 16)
        self.ids = torch.randint(0, 64, (2, 8))
        self.mask = torch.ones(2, 8, dtype=torch.long)

    def event(self):
        return self.model.encoder(self.numeric, self.timing, self.pixels, self.ids, self.mask)

    def test_both_modalities_reach_state_and_action_with_trainable_projections(self):
        event = self.event()
        state = self.model.initial(2, "cpu")
        observed = self.model.observe(state, event)
        torch.testing.assert_close(observed["z"], state["z"])  # observation does not reset work
        action0 = self.model.readout(observed, torch.zeros(2, 8), torch.zeros(2))[0]
        refined = self.model.refine(observed, event, torch.zeros(2, 3, 5), torch.tensor([0, 2]))
        torch.testing.assert_close(refined["z"][0], state["z"][0])
        self.assertGreater((refined["z"][1] - state["z"][1]).abs().sum().item(), 0)
        action1, activity, value, mean, _ = self.model.readout(refined, torch.zeros(2, 8), torch.zeros(2))
        loss = action0.square().mean() + action1.square().mean() + activity.square().mean() \
            + value.square().mean() + mean.square().mean()
        loss.backward()
        self.assertFalse(self.model.encoder.backbone.training)
        self.assertTrue(all(p.grad is None for p in self.model.encoder.backbone.parameters()))
        for layer in (self.model.encoder.image, self.model.encoder.text,
                      self.model.core.gru, self.model.core.block):
            self.assertGreater(sum(p.grad.abs().sum().item() for p in layer.parameters() if p.grad is not None), 0)
        for remove in (slice(2, 6), slice(6, None)):
            present = event.present.clone()
            present[:, remove] = False
            changed = self.model.observe(state, EncodedEvent(event.tokens, present))
            self.assertGreater((observed["h"] - changed["h"]).abs().sum().item(), 1e-5)

    def test_masked_features_cannot_change_fast_or_refined_state(self):
        event = self.event()
        event.present[0, 2:] = False
        changed = event.tokens.clone()
        changed[0, 2:] = torch.randn_like(changed[0, 2:]) * 1000
        other = EncodedEvent(changed, event.present)
        initial = self.model.initial(2, "cpu")
        a, b = self.model.observe(initial, event), self.model.observe(initial, other)
        torch.testing.assert_close(a["h"], b["h"])
        args = (torch.zeros(2, 3, 5), torch.tensor([2, 1]))
        a, b = self.model.refine(a, event, *args), self.model.refine(b, other, *args)
        torch.testing.assert_close(a["z"], b["z"])

    def test_cached_features_match_live_features_and_missing_modalities_work(self):
        enc = self.model.encoder
        images, texts = enc.encode_image(self.pixels), enc.encode_text(self.ids, self.mask)
        cached = enc.project(self.numeric, self.timing, images, texts, text_present=self.mask)
        torch.testing.assert_close(cached.tokens, self.event().tokens)
        numeric_only = enc(self.numeric, self.timing)
        self.assertEqual(numeric_only.tokens.shape, (2, 2, 32))
        old = self.model.initial(2, "cpu")
        new = self.model.observe(old, numeric_only)
        new = self.model.observe(new, numeric_only)
        reset = self.model.observe(self.model.initial(2, "cpu"), numeric_only)
        self.assertGreater((new["h"] - reset["h"]).abs().sum().item(), 1e-5)
        torch.testing.assert_close(new["A"], old["A"])
        torch.testing.assert_close(new["B"], old["B"])


if __name__ == "__main__":
    unittest.main()
