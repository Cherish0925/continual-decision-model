"""Pretrained-weight reuse and a native multimodal model's latent connection."""
from dataclasses import asdict
from importlib.util import find_spec
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import torch

HAS_TRANSFORMERS = find_spec("transformers") is not None
if HAS_TRANSFORMERS:
    from transformers import SiglipConfig, SiglipModel, Qwen3_5Config, Qwen3_5ForConditionalGeneration

from cdm.core import ContinualCore, CoreConfig
from cdm.multimodal import MultimodalCore, MultimodalEncoder, join_events
from cdm.semantic import SemanticExpert, SemanticReader


@unittest.skipUnless(HAS_TRANSFORMERS, "Install .[multimodal] for pretrained integration tests")
class PretrainedIntegrationTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(19)
        torch.set_num_threads(1)
        vision = dict(hidden_size=32, intermediate_size=64, num_hidden_layers=1,
                      num_attention_heads=4, image_size=16, patch_size=4)
        text = dict(hidden_size=32, intermediate_size=64, num_hidden_layers=1,
                    num_attention_heads=4, vocab_size=64, max_position_embeddings=8)
        self.encoder = MultimodalEncoder(SiglipModel(SiglipConfig(text_config=text, vision_config=vision)),
                                         numeric_dim=6, width=32, image_grid=2)
        self.config = CoreConfig(event_dim=6, record_dim=5, actions=3, width=32, slots=4,
                                 heads=4, records=3, hidden=32, action_width=16, sensor_noise=True,
                                 learned_step=True, activity_values=True, gated_workspace=True)
        self.source = ContinualCore(self.config, torch.tensor([-1., 0., 1.])).eval()

    def load_source(self):
        with TemporaryDirectory() as folder:
            path = Path(folder) / "model.pt"
            torch.save({"config": asdict(self.config), "model": self.source.state_dict()}, path)
            return MultimodalCore.from_lifetime_checkpoint(path, self.encoder)

    def test_checkpoint_preserves_original_state_refinement_and_decisions(self):
        model = self.load_source().eval()
        actual = expected = model.initial(2, "cpu")
        records, counts = torch.randn(2, 3, 5), torch.tensor([1, 2])
        scalars, goal = torch.randn(2, self.config.scalar_dim), torch.randn(2)
        for _ in range(3):
            raw = torch.randn(2, 6)
            h = self.source.fast(expected["h"], raw)
            z = self.source.refine(expected["z"], h, records, counts)
            expected = {**expected, "h": h, "z": z}
            actual = model.refine(model.observe(actual, None, raw), None, records, counts)
            torch.testing.assert_close(actual["h"], expected["h"])
            torch.testing.assert_close(actual["z"], expected["z"])
            mean, lv = self.source.predict_actions(self.source.state_features(h, z), expected["A"], expected["B"])
            offset = mean - goal[:, None]
            heads = self.source.heads(self.source.trunk(h, z, offset, lv, scalars), offset, lv)
            for a, b in zip(model.readout(actual, scalars, goal), (*heads, mean, lv)):
                torch.testing.assert_close(a, b)

    def test_new_input_gain_can_learn_without_changing_initial_fast_behavior(self):
        model = self.load_source().train()
        state, raw = model.initial(2, "cpu"), torch.randn(2, 6)
        event = model.encoder(torch.randn(2, 6), torch.zeros(2, 3))
        updated = model.observe(state, event, raw)
        torch.testing.assert_close(updated["h"], self.source.fast(state["h"], raw))
        updated["h"].square().mean().backward()
        self.assertGreater(model.input_gain.grad.abs().item(), 0)
        self.assertTrue(all(p.grad is None for p in model.encoder.backbone.parameters()))

    def tiny_expert(self):
        text = dict(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=4,
                    num_attention_heads=4, num_key_value_heads=2, head_dim=8,
                    linear_num_key_heads=2, linear_num_value_heads=2, linear_key_head_dim=8,
                    linear_value_head_dim=8, linear_conv_kernel_dim=4,
                    layer_types=["linear_attention"] * 3 + ["full_attention"],
                    rope_parameters={"rope_type": "default", "rope_theta": 10000.,
                                     "mrope_section": [1, 1, 2], "partial_rotary_factor": 1.0},
                    pad_token_id=0, eos_token_id=2)
        vision = dict(depth=1, hidden_size=32, intermediate_size=64, num_heads=4,
                      patch_size=4, temporal_patch_size=1, spatial_merge_size=2,
                      out_hidden_size=32, num_position_embeddings=64, deepstack_visual_indexes=[])
        cfg = Qwen3_5Config(text_config=text, vision_config=vision, image_token_id=60,
                            video_token_id=63, vision_start_token_id=61, vision_end_token_id=62)
        return SemanticExpert(Qwen3_5ForConditionalGeneration(cfg))

    def test_native_vision_language_features_reach_cdm_and_train_only_the_reader(self):
        expert = self.tiny_expert().train()
        ids = torch.tensor([[61, 60, 60, 60, 60, 62, 5, 6], [61, 60, 60, 60, 60, 62, 7, 8]])
        mask = torch.ones_like(ids)
        pixels = torch.randn(32, 48)
        grid = torch.tensor([[1, 4, 4], [1, 4, 4]])
        features = expert.encode(input_ids=ids, attention_mask=mask, pixel_values=pixels, image_grid_thw=grid,
                                 mm_token_type_ids=(ids == 60).long())
        changed = expert.encode(input_ids=ids, attention_mask=mask, pixel_values=pixels + 1, image_grid_thw=grid,
                                mm_token_type_ids=(ids == 60).long())
        self.assertGreater((features - changed).abs().sum().item(), 1e-5)
        self.assertFalse(expert.backbone.training)
        self.assertFalse(features.requires_grad)
        model = self.load_source()
        reader = SemanticReader(32, width=32, tokens=4, heads=4)
        state = model.initial(2, "cpu")
        visual = model.encoder(torch.randn(2, 6), torch.zeros(2, 3))
        semantic = reader(features, mask, state["h"], torch.tensor([0., 0.5]))
        combined = join_events(visual, semantic)
        without = model.observe(state, visual)
        with_semantics = model.observe(state, combined)
        self.assertGreater((with_semantics["h"] - without["h"]).abs().sum().item(), 1e-5)
        state = model.refine(with_semantics, combined, torch.zeros(2, 3, 5), torch.tensor([1, 1]))
        action = model.readout(state, torch.zeros(2, self.config.scalar_dim), torch.zeros(2))[0]
        action.square().mean().backward()
        self.assertGreater(reader.source[1].weight.grad.abs().sum().item(), 0)
        self.assertGreater(reader.queries.grad.abs().sum().item(), 0)
        self.assertTrue(all(p.grad is None for p in expert.parameters()))

    def test_unavailable_and_padded_expert_results_do_not_change_observation(self):
        reader = SemanticReader(32, width=32, tokens=4, heads=4)
        model = self.load_source().eval()
        state = model.initial(2, "cpu")
        features, present = torch.randn(2, 8, 32), torch.zeros(2, 8, dtype=torch.bool)
        visual = model.encoder(torch.randn(2, 6), torch.zeros(2, 3))
        missing = reader(features, present, state["h"], torch.ones(2))
        a = model.observe(state, visual)
        b = model.observe(state, join_events(visual, missing))
        torch.testing.assert_close(a["h"], b["h"])
        self.assertTrue(torch.isfinite(missing.tokens).all())
        present[:, :4] = True
        s1 = reader(features, present, state["h"], torch.ones(2))
        features[:, 4:] += 1000
        s2 = reader(features, present, state["h"], torch.ones(2))
        torch.testing.assert_close(s1.tokens, s2.tokens)


if __name__ == "__main__":
    unittest.main()
