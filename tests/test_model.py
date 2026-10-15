"""Model semantics shared by CPU and GPU implementations."""

import unittest

import torch

from cdm import CDMConfig, ContinualDecisionModel


class ModelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        torch.set_num_threads(1)
        self.model = ContinualDecisionModel(CDMConfig(
            observation_dim=7, action_dim=5, goal_dim=2, environment_dim=3,
            hidden_dim=16, slots=3, memory_size=2, heads=2,
        ))
        self.obs = torch.randn(2, 7)
        self.goal = torch.randn(2, 2)
        self.environment = torch.randn(2, 3)

    def test_fresh_context_changes_action_without_refinement(self):
        state = self.model.initialize(2)
        first = self.model.assimilate(state, self.obs, self.goal, self.environment)
        second = self.model.assimilate(state, self.obs + 2, self.goal, self.environment)
        self.assertTrue(torch.equal(first.slots, second.slots))
        self.assertFalse(torch.allclose(self.model.readout(first).action_logits,
                                        self.model.readout(second).action_logits))

    def test_batch_items_are_independent(self):
        self.model.eval()
        together, _ = self.model(self.obs, self.goal, self.environment, steps=2)
        for i in range(2):
            alone, _ = self.model(self.obs[i:i + 1], self.goal[i:i + 1],
                                 self.environment[i:i + 1], steps=2)
            torch.testing.assert_close(together.action_logits[i], alone.action_logits[0], atol=1e-6, rtol=1e-5)

    def test_refinement_continues_persistent_state(self):
        state = self.model.assimilate(self.model.initialize(2), self.obs, self.goal, self.environment)
        once = self.model.refine(state)
        continued = self.model.refine(once)
        twice = self.model.refine(state, steps=2)
        torch.testing.assert_close(continued.slots, twice.slots)
        reset = self.model.refine(self.model.reset_workspace(once))
        self.assertFalse(torch.allclose(continued.slots, reset.slots))

    def test_memory_is_bounded_and_version_conditioned(self):
        state = self.model.initialize(2)
        for _ in range(5):
            _, state = self.model(self.obs, self.goal, self.environment, state=state)
        self.assertEqual(tuple(state.memory.shape), (2, 2, 3, 16))
        self.assertTrue(state.memory_valid.all())
        torch.testing.assert_close(state.memory_cursor, torch.ones(2, dtype=torch.long))
        versioned = self.model.set_model_version(state, 1)
        fresh, info = self.model.read_memory(versioned)
        self.assertFalse(info.valid.any())
        self.assertTrue(torch.equal(info.weights, torch.zeros_like(info.weights)))
        self.assertTrue(torch.equal(info.gates[..., 2], torch.ones_like(info.gates[..., 2])))
        self.assertTrue(torch.isfinite(fresh.slots).all())
        fresh = self.model.write_memory(fresh)
        _, info = self.model.read_memory(fresh)
        self.assertEqual(info.valid.sum(dim=1).tolist(), [2, 2])

    def test_device_and_gradients_through_memory(self):
        for device in ["cpu"] + (["cuda"] if torch.cuda.is_available() else []):
            model = self.model.to(device)
            obs, goal, environment = [x.to(device) for x in (self.obs, self.goal, self.environment)]
            _, state = model(obs, goal, environment, steps=2)
            output, state = model(obs + 0.1, goal + 0.1, environment, state=state)
            loss = output.action_logits.square().mean() + output.activity_logits.square().mean() + output.value.square().mean()
            model.zero_grad(set_to_none=True)
            loss.backward()
            self.assertEqual(state.memory.device.type, device)
            for name, parameter in model.named_parameters():
                self.assertIsNotNone(parameter.grad, name)
                self.assertTrue(torch.isfinite(parameter.grad).all(), name)
            self.assertFalse(state.detach().slots.requires_grad)

    def test_selected_streams_write_independently(self):
        state = self.model.assimilate(self.model.initialize(2), self.obs, self.goal, self.environment)
        state = self.model.refine(state)
        written = self.model.write_memory(state, mask=torch.tensor([True, False]))
        self.assertEqual(written.memory_valid.sum(dim=1).tolist(), [1, 0])
        self.assertEqual(written.memory_cursor.tolist(), [1, 0])
        torch.testing.assert_close(written.memory[0, 0], state.slots[0])
        torch.testing.assert_close(written.memory[1], state.memory[1])

    def test_observation_prefix_and_zero_refinement(self):
        state = self.model.initialize(2)
        first = self.model.assimilate(state, self.obs, self.goal, self.environment)
        self.assertEqual(first.event_index.tolist(), [1, 1])
        self.assertEqual(first.workspace_event_index.tolist(), [0, 0])
        self.assertIs(self.model.refine(first, steps=0), first)
        refined = self.model.refine(first)
        self.assertEqual(refined.workspace_event_index.tolist(), [1, 1])
        stored = self.model.write_memory(refined)
        self.assertEqual(stored.memory_event_index[:, 0].tolist(), [1, 1])
        second = self.model.assimilate(stored, self.obs + 1, self.goal, self.environment)
        self.assertEqual(second.event_index.tolist(), [2, 2])
        self.assertEqual(second.workspace_event_index.tolist(), [1, 1])
        read, _ = self.model.read_memory(second)
        self.assertEqual(read.workspace_event_index.tolist(), [2, 2])
        self.assertEqual(read.memory_event_index[:, 0].tolist(), [1, 1])


if __name__ == "__main__":
    unittest.main()
