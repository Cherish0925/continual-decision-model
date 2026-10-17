"""Training objectives must describe what the deployed agent did: its precise readings, its adapters, its workspace."""
import unittest

import torch

from cdm.core import CoreConfig, ContinualCore
from cdm.instrument import InstrumentFleet, InstrumentSpec, action_values
from experiments.lifetime.objectives import (deployed_adapters, online_nll, predictions_per_tick,
                                             recompute_states, window_support)
from experiments.lifetime.runtime import (SCALAR_DIM, Costs, feature_dims, goal_kinds, new_agent, tick)
from experiments.lifetime.train import attach_returns, first_observation


def core(spec, **options):
    goals = goal_kinds(spec)
    event_dim, record_dim = feature_dims(spec, goals)
    config = CoreConfig(event_dim=event_dim, record_dim=record_dim, actions=spec.actions, action_dim=spec.dims,
                        width=16, slots=2, heads=2, records=4, rank=2, hidden=16, action_width=8,
                        scalar_dim=SCALAR_DIM + goals - 1, goals=goals, **options)
    torch.manual_seed(3)
    return ContinualCore(config, action_values(spec)).eval()


def collect(model, spec, warm, length, streams=6, buffer=32, every=3):
    """Run ``warm`` ticks, then a window of ``length``; return the agent before the window, the window's steps,
    the adapters each tick predicted with, and the final agent."""
    env = InstrumentFleet(streams, spec, seed=5)
    agent, obs = new_agent(model, streams, buffer, "cpu"), first_observation(env)
    torch.manual_seed(7)
    steps, used = [], []
    for t in range(warm + length):
        if t == warm:
            start, steps, used = agent, [], []
        used.append((agent.A, agent.B))
        agent, obs, step = tick(model, agent, env, obs, "fixed_learn", Costs(), 0.05, every, learn_ticks=2,
                                support="buffer", precise_targets=True)
        steps.append(step)
        attach_returns(steps, spec.delay)
    return start, steps, used, agent


class ObjectiveTests(unittest.TestCase):
    def test_precise_reading_replaces_the_cheap_reading_of_its_transition(self):
        spec = InstrumentSpec(precise_noise=0.0)
        model = core(spec)
        start, steps, _, agent = collect(model, spec, warm=0, length=24)
        S = lambda k: torch.stack([s[k] for s in steps])
        T, L = agent.learn_y.shape[1], len(steps)
        accepted, output = S("accepted"), S("output")
        measured = accepted[:L - spec.delay].nonzero()
        self.assertGreater(len(measured), 0)
        for t, i in measured.tolist():
            position = T - L + t
            self.assertTrue(bool(agent.learn_p[i, position]))
            self.assertAlmostEqual(float(agent.learn_y[i, position]), float(output[t, i]), places=5)
            self.assertEqual(int(S("precise_at")[t, i]), t + spec.delay)
        self.assertEqual(int(agent.learn_p.sum()), len(measured))

    def test_replayed_adapters_are_the_adapters_the_agent_predicted_with(self):
        for blocks in (1, 3):
            with self.subTest(blocks=blocks):
                self.replay(blocks)

    def replay(self, blocks):
        spec = InstrumentSpec(search_share=0.5)
        model = core(spec, sensor_noise=True, step_rule="evidence", gated_workspace=True, adapter_blocks=blocks)
        start, steps, used, agent = collect(model, spec, warm=40, length=30)
        S = lambda k: torch.stack([s[k] for s in steps])
        L, T = len(steps), agent.learn_s.shape[1]
        learn = S("learn")
        self.assertGreater(int(learn.sum()), 10)
        self.assertGreater(int(S("arrived").sum()), 2)
        states = agent.learn_s[:, T - L:].transpose(0, 1)                   # the states the window stored
        support = window_support(model, start, states, S("action"), S("target"), S("precise_value"),
                                 S("precise_at"), S("arrived"), S("precise"), S("measured_back"))
        version, As, Bs = deployed_adapters(model, start, learn, support, 2, 0.05, differentiable=False)
        for t in range(L):
            for i in range(learn.shape[1]):
                torch.testing.assert_close(As[version[t, i]][i], used[t][0][i], rtol=1e-5, atol=1e-6)
                torch.testing.assert_close(Bs[version[t, i]][i], used[t][1][i], rtol=1e-5, atol=1e-6)
        mean, _ = predictions_per_tick(model, states.float(), As, Bs, version)
        for t in (0, L // 2, L - 1):
            expected = model.predict_actions(states[t].float(), *used[t])[0]
            torch.testing.assert_close(mean[t], expected, rtol=1e-4, atol=1e-5)

    def test_deployed_loss_trains_the_step_and_the_sensors_through_every_step(self):
        spec = InstrumentSpec()
        model = core(spec, sensor_noise=True, step_rule="evidence")
        start, steps, _, agent = collect(model, spec, warm=40, length=20)
        S = lambda k: torch.stack([s[k] for s in steps])
        L, T = len(steps), agent.learn_s.shape[1]
        states = agent.learn_s[:, T - L:].transpose(0, 1)
        support = window_support(model, start, states, S("action"), S("target"), S("precise_value"),
                                 S("precise_at"), S("arrived"), S("precise"), S("measured_back"))
        version, As, Bs = deployed_adapters(model, start, S("learn"), support, 2, 0.05, differentiable=True)
        self.assertGreater(len(As), 3)
        nll = online_nll(model, states.float(), S("action"), As, Bs, version, S("target"),
                         torch.zeros_like(S("arrived")))
        nll.mean().backward()
        for name in ("log_step", "log_kappa", "log_noise"):
            grad = getattr(model, name).grad
            self.assertIsNotNone(grad, name)
            self.assertGreater(float(grad.abs().sum()), 0.0, name)
        self.assertGreater(float(model.pred_in.weight.grad.abs().sum()), 0.0)

    def test_workspace_unrolled_over_events_matches_collection(self):
        spec = InstrumentSpec()
        model = core(spec, gated_workspace=True)
        L, n = 9, 3
        event = torch.randn(L, n, model.config.event_dim)
        records = torch.randn(L, n, model.config.records, model.config.record_dim)
        counts = torch.randint(0, 3, (L, n))
        dones = torch.zeros(L, n, dtype=torch.bool)
        dones[4, 1] = True
        h, z = torch.zeros(n, 16), model.initial(n, "cpu")["z"]
        hp, zp = [], []
        with torch.no_grad():
            for t in range(L):
                if t and dones[t - 1].any():
                    reset = dones[t - 1]
                    h = torch.where(reset[:, None], torch.zeros_like(h), h)
                    z = torch.where(reset[:, None, None], model.initial(n, "cpu")["z"], z)
                hp.append(h)
                zp.append(z)
                h = model.fast(h, event[t])
                z = model.refine(z, h, records[t], counts[t])
        hp, zp = torch.stack(hp), torch.stack(zp)
        for segment in (1, 2, 4, 9):
            _, zread = recompute_states(model, hp, zp, event, records, counts, dones, segment)
            torch.testing.assert_close(zread, zp, rtol=1e-5, atol=1e-5)
        zp = zp.clone().requires_grad_()
        _, zread = recompute_states(model, hp, zp, event, records, counts, dones, 4)
        zread[3].sum().backward()
        self.assertGreater(float(zp.grad[0].abs().sum()), 0.0)             # three events back, same segment
        self.assertEqual(float(zp.grad[1:].abs().sum()), 0.0)


class SearchRequestTests(unittest.TestCase):
    def test_a_request_for_the_largest_output_scores_the_shortfall(self):
        spec = InstrumentSpec(search_share=1.0, request_rate=1.0, cheap_noise=0.0, process_noise=0.0, drift=0.0)
        env = InstrumentFleet(5, spec, seed=1)
        out = env.step(torch.zeros(5, dtype=torch.long), torch.zeros(5, dtype=torch.bool))
        self.assertTrue(bool(out["search"].all()))
        self.assertEqual(float(out["setpoint"].abs().sum()), 0.0)
        out = env.step(torch.zeros(5, dtype=torch.long), torch.zeros(5, dtype=torch.bool))
        best = env.steady(env.current(), env.table.expand(5, -1, -1)).max(1).values
        torch.testing.assert_close(out["error"], (best - out["output"]).clamp(min=0))
        torch.testing.assert_close(out["transition_setpoint"], best)


if __name__ == "__main__":
    unittest.main()


class BranchTests(unittest.TestCase):
    def test_copies_of_a_process_share_its_randomness(self):
        env = InstrumentFleet(5, InstrumentSpec(search_share=0.5, dwell=20.0, request_rate=0.2), seed=2)
        for _ in range(10):
            env.step(torch.randint(0, 81, (5,)), torch.zeros(5, dtype=torch.bool))
        copies = env.branch(torch.tensor([1, 3]), 3, seed=8)
        outs = []
        for _ in range(60):
            out = copies.step(torch.full((6,), 40), torch.zeros(6, dtype=torch.bool))
            outs.append(torch.stack((out["cheap"], out["setpoint"], out["error"])))
        outs = torch.stack(outs).view(60, 3, 2, 3)
        torch.testing.assert_close(outs[..., 0], outs[..., 1], rtol=0, atol=0)
        torch.testing.assert_close(outs[..., 0], outs[..., 2], rtol=0, atol=0)
        self.assertGreater(float((outs[:, :, 0, 0] - outs[:, :, 1, 0]).abs().max()), 0.0)

    def test_each_copy_takes_its_option_and_invalid_options_are_marked(self):
        from experiments.lifetime.runtime import OPTIONS, branch_values
        spec = InstrumentSpec()
        model = core(spec)
        env = InstrumentFleet(4, spec, seed=1)
        agent, obs = new_agent(model, 4, 16, "cpu"), first_observation(env)
        torch.manual_seed(0)
        for _ in range(20):
            agent, obs, _ = tick(model, agent, env, obs, "full", Costs(), 0.05, 8, support="buffer")
        env.pending[:] = 0
        agent.busy[0], env.pending[1] = 1, 2                       # stream 0 is learning, stream 1 is measuring
        rows = torch.tensor([0, 1, 2])
        args = (model, agent, env, obs, rows, "full", Costs(), 0.05, 8, 6, 0.99, 0.1, 5)
        q, valid = branch_values(*args, support="buffer")
        again, _ = branch_values(*args, support="buffer")
        torch.testing.assert_close(q, again)
        self.assertEqual(tuple(q.shape), (3, OPTIONS))
        self.assertTrue(torch.equal(valid[0], torch.tensor([True, True, True, False, False, False, False])))
        self.assertFalse(bool(valid[1, 1]))
        self.assertTrue(bool(valid[2].all()))
        self.assertGreater(float(q[2].max() - q[2].min()), 0.0)
