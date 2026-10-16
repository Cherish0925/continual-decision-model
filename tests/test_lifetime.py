"""Instrument invariants, equivalences inside the continual core, and the timing of activities."""
import sys
import unittest
from dataclasses import replace
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments" / "lifetime"))

from cdm.core import CoreConfig, ContinualCore  # noqa: E402
from cdm.instrument import InstrumentFleet, InstrumentSpec, action_values, response  # noqa: E402
from runtime import LEARN, MODES, Costs, feature_dims, masks, new_agent, tick  # noqa: E402

QUIET = InstrumentSpec(precise_noise=0.0, cheap_noise=0.0, process_noise=0.0, drift=0.0)


def small_model(spec=InstrumentSpec(), width=32):
    event_dim, record_dim = feature_dims(spec)
    config = CoreConfig(event_dim=event_dim, record_dim=record_dim, actions=spec.actions, action_dim=spec.dims,
                        width=width, slots=4, heads=4, records=8, rank=4, hidden=32)
    torch.manual_seed(0)
    return ContinualCore(config, action_values(spec)).eval()


def first(env):
    return env.step(torch.full((env.n,), env.spec.actions // 2), torch.zeros(env.n, dtype=torch.bool))


class InstrumentTests(unittest.TestCase):
    def test_action_table(self):
        for dims, levels in ((1, 11), (2, 9)):
            spec = InstrumentSpec(dims=dims, levels=levels)
            table = action_values(spec)
            self.assertEqual(tuple(table.shape), (levels ** dims, dims))
            torch.testing.assert_close(table[spec.actions // 2], torch.zeros(dims))

    def test_same_seed_same_lifetime(self):
        a, b = InstrumentFleet(64, seed=3), InstrumentFleet(64, seed=3)
        gen = torch.Generator().manual_seed(0)
        for _ in range(200):
            act = torch.randint(0, a.spec.actions, (64,), generator=gen)
            meas = torch.rand(64, generator=gen) < 0.2
            oa, ob = a.step(act, meas), b.step(act, meas)
            for key in oa:
                self.assertTrue(torch.equal(oa[key], ob[key]), key)

    def test_precise_reading_returns_the_output_at_request_time(self):
        env = InstrumentFleet(256, QUIET, seed=1)
        gen = torch.Generator().manual_seed(0)
        sent = {}
        for t in range(300):
            act = torch.randint(0, env.spec.actions, (256,), generator=gen)
            out = env.step(act, torch.rand(256, generator=gen) < 0.5)
            for i in out["arrived"].nonzero().squeeze(1).tolist():         # a new request may leave in the same tick
                t0, value = sent.pop(i)
                self.assertEqual(t - t0, env.spec.delay)
                self.assertAlmostEqual(float(out["precise"][i]), value, places=5)
            for i in out["accepted"].nonzero().squeeze(1).tolist():
                sent[i] = (t, float(out["output"][i]))

    def test_setpoints_are_reachable(self):
        for dims in (1, 2):
            env = InstrumentFleet(512, replace(QUIET, dims=dims, levels=11 if dims == 1 else 9), seed=2)
            for _ in range(200):
                out = env.step(torch.full((512,), env.spec.actions // 2), torch.zeros(512, dtype=torch.bool))
                if out["started"].any():
                    r = response(env.spec, env.current().unsqueeze(1), env.table.unsqueeze(0), env.drift.unsqueeze(1))
                    gap = (r - env.setpoint.unsqueeze(1)).abs().min(1).values[out["started"]]
                    self.assertLess(float(gap.max()), 1e-5)

    def test_reset_starts_a_new_lifetime(self):
        env = InstrumentFleet(32, seed=4)
        for _ in range(100):
            env.step(torch.zeros(32, dtype=torch.long), torch.ones(32, dtype=torch.bool))
        mask = torch.arange(32) % 2 == 0
        env.reset(mask)
        self.assertFalse(env.active[mask].any())
        self.assertFalse(env.pending[mask].any())
        self.assertTrue(torch.equal(env.seen[mask].sum(1), torch.ones(16, dtype=torch.long)))


class CoreTests(unittest.TestCase):
    def test_all_action_prediction_matches_explicit_inputs(self):
        model = small_model()
        n = 5
        s = model.initial(n, "cpu")
        A, B = torch.randn_like(s["A"]), torch.randn_like(s["B"])
        state = model.state_features(torch.randn(n, 32), torch.randn(n, 4, 32))
        x = torch.cat((state.unsqueeze(1).expand(n, model.config.actions, -1),
                       model.action_features().unsqueeze(0).expand(n, -1, -1)), -1)
        for adapter in ((None, None), (A, B)):
            fast, slow = model.predict_actions(state, *adapter), model.predict(x, *adapter)
            for a, b in zip(fast, slow):
                torch.testing.assert_close(a, b, rtol=1e-4, atol=1e-4)

    def test_refinement_of_chosen_rows_matches_refining_all(self):
        model = small_model()
        n = 9
        z, h, records = torch.randn(n, 4, 32), torch.randn(n, 32), torch.randn(n, 8, model.config.record_dim)
        counts = torch.tensor([0, 1, 2, 4, 0, 4, 2, 1, 0])
        dense, ctx = z, model.context(h, records)
        for step in range(4):
            dense = torch.where((counts > step).view(-1, 1, 1), model.norm_slots(model.block(dense, ctx)), dense)
        torch.testing.assert_close(model.refine(z, h, records, counts), dense, rtol=1e-5, atol=1e-5)

    def test_workspace_keeps_its_scale(self):
        model = small_model()
        z, h, records = model.initial(3, "cpu")["z"], torch.randn(3, 32), torch.randn(3, 8, model.config.record_dim)
        with torch.no_grad():
            for _ in range(500):
                z = model.refine(z, h, records, torch.full((3,), 4))
        self.assertLess(float(z.norm(dim=-1).max()), 4 * 32 ** 0.5)

    def test_sequence_and_tick_fast_states_agree(self):
        model = small_model()
        events = torch.randn(6, 3, model.config.event_dim)
        h = torch.zeros(3, 32)
        seq = model.fast_sequence(h, events)
        for t in range(6):
            h = model.fast(h, events[t])
            torch.testing.assert_close(h, seq[t], rtol=1e-5, atol=1e-5)


class RuntimeTests(unittest.TestCase):
    def test_modes_restrict_options(self):
        model = small_model()
        agent = new_agent(model, 4, 16, "cpu")
        agent.filled[:] = 16
        pending = torch.tensor([0, 2, 0, 0])
        agent.busy[:] = torch.tensor([0, 0, 1, 0])
        for mode in MODES:
            for t in (0, 3):
                m = masks(agent, pending, mode, t, every=8)
                self.assertFalse(m[1, 1])                                   # measurement in flight
                self.assertTrue(torch.equal(m[2, 2:], torch.tensor([True, False, False, False, False])))
                learn = m[:, 2 + LEARN]
                if mode in ("frozen", "recurrent"):
                    self.assertFalse(learn.any())
                if mode == "recurrent":
                    self.assertFalse(m[:, 3:2 + LEARN].any())
                if mode in ("fixed_learn", "world_model"):
                    self.assertEqual(bool(learn[0]), t == 0)
                if mode == "world_model":
                    self.assertTrue(torch.equal(m[:, :2].sum(1), torch.ones(4, dtype=torch.long)))
                    self.assertTrue(torch.equal(m[:, 2:].sum(1), torch.ones(4, dtype=torch.long)))

    def test_learning_occupies_compute_and_lands_on_completion(self):
        spec = InstrumentSpec()
        model = small_model(spec)
        env = InstrumentFleet(3, spec, seed=0)
        agent, obs = new_agent(model, 3, 16, "cpu"), first(env)
        while not (agent.filled >= 8).all():
            agent, obs, step = tick(model, agent, env, obs, "frozen", Costs(), 0.05, every=1)
        before = agent.A.clone()
        agent, obs, step = tick(model, agent, env, obs, "fixed_learn", Costs(), 0.05, every=1, learn_ticks=2)
        self.assertTrue(step["learn"].all())
        self.assertTrue(torch.equal(agent.A, before))                       # still running
        self.assertTrue(torch.equal(agent.busy, torch.ones(3, dtype=torch.long)))
        agent, obs, step = tick(model, agent, env, obs, "fixed_learn", Costs(), 0.05, every=1, learn_ticks=2)
        self.assertTrue((step["compute"] == 0).all())                       # compute occupied: idle only
        self.assertFalse(torch.equal(agent.A, before))                      # completed: used from the next tick
        self.assertTrue(torch.equal(agent.busy, torch.zeros(3, dtype=torch.long)))


if __name__ == "__main__":
    unittest.main()
