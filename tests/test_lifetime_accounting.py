"""Lifetime boundaries must describe the same states in collection, training and evaluation."""
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments" / "lifetime"))

import evaluate as evaluation
from cdm.core import ContinualCore, CoreConfig
from cdm.instrument import InstrumentFleet, InstrumentSpec, action_values
from cdm.thermal import ThermalSpec
from evaluate import ResponseTimes
from latency import clock_statistics
from runtime import Costs, distributions, feature_dims, new_agent, tick
from train import first_observation, recompute_states


def small_core():
    spec = InstrumentSpec(dims=1, levels=3)
    event_dim, record_dim = feature_dims(spec)
    config = CoreConfig(event_dim, record_dim, spec.actions, action_dim=1, width=8,
                        slots=2, heads=2, records=3, rank=2, hidden=8, action_width=8)
    torch.manual_seed(11)
    return ContinualCore(config, action_values(spec)).eval()


class RecurrentTrainingTests(unittest.TestCase):
    def trajectory(self, resets, zero_compute=False):
        model = small_core()
        length, streams = resets.shape
        event = torch.randn(length, streams, model.config.event_dim)
        records = torch.randn(length, streams, model.config.records, model.config.record_dim)
        counts = torch.zeros(length, streams, dtype=torch.long) if zero_compute else \
            torch.tensor([[0, 1, 2], [1, 0, 1], [2, 1, 0], [1, 2, 1], [0, 1, 2], [1, 0, 1]])
        h = torch.randn(streams, model.config.width)
        z = torch.randn(streams, model.config.slots, model.config.width)
        fresh_z = model.initial(streams, "cpu")["z"]
        hp, zp, expected_h, expected_z = [], [], [], []
        with torch.no_grad():
            for t in range(length):
                if t:
                    h = torch.where(resets[t - 1, :, None], torch.zeros_like(h), h)
                    z = torch.where(resets[t - 1, :, None, None], fresh_z, z)
                hp.append(h.clone())
                zp.append(z.clone())
                h = model.fast(h, event[t])
                expected_h.append(h.clone())
                expected_z.append(z.clone())
                z = model.refine(z, h, records[t], counts[t])
        args = (model, torch.stack(hp), torch.stack(zp), event, records, counts, resets)
        return args, torch.stack(expected_h), torch.stack(expected_z)

    def test_collection_and_ppo_reconstruction_agree_across_resets(self):
        for reset, zero_compute in ((False, False), (True, False), (True, True)):
            with self.subTest(reset=reset, zero_compute=zero_compute):
                done = torch.zeros(6, 3, dtype=torch.bool)
                if reset:
                    done[1, 0], done[3, 1], done[3, 2] = True, True, True
                args, expected_h, expected_z = self.trajectory(done, zero_compute)
                H, Z = recompute_states(*args)
                torch.testing.assert_close(H, expected_h, rtol=1e-5, atol=1e-6)
                torch.testing.assert_close(Z, expected_z, rtol=1e-5, atol=1e-6)
                model = args[0]
                n = H.shape[0] * H.shape[1]
                offset, logvar = torch.randn(n, model.config.actions), torch.randn(n, model.config.actions)
                scalars = torch.randn(n, model.config.scalar_dim)
                allowed = torch.ones(n, 2 + model.config.compute_options, dtype=torch.bool)
                def log_prob(h, z):
                    cl, al, _ = model.heads(model.trunk(h.flatten(0, 1), z.flatten(0, 1),
                                                       offset, logvar, scalars), offset, logvar)
                    return sum(d.log_prob(torch.zeros(n, dtype=torch.long))
                               for d in distributions(cl, al, allowed))
                torch.testing.assert_close(log_prob(H, Z), log_prob(expected_h, expected_z),
                                           rtol=1e-5, atol=1e-6)

    def test_fixture_exposes_the_previous_reset_omission(self):
        done = torch.zeros(6, 3, dtype=torch.bool)
        done[1, 0] = True
        args, expected_h, expected_z = self.trajectory(done)
        model, hp, zp, event, records, counts, _ = args
        with torch.no_grad():
            old_h = model.fast_sequence(hp[0], event)
            old_out = model.refine(zp.flatten(0, 1), old_h.flatten(0, 1), records.flatten(0, 1),
                                   counts.flatten(0, 1)).view_as(zp)
            old_read = torch.cat((zp[:1], old_out[:-1]), 0)
        self.assertGreater(float((old_h[2, 0] - expected_h[2, 0]).abs().max()), 0.01)
        self.assertGreater(float((old_read[2, 0] - expected_z[2, 0]).abs().max()), 0.1)

    def test_fast_state_gradients_stop_at_each_streams_reset(self):
        done = torch.zeros(6, 3, dtype=torch.bool)
        done[1, 0] = True
        args, _, _ = self.trajectory(done)
        args = list(args)
        args[3] = args[3].requires_grad_()
        H, _ = recompute_states(*args)
        H[-1].square().sum().backward()
        grad = args[3].grad
        self.assertEqual(float(grad[:2, 0].abs().sum()), 0.0)
        self.assertGreater(float(grad[2:, 0].abs().sum()), 0.0)
        self.assertGreater(float(grad[:2, 1].abs().sum()), 0.0)


class TransitionTests(unittest.TestCase):
    def test_completed_transition_retains_request_and_condition_metadata(self):
        for old_active, new_active, new_request in ((False, True, True), (True, False, False), (True, True, True)):
            with self.subTest(old=old_active, new=new_active, replacement=new_request):
                spec = InstrumentSpec(cheap_noise=0, precise_noise=0, process_noise=0, drift=0)
                env = InstrumentFleet(1, spec, seed=9)
                env.active[:] = old_active
                env.setpoint[:] = 7.0
                env.since_switch[:] = 19
                env.returned[:] = True
                def change_condition():
                    env.since_switch[:] = 0
                    env.returned[:] = False
                def change_request():
                    env.active[:] = new_active
                    env.setpoint[:] = -3.0
                    return torch.tensor([new_request])
                env._switch, env._requests = change_condition, change_request
                out = env.step(torch.tensor([spec.actions // 2]), torch.tensor([False]))
                self.assertEqual(bool(out["transition_active"][0]), old_active)
                self.assertEqual(float(out["transition_setpoint"][0]), 7.0)
                self.assertEqual(int(out["transition_since_switch"][0]), 19)
                self.assertTrue(bool(out["transition_returned"][0]))
                self.assertEqual(bool(out["active"][0]), new_active)
                self.assertEqual(float(out["setpoint"][0]), -3.0)
                expected = (out["output"] - 7).abs() if old_active else torch.zeros(1)
                torch.testing.assert_close(out["error"], expected)

    def test_runtime_keeps_next_observation_separate_from_completed_action(self):
        model = small_core()
        spec = InstrumentSpec(dims=1, levels=3, request_rate=1, request_ticks=2, dwell=1)
        env = InstrumentFleet(2, spec, seed=8)
        agent, obs = new_agent(model, 2, 8, "cpu"), first_observation(env)
        for _ in range(6):
            active, started, goal = obs["active"].clone(), obs["started"].clone(), obs["setpoint"].clone()
            since, returned = env.since_switch.clone(), env.returned.clone()
            agent, obs, step = tick(model, agent, env, obs, "recurrent", Costs(), 0.05, 8)
            for key, expected in (("active", active), ("started", started), ("setpoint", goal),
                                   ("since_switch", since), ("returned", returned)):
                torch.testing.assert_close(step[key], expected)
            torch.testing.assert_close(step["next_active"], obs["active"])
            torch.testing.assert_close(step["next_started"], obs["started"])


class EvaluationTests(unittest.TestCase):
    @staticmethod
    def step(started=False, active=True, error=0.3, next_active=True, next_started=False):
        return {k: torch.tensor([v]) for k, v in dict(started=started, active=active, error=error,
                 next_active=next_active, next_started=next_started).items()}

    def test_response_counts_first_and_last_action_and_immediate_replacement(self):
        response = ResponseTimes(1, "cpu", 3)
        response.add(0, self.step(started=True))
        response.add(1, self.step())
        response.add(2, self.step(next_started=True))  # failed request replaced without an idle tick
        response.add(3, self.step(started=True, error=0.02))
        response.add(4, self.step(next_active=False))
        self.assertEqual(response.values, [3, 1])
        self.assertEqual(response.summary()["successful_requests"], 1)
        self.assertEqual(response.summary()["unfinished_requests"], 0)
        last = ResponseTimes(1, "cpu", 3)
        last.add(0, self.step(started=True))
        last.add(1, self.step())
        last.add(2, self.step(error=0.02, next_active=False))
        self.assertEqual(last.values, [3])
        self.assertEqual(last.summary()["successful_requests"], 1)

    def test_unfinished_request_is_not_reported_as_a_completed_failure(self):
        response = ResponseTimes(1, "cpu", 3)
        response.add(0, self.step(started=True))
        self.assertEqual(response.summary()["requests"], 0)
        self.assertEqual(response.summary()["unfinished_requests"], 1)

    def test_partial_thousand_tick_bin_uses_its_actual_sample_count(self):
        devices, ticks = 2, 1003
        def fake_tick(*args, **kwargs):
            step = {key: torch.ones(devices) for key in
                    ("error", "prediction_error", "shared_prediction_error", "reward", "counts")}
            step.update({key: torch.ones(devices, dtype=torch.bool) for key in ("active", "next_active")})
            step.update({key: torch.zeros(devices, dtype=torch.bool) for key in
                         ("returned", "learn", "accepted", "started", "next_started")})
            step["since_switch"] = torch.zeros(devices, dtype=torch.long)
            return None, None, step
        with patch.object(evaluation, "make_fleet"), patch.object(evaluation, "new_agent"), \
                patch.object(evaluation, "first_observation"), patch.object(evaluation, "tick", fake_tick):
            result = evaluation.evaluate(None, dict(mode="full", buffer=8, adapter_lr=0.05, every=8, learn_ticks=2),
                                         {"instrument": InstrumentSpec(), "thermal": ThermalSpec()}, "instrument",
                                         devices, ticks, 0, "cpu", "full")
        self.assertEqual(result["prediction_error_by_thousand_ticks"], [1.0, 1.0])
        self.assertEqual(result["reward_per_100_by_thousand_ticks"], [100.0, 100.0])

    def test_deadlines_include_lateness_from_the_previous_control(self):
        result = clock_statistics([15.0, 6.0], [15.0, 11.0], 10.0)
        self.assertEqual(result["missed_periods"], 2)
        self.assertEqual(result["latency_ms_median"], 10.5)
        self.assertEqual(result["response_ms_median"], 13.0)


if __name__ == "__main__":
    unittest.main()
