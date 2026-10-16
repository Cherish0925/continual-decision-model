"""Maze invariants and the semantics of every continuation mode."""
import unittest
from collections import deque
from dataclasses import replace

import numpy as np
import torch

from cdm.maze import MazeSpec, REGIMES, distance_maps, make_batch, make_episode, move, optimal_actions
from cdm.planner import MODES, PlannerConfig, build_planner


def bfs(blocked, goal):
    h, w = blocked.shape
    d = np.full(h * w, np.inf)
    d[goal] = 0
    q = deque([goal])
    while q:
        s = q.popleft()
        r, c = divmod(s, w)
        for dr, dc in ((-1, 0), (0, 1), (1, 0), (0, -1)):
            rr, cc = r + dr, c + dc
            if 0 <= rr < h and 0 <= cc < w and not blocked[rr, cc] and np.isinf(d[rr * w + cc]):
                d[rr * w + cc] = d[s] + 1
                q.append(rr * w + cc)
    return d.reshape(h, w)


class MazeTests(unittest.TestCase):
    def test_every_free_cell_reaches_the_goal(self):
        rng = np.random.default_rng(1)
        for rooms in (3, 5, 7):
            batch = make_batch(rng, MazeSpec(rooms), list(REGIMES) * 2, ticks=40)
            blocked = batch["blocked"].reshape(-1, *batch["blocked"].shape[2:])
            goal = batch["goal"].reshape(-1)
            dist = distance_maps(torch.from_numpy(blocked), torch.from_numpy(goal)).numpy()
            self.assertTrue(np.isfinite(dist[~blocked]).all())
            self.assertFalse(blocked.reshape(len(goal), -1)[np.arange(len(goal)), goal].any())
            for i in rng.choice(len(goal), 25, replace=False):
                np.testing.assert_array_equal(dist[i], bfs(blocked[i], goal[i]))

    def test_regimes_change_as_specified(self):
        rng = np.random.default_rng(2)
        spec = MazeSpec(5)
        static = make_episode(rng, spec, "static", 30)
        self.assertFalse(static["changed"].any())
        full = make_episode(rng, spec, "full_return", 120, (6, 6))
        maps = full["blocked"].reshape(121, -1)
        # the third configuration after a change is a complete return
        keys = [(m.tobytes(), g) for m, g in zip(maps, full["goal"])]
        self.assertEqual(len(set(keys)), 3)
        cont = make_episode(rng, spec, "continuous", 100)
        self.assertGreater(cont["changed"].sum(), 20)

    def test_optimal_actions_decrease_distance(self):
        rng = np.random.default_rng(3)
        batch = make_batch(rng, MazeSpec(4), ["continuous"] * 4, ticks=5)
        blocked = torch.from_numpy(batch["blocked"][:, 0])
        goal = torch.from_numpy(batch["goal"][:, 0])
        dist = distance_maps(blocked, goal)
        opt = optimal_actions(dist, blocked)
        b, _, h, w = opt.shape
        for i in range(b):
            for cell in np.flatnonzero(~blocked[i].numpy().reshape(-1))[:40]:
                for action in np.flatnonzero(opt[i, :, cell // w, cell % w].numpy()):
                    nxt = int(move(blocked[i:i + 1], torch.tensor([cell]), torch.tensor([action]))[0])
                    expected = 0 if cell == int(goal[i]) else dist[i].view(-1)[cell] - 1
                    self.assertEqual(float(dist[i].view(-1)[nxt]), float(expected))


class PlannerTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        rng = np.random.default_rng(4)
        batch = make_batch(rng, MazeSpec(3), ["full_return"] * 3, ticks=30, phase_range=(3, 3))
        self.blocked = torch.from_numpy(batch["blocked"])
        self.goal = torch.from_numpy(batch["goal"])

    def model(self, mode, **kwargs):
        return build_planner(PlannerConfig(mode=mode, channels=16, key_dim=8, memory_size=4,
                                           warmup=2, **kwargs)).eval()

    @torch.no_grad()
    def rollout(self, model, ticks, think=1):
        state = model.start(self.blocked[:, 0], self.goal[:, 0])
        history = []
        for t in range(1, ticks + 1):
            state = model.observe(state, self.blocked[:, t], self.goal[:, t])
            history.append(state)
            state, _ = model.think(state, think)
        return state, history

    def test_carry_keeps_workspace_and_restart_reinitialises(self):
        carry, restart = self.model("carry"), self.model("restart")
        restart.load_state_dict(carry.state_dict())
        state = carry.start(self.blocked[:, 0], self.goal[:, 0])
        state, _ = carry.think(state, 2)
        t = int(np.flatnonzero((self.blocked[0, 1:] != self.blocked[0, :-1]).flatten(1).any(1)
                               | (self.goal[0, 1:] != self.goal[0, :-1]))[0]) + 1
        state = replace(state, blocked=self.blocked[:, t - 1], goal=self.goal[:, t - 1])
        kept = carry.observe(state, self.blocked[:, t], self.goal[:, t])
        torch.testing.assert_close(kept.z, state.z)
        reset = restart.observe(state, self.blocked[:, t], self.goal[:, t])
        torch.testing.assert_close(reset.z[0], restart.initialise(restart.encode(self.blocked[:, t], self.goal[:, t]))[0])

    def test_goal_restart_reinitialises_only_on_goal_moves(self):
        model = self.model("goal_restart")
        state = model.start(self.blocked[:, 0], self.goal[:, 0])
        state, _ = model.think(state, 2)
        remapped = model.observe(state, ~self.blocked[:, 0], self.goal[:, 0])
        torch.testing.assert_close(remapped.z, state.z)
        goal = (self.goal[:, 0] + 1) % self.blocked[0].numel()
        moved = model.observe(state, self.blocked[:, 0], goal)
        torch.testing.assert_close(moved.z, model.initialise(model.encode(self.blocked[:, 0], goal)))

    def test_snapshot_restores_an_exact_return(self):
        model = self.model("snapshot")
        _, history = self.rollout(model, 12)
        returned = [h for h in history if h.tick > 6 and h.mem_valid.any()]
        self.assertTrue(returned)
        for h in history:
            self.assertTrue(torch.isfinite(h.z).all())
        # after three configurations every return finds its own stored computation
        self.assertTrue(bool(history[-1].mem_valid[:, :3].all()))

    def test_every_mode_is_finite_and_batch_independent(self):
        for mode, extra in [(mode, {}) for mode in MODES] + [("factorized", {"map_memory": 4}),
                                                            ("carry", {"goal_read": True}),
                                                            ("carry", {"goal_read": True, "change_gate": True})]:
            model = self.model(mode, **extra)
            state, _ = self.rollout(model, 8, think=2)
            self.assertTrue(torch.isfinite(state.z).all(), mode)
            single = build_planner(model.config).eval()
            single.load_state_dict(model.state_dict())
            blocked, goal = self.blocked, self.goal
            self.blocked, self.goal = blocked[1:2], goal[1:2]
            alone, _ = self.rollout(single, 8, think=2)
            self.blocked, self.goal = blocked, goal
            torch.testing.assert_close(alone.z[0], state.z[1], atol=1e-5, rtol=1e-4)

    def test_factorized_goal_change_keeps_map_computation(self):
        model = self.model("factorized")
        c = model.config.channels
        state = model.start(self.blocked[:, 0], self.goal[:, 0])
        state, _ = model.think(state, 3)
        moved = model.observe(state, self.blocked[:, 0], (self.goal[:, 0] + 1) % self.blocked[0].numel())
        torch.testing.assert_close(moved.z[:, :c], state.z[:, :c])
        self.assertFalse(torch.allclose(moved.z[:, c:], state.z[:, c:]))
        remapped = model.observe(state, ~self.blocked[:, 0], self.goal[:, 0])
        torch.testing.assert_close(remapped.z, state.z)
        kept = self.model("factorized", reseed=False)
        kept.load_state_dict(model.state_dict())
        carried = kept.observe(state, self.blocked[:, 0], (self.goal[:, 0] + 1) % self.blocked[0].numel())
        torch.testing.assert_close(carried.z, state.z)

    def test_factorized_map_records_restore_a_returning_map(self):
        model = self.model("factorized", map_memory=4)
        c, cells = model.config.channels, self.blocked[0, 0].numel()
        state = model.start(self.blocked[:, 0], self.goal[:, 0])
        state, _ = model.think(state, 2)
        kept_m = state.z[:, :c]
        away = model.observe(state, ~self.blocked[:, 0], self.goal[:, 0])
        torch.testing.assert_close(away.z, state.z)
        away, _ = model.think(away, 2)
        goal = (self.goal[:, 0] + 1) % cells
        back = model.observe(away, self.blocked[:, 0], goal)
        torch.testing.assert_close(back.z[:, :c], kept_m)
        x = model.encode(self.blocked[:, 0])
        torch.testing.assert_close(back.z[:, c:], model._seed(kept_m, x, goal))

    def test_goal_read_sees_the_workspace_at_the_goal(self):
        model = self.model("carry", goal_read=True)
        state = model.start(self.blocked[:, 0], self.goal[:, 0])
        moved = replace(state, goal=(self.goal[:, 0] + 1) % self.blocked[0, 0].numel())
        self.assertFalse(torch.allclose(model.advance(state, 1).z, model.advance(moved, 1).z))

    def test_change_gate_acts_only_when_the_input_changes(self):
        model = self.model("carry", goal_read=True, change_gate=True)
        state = model.start(self.blocked[:, 0], self.goal[:, 0])
        same = model.observe(state, self.blocked[:, 0], self.goal[:, 0])
        torch.testing.assert_close(same.z, state.z)
        moved = model.observe(state, self.blocked[:, 0], (self.goal[:, 0] + 1) % self.blocked[0, 0].numel())
        self.assertEqual(moved.gates.shape, state.z.shape)
        self.assertFalse(torch.allclose(moved.z, state.z))

    def test_advance_matches_the_last_readout_of_think(self):
        for mode in ("carry", "factorized"):
            model = self.model(mode)
            state = model.start(self.blocked[:, 0], self.goal[:, 0])
            thought, outputs = model.think(state, 3)
            advanced = model.advance(state, 3)
            torch.testing.assert_close(advanced.z, thought.z)
            torch.testing.assert_close(model.readout_state(advanced)[0], outputs[-1][0])

    def test_learned_reader_gates_are_distributions(self):
        for mode in ("memory_local", "memory_global"):
            _, history = self.rollout(self.model(mode), 10)
            gates = history[-1].gates
            torch.testing.assert_close(gates.sum(1), torch.ones_like(gates[:, 0]))



class OnlineTests(unittest.TestCase):
    def test_one_online_update_changes_parameters_with_either_teacher(self):
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments" / "dynamic_maze"))
        import online
        batch = make_batch(np.random.default_rng(5), MazeSpec(3), ["continuous", "partial_return"], ticks=4)
        blocked, goal = torch.from_numpy(batch["blocked"]), torch.from_numpy(batch["goal"])
        for teacher in online.TEACHERS:
            torch.manual_seed(0)
            model = build_planner(PlannerConfig(mode="carry", channels=8, warmup=2))
            frozen = build_planner(model.config).requires_grad_(False)
            before = [p.detach().clone() for p in model.parameters()]
            loss = online.update_loss(model, [(blocked, goal)], teacher, 1, 0.5, frozen, depth=4)
            loss.backward()
            torch.optim.SGD(model.parameters(), lr=0.1).step()
            self.assertTrue(torch.isfinite(loss), teacher)
            self.assertTrue(any(not torch.equal(a, b) for a, b in zip(before, model.parameters())), teacher)


if __name__ == "__main__":
    unittest.main()
