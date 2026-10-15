import unittest
import numpy as np

from grid import (Context, GridWorld, choose_action, geometric_values, make_trace,
                  optimal_action, reference_values, transition_table)
from planners import METHODS, Planner


class GridTests(unittest.TestCase):
    def test_time_advances_during_computation(self):
        world = GridWorld(make_trace(0, "continuous"))
        planner = Planner("carry")
        previous = world.observe()
        planner.compute(previous)
        fresh = world.advance_time()
        self.assertEqual(fresh.tick, previous.tick + 1)
        world.move(choose_action(planner.values, fresh, 9))
        self.assertEqual(world.tick, 1)

    def test_every_method_uses_new_obstacle_and_new_goal(self):
        blocked = np.zeros(81, dtype=bool)
        original = Context(blocked, 44, 40, 0, 0)
        changed = blocked.copy()
        changed[41] = True
        fresh = Context(changed, 39, 40, 1, 1)
        for method in METHODS:
            planner = Planner(method)
            planner.compute(original)
            # Stale values strongly request the newly blocked right-hand cell.
            planner.values[:] = 0
            planner.values[41] = 100
            action = choose_action(planner.values, fresh, 9)
            self.assertEqual(action, 4)

    def test_equal_cell_budget(self):
        for fraction in (.25, 1.):
            totals = []
            for method in METHODS:
                planner = Planner(method, budget_fraction=fraction)
                world = GridWorld(make_trace(1, "continuous"))
                for _ in range(11):
                    planner.compute(world.observe())
                    world.advance_time()
                totals.append(planner.total_backups)
            self.assertEqual(len(set(totals)), 1)
            self.assertEqual(totals[0], 11 * round(81 * fraction))

    def test_snapshot_restores_workspace_not_agent_or_goal(self):
        planner = Planner("snapshot")
        blocked = np.zeros(81, dtype=bool)
        first = Context(blocked, 80, 0, 0, 0)
        planner.compute(first)
        stored = planner.values.copy()
        other_map = blocked.copy()
        other_map[40] = True
        planner.compute(Context(other_map, 80, 1, 0, 1))
        fresh = Context(blocked, 80, 30, 0, 2)
        retrieved = planner._read(fresh)
        np.testing.assert_array_equal(stored, retrieved)
        self.assertEqual(fresh.position, 30)
        self.assertEqual(fresh.goal, 80)

    def test_factorized_read_does_not_use_reference(self):
        planner = Planner("factorized")
        context = GridWorld(make_trace(2, "partial_return")).observe()
        planner.compute(context)
        changed = context.blocked.copy()
        changed[0] = ~changed[0]
        result = planner._read(Context(changed, context.goal, 60, 1, 1))
        self.assertEqual(result.shape, (81,))
        self.assertGreater(planner.local_reads, 0)
        self.assertEqual(planner.total_backups, 81)

    def test_reference_follows_current_map_shortest_path(self):
        trace = make_trace(3, "full_return")
        context = GridWorld(trace).observe()
        for _ in range(81):
            if context.position == context.goal:
                break
            action = optimal_action(context, 9)
            nxt, legal = transition_table(context.blocked, 9)
            self.assertTrue(legal[context.position, action])
            context = Context(context.blocked, context.goal, int(nxt[context.position, action]), 0, 0)
        self.assertEqual(context.position, context.goal)


if __name__ == "__main__":
    unittest.main()
