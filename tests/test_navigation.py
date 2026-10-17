"""Vehicles behind walls: motion, path distances, the field, and their place among the other families."""
import unittest

import torch

from cdm.instrument import InstrumentSpec
from cdm.navigation import FIELDS, NavigationFleet, NavigationSpec, path_distance
from cdm.thermal import ThermalSpec
from experiments.lifetime.runtime import make_fleet
from experiments.lifetime.train import first_observation

QUIET = NavigationSpec(walls=1, request_rate=0.0, dwell=1e9, jitter=0.0, process_noise=0.0, drift=0.0)


def condition(wall):
    """Height 1, floor 0, width 0.6, speed 0.1, beacon at (0.5, 0), one wall."""
    p = torch.zeros(FIELDS + 4)
    p[:FIELDS] = torch.tensor([1.0, 0.0, 0.6, 0.1, 0.5, 0.0])
    p[FIELDS:] = torch.tensor(wall)
    return p


def fleet(walls):
    env = NavigationFleet(len(walls), QUIET, seed=0)
    env.params[:] = torch.stack([condition(w) for w in walls]).unsqueeze(1)
    env.distance = env._distances(env.params)
    env.position = torch.tensor([[-0.5, 0.0]] * len(walls))
    return env


class NavigationTests(unittest.TestCase):
    def test_a_wall_stops_the_vehicle_and_the_path_bends_around_it(self):
        env = fleet([[0.0, 0.0, -0.5, 0.5], [0.0, 0.0, 2.0, 3.0]])     # a wall at x = 0, and one outside
        east = int((env.table - torch.tensor([1.0, 0.0])).norm(dim=1).argmin())
        for _ in range(15):
            out = env.step(torch.full((2,), east), torch.zeros(2, dtype=torch.bool))
        self.assertLess(float(env.position[0, 0]), 0.0)
        self.assertGreater(float(env.position[0, 0]), -1e-2)
        self.assertAlmostEqual(float(env.position[1, 0]), 1.0, places=5)
        D = path_distance(env.params[:, 0], 25, 1)
        start = (12 - 6, 12)                                          # the cell at (-0.5, 0)
        self.assertGreater(float(D[0][start]), 1.4)
        self.assertAlmostEqual(float(D[1][start]), 1.0, places=4)
        self.assertLess(float(out["output"][0]), float(out["output"][1]))

    def test_waypoints_reach_the_beacon_behind_the_wall(self):
        env = fleet([[0.0, 0.0, -0.5, 0.5]])
        point = lambda x, y: int((env.table - torch.tensor([x, y])).norm(dim=1).argmin())
        for target in [point(-0.5, 0.75)] * 8 + [point(0.5, 0.75)] * 12 + [point(0.5, 0.0)] * 10:
            out = env.step(torch.tensor([target]), torch.zeros(1, dtype=torch.bool))
        torch.testing.assert_close(env.position[0], torch.tensor([0.5, 0.0]), atol=1e-4, rtol=0)
        best = env.steady(env.current(), env.table.expand(1, -1, -1)).max(1).values
        self.assertGreater(float(out["output"][0]), float(best[0]) - 0.05)

    def test_expected_next_is_the_noise_free_step(self):
        env = fleet([[1.0, 0.3, -0.8, 0.4]])
        before = env.expected_next(env.table.expand(1, -1, -1))
        for action in (0, 40, 77):
            copy = fleet([[1.0, 0.3, -0.8, 0.4]])
            out = copy.step(torch.tensor([action]), torch.zeros(1, dtype=torch.bool))
            self.assertAlmostEqual(float(out["output"][0]), float(before[0, action]), places=5)

    def test_three_families_run_as_one_fleet_and_copies_share_randomness(self):
        specs = {"instrument": InstrumentSpec(search_share=0.5), "thermal": ThermalSpec(),
                 "navigation": NavigationSpec(search_share=0.5, request_rate=0.1, dwell=30.0)}
        env = make_fleet("all", 9, 4, "cpu", specs)
        obs = first_observation(env)
        self.assertEqual(obs["cheap"].shape, (9,))
        copies = env.branch(torch.tensor([1, 7]), 2, seed=3)
        outs = [copies.step(torch.full((4,), 50), torch.zeros(4, dtype=torch.bool))["cheap"] for _ in range(40)]
        outs = torch.stack(outs)
        torch.testing.assert_close(outs[:, 0], outs[:, 1], rtol=0, atol=0)
        torch.testing.assert_close(outs[:, 2], outs[:, 3], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
