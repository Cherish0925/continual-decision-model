"""Compare the fast reader with native observations during real instrument use."""
import base64
import contextlib
from importlib.util import find_spec
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from experiments.discoveryworld.observation import get_observation


@unittest.skipUnless(find_spec("discoveryworld"), "Install the pinned DiscoveryWorld simulator")
class DiscoveryWorldObservationTests(unittest.TestCase):
    def test_native_png_and_rgb_match_through_instrument_actions(self):
        import numpy as np
        from PIL import Image
        from discoveryworld.DiscoveryWorldAPI import DiscoveryWorldAPI

        fixture = Path(__file__).resolve().parents[1] / "experiments/discoveryworld/fixtures/reactor_measurement_seed100.jsonl"
        actions = [json.loads(line) for line in fixture.read_text().splitlines()]
        traces = {}
        # Freeze only the decorative UI blink, not the simulator's world clock.
        with TemporaryDirectory() as temporary, patch("pygame.time.get_ticks", return_value=0):
            for mode in ("native", "png", "rgb"):
                with contextlib.redirect_stdout(io.StringIO()):
                    api = DiscoveryWorldAPI(threadID=0)
                    api.FRAME_DIR = str(Path(temporary) / mode)
                    self.assertTrue(api.loadScenario("Reactor Lab", "Normal", randomSeed=100, numUserAgents=1))
                traces[mode] = []
                for action in [None, *actions]:
                    result = None
                    if action is not None:
                        result = api.performAgentAction(agentIdx=0, actionJSON=action)
                        self.assertTrue(result["success"])
                        self.assertTrue(api.tick()["success"])
                    raw = api.getAgentObservation(0) if mode == "native" else get_observation(api, rgb=mode == "rgb")
                    pixels = []
                    for suffix in ("no_grid", "with_grid"):
                        if mode == "rgb":
                            pixels.append(raw["vision"][f"rgb_{suffix}"])
                        else:
                            data = raw["vision"][f"base64_{suffix}"].split(",", 1)[1]
                            pixels.append(np.array(Image.open(io.BytesIO(base64.b64decode(data))).convert("RGB")))
                    traces[mode].append((raw["ui"], result, pixels))
                    self.assertEqual(api.getStepCounter(), raw["ui"]["world_steps"])
                if mode != "native":
                    self.assertEqual(list(Path(api.FRAME_DIR).iterdir()), [])

        for mode in ("png", "rgb"):
            for expected, actual in zip(traces["native"], traces[mode]):
                self.assertEqual(expected[:2], actual[:2])
                for native_pixels, fast_pixels in zip(expected[2], actual[2]):
                    np.testing.assert_array_equal(native_pixels, fast_pixels)


if __name__ == "__main__":
    unittest.main()

