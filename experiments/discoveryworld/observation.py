"""Native DiscoveryWorld observations without per-step frame-file writes.

Uses the official renderer and public UI at the revision pinned in adapter.py.
PNG mode keeps the native observation interface. RGB mode supplies uint8 HWC
arrays directly to an image processor; its vision keys are rgb_no_grid and
rgb_with_grid. Both modes preserve both views and consume the UI message queue
once, just as the native observation call does.
"""
from __future__ import annotations

import base64
from io import BytesIO


def get_observation(api, agentIdx: int = 0, *, rgb: bool = False) -> dict:
    """Read one observation, without advancing the world or saving debug frames."""
    import pygame
    from PIL import Image

    result = {"errors": [], "ui": {}, "vision": {}}
    if not 0 <= agentIdx < api.numUserAgents:
        result["errors"].append("Agent index out of range")
        return result
    if api.world is None:
        result["errors"].append("World is not initialized")
        return result

    ui = api.ui[agentIdx]
    x, y = ui.currentAgent.getWorldLocation()
    start_x = x - int(api.viewportSizeX / 2)
    start_y = y - int(api.viewportSizeY / 2)
    width = int(api.viewportSizeX * 32 * api.renderScale)
    height = int(api.viewportSizeY * 32 * api.renderScale)

    for grid, suffix in ((False, "no_grid"), (True, "with_grid")):
        api.window.fill((0, 0, 0))
        api.world.renderViewport(api.window, start_x, start_y,
                                 api.viewportSizeX, api.viewportSizeY,
                                 0, 0, api.renderScale, includeGrid=grid)
        # Rendering also refreshes accessible argument objects and dialog state.
        ui.render()
        viewport = api.window.subsurface((0, 0, width, height))
        if rgb:
            pixels = pygame.surfarray.array3d(viewport).transpose(1, 0, 2).copy()
            result["vision"][f"rgb_{suffix}"] = pixels
        else:
            pixels = Image.frombytes("RGB", (width, height), pygame.image.tobytes(viewport, "RGB"))
            encoded = BytesIO()
            pixels.save(encoded, format="PNG", compress_level=1)
            payload = base64.b64encode(encoded.getvalue()).decode("ascii")
            result["vision"][f"base64_{suffix}"] = "data:image/png;base64," + payload

    result["ui"] = ui.renderJSON()
    api.taskProgress = result["ui"]["taskProgress"]
    api.steps = result["ui"]["world_steps"]
    return result
