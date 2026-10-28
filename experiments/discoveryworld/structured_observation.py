"""Public DiscoveryWorld observations for numerical data collection.

This adapter omits images while preserving public UI, action bindings and queue
consumption. Visual policies continue to use DiscoveryWorldEnv's PNG path.
"""
from __future__ import annotations

from copy import deepcopy
from time import perf_counter

from .adapter import DiscoveryWorldEnv


def get_structured_observation(api, agent_index=0):
    ui = api.ui[agent_index]
    inventory = ui.currentAgent.getInventory()
    accessible = ui.currentAgent.getObjectsAgentFacing(respectContainerStatus=True)
    accessible = ui._filterEnvObjects(inventory, accessible)
    arguments = ui._filterDuplicateObjects(inventory + accessible)
    ui.updateArgumentObjects(arguments)
    for box in (1, 2):
        attr = f"curSelectedArgument{box}Idx"
        if getattr(ui, attr) >= len(arguments):
            setattr(ui, attr, len(arguments) - 1)
        ui.changeArgumentBox(delta=0, whichBox=box)
    ui.inModal = ui.dialogToDisplay is None and bool(ui.messageQueueText)
    public = ui.renderJSON()
    api.taskProgress = public["taskProgress"]
    api.steps = public["world_steps"]
    return {"errors": [], "ui": public, "vision": {}}


class StructuredDiscoveryWorldEnv(DiscoveryWorldEnv):
    """Native actions and ticks with public numerical/text observations only."""

    def __init__(self, output, max_steps=1000):
        super().__init__(output, max_steps)
        self.observation_mode = "structured"

    def _observe(self):
        capture = perf_counter() - self.started
        if self.api.isAgentInDialog(agentIdx=0):
            ui = self.api.ui[0]
            text, options = ui.currentAgent.getAgentInDialogWith().dialogTree.getCurrentDialog()
            ui.dialogToDisplay = {"dialogText": text, "dialogOptions": options}
        result = get_structured_observation(self.api)
        return {"ui": deepcopy(result["ui"]), "vision": {},
                "capture_seconds": capture, "arrival_seconds": perf_counter() - self.started}
