"""A single-stream event interface; tools may return after internal computation."""
from __future__ import annotations

import torch

from cdm.continual_experiment import Activity


class ContinualSession:
    def __init__(self, model, device="cpu"):
        self.model = model
        self.state = model.initial(1, device)
        self.observation = None
        self.last_event = -1
        self.request = 0
        self.pending_request = None

    @torch.no_grad()
    def receive(self, event_id, observation, *, request_id=None, feedback=None):
        """Event IDs are monotonically increasing. Feedback matches one request.

        Unsolicited observations can arrive while a tool is running. They update
        cognition without treating that tool as complete.
        """
        if event_id <= self.last_event:
            raise ValueError("An observation must be received exactly once, in arrival order")
        if feedback is not None:
            if self.pending_request is None or request_id != self.pending_request:
                raise ValueError("Feedback must identify the pending request")
            self.state = self.model.feedback(self.state, feedback)
            self.pending_request = None
        elif request_id is not None:
            raise ValueError("A request result requires feedback")
        self.state = self.model.receive(self.state, observation)
        self.observation, self.last_event = observation, event_id

    @torch.no_grad()
    def tick(self):
        if self.observation is None:
            return {"activity": "WAIT"}
        out = self.model.readout(self.state, self.observation["candidates"], self.observation["action_mask"])
        selected = out["activity_logits"].argmax(-1)
        self.state = self.model.internal(self.state, selected, self.observation)
        activity = Activity(int(selected[0]))
        result = {"activity": activity.name}
        if activity == Activity.ACT:
            action = out["logits"].argmax(-1)
            self.state = self.model.begin_action(self.state, out["features"], action)
            self.request += 1
            self.pending_request = self.request
            result.update(request_id=self.request, action=int(action[0]))
        return result
