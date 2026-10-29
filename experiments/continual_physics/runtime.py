"""Persistent physical inference with explicit, once-only tool-result delivery."""
from __future__ import annotations

from dataclasses import dataclass

import torch

from cdm.continual_experiment import Activity
from experiments.continual_experiment.runtime import ContinualSession


@dataclass
class PendingResult:
    event_id: int
    request_id: int
    observation: dict
    feedback: torch.Tensor
    physical_feedback: dict
    remaining_ticks: int


class ContinualPhysicsSession(ContinualSession):
    """Reuse the event runtime with physical outcomes and delayed delivery.

    Holding a result changes its delivery time, not the simulator's clock.
    Undelivered observation tensors are never used by ``tick``.
    """

    def __init__(self, model, device="cpu"):
        super().__init__(model, device)
        self.delivery = None
        self.delivered_results = 0
        self.pending_ticks = 0

    @torch.no_grad()
    def receive(self, event_id, observation, *, request_id=None, feedback=None,
                physical_feedback=None):
        if event_id <= self.last_event:
            raise ValueError("An observation must be received exactly once, in arrival order")
        if feedback is not None:
            if self.pending_request is None or request_id != self.pending_request:
                raise ValueError("Feedback must identify the pending request")
            supplied = physical_feedback
            if supplied is not None and not supplied["mask"].any():
                supplied = None
            self.state = self.model.feedback(self.state, feedback,
                                             physical_feedback=supplied)
            self.pending_request = None
            self.delivered_results += 1
        elif request_id is not None or physical_feedback is not None:
            raise ValueError("A request result requires its execution feedback")
        self.state = self.model.receive(self.state, observation)
        self.observation, self.last_event = observation, event_id

    @torch.no_grad()
    def tick(self):
        if self.observation is None:
            return {"activity": "WAIT"}
        out = self.model.readout(self.state, self.observation["candidates"],
                                 self.observation["action_mask"])
        if "physical_lower_hz" in self.observation:
            allowed = ((out["physical_settings"] >= self.observation["physical_lower_hz"][:, None])
                       & (out["physical_settings"] <= self.observation["physical_upper_hz"][:, None]))
            out["logits"] = torch.cat((out["logits"][:, :20],
                                        out["logits"][:, 20:].masked_fill(~allowed, -torch.inf)), -1)
            available = out["logits"].isfinite().any(-1)
            out["activity_logits"][:, Activity.ACT] = out["activity_logits"][:, Activity.ACT].masked_fill(~available, -torch.inf)
            out["activity_logits"][:, Activity.WAIT] = torch.where(
                ~available, 0., out["activity_logits"][:, Activity.WAIT])
        selected = out["activity_logits"].argmax(-1)
        self.state = self.model.internal(self.state, selected, self.observation)
        activity = Activity(int(selected[0]))
        result = {"activity": activity.name, "prediction": out}
        if activity == Activity.ACT:
            action = out["logits"].argmax(-1)
            self.state = self.model.begin_action(self.state, out["features"], action,
                                                 physical=out)
            self.request += 1
            self.pending_request = self.request
            result.update(request_id=self.request, action=int(action[0]))
        return result

    def queue_result(self, event_id, request_id, observation, feedback,
                     physical_feedback, *, delay_ticks=0):
        if self.delivery is not None:
            raise ValueError("Deliver the pending result before queueing another")
        if request_id != self.pending_request or self.pending_request is None:
            raise ValueError("A queued result must identify the dispatched request")
        if delay_ticks < 0:
            raise ValueError("A result delivery delay cannot be negative")
        self.delivery = PendingResult(event_id, request_id, observation, feedback,
                                      physical_feedback, delay_ticks)

    def poll_result(self):
        """Run one internal tick or deliver one result; never dispatch a new tool."""
        if self.delivery is None:
            return {"delivered": False, "activity": "WAIT"}
        pending = self.delivery
        if pending.remaining_ticks:
            result = self.tick()
            if "action" in result:
                raise RuntimeError("The model dispatched while a tool result was pending")
            pending.remaining_ticks -= 1
            self.pending_ticks += 1
            return {"delivered": False, "activity": result["activity"]}
        self.receive(pending.event_id, pending.observation,
                     request_id=pending.request_id, feedback=pending.feedback,
                     physical_feedback=pending.physical_feedback)
        self.delivery = None
        return {"delivered": True}


def setting_outcomes(decision):
    """Distinct public outcomes for the requested object and device."""
    attempts = []
    if decision["action_index"] >= 20:
        result = decision["result"]
        attempts.extend(result.get("attempts", []))
        setting = result.get("setting")
        request = result.get("request", {})
        if (setting is not None and setting.get("object_handle") == request.get("target_handle")
                and setting.get("reactor_handle") == request.get("reactor_handle")):
            attempts.append(setting)
    return list(dict.fromkeys((float(row["frequency_hz"]), row["activated"])
                                 for row in attempts
                                 if row.get("activated") in (True, False)
                                 and row.get("frequency_hz") is not None))


def physical_feedback(decision, device, *, tolerance_hz=2.):
    """Intervals from actual returned settings, never the requested setpoint.

    ``tolerance_hz`` is the native task's activation rule, separate from the
    available dial step. Failed and successful settings share that same rule.
    Measurement actions return no setting interval.
    """
    received = setting_outcomes(decision)
    intervals = torch.tensor([[[value - tolerance_hz, value + tolerance_hz]
                               for value, _ in received]], device=device,
                              dtype=torch.float64).reshape(1, -1, 2)
    success = torch.tensor([[bool(activated) for _, activated in received]],
                            device=device, dtype=torch.bool)
    return {"intervals": intervals, "success": success,
            "mask": torch.ones_like(success)}
