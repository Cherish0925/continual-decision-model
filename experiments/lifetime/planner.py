"""Control by imagined rollouts of a trained model: a model-based alternative to its control head.

For every action, the model imagines holding it for ``ticks`` ticks: each imagined tick feeds the fast state an
event whose cheap reading is the reading the model predicted for that action, with no precise reading, no new
request and no activity, and then predicts the next reading from the new state with the instance's adapter. The
workspace is held as it is. The action whose imagined readings stay closest to the goal over the rollout is
chosen: the setpoint, or for a request for the largest output, the largest imagined reading. One imagined tick is
the greedy readout of the model's own predictions. The rollout uses the commanded actuator values as the
previous control, which is what instruments and thermal processes report; vehicles report their position, which
the model does not predict, so this planner applies to the first two.
"""
from __future__ import annotations

from types import SimpleNamespace

import torch

from experiments.lifetime.runtime import event_features


def imagined_control(ticks: int, discount: float = 1.0):
    """A control override for ``runtime.tick``: the action whose imagined rollout best reaches the goal."""

    def choose(model, agent, h, obs, pending, mean):
        n, K = mean.shape
        goals, w = model.config.goals, model.config.width
        values = model.action_values                                         # [K, dims]
        controls = values.unsqueeze(0).expand(n, -1, -1).reshape(n * K, -1)
        repeat = lambda x: x.repeat_interleave(K, 0)
        hk = repeat(h.float())
        with torch.autocast(h.device.type, dtype=torch.bfloat16, enabled=h.device.type == "cuda"):
            held = repeat(model.norm_z(agent.z.mean(1)))                     # the workspace, held
        reading = mean.float().reshape(n * K)
        cost = torch.zeros(n, K, device=mean.device)
        weight = 1.0
        for j in range(ticks):
            if j:
                event_obs = {"cheap": reading, "precise": torch.zeros_like(reading),
                             "arrived": torch.zeros_like(reading, dtype=torch.bool), "setpoint": repeat(obs["setpoint"]),
                             "active": repeat(obs["active"]), "started": torch.zeros_like(reading, dtype=torch.bool),
                             "search": repeat(obs["search"])}
                imagined = SimpleNamespace(busy=repeat(agent.busy), since_measure=repeat(agent.since_measure) + j,
                                           predicted=reading,
                                           last=torch.cat((controls, torch.zeros(n * K, 3, device=h.device)), 1))
                with torch.autocast(h.device.type, dtype=torch.bfloat16, enabled=h.device.type == "cuda"):
                    hk = model.fast(hk, event_features(event_obs, imagined, repeat(pending), goals)).float()
                    state = torch.cat((model.norm_h(hk), held), -1).view(n, K, -1)
                    x = torch.cat((state, model.action_features().unsqueeze(0).expand(n, -1, -1).to(state.dtype)), -1)
                    reading = model.predict(x, agent.A, agent.B)[0].float().reshape(n * K)
            r = reading.view(n, K)
            goal = torch.where(obs["search"], r.max(1).values, obs["setpoint"]).unsqueeze(1)
            cost = cost + weight * (r - goal).abs()
            weight *= discount
        return cost.argmin(1)

    return choose
