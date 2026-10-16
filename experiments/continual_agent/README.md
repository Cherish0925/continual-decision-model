# Continual agent: one running process that acts, looks, thinks and learns

This experiment puts perception, refinement, action, and parameter updates in one simulation process. It trains the choice of refinement count, surveys, and movement; the learning-update schedule remains a program rule. It builds on the planners of the [changing-maze study](../dynamic_maze/README.md).

## Task

A long stream of room-maze sites. Doors toggle and obstacles move continually. Requests (goals) arrive every 20–50 ticks; after reaching a request the agent is idle until the next one, so busy and idle periods alternate. The agent sees the true map within two cells and keeps the last map it saw elsewhere, with the age of every cell. A survey returns the whole map after a delay of two ticks; the agent keeps moving and computing while it waits. Reward: +1 on reaching a request, −0.02 per tick while a request is pending, minus a cost per refinement and per survey.

## Agent

| Part | Content | Update |
|---|---|---|
| Workspace | Spatial feature map of the planner (goal read; trained on partial views with an age plane, or with a learned keep-or-rebuild gate) | Shared refinements; the gate or rebuild when inputs change |
| Information state | Last seen map and the age of each cell | Local view every tick, survey results when they arrive |
| Activity head | Policy and value over `{0, 1, 2, 4} refinements × {survey, no survey} × {move, stay}` from the planner's state at the agent's cell, its readout, and counters (ticks since survey, since request, pending survey, mean map age) | PPO on reward minus costs over 2,048 parallel streams |
| Parameters | Planner weights | Online learning steps paid from unused compute ([`operate.py`](operate.py)) |

Computation and action happen in the same simulated tick: the chosen refinements run, then the agent moves along the refreshed plan or stays. Surveys return after a specified number of ticks. The environment does not run on an independent clock during refinement or gradient updates.

Each site starts from the full map with a newly initialized planner and 16 warm-up refinements; observations become partial after initialization. Workspace and belief persist across requests and idle periods within that site. A new site resets them; `operate.py` retains learned weights and replay across sites. The reported activity evaluations use four sites of 128 ticks, not one uninterrupted long-running state.

[`agent.py`](agent.py) trains the activity head with the planner fixed and compares it with hand-written schedules: one refinement and a step every tick; four refinements every tick; one refinement with a survey every 16 ticks; and a scaled schedule that computes and surveys only while a request is pending, with four refinements right after a request arrives or while the readout is still changing.

[`operate.py`](operate.py) runs one long stream with a shared compute budget of two refinements per stream per tick. Refinements the activity head leaves unused pay for learning steps of the planner. The stream changes its rules in phases: the training distribution, then doors that change three times faster, then 9×9 mazes, then the training distribution again. The published operation results use the hand-written `scaled` schedule; loading a trained activity head is supported by `--policy` but has not supplied the reported online-learning result. Learning targets:

- **distill**: on replayed sequences of the agent's own beliefs and goals, each one-refinement readout moves toward a frozen copy's readout 16 refinements deeper by default. The live learner as teacher is a separate comparison. This trains existing deep computation into a faster readout.
- **hindsight**: when a survey returns, the belief, ages and goal held when it was requested are paired with the optimal actions on the returned map. This carries what the world actually did, including a change in how fast doors change.

Conditions: no learning; learning from unused compute with each target and with both; learning at a fixed rate regardless of the activity head.

## Trained activity heads

Release [v0.2.1](https://github.com/Cherish0925/continual-decision-model/releases/tag/v0.2.1) holds the nine activity heads of the results (three planners, three PPO seeds each) as `runs/continual_agent/<planner>_seed<k>.pt`; the planners they run on are in release [v0.2.0](https://github.com/Cherish0925/continual-decision-model/releases/tag/v0.2.0). [`behaviour.py`](behaviour.py) and `operate.py --policy` read them.

## Run

```bash
python experiments/continual_agent/agent.py --model runs/dynamic_maze/main/partial_read_age/seed0/model.pt \
    --streams 2048 --site-ticks 128 --iterations 120 --output runs/continual_agent/partial_read_age_s0.json
python experiments/continual_agent/operate.py --model runs/dynamic_maze/main/partial_read_age/seed0/model.pt \
    --condition spare --targets distill hindsight --replay-old --output runs/continual_agent/operate/spare.json
```
