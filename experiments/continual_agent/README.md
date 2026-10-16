# Continual agent: one running process that acts, looks, thinks and learns

This experiment puts the CDM operations into one continuously running agent and trains the choice among them. It builds on the planners of the [changing-maze study](../dynamic_maze/README.md).

## Task

A long stream of room-maze sites. Doors toggle and obstacles move continually. Requests (goals) arrive every 20–50 ticks; after reaching a request the agent is idle until the next one, so busy and idle periods alternate. The agent sees the true map within two cells and keeps the last map it saw elsewhere, with the age of every cell. A survey returns the whole map after a delay of two ticks; the agent keeps moving and computing while it waits. Reward: +1 on reaching a request, −0.02 per tick while a request is pending, minus a cost per refinement and per survey.

## Agent

| Part | Content | Update |
|---|---|---|
| Workspace | Spatial feature map of the planner (goal read; trained on partial views with an age plane, or with a learned keep-or-rebuild gate) | Shared refinements; the gate or rebuild when inputs change |
| Information state | Last seen map and the age of each cell | Local view every tick, survey results when they arrive |
| Activity head | Policy and value over `{0, 1, 2, 4} refinements × {survey, no survey} × {move, stay}` from the planner's state at the agent's cell, its readout, and counters (ticks since survey, since request, pending survey, mean map age) | PPO on reward minus costs over 2,048 parallel streams |
| Parameters | Planner weights | Online learning steps paid from unused compute ([`operate.py`](operate.py)) |

Computation and action happen in the same tick: the chosen refinements run, then the agent moves along the refreshed plan or stays. Surveys are asynchronous.

[`agent.py`](agent.py) trains the activity head with the planner fixed and compares it with hand-written schedules: one refinement and a step every tick; four refinements every tick; one refinement with a survey every 16 ticks; and a scaled schedule that computes and surveys only while a request is pending, with four refinements right after a request arrives or while the readout is still changing.

[`operate.py`](operate.py) runs one long stream with a shared compute budget of two refinements per stream per tick. Refinements the activity head leaves unused pay for learning steps of the planner. The stream changes its rules in phases: the training distribution, then doors that change three times faster, then 9×9 mazes, then the training distribution again. Learning targets:

- **distill**: on replayed sequences of the agent's own beliefs and goals, each one-refinement readout moves toward the same network's readout 16 refinements deeper. This makes existing computation faster to use.
- **hindsight**: when a survey returns, the belief, ages and goal held when it was requested are paired with the optimal actions on the returned map. This carries what the world actually did, including a change in how fast doors change.

Conditions: no learning; learning from unused compute with each target and with both; learning at a fixed rate regardless of the activity head.

## Trained activity heads

The three activity heads of the results are in release [v0.2.0](https://github.com/Cherish0925/continual-decision-model/releases/tag/v0.2.0) as `runs/continual_agent/<planner>_s0.pt`, together with the planners they run on; [`behaviour.py`](behaviour.py) and `operate.py --policy` read them.

## Run

```bash
python experiments/continual_agent/agent.py --model runs/dynamic_maze/main/partial_read_age/seed0/model.pt \
    --streams 2048 --site-ticks 128 --iterations 120 --output runs/continual_agent/partial_read_age_s0.json
python experiments/continual_agent/operate.py --model runs/dynamic_maze/main/partial_read_age/seed0/model.pt \
    --condition spare --targets distill hindsight --replay-old --output runs/continual_agent/operate/spare.json
```
