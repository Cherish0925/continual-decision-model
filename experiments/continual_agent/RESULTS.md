# Results: continual agent

Task and agent are described in [README.md](README.md). Rewards are per 100 ticks per stream; each evaluation covers four fresh sites of 128 ticks with 2,048 streams (about one million agent-ticks). One PPO seed per planner.

## Learning how much to compute and when to look

The activity head chooses each tick how many refinements to run (0, 1, 2 or 4), whether to survey, and whether to move. It was trained with PPO for 120 iterations against reward minus costs (0.002 per refinement, 0.01 per survey). Greedy evaluation:

| Policy | Carry with goal read and gate | Factorized, partial view, age input | Goal read, partial view, age input | Refinements per tick | Surveys per 100 ticks |
|---|---:|---:|---:|---:|---:|
| One refinement and a step every tick | 1.453 | 1.449 | 1.478 | 1.00 | 0 |
| Four refinements every tick | 0.857 | 0.872 | 0.892 | 4.00 | 0 |
| One refinement, survey every 16 ticks | 1.491 | 1.473 | 1.479 | 1.00 | 5.5 |
| Hand-written: compute and look only for a pending request, four refinements right after it arrives | 1.575 | 1.578 | 1.581 | 0.68 | 3.5 |
| Learned activity head | 1.587 | 1.562 | 1.556 | 0.32–0.51 | 7.7–8.9 |

The learned head reaches a net return close to the best hand-written schedule (−0.025 to +0.012) with 24–54% fewer refinements, and beats the fixed one-refinement schedule by 0.08–0.13. It requests more surveys; the reduction in refinements is not a measurement of total runtime or power savings. Request completion is 80.92–82.54% across the three learned heads. On the gated planner it is 82.54%, compared with 82.00% for the hand-written schedule and 80.01% for fixed one-refinement operation. Where it spends its operations (carry with goal read and gate, two fresh sites):

| State | Refinements per tick | Surveys per 100 ticks | Moves per tick |
|---|---:|---:|---:|
| Idle, no request | 0.27 | 7.1 | 0.47 |
| Request pending | 0.74 | 9.5 | 1.00 |
| Tick the request arrives | 2.59 | 23.1 | 1.00 |
| 1–3 ticks later | 0.87–1.01 | 5.1–7.2 | 1.00 |
| 8 or more ticks later | 0.52 | 10.3 | 1.00 |
| Pending, mean map age 0–3 ticks | 0.87 | 5.7 | 1.00 |
| Pending, mean map age 8–11 ticks | 0.59 | 14.7 | 1.00 |
| Pending, mean map age 16–19 ticks | 0.61 | 26.8 | 1.00 |

Nothing in the reward names these states. The head learned to burst computation and information gathering when a new goal arrives, to keep computing at a low level otherwise, to survey more as its map ages, and to keep surveying while idle so that the next request starts from a fresh map.

## Learning while operating

One stream of 512 parallel agents runs through four phases of four sites each: 5×5 rooms with doors toggling at 0.3 per tick (the planner's training conditions), doors toggling at 1.0, 9×9 rooms, and 5×5 at 0.3 again. The activity schedule is the hand-written one; the planner (goal read, partial view, age input) learns only from unused refinements of a budget of two per stream per tick (four in one condition). Mean reward per 100 ticks in each phase:

| Learning | 5×5 | faster doors | 9×9 | 5×5 again | Learning steps |
|---|---:|---:|---:|---:|---:|
| None | 1.587 | 1.505 | 0.273 | 1.567 | 0 |
| Hindsight from surveys | 1.573 | 1.516 | 0.264 | 1.535 | 324 |
| Distillation from a frozen copy | 1.594 | 1.504 | 0.274 | 1.570 | 324 |
| Both | 1.582 | 1.504 | 0.283 | 1.564 | 324 |
| Both, budget of four | 1.580 | 1.508 | 0.258 | 1.560 | 836 |
| Distillation from the live network, with hindsight, fixed rate | 1.576 | 1.502 | 0.118 | −0.330 | 250 |
| Distillation from the live network | −0.540 | −2.209 | −2.267 | −2.229 | 228 |

In this stream none of the anchored learning signals changed reward beyond site-to-site variation (about ±0.03). Distillation toward the network's own live deeper readout destroys the planner within a few hundred steps, as in the full-view deployment of [`online.py`](../dynamic_maze/RESULTS.md#6-learning-during-operation). There, 200 updates on batches of full-view 9×9 episodes cut the factorized planner's non-optimal moves from 1.90% to 0.91% with the frozen copy as teacher. The difference points at two limits of the continuous stream: the partial view (most errors come from stale maps, which distillation cannot repair) and the small amount of experience the learner sees (64 recorded streams; tens of survey returns per phase for hindsight).

## What this establishes and what it does not

Measured in this task: the agent chooses move/stay every simulated tick, keeps computation and information state across requests and idle periods within a site, receives surveys after a tick delay, and selects refinement and observation through a learned head trained on reward minus costs. Its net return is close to the hand-written schedule with fewer refinements and more surveys. A separate full-view adaptation experiment improves a deployed planner on a new maze size with exact-solver or frozen-copy targets; the tested live-self-distillation runs collapse.

A new site resets the workspace and starts with a full map and 16 warm-up refinements. Weights and replay can persist across sites during online learning. The current serial simulation does not establish independently timed control during GPU computation or learning. The released activity training also uses the last pre-action feature at nonterminal PPO rollout boundaries where the next decision state's value is required; a corrected training comparison remains to run (see [STATUS.md](../../STATUS.md)).

Not yet: learning during operation that improves the partially observing agent's reward; a learned decision about when to spend compute on learning; the activity head and the planner trained together; a second domain.
