# Results: continual agent

Task and agent are described in [README.md](README.md). Rewards are per 100 ticks per stream. Each evaluation covers four fresh sites of 128 ticks with 2,048 parallel streams: about one million agent-ticks of parallel samples, not one uninterrupted run. Each site starts from a full map, a new workspace and 16 warm-up refinements.

## Learning how much to compute and when to look

The activity head chooses each tick how many refinements to run (0, 1, 2 or 4), whether to survey, and whether to move. It is trained with PPO for 120 iterations on reward minus costs (0.002 per refinement, 0.01 per survey); at the end of each 32-tick segment the value bootstrap uses the decision state reached after the segment's last action. Three PPO seeds per planner, planner fixed. Greedy evaluation, mean ± standard error over seeds:

| Policy | Carry with goal read and gate | Factorized, partial view, age input | Goal read, partial view, age input | Refinements per tick | Surveys per 100 ticks |
|---|---:|---:|---:|---:|---:|
| One refinement and a step every tick | 1.453 | 1.449 | 1.478 | 1.00 | 0 |
| Four refinements every tick | 0.857 | 0.872 | 0.892 | 4.00 | 0 |
| One refinement, survey every 16 ticks | 1.491 | 1.473 | 1.479 | 1.00 | 5.5 |
| Hand-written: compute and look only for a pending request, four refinements right after it arrives | 1.575 | 1.578 | 1.581 | 0.68 | 3.5 |
| Learned activity head | **1.623 ± 0.005** | 1.571 ± 0.008 | **1.595 ± 0.003** | 0.33–0.39 | 6.7–6.8 |

The learned head beats the best hand-written schedule by 0.048 and 0.014 on two planners and ties it on the third (−0.007), with 42–51% fewer refinements and about twice as many surveys. Fewer refinements alone does not measure total runtime or power. Requests reached:

| Policy | Gated planner | Factorized, age | Goal read, age |
|---|---:|---:|---:|
| One refinement every tick | 80.01% | 79.82% | 80.46% |
| Hand-written schedule | 82.00% | 82.05% | 82.12% |
| Learned activity head | 82.45% | 81.03% | 81.91% |

These heads replace the single-seed heads of release v0.2.0, whose training bootstrapped segment ends from the state before the last action; those heads scored 1.556–1.587.

Where the learned heads of the gated planner spend their operations (three heads, two fresh sites each; mean, with range over heads):

| State | Refinements per tick | Surveys per 100 ticks | Moves per tick |
|---|---:|---:|---:|
| Idle, no request | 0.06 (0.05–0.07) | 4.8 (3.0–6.0) | 0.38 |
| Request pending | 0.69 (0.59–0.78) | 8.7 (7.3–9.5) | 1.00 |
| Tick the request arrives | 2.18 (1.68–2.88) | 31.2 (21.3–50.0) | 1.00 |
| 1–2 ticks later | 1.03–1.07 | 5.8–10.3 | 1.00 |
| 8 or more ticks later | 0.51 (0.43–0.57) | 8.3 (7.1–10.4) | 1.00 |
| Pending, mean map age 0–3 ticks | 0.82 | 4.4 | 1.00 |
| Pending, mean map age 8–11 ticks | 0.56 | 11.8 | 1.00 |
| Pending, mean map age 16–19 ticks | 0.94 | 29.5 | 1.00 |

The reward does not name these states. All three heads learn the same pattern: almost no computation while idle, while still surveying so that the next request starts from a recent map; a burst of computation and observation when a request arrives; and surveys that become more frequent as the stored map ages.

## Learning while operating

The planner learns only from refinements the activity schedule leaves unused, out of a budget of four refinements per stream per tick. Its target is the readout of a frozen copy of itself 16 refinements deeper, on replayed sequences of the agent's own experience; when surveys return, it can also learn the optimal actions on the returned map ("hindsight"). The activity schedule is the hand-written one. Comparisons are paired by site: each condition meets the same sites.

**Full view, larger mazes.** Planners trained on 5×5 rooms run four sites of 9×9 rooms, then four of 5×5 rooms, with 512 streams. Reward per 100 ticks, mean of the four sites in each phase:

| Planner, seed | 9×9, no learning | 9×9, learning | Difference | 5×5 afterwards, difference |
|---|---:|---:|---:|---:|
| Carry with goal read, 0 | 0.341 | 0.345 | +0.004 | +0.001 |
| Carry with goal read, 1 | 0.332 | 0.350 | +0.018 | −0.001 |
| Carry with goal read, 2 | 0.334 | 0.347 | +0.013 | +0.003 |
| Factorized, 0 | 0.324 | 0.350 | +0.026 | −0.002 |
| Factorized, 1 | 0.298 | 0.335 | +0.037 | −0.003 |
| Factorized, 2 | 0.279 | 0.335 | +0.055 | −0.002 |

Learning while acting raises 9×9 reward by 0.026 on average (standard error 0.0075 over the six runs). For the factorized planner the gain is positive on every site and shows from the first one; 5×5 performance afterwards is unchanged. With about 300 learning steps, the factorized planner closes most of its gap to the single workspace with the goal read.

**Partial view.** One planner (goal read, partial view, age input) runs four phases of four sites each: 5×5 at the training door rate of 0.3 per tick, doors toggling at 1.0, 9×9 rooms, and 5×5 again. The budget is two refinements per stream per tick (four in one condition), and the learner sees 64 recorded streams. Mean reward per 100 ticks in each phase:

| Learning | 5×5 | faster doors | 9×9 | 5×5 again | Learning steps |
|---|---:|---:|---:|---:|---:|
| None | 1.587 | 1.505 | 0.273 | 1.567 | 0 |
| Hindsight from surveys | 1.573 | 1.516 | 0.264 | 1.535 | 324 |
| Distillation from a frozen copy | 1.594 | 1.504 | 0.274 | 1.570 | 324 |
| Both | 1.582 | 1.504 | 0.283 | 1.564 | 324 |
| Both, budget of four | 1.580 | 1.508 | 0.258 | 1.560 | 836 |
| Distillation from the live network, with hindsight, fixed rate | 1.576 | 1.502 | 0.118 | −0.330 | 250 |
| Distillation from the live network | −0.540 | −2.209 | −2.267 | −2.229 | 228 |

A rerun on 9×9 with a learning rate of 3·10⁻⁴, all 512 streams recorded, a budget of four and about 150 learning steps in the 9×9 phase gave paired differences within ±0.04 per site, +0.005 on average.

**Replay sampling correction after these runs.** At `a230e4e`, replay concatenated randomly selected records and then took the first `learn_batch` rows. With 512 streams per record and a batch of 32, every distillation update used only streams 0–31 of the first selected record. Increasing recording from 64 to 512 streams therefore did not expand the streams used by distillation. Survey hindsight also took a prefix of its selected records. The updated sampler draws rows without replacement across the selected records, keeps their fields paired, and gathers only the sampled rows. Four CPU sampling tests cover all 512 stream indices, multiple records per minibatch, unequal survey batches, and repeatability. The tables above retain the measured results from before this correction; no corrected GPU learning run is reported here.

Random changes outside the local view may limit the value of additional planning, but these experiments do not establish that every remaining error is unavoidable or that experience contains nothing useful to learn. The replay restriction, prediction from observation history, and the choice of learning target have not been separated in this comparison. With a full view, the measured gains show that deeper fixed-teacher predictions can improve the shallow planner on larger maps. Learning toward the network's own live deeper readout destroys the planner within a few hundred steps in the tested settings, as in the separate full-view deployment of [`online.py`](../dynamic_maze/RESULTS.md#6-learning-during-operation).

## What this establishes and what it does not

Measured in this task: the agent acts every simulated tick, keeps computation and information state across requests and idle periods within a site, and receives surveys after a two-tick delay; a head trained on reward minus costs chooses its computation and observation and beats or matches the best hand-written schedule with about half the refinements; learning paid from compute the agent leaves unused improves the running agent on larger mazes when it sees the whole map, without loss on the training size.

Not yet: learning that improves the agent under a partial view, which needs a world whose regularities change in ways the agent can learn; a learned decision about when and what to learn; joint training of the activity head and the planner; independently timed control during GPU computation or learning; one state carried across sites for hours; a second domain.

