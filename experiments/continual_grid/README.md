# Continual grid planning

A NumPy experiment on reusing ongoing computation while a navigation environment changes. All five tested planners perform the same number of Bellman cell backups at a given budget. Their differences are how they initialize and reuse the value workspace. There are no learned parameters in this experiment.

## Run

From the repository root:

```bash
python -m pip install -r experiments/continual_grid/requirements.txt
python -m unittest discover -s experiments/continual_grid -p 'test_*.py' -v
python experiments/continual_grid/run_experiment.py --output-dir runs/continual-grid
```

The default run uses five paired seeds, three regimes, 192 action ticks, and budgets of 20 or 81 cell backups per tick. It completes 180 episodes including an unrestricted current-map reference. The archived run is in [results/cpu_20261015](results/cpu_20261015); numerical findings are in [RESULTS.md](RESULTS.md).

## World and computation clock

The 9×9 world has two walls with changing doors, one moving obstacle, and a goal that changes every 24 ticks. A goal phase gives +1 on its first arrival and −0.02 per tick until arrival. Reaching a goal does not teleport the agent or reset its planner. The world is fully observed.

Each action tick has four operations:

1. Observe the current map and goal.
2. Read computational memory and update the allocated 20 or 81 value cells.
3. Advance the environment by one tick.
4. Select and execute one action using the new map and goal.

Every method uses the same fresh-observation action head. Its immediate reward and movement feasibility come from the current context; its continuation values come from the workspace. A goal-aware geometric rule breaks equal-value ties. A newly blocked cell cannot be entered. If an obstacle moves onto the agent's current cell, the agent can leave it.

The two budgets represent different cell allocations within one bounded computation block. Environment time is an abstract clock, not CPU wall time. CPU measurements include memory reading/writing, backups, observation handling after the block, and action selection. Comparing total latency is necessary because equal Bellman work does not make memory lookup free.

## Methods

| Method | Workspace on map or goal changes |
|---|---|
| Restart | Reinitialize from obstacle-free geometric values; retain progress while context is unchanged. |
| Carry | Retain values across map changes; reinitialize when the goal changes. |
| Snapshot | Restore a stored value array when the complete map and goal match; otherwise use carry. |
| Nearest | Read the stored array with the smallest map Hamming distance among records with the same goal. |
| Factorized | Read per-cell values from records matching that cell's radius-two local map and current goal; break ties by global map distance and recency. |

All memory methods share a capacity of 32 records and restore an exact-context snapshot when available. Each record contains its map, goal, approximate value array, computation count and last update tick. Retrieval changes the computational workspace; the current agent position and goal always come from the environment.

The factorized method is an explicit local-context heuristic. A cell's planning value can depend on a distant door, so local map agreement does not establish that its stored value is correct. Its purpose is to test whether this simple reusable representation is sufficient before introducing a learned one.

The separate **current-map reference** computes the exact shortest-path value for the currently visible map with unlimited planning work at action selection. It uses no future maps. It supplies a stationary current-map comparison and action-value gaps; it is outside the equal-work comparison. None of its values enter the five tested planners.

## Regimes and measurements

- **Full return:** complete map-goal pairs repeat after four goal phases.
- **Partial return:** familiar goals and door configurations recombine with changed obstacles.
- **Continuous:** the obstacle changes every three ticks and doors every nine ticks, during ongoing planning.

Each seed supplies the same exogenous map-goal trace and initial position to every method. Agent positions subsequently depend on their own actions. Measurements include total return, fraction of goal phases completed, arrival time capped at phase duration, current-map action-value gap, allocated cell backups, and CPU mean/p95 decision latency. Reading and writing times are retained separately. Current-map action gaps are descriptive comparisons with a stationary planner, not optimal values for the unknown future event stream.

Files written by the CLI:

| File | Contents |
|---|---|
| `config.json` | Parameters, versions and clock convention |
| `episodes.csv` | One row per seed, regime, method and budget |
| `phases.csv` | Goal-phase outcomes and arrival times |
| `decisions.csv.gz` | Every observation time, position, action, reward, gap and measured latency |
| `summary.csv`, `summary.json` | Means and sample standard deviations across seeds |

Compression uses a fixed gzip timestamp. Task outcomes repeat with the same seeds and versions; measured CPU times vary between runs. Unit tests cover time advancement, fresh-context action selection for all methods, equal cell budgets, snapshot retrieval, local-memory reading, and current-map shortest-path planning.
