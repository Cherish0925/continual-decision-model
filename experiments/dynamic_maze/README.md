# Continuing computation in a changing maze

A planner that improves its decision by iterating must keep acting while the world changes under it. This study trains recurrent latent planners on room mazes whose doors, obstacles and goals change during planning. Each planner gets a fixed number of refinements per tick and acts every tick. The study asks:

1. After a change, which computation should the next refinement start from?
2. How does a change reach every computation that depends on it, and can the model learn which computation to keep and which to rebuild?
3. When the agent sees only its surroundings and many agents share one accelerator, who should get the next refinement and who should look?
4. Can the deployed planner keep learning from its own operation?

Results are in [RESULTS.md](RESULTS.md).

## World

An `n × n` grid of 3×3 rooms (`side = 4n + 1`; 21×21 cells for the 5×5 training mazes). A random spanning tree of doors is always open, so every cell stays reachable. The other doors are shortcuts that open and close. One obstacle per room may sit on a corner or the centre, which never blocks a doorway. Goals sit on the middle cell of a room edge. The mean distance to the goal is 16 moves in a 5×5 maze and the longest exceeds 60, so a plan takes many refinements to build.

Four schedules drive the world. The schedule never depends on the agent:

| Regime | What changes |
|---|---|
| Full return | Three full configurations (doors, obstacles, goal) repeat in a cycle |
| Partial return | Door, obstacle and goal settings each come from a pool of three and recombine; every phase changes at least one factor |
| Continuous change | Each tick a shortcut toggles with probability 0.3 and an obstacle moves with probability 0.3; the goal jumps every 12–48 ticks |
| Static | Nothing changes (used to train one control) |

Phases last 6–12 ticks in training and 12–24 ticks in evaluation.

Most of a maze's optimal policy does not depend on the goal: for every free cell, one fixed action is optimal for most goals, and that action is optimal on 83–84% of free cells averaged over goals ([`goal_blind.py`](goal_blind.py)). Map accuracy therefore starts near 83% even without any goal information. The evaluation scores the cells whose optimal action a change actually altered.

## Planners

[`cdm/planner.py`](../../cdm/planner.py). An encoder maps the blocked map and goal to features `x`. The workspace `z` is a 96-channel map refined by one shared gated convolutional block, `z ← (1−u)·z + u·tanh(c)`, with `u, c` computed from `[z, x]` by two 3×3 convolutions. One refinement moves information about two cells. Heads read every cell: logits over the five actions (stay, up, right, down, left) and the distance to the goal. Each episode starts with 16 refinements; afterwards the planner gets `k` refinements per tick and acts from the readout at its own cell. The model has 0.60 million parameters.

When the map or goal changes, the continuation mode sets the workspace for the next refinement:

| Mode | Workspace after a change |
|---|---|
| Restart | Initialise from the new context |
| Carry | Keep the current workspace; refinement uses the new context |
| Restart on goal moves | Initialise when the goal moves; keep the workspace when only the map changes |
| Exact snapshot | Load a stored workspace whose map and goal match exactly; otherwise carry |
| Nearest context | Load the stored workspace with the same goal and the closest map, or carry if that is closer |
| Learned memory, per cell | For every cell, attend over the carried and stored workspaces with keys from the context features and the goal; a per-cell gate mixes carried, retrieved and freshly initialised computation |
| Learned memory, whole map | The same reader with one attention weight per stored workspace |

All memory modes store the finished workspace of each context when it ends, in a ring of eight records with the map and goal that produced it. Every mode reads the current map and goal at every refinement and at the action readout.

### Computation organised by premise

The factorized planner splits the workspace by what each part depends on. A map workspace `m` is encoded and refined from the blocked map alone, so a goal move leaves it valid. A goal workspace `g` holds the goal-specific plan:

- `m ← gated(m, [m, x_map])`
- `g ← gated(g, [g, m, x_map, m(goal), 1_goal])`

`m(goal)` is the map workspace read at the goal cell and broadcast to every cell; `1_goal` marks the goal. When the goal moves, `g` is rebuilt in one step from `[m, m(goal), x_map, 1_goal]` and `m` is kept. When the map changes, both continue. The action head reads `[g, m, m(goal)]`. The read at the goal cell is the only non-local operation: it lets a goal change reach every cell at once, where local refinement would need a wave of updates across the maze. With 60 channels per workspace a refinement costs 324,540 multiply–accumulates per cell, within 2% of the 96-channel planner (331,776); the model has 0.48 million parameters.

Ablations and single-workspace counterparts isolate the parts:

| Variant | Change |
|---|---|
| Factorized, no goal read | `m(goal)` is replaced by zeros; goal information spreads only through local refinement |
| Factorized, no rebuild | `g` is kept when the goal moves instead of being rebuilt from `m` |
| Factorized with map records | map workspaces are stored with their map and restored when that map returns, whatever the goal; `g` is then rebuilt |
| Carry with goal read | one 86-channel workspace; every refinement and readout also receives the workspace read at the goal cell |
| Goal read and learned rebuild gate | as above, plus a gate that, when the input changes, sets for every cell and channel `z ← (1−r)·z + r·ẑ`, with `r` and `ẑ` computed from `[z, x, x − x_prev, z(goal), z(previous goal)]` |

All single-workspace variants cost within 2.5% of 332k multiply–accumulates per cell per refinement; the gate adds a 1×1 computation only at ticks where the input changed.

## Training

Each step samples 48 episodes of 32 ticks from the three changing regimes and one refinement budget `k ∈ {1, 2, 4}`. After every refinement the loss scores every free cell: cross-entropy against the set of optimal actions plus a distance regression. Gradients run through the whole episode, so the shared block learns to repair its own computation after changes. AdamW, learning rate 2·10⁻³ with warm-up and cosine decay, 3,000 steps, bfloat16 on one H100. Five seeds per variant. *Carry, static training* uses carry but sees only static mazes during training.

## Evaluation

Every model meets the same 1,024 held-out episodes of 96 ticks per regime and maze size. The agent moves from the readout at its own cell after `k` refinements; an illegal move leaves it in place.

- **Optimal agent actions**: share of ticks on which the chosen move is optimal on the current map.
- **Return and goal arrivals**: reward +1 on arrival, −0.02 per tick before it; share of goal phases in which the agent reaches the goal. The optimal policy on the current map gives the reference.
- **Affected cells**: after each change, the free cells where none of the previously optimal actions is still optimal. Their accuracy as a function of ticks since the change, split by what changed (goal, doors, obstacles, goal and map) and by whether the new context occurred earlier in the episode.
- **Larger mazes**: 7×7 and 9×9 rooms with the 5×5-trained weights.
- **Latency**: wall time per tick on an otherwise idle H100 for one agent and for 256 agents.
- **Action fields**: every cell's action after one goal move and after one door closure ([`fields.py`](fields.py)).

### Controlled changes

[`probe.py`](probe.py) lets each planner converge for 32 refinements on a static maze, applies exactly one change (the goal moves to another admissible cell, three shortcut doors toggle, or both), and scores the affected cells before any further refinement and after each of eight more, on 512 mazes of 5×5 and of 9×9 rooms. [`gates.py`](gates.py) records what the learned gate rebuilds in the same setting.

### Partial view

[`train_partial.py`](train_partial.py) trains planners on what a moving agent sees: the true map within two cells of the agent, the whole map when it surveys (probability 1/12 per tick during training), and the last seen map elsewhere. A scripted agent walks the training episodes along shortest paths on its own map with 10% random steps. With `--age-input` the planner also reads how many ticks ago each cell was last seen. Targets are optimal actions and distances on the true map.

### Activities

[`active.py`](active.py) gives one agent a choice each tick between moving, thinking (staying for three further refinements) and surveying (staying and receiving the whole map at the next tick). Activity values are trained on branched rollouts: from the same state, each first activity is followed by a fixed continuation for 24 ticks of the same world, and a small head on the planner's state predicts the return differences.

[`shared.py`](shared.py) lets 1,024 agents with partial views share budgets instead: each tick every agent gets one refinement and moves; extra refinements and surveys go to the agents with the highest priority. Priorities are fixed rules (uniform, largest change of the action distribution in the latest refinement, fixed rotation of surveys, longest since the last survey), learned values, or a label-reading reference for refinements. The refinement value is trained on the one-step change in correctness of the agent's action; the survey value on the return difference of two branches of the same world over 24 ticks, with and without a survey.

### Shared budget, full view

[`allocate.py`](allocate.py) runs 1,024 agents with one planner. Each tick every agent gets one refinement and a further `(b − 1)·N` refinements go, in four rounds, to the agents with the highest priority. The priority signals are: none (the same number for every agent), random, refinements since the last change, low action probability at the agent's cell, the change of the action distribution at the agent's cell in its latest refinement, and, as a reference that reads the labels, whether the current action is wrong.

## Trained models

Release [v0.2.0](https://github.com/Cherish0925/continual-decision-model/releases/tag/v0.2.0) holds every trained planner and activity head of this study and of the [continual agent](../continual_agent/README.md):

| File | Contents | SHA-256 |
|---|---|---|
| `cdm-maze-checkpoints-v0.2.0.tar.gz` (170 MB) | `runs/dynamic_maze/main/<variant>/seed<k>/` with `model.pt`, `config.json` and `train_log.jsonl` for all 79 planners; `runs/continual_agent/<planner>_s0.pt` for the three activity heads | `0f9559b1cf9084c65f9dca72e18587a4f75e2148008a30c31e656aea9feca5f7` |
| `cdm-maze-evaluations-v0.2.0.tar.gz` (5 MB) | `runs/dynamic_maze/eval/<variant>/seed<k>.json` and `oracle.json`: every evaluation file with its recovery curves | `c69c97c2d200fe1f138a9bd65d1c818a6525bcea27edcde1adc7716d25933081` |

Extracted at the repository root, the files land where the commands below expect them:

```bash
curl -LO https://github.com/Cherish0925/continual-decision-model/releases/download/v0.2.0/cdm-maze-checkpoints-v0.2.0.tar.gz
tar -xzf cdm-maze-checkpoints-v0.2.0.tar.gz
```

`evaluate.load(path, device)` rebuilds a planner from its `model.pt`. [results/training.json](results/training.json) lists the parameters, configuration, final training loss and map accuracy, and training time of every run.

## Run

```bash
python -m unittest discover -s tests -p 'test_dynamic_maze.py'
python experiments/dynamic_maze/launch.py --root runs/dynamic_maze/main --per-gpu 8 --min-free-gb 16 --stagger 120 \
    --extra --steps 3000                     # every variant, five seeds; partial-view variants use train_partial.py
python experiments/dynamic_maze/evaluate_all.py --root runs/dynamic_maze/main --out runs/dynamic_maze/eval \
    --per-gpu 3 --min-free-gb 25 --extra --rooms 5 9 --think 1 2 4 --episodes 1024 --batch 1024
python experiments/dynamic_maze/evaluate.py --latency-only --model runs/dynamic_maze/main/{carry,carry_read,factorized}/seed0/model.pt \
    --think 1 2 4 --output runs/dynamic_maze/latency.json      # on an otherwise idle GPU
python experiments/dynamic_maze/probe.py --model runs/dynamic_maze/main/{carry,carry_read,carry_gate,factorized}/seed*/model.pt \
    --mazes 512 --output runs/dynamic_maze/probe/probe.json
python experiments/dynamic_maze/gates.py --model runs/dynamic_maze/main/carry_gate/seed*/model.pt --output runs/dynamic_maze/gates/gates.json \
    --figure experiments/dynamic_maze/results/figures/gates
python experiments/dynamic_maze/allocate.py --model runs/dynamic_maze/main/carry/seed{0,1,2}/model.pt --rooms 5 9 \
    --budgets 1 1.5 2 4 --output runs/dynamic_maze/allocation/carry.json
python experiments/dynamic_maze/shared.py --model runs/dynamic_maze/main/carry_read/seed{0,1,2}/model.pt \
    --output runs/dynamic_maze/shared/carry_read_r5.json
python experiments/dynamic_maze/partial_eval.py --model runs/dynamic_maze/main/partial_read_age/seed*/model.pt \
    --output runs/dynamic_maze/partial_eval.json
python experiments/dynamic_maze/online.py --model runs/dynamic_maze/main/factorized/seed0/model.pt --condition online-new \
    --teacher frozen --output runs/dynamic_maze/online/factorized_seed0_online-new_frozen.json
python experiments/dynamic_maze/fields.py --model restart=runs/dynamic_maze/main/restart/seed0/model.pt \
    carry=runs/dynamic_maze/main/carry/seed0/model.pt carry_read=runs/dynamic_maze/main/carry_read/seed0/model.pt \
    carry_gate=runs/dynamic_maze/main/carry_gate/seed0/model.pt factorized=runs/dynamic_maze/main/factorized/seed0/model.pt \
    --out experiments/dynamic_maze/results/figures/fields
python experiments/dynamic_maze/report.py --eval-root runs/dynamic_maze/eval --latency runs/dynamic_maze/latency.json \
    --probe runs/dynamic_maze/probe/*.json --allocation runs/dynamic_maze/allocation/*.json --out experiments/dynamic_maze/results
```

One training run takes about 9 minutes alone on an H100 and 25–60 minutes when 12–16 runs share two GPUs; one evaluation takes a few minutes on an idle GPU and needs up to 25 GB for the memory modes on 9×9 mazes with 1,024 episodes per batch.
