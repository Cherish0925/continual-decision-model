# Continual learning from chemical experiments

Combinatorial Chemistry provides a second native scientific task with a
qualitatively different action and observation structure. An experiment selects
integer amounts from the visible dispensers, mixes them in a jar, treats a
rusted key and receives its new visible rust level. After cleaning the key, the
agent must open the door and leave. The existing public apparatus executor is
reused without changing the simulator or exposing its hidden solution.

The input is the visible component list, current rust level, previously
executed recipes and their returned rust changes/costs, and the model's own
remaining action budget. Candidate actions are nonempty recipes with at most
six parts, plus opening the door after the key is visibly clean. These are
experiment-level actions: the shared executor performs the corresponding
navigation, cleaning, dispensing, mixing and key manipulation, and every native
action is charged. The task score and scenario seed are evaluator metadata,
not input features.

Rust changes persist. A recipe applied to a heavily rusted key and the same
recipe applied after partial cleaning are different learning examples. The
model therefore predicts the returned rust state conditional on the actual
preceding state and history. The exploratory recipe order is not treated as an
optimal policy target.

## Shared model and training path

`observations.py` produces the existing continuing-core interface:

| Tensor | Shape | Content |
|---|---|---|
| `event` | `[B,36]` | Last returned recipe, before/after rust, cost, own progress and budget |
| `evidence` | `[B,N,20]` | Current state and already returned experimental history |
| `candidates` | `[B,A,30]` | Component counts/proportions, public context and previous returns for that recipe |
| masks | Variable | Available evidence and executable recipes/door action |

The GRU and Transformer workspace weights are reused from the existing active
core. Chemical input/output projections are initialized separately. The
four-class outcome head predicts visible rust levels 0–3; the existing
progress/cost adapter receives actual rust reduction and native operation cost.
There is no affine reactor model in this task path. The input is currently
structured; these experiments do not establish multimodal chemical reasoning.

`train.py` trains conditional rust-class prediction and returned progress/cost.
It uses the executed action only to select its observed outcome. At deployment, the model chooses a recipe by expected rust reduction from
its learned conditional rust distribution divided by predicted native cost.
The randomly initialized action head is not used. This is a myopic
prediction-guided rule; learning long-term action value and cross-task
improvement remain subsequent experiments. Exhaustive recipe ordering is not
used as a success-policy label.

Training rotates through all contiguous windows of every episode. A later
window first replays its complete, actually observed prefix with current
weights, then applies truncated backpropagation within that window. Later
successful recipes and partial-cleaning states are included. Twelve short CPU
updates verified a terminal-success window after eight replayed transitions.
Saved/reloaded outcome-based scores match exactly; after the initial two CPU
updates, candidate scores change by up to 0.0141 on the checked state (the top
choice remains the same). These checks establish the causal training-to-score
path, not a task improvement.

## Existing data and verified execution

The reusable local collection contains 15 complete native reference episodes
and 262 experimental transitions: 141 Normal training transitions, 36 already
explored Challenge transitions, 71 Normal validation transitions and 14 pilot
transitions. The Challenge examples are training material, not an unseen-law
validation result. The first failed Challenge interface attempts are excluded
by the existing collection manifest.

New execution checks used different instances:

| Execution | Native actions | Experiments | Result |
|---|---:|---:|---|
| Reference, Normal | 303 | 24 | Key cleaned and task completed |
| Reference, Challenge | 346 | 28 | Key cleaned and task completed |
| Newly initialized action head, four experiments | 54 | 4 | Rust reduced to level 2; task not completed |

Each returned experiment changes the shared persistent state and enters the
feedback store exactly once. The reference runs include one additional door
execution after the last chemical experiment. Two short CPU optimizer updates
on four-transition training prefixes produce nonzero gradients in the GRU,
Transformer workspace and rust outcome head. The saved checkpoint was then
loaded through the same native evaluator and executed four new chemical
experiments with once-only feedback. This verifies the complete
training path; it is not a task-performance or transfer claim.

The exact records are in [results/entry.json](results/entry.json). Full native
traces and the small training-path checkpoint remain under
`runs/continual_physics/chemistry_entry/`.

```bash
python -m experiments.continual_chemistry.dataset \
  --output runs/continual_physics/chemistry_entry/dataset.json
python -m experiments.continual_chemistry.native \
  --seed 7401 --policy reference \
  --output runs/continual_physics/chemistry_entry/reference_normal_seed7401
python -m experiments.continual_chemistry.train \
  --updates 100 --sequence-length 8 --device cuda \
  --output runs/continual_physics/chemistry_training
```

To execute a saved chemical model, pass its `last.pt` to `native --checkpoint`
and select `--policy model`. The same evaluator accepts the original reactor
core checkpoint, in which case chemical heads are newly initialized.

The CUDA command is the training entry, not a claim that such training has
already run. This integration does not change the frozen reactor confirmation
cohort or its running experiments.

## Source and attribution

Execution uses the installed official [DiscoveryWorld source](https://github.com/allenai/discoveryworld)
at revision `fd591323920be0d3786ef350955de1945aa571e5`. Its installed package
includes the Apache-2.0 license. The public task descriptions and simulator
observations in derived trajectories retain their upstream attribution. We do
not copy hidden recipe dictionaries into training records or model inputs.
Raw trajectories also contain evaluator fields for scoring; the dataset loader
selects only the explicitly listed public-state fields for observations.

## Expanded experience and recipe-separated development

The subsequent CPU collection contains 100 new Normal and 40 new Challenge
instances. The public exploratory reference completes all 140 tasks, recording
2,926 experiments including 2,786 unsuccessful recipes. The largest Challenge
run costs 882 native actions under the 1,000-action collection budget. These
are reference trajectories, not trained CDM task results.

The official generator uses three or four available substances, two or three
active components and three or four total parts. Canonically reduced recipes
identify 35 distinct groups in this collection. There are only 42 distinct
public experimental sequences under the fixed enumeration order; repeated
seeds therefore do not supply 140 independent reasoning patterns.

The frozen manifest allocates 110 episodes to training and 30 to development.
Eighteen development instances reserve complete recipe groups absent from both
the new training set and all prior explored data. Twelve retain a known recipe
under a different instance seed. This is a new-recipe-parameter comparison
within the same official reaction rule, not an unseen physical mechanism.
The simulator prints its solution in diagnostic logs; `collect.py` reads that
label only after execution to form groups. Those logs and recipe group labels
are never passed to the observation encoder or prediction model.

[The data manifest](results/data_manifest.json) records every instance, split,
recipe group and duplicate sequence group. Raw experiments remain in
`runs/continual_physics/chemistry_data/`; pass that directory to `train --data-root`
to use its training split. All native transitions, including unsuccessful
recipes and partial cleaning, are retained. The next sampling work should
increase histories and experiment-order diversity rather than repeat the same
fixed sequence under further seeds.
