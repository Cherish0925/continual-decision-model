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
2,926 experiments including 2,786 non-terminating trials. The largest Challenge
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
to use its training split. All native transitions, including non-terminating
trials and partial cleaning, are retained. The next sampling work should
increase histories and experiment-order diversity rather than repeat the same
fixed sequence under further seeds.

## Fixed-endpoint chemistry comparison

The expanded data contain 110 training episodes and 30 development episodes:
18 use seven recipe parameter groups absent from the training and earlier
exploration records; 12 are new instances of seen recipes. All use the same
native chemical mechanism. Of 2,786 non-terminating trials, 214 visibly remove
some rust; they are useful conditional feedback, not uniformly failed trials.

Two runs initialize the chemical interfaces independently (seeds 741 and 742)
from the same public recurrent core, and train for exactly 500 updates of up to
eight transitions. Each episode's windows are traversed in seeded random order with
the complete earlier observed prefix reconstructed before the trained window.
`initial.pt` preserves each exact untrained interface; `last.pt` is the fixed
endpoint. No development result selects an endpoint.

All native conditions share 1,000 actual operations, at most 209 recipe trials,
the same tools and initial environment random states. An identical returned
recipe is unavailable when its observed endpoint cannot improve the current
rust state: the native response is deterministic and rust only decreases. This
rule applies to the reference, both initial models and both trained models.
The reference enumerates available recipes by total quantity. The model ranks
recipes by predicted immediate rust reduction per predicted operation cost;
its action scores use the trained outcome and cost predictions directly.

Run the native comparison with a fixed Python hash seed:

```bash
PYTHONHASHSEED=0 python -m experiments.continual_chemistry.evaluate \
  --training-root runs/continual_physics/chemistry_training --workers 8
```

The evaluator checks the actual recipe identity against the frozen split only
after execution. Those identities, simulator diagnostics and task scores never
enter model inputs. Set iteration changes the order of both vectors in the
native cosine calculation together; recipe generation itself uses a seeded
ordered-name list. The comparison additionally fixes Python hashing and shares
initial random states across methods.

This experiment concerns conditional learning and its effect on chemistry
choices. It uses structured observations and a new task interface over the
existing core; it does not demonstrate a jointly trained two-task model or
learned autonomous computation scheduling.


### Development result

All 150 native executions completed without interface errors; actual recipe
identities matched the frozen groups in every execution. Failures below used
the common 1,000-operation budget.

| Policy | All 30: completed / mean operations | Untrained recipes, 18 | Seen recipes, 12 |
| --- | ---: | ---: | ---: |
| Integer-recipe reference | 30 / 260.83 | 18 / 282.06 | 12 / 229.00 |
| Initial heads, 741 | 25 / 613.07 | 15 / 566.89 | 10 / 682.33 |
| Initial heads, 742 | 30 / 338.07 | 18 / 379.83 | 12 / 275.42 |
| Trained, 741 | 26 / 702.50 | 14 / 814.33 | 12 / 534.75 |
| Trained, 742 | 27 / 688.67 | 15 / 812.56 | 12 / 502.83 |

The two trained models do not improve on the reference. Relative to their
paired initial heads, neither delivers a consistent gain in completion and
operation cost. These are development results for this fixed training run,
not independent confirmation of a selected best model.

On the recorded development histories, training reduces mean per-episode rust
negative log likelihood from 1.395/1.424 to 0.609/0.622. A train-only frequency
table conditioned just on component count and current rust achieves 0.457;
its 80.3% class accuracy also exceeds the models' 77.9/78.7%. Predicting unchanged
rust already gives 79.7%. Thus the neural loss reduction does not establish
learned chemical relationship inference.

Two deployment findings are directly recorded. All 1,426/1,401 returned trials
from the trained policies have predicted costs below the scoring rule's
one-operation floor, so the denominator does not discriminate their choices.
The training split contains only one-to-four-part recipes; the models select
five/six-part recipes in 855/817 trials, respectively. Their outcome forecasts
are therefore used extensively outside the collected recipe-size range.
These facts identify ineffective cost calibration and unsupported prediction
use; they do not isolate their respective causal shares of the task deficit.

The next learning design must cover the choices it will actually rank, make
cost predictions numerically usable, and test relationship-sensitive prediction
against the strong conditional frequency reference. The frozen models and
results are preserved. Machine-readable [aggregate comparisons](results/development.json),
[per-instance results](results/development_instances.csv) and
[compact tables](results/development.csv) include both initializations and
recipe-group paired uncertainty. Conditional prediction errors can be
regenerated with `prediction.py`; `summarize.py` rebuilds these tables from the
saved native records.

## Broader native experiment experience

The next data batch uses 320 new native instances, split equally between Normal
and Challenge. Its candidate space contains each primitive integer proportion
once (55 / 161 candidates through six total parts). Scaling a recipe does not
create a distinct composition; the actual ingredient count and tool operations
still determine its cost. Rust-free keys end an episode rather than creating
uninformative zero-rust experiments or an artificial reset.

At each trial, the collector either samples an untried proportion uniformly
within a randomly selected total-parts stratum, or chooses by predictive entropy
plus disagreement of the two preceding outcome models. These are sampling
heuristics, not claimed optimal experiment design. Their predictions select an
experiment only; all training targets are returned native rust states and
actual operation counts. Partial cleaning changes the next experiment's input.

Before collecting outcomes, an evaluator-only generator pass freezes recipe
groups and instance seeds. Six recipe groups absent from the earlier 35 groups
are reserved for 30 new confirmation instances, with another 30 new instances
of training-group recipes. Neither these six groups nor the seven earlier
development groups enter the new training batch. Generator labels are used
only for this split and post-execution verification. The collector receives
only a scenario, instance seed and behavior random seed. The
[fixed plan](results/exploration_plan.json) preserves the exact separation.

```bash
PYTHONHASHSEED=0 python -m experiments.continual_chemistry.exploration_plan
PYTHONHASHSEED=0 python -m experiments.continual_chemistry.explore --workers 12
```

The first command creates worlds for grouping without performing experiments.
The second records every executed operation and completed experimental return,
including budget-limited runs. The resulting manifest reports total collection
cost, recipe-size coverage and the complete distribution of rust transitions.

The split also records composition structure after removing substance names.
Training contains 33 named compositions spanning five sorted positive ratio
structures. The 30 unseen-composition confirmation instances contain six named
compositions and three structures, all three present in training. Thus this is
held-out ingredient–ratio combination transfer, not an unseen ratio structure
or a new reaction law. This clarification does not change any frozen seed or
recipe group.

The first mixed batch is complete: 320 episodes produced 10,610 returned
experiments at a total cost of 154,240 native operations. The behavior completed
270 tasks; 50 exhausted their budgets, and none had an execution error. These
are data-collection outcomes, not the performance of a newly trained model.
Five/six-part compositions account for 2,660/2,383 observations. There are 461
partial-cleaning improvements. Most feedback is still censored: 9,574 trials
start and finish at rust level one. Such a trial establishes that the mixture
did not fully clean this key; it does not establish an intrinsic reaction
endpoint of one. The [data manifest](results/exploration_data.json) preserves
all episodes, costs, transition counts and split coverage.

## Learning responses from experimental history

`ReactionExperimentPolicy` retains the continuing GRU, Transformer workspace,
feedback store and public observation interface. Each candidate now queries the
complete returned experiment history through a shared composition encoder.
Its input describes ingredient proportions; total material quantity remains in
the separate cost interface. Jointly permuting ingredient names leaves this
direct relation representation unchanged. This is a representation property,
not a claim that the inherited recurrent core is already permutation invariant.
Training consistently renames ingredients throughout an observed prefix.

The visible rust state only decreases. Let R be a recipe's attainable rust
endpoint and r the current rust; the observed return is Y=min(r,R). An improving
return identifies R=Y. An unchanged return only identifies R>=r. The model
predicts a distribution over R, sums its censored tail to obtain P(Y=r), and
optimizes the ordinary log likelihood of the actual return. It receives no
hidden recipe, reaction formula or counterfactual response label. Separate
attention pools for exact improvements and censored returns keep rare useful
evidence available without class weighting or distorted output probabilities.
The queried history summary interacts with the same continuing h/z features
that are updated after each received experiment.

The initial visible probabilities equal the train-only component-count/current-
rust frequency reference. Progress and log-cost outputs start at their training
means; the regression loss uses training standard deviations while keeping
runtime feedback in its original units. The inherited core uses learning rate
3e-5 and the new interfaces use 1e-3. Training computes losses only for the
actually observed query and reconstructs its complete earlier prefix. Native
execution uses primitive integer proportions and the observed-response
probabilities divided by predicted native cost. `--primitive-candidates` applies
the same candidate family to the enumeration reference and earlier models.

```bash
python -m experiments.continual_chemistry.train_reaction \
  --data-root runs/continual_physics/chemistry_exploration \
  --output runs/continual_physics/chemistry_reaction/trained \
  --updates 500 --device cpu
python -m experiments.continual_chemistry.prediction_reaction \
  --checkpoint runs/continual_physics/chemistry_reaction/trained/last.pt \
  --output runs/continual_physics/chemistry_reaction/trained/prediction.json
```

A 220-update CPU development pilot on the *previous* 110 training trajectories
took 87.7 seconds. On their 30 recorded development histories, episode-mean
rust NLL was 0.4653, versus 0.4543 for the valid-class frequency reference.
Removing direct history gave 0.4643, reversing mixture/response correspondence
gave 0.4653, and suppressing the response head's continuing-state input gave
0.4485. Cost MAE was 1.41 native operations. Thus numerical cost prediction and
the joint gradient path are working, but this pilot does not establish useful
relationship inference or a benefit from the continuing state. The pilot
preceded ingredient-renaming augmentation. It is preserved as a development
result; no native confirmation or endpoint selection followed from it.
`prediction_reaction.py` compares these input paths on identical public prefixes,
including improvement calibration and ranking. The new mixed-order collection
is needed to examine history-conditioned learning over the full proposal family.

The subsequent collection supplies 10,610 real returns from 320 training
instances, with randomized/active experiment order and the complete primitive
proposal family. A fixed twelve-epoch batched warmup covers 127,320 labels in
63.2 CPU seconds, including 25.1 seconds to cache frozen-core prefix features.
The cache agrees with the received-event runtime within 1.91e-6 on ten checked
queries. Core weights are frozen during this stage; history encoders, response
readout and cost interfaces learn jointly.

On the same old 30 development histories, episode-mean NLL is now 0.3384 versus
0.4685 for the new training-frequency reference. Removing direct history gives
0.4384, removing the response readout's core input gives 0.3559, and reversing
mixture/return correspondence while preserving both marginals gives 0.9677.
For the 18 untrained-recipe instances, the corresponding values are 0.4018,
0.4882, 0.4213, 0.4213 and 0.4696. The stronger correspondence effect on the
12 seen-recipe instances is retained separately in
[the warmup results](results/reaction_warmup.json). Overall improvement-event
average precision is 0.716 versus 0.502 for the frequency reference; its Brier
score is 0.0565 versus 0.0717. Predicted native cost MAE is 0.135 operations.
These measurements support history-conditioned response prediction on recorded
prefixes. They are not native task-completion or jointly trained core results.

`warmup_reaction.py` performs complete uniformly shuffled transition epochs.
`joint_reaction.py` then updates the inherited core and the response together,
with batched sequence windows and full current-weight public-prefix replay.
Both stages save their fixed endpoint and training quantities. The new 60-case
confirmation cohort is not used to set either endpoint.
