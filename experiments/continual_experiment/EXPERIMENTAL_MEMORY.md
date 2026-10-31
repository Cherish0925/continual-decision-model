# Learning when previous experiments apply

The response-learning study established useful adaptation within one apparatus.
Its fitted correction reduced chemical operations in both joint-training
repetitions, but the supporting experimental ledger disappeared at the next
task. This study preserves those observations and learns to judge their
applicability from the results of new experiments.

The question is whether an agent can recover useful experimental knowledge
after interference, while adapting promptly when a relationship is new.
Returning tasks and new tasks are measured separately: rapid reuse can coexist
with expensive commitment to an inappropriate memory.

## Model and learning

One set of shared core weights serves physical control and chemistry, with
separate recurrent states for the two interfaces. Each chemistry task has its
own current ledger. At a task boundary, a six-slot archive retains up to the
latest 32 returned observations per task, replacing its oldest slot. Archive
entries contain public input proportions, available components, and observed
before/after outcomes. No recipe identity, instance seed, or scheduled return
label enters the model.

Each archived ledger defines a response predictor using the current shared
conditional-response network and the existing four-step functional update.
The predictor is reconstructed from observations, rather than an old hidden
state or a saved correction matrix. A separate predictor uses only evidence
from the current task. Mixture probabilities combine these possibilities.

For predictor j, the evidence accumulated before decision t is

```text
L_j(t) = sum_{s < t} log p_j(y_s | x_s, history_before_s).
```

Each term is computed before receiving the observation it scores. The learned
reader uses evidence relative to the current-task predictor, the number of
current returns, archive length, and archive age. Old predictors share a fixed
total prior mass; adding duplicate records cannot increase that mass by itself.
The current-task predictor remains an explicit alternative. Incompatible
public ingredient sets exclude an archive from the mixture.

The returned measurement is sometimes censored: unchanged rust c reveals only
that the latent response is at least c. Predictions are mixed at the latent
response level, then mapped to the observable outcome. Inner response fitting
and expert evidence use the same observation semantics.

The next actual return supplies the proper likelihood loss. It trains the
response network, functional update, and memory reader. The action policy also
trains from subsequent native-operation rewards and uses the mixed prediction
to rank experiments. In the inherited action interface, predicted probabilities
are detached from the policy-gradient path; the reader receives prediction
gradients, not a direct action-reward gradient. Shared core and action
parameters participate in joint training.

Task boundaries and archive writes are explicit. Writes are automatic, and
response fitting uses four internal updates. The inherited LEARN/CONSOLIDATE
activities continue to update the separate cost model. This experiment tests
learned knowledge reuse; it does not implement autonomous archive writing,
response-compute scheduling, or consolidation into long-term neural weights.

## Training and independent execution

Two existing jointly trained parents each initialize a memory condition and
a no-memory condition. These are paired continuations, not four independently
pretrained models. All four use 200 optimizer windows, batch 32, and 16 physical
plus 16 chemical requests per window. Chemistry selects and executes its own
experiments. Current responses, actual operation costs, and subsequent returns
provide training targets; unchosen experiments supply no outcome labels.

Training lifetimes contain twelve tasks with sampled novelty and return lags.
Raw archives persist between gradient windows; neural graphs detach between
windows. The matched no-memory condition performs the archive computation but
does not use archived forecasts. Its network and within-task update are trained
separately from the same starting weights and request budget.

The native cohort contains twenty twelve-task streams. Every stream uses one
public ingredient-count setting, so ingredient count alone cannot identify
which of its six chemical relationships has returned. The order is
`A B C A D E B F A C F D`, with a different native instance for every occurrence.
Physical tasks are interleaved between chemistry tasks. Model states remain
separate across domains and weights are shared.

There are 240 chemical and 240 physical instances per full condition. Of the
chemical tasks, 120 return to a previous relationship and 107 use named
compositions absent from training. The response family is still the official
cosine-threshold family; this is not transfer to unseen chemical laws. The
cohort was selected from new seed ranges before candidate execution, without
selecting outcomes. Budget is 400 physical or 1,000 chemical native operations.

| Comparison | What changes |
|---|---|
| Initial parent | Before this continuation, with no archive use |
| Trained no-memory model | Same initial weights and request budget; current-task response fitting retained |
| Learned experimental memory | Response predictors and applicability reader jointly trained |
| Same weights, archive disabled | Removes historical predictions from the trained memory model |
| Same weights, fixed evidence reader | Replaces the learned reader by its initialized evidence-weighting rule, including the same initial prior |
| Direct recipe retrieval | Tries recent observed-successful recipes still legal in this task, then uses the independently trained no-memory model |
| Same weights, neural state reset | Clears chemistry h/z at task boundaries, retaining its experimental archives |
| Bayesian memory / reset | Known reaction-family reference, with or without six archived task posteriors |

The fixed reader starts with current-task log prior 1 and total archived log
prior 0. It then accumulates predictive evidence with unit coefficient. This
matches the learned reader's initialization, avoiding a changed prior in the
comparison of learning itself. The Bayesian reference has a stronger explicit
prior: all official cosine-response hypotheses through six mixture parts. Its
memory uses half uniform prior and half the average of past task posteriors;
two-step experimental selection charges actual native operations.

All final endpoints are retained. Interventions confined to chemistry skip
repeating independent, unchanged physical executions. Completion, operation
counts including failures, and measured runtime are reported. Paired intervals
resample whole twelve-task streams, preserving dependence between task returns.
Same-history prediction analysis compares cached forecasts before each actual
return, separating memory's predictive effect from changed experiment choice.

## Reproduction

From the repository root, with the published parent assets restored under their
documented relative paths:

```bash
python -m experiments.continual_experiment.launch_experimental_memory \
  --initial-root runs/continual_physics/response_replay_20261031/joint \
  --output runs/experimental_memory/train --updates 200 --batch 32

PYTHONHASHSEED=0 python -m experiments.continual_experiment.memory_manifest \
  --output runs/experimental_memory/cohort --workers 20

python -m experiments.continual_experiment.evaluate_memory \
  --training-root runs/experimental_memory/train \
  --manifest runs/experimental_memory/cohort/manifest.json \
  --output runs/experimental_memory/native --workers 10

PYTHONHASHSEED=0 OPENBLAS_NUM_THREADS=1 python -m experiments.continual_experiment.memory_reference \
  --manifest runs/experimental_memory/cohort/manifest.json \
  --output runs/experimental_memory/references \
  --initial-rng-root runs/experimental_memory/native/initial_rng --workers 20

python -m experiments.continual_experiment.analyze_memory_predictions \
  --native-root runs/experimental_memory/native \
  --manifest runs/experimental_memory/cohort/manifest.json \
  --output runs/experimental_memory/analysis

python -m experiments.continual_experiment.memory_controls \
  --training-root runs/experimental_memory/train \
  --manifest runs/experimental_memory/cohort/manifest.json \
  --output runs/experimental_memory/native --workers 8

python -m experiments.continual_experiment.export_memory_results \
  --input runs/experimental_memory
```

## Development comparison: retaining exact observations

An additional comparison asks whether re-fitting an archived ledger loses
important precision at already measured inputs. Within a candidate context,
an improving observation fixes the latent response; a non-improving observation
places a lower bound on it. The optional observed-response constraint applies
these facts to matching proportions, leaving other predictions unchanged.
Context selection still uses current experimental evidence.

This comparison uses eight six-task development streams, including instances
evaluated in earlier work. Each continuation uses one fixed intermediate
checkpoint for every condition (window 50 for continuation 1 and 100 for
continuation 2). These endpoints are not selected for the final comparison.

| Development condition | Operations, continuation 1 | Operations, continuation 2 |
|---|---:|---:|
| Experimental memory | 62.08 | 62.40 |
| Preserve exact observed responses | 62.08 | 62.40 |
| Same weights, archives disabled | 85.38 | 93.42 |
| Same weights, fixed evidence reader | 62.08 | 62.67 |

All 48 tasks complete in every condition. The exact-observation intervention
changes the tails of archived forecasts on repeated inputs, but neither the
chosen experiments nor their predicted actual outcomes change. The existing
functional fit already preserves these observations sufficiently for this
development comparison. More exact storage alone gives no additional task
benefit here. The complete comparison uses separately trained references,
new instances and the fixed final window-200 endpoints.

## Relation to prior work

[MetaFun](https://proceedings.mlr.press/v119/xu20i.html) studies learned iterative
functional updates. [Episodic recall in meta-learning](https://proceedings.mlr.press/v80/ritter18a.html)
addresses repeated tasks and recovery of previous solutions.
[VariBAD](https://www.jmlr.org/papers/v22/21-0657.html) connects inferred task
uncertainty to action selection, and
[CN-DPM](https://openreview.net/forum?id=SJxSOJStPr) develops expert mixtures
for continual learning. Kernel adaptation, episodic recall, and task inference
are established ideas. This study asks whether a shared experimental model can
learn useful applicability judgments from its own returned observations, and
whether those judgments improve native task costs beyond fixed retrieval and
separately trained references. Its scientific contribution depends on that
comparison and the conditions under which reuse succeeds or fails.
