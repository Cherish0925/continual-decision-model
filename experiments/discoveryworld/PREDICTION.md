# Learning to use action consequences

The previous multi-depth imitation experiment did not make additional workspace
updates useful: all tested depths executed the same actions. This continuation
gives those updates a concrete learning task. The model predicts what a candidate
action will return and what progress follows under the recorded behavior, reads
these predictions in its workspace, and revises its action ranking.

## Model and learning objective

The existing GRU receives each observation once. One ordinary Transformer update
processes that observation, preserving the reference policy's input path. An
action-conditioned predictor then reads the continuing state, public action
features and experimental evidence. It predicts:

- operation success, returned numerical contents and their presence;
- observed reward and task termination, as training targets;
- discounted progress minus action cost over 1, 8, 32 and 128 subsequent actions.

At each additional computation cycle, predictions for the eight highest-scoring
public actions enter the workspace as explicitly predicted content. The same
Transformer updates its state, and the predictor and action reader reconsider
the candidates. Numerical fitting and native scalar execution remain the existing
explicit tools. There is no simulator lookahead during model execution.

Training covers zero, one and two prediction cycles. Successful executions
provide action-imitation targets; **all** executions provide outcome targets.
An uncompleted time-limit tail is masked when the requested future horizon has
not been observed. Returned observations never enter the input of the decision
that precedes them. GRU, workspace, projections, action reader and predictor are
trained jointly; visual perception and the numerical prior remain fixed.

The returns describe recorded continuation policies. Without a matched
intervention, they do not identify the causal value of an unexecuted action.
The additional cycles revise predictions; they are not claimed to be successive
steps of an accurate imagined environment rollout.

This design takes the use of predicted consequences as decision context from
[Imagination-Augmented Agents](https://arxiv.org/abs/1707.06203). The broader
connection between outcome learning and behavior learning is also developed in
[Dreamer](https://www.nature.com/articles/s41586-025-08744-2). Our experiment asks
whether consequence prediction makes the existing persistent workspace useful
for scientific action selection and further computation; those prior results do
not answer this question for CDM.

## Data and fixed comparison

The starting set contains 45 successful reference training traces and eight
validation traces. The new collection adds:

| Source | Instances | Purpose |
| --- | ---: | --- |
| Public-observation reference, seeds 2000–2078 excluding remainder 4 modulo 5 | 64 | Successful and failed training experiments |
| Deployed CDM reader, seeds 2100–2138 excluding remainder 4 | 32 | Actual policy behavior, including failures |
| Reference, seeds 2200–2218 excluding remainder 4 | 16 | New-instance validation |

The reference completed 63 of its 64 training instances. The deployed reader
completed 25 of its 32 training instances. All 16 new reference validation
instances completed. These collection outcomes are not results of the new model.

Two matched training arms use these same data, initialization and optimization:

1. **Prediction feedback:** predicted consequences enter workspace refinement and
   action scoring.
2. **Prediction without workspace feedback:** the same prediction targets and
   action scoring, with ordinary observation-only workspace refinement.

Both train for three epochs, seed 151, BF16 dense operations, FP32 recurrent
state, sequence windows of eight decisions, and batches of eight episodes.
Core learning rate is 0.00001; new modules and action reader use 0.0001. The
prediction-loss coefficient is 0.5; discount is 0.99 and native action cost is
0.001. Each consecutive repeated-action run contributes one unit of imitation
mass, as in the reference reader. The final epoch is the preselected comparison
checkpoint; validation diagnoses prediction and action learning.

Native evaluation starts from reset, uses a 400-action limit and pairs the
simulator's initial random streams across methods. The planned new-instance
cohort is 2400–2418 excluding remainder 4 (16 instances). The separate unseen
relation-family cohort is 2404, 2409, 2414, 2419, 2424, 2429, 2434 and 2439.
Neither cohort enters training or model selection. Compare the reference reader,
both learned arms, and the same feedback model at zero extra prediction cycles.

Report task completion, progress and actions including failures, prediction
error at each cycle, and action changes induced by computation. More changed
actions or lower offline prediction error do not by themselves establish better
scientific decisions. If the model's depths disagree on training histories,
paired native continuations can measure the consequence of that disagreement
and supply subsequent activity-learning targets.

## Reproduction

The source reader is the `belief_actions/full_seed61/last.pt` checkpoint in the
[v0.5.0 reference package](RELEASE.md). Collection uses `reference.py` and
`evaluate.py`. `train_predictive_actions --prepare` derives prediction targets
from actual transitions and reuses frozen feature caches. Example training:

```bash
python -m experiments.discoveryworld.train_predictive_actions \
  --checkpoint runs/discoveryworld/belief_actions/full_seed61/last.pt \
  --prepared runs/discoveryworld/predictive_actions/prepared \
  --output runs/discoveryworld/predictive_actions/feedback \
  --device cuda:0
```

Use `--no-feedback` for the matched second arm. Evaluation uses
`evaluate --policy-kind predictive`; `--refinements 0` means zero **additional
prediction cycles**, retaining the ordinary observation update. The
`--no-use-forecasts` intervention removes both forecast scoring and forecast
feedback while retaining the rest of the model. It measures reliance on the
trained prediction path, not the performance of an independently trained
alternative.

The experiment output directory contains the collected trajectories, prepared
data, configurations, epoch checkpoints and measured learning curves. New native
results will be reported in [RESULTS.md](RESULTS.md) when execution finishes.
Learning when to compute and retaining learned experimental skills across task
switches remain further questions; they are not implied by this training run.
