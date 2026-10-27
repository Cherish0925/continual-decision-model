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
simulator's initial random streams across methods. The new-instance
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
`evaluate.py`. The [data list](results/prediction_data.json) names every training
and validation trajectory. `train_predictive_actions --prepare` derives targets
and rebuilds frozen features from these released records:

```bash
for split in train validation; do
  python -m experiments.discoveryworld.train_predictive_actions --prepare \
    --checkpoint runs/discoveryworld/belief_actions/full_seed61/last.pt \
    --dataset experiments/discoveryworld/results/prediction_data.json --split "$split" \
    --output runs/discoveryworld/predictive_actions/prepared --device cuda:0
done
```

Use `--parts 2 --part 0` and `--parts 2 --part 1` on separate GPUs to divide
preparation. Example training:

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
data, configurations, epoch checkpoints and measured learning curves. The completed native
comparison is reported in [RESULTS.md](RESULTS.md#action-consequence-prediction--27-october).
Learning when to compute and retaining learned experimental skills across task
switches remain further questions; they are not implied by this training run.

## Learning from alternative actions

The follow-up collects 96 native continuations from 48 training states. Sixteen
training histories each contribute the smallest top-two action margin in three
preassigned windows: decisions 48–111, 112–175 and 176–239. Numerical dialogues
are excluded because the explicit scalar tool owns their execution. Each pair
replays the same observations and random stream, forces one of the two candidate
actions, then returns control to the same predictive policy until completion or
400 total actions. Selection uses current policy scores, not future outcomes.

The learning target combines the two actions' actual returned observations,
finite-horizon returns and the difference in final progress minus 0.001 per
subsequent action. Preference weights preserve that measured scale: a one-action
saving is not rescaled into a task-completion gain. Both arms replay the parent
policy distribution and successful training actions from the same incoming
states. The paired arm additionally learns the observed consequences and action
preferences. The GRU, workspace, projections, reader and predictor are updated
through one observed transition; preceding recurrent states are cached from the
parent. This is not full-lifetime backpropagation.

Two epochs, seed 161, core learning rate 0.000001 and other learning rate 0.00001
are fixed before evaluation. Native comparison uses the original reference,
the predictive parent, matched replay and paired-consequence learning on sixteen
new instances in 2600–2618 excluding remainder 4, and eight reserved-family
instances 2604–2639 with remainder 4. Final-epoch weights are used for every arm.
No held-out execution is used as a training example.

```bash
python -m experiments.discoveryworld.action_consequences \
  --checkpoint runs/discoveryworld/predictive_actions/feedback/last.pt \
  --source runs/discoveryworld/predictive_actions/interactions \
  --output runs/discoveryworld/action_consequences \
  --seeds 2100 2101 2102 2103 2105 2106 2107 2108 2110 2111 2112 2113 2115 2116 2117 2118
python -m experiments.discoveryworld.consequence_campaign
```

The campaign waits for all selected branches, prepares parent-state contexts,
runs matched trainings on two GPUs, and evaluates the three learned alternatives
in parallel after collecting a shared reference random stream. Raw collection
and learning outputs are in `runs/discoveryworld/action_consequences/` and
`runs/discoveryworld/consequence_learning/`.
