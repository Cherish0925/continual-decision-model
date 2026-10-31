# Learning from an experiment to predict the next one

The active-feedback study found that its retained chemical corrections worsened
next-outcome prediction. More observations improved than deteriorated, but the
average deterioration was approximately four times larger. The response update
also shared a LEARN activity with a separate progress/cost update. This study
first isolates the response update, keeping the core and cost policy fixed.

The question is how a returned observation should change predictions for
different experiments. The old update carries a matrix fitted to hidden
features and base predictions saved at earlier histories. Both representations
and base predictions change when subsequent observations arrive. Furthermore,
the feature's constant coordinate permits a correction shared by all queries.
These properties motivate comparisons; neither is yet an established cause.

## Update and comparison

For an earlier experiment with latent response probabilities p and an exact
return y, the logit gradient is p minus the one-hot outcome. A non-improvement
at rust c only reveals R >= c; its gradient is p minus p conditioned on that
tail. The actual next return is withheld from every inner update.

The correction is a sum of support gradients transferred through a kernel:

    correction(query) = sum_i K(query, earlier_input_i) coefficient_i.

Four differentiable support-update steps fit the coefficients. Subsequent
prediction likelihood trains the step sizes, shrinkage and response calibration.
The relation condition additionally learns a kernel length scale. These
functional-gradient and meta-learning operations have direct prior work,
including [MetaFun (ICML 2020)](https://proceedings.mlr.press/v119/xu20i.html).
Their use here tests a concrete failure of experimental learning; this study
does not claim the general operations as new inventions.

| Condition | Evidence used to adjust the next prediction |
|---|---|
| Original | The released model's accumulated online correction |
| Zero | The same model without its response correction; cost learning continues |
| Calibration | Learned probability temperature and class bias, with no within-task update |
| Stored | Four updates using the saved support features and base predictions |
| Refreshed | Four updates after re-encoding the support at the current history and core state |
| Relation | Refreshed support predictions, with correction transfer controlled by mixture similarity |

Stored and refreshed use a feature inner-product kernel. Relation uses an RBF
kernel on normalized mixture proportions. Common ingredient renaming leaves
its distance unchanged. All fitted conditions include the same probability
calibration, so calibration alone remains an explicit reference.

## Training and actual execution

Two frozen parents, 911 and 912, each produce chronological features for the
320 training exploration episodes and the 30 existing development episodes.
Each fitted condition trains for 30 complete epochs with batch 128, Adam rate
0.01, and a fixed final endpoint. Training sees only the training episodes.
The next returned experiment supplies the outer target; previous observations
are the support. Parent weights, action heads and cost learning remain fixed.

The native cohort is fixed to twenty Normal and twenty Challenge instances,
with seeds 241000 + 7i, i = 0,...,19, for each difficulty. Every condition starts
at the same initial task and has 1,000 native operations. Both parents and all
four fitted methods are included. Operations, completion and inference time
are reported together. Re-encoding and four inner updates consume computation;
this is not an equal-compute comparison. Inference time includes that work.

Parameter correction is rebuilt from current evidence at each readout. It is
separate from the inherited cost-learning activity. Autonomous selection of a
response update and cross-task consolidation are not evaluated by this first
comparison. Task state resets at each apparatus. The result determines which
update, if any, should be trained as an available activity in the longer task
stream. A calibration-only gain would not establish useful online learning.

## Completed comparisons

Sixteen fixed-endpoint update trainings and 1,040 native task executions are
complete. The first eight fit the recorded exploration histories. Their
development predictions improve slightly, but their native actions do not
improve over the original or zero-update models. The second eight fit
experiments selected by the frozen parents: 14,705 and 14,639 training returns,
with 20% uniformly sampled legal experiments for coverage. Development uses
greedy selected experiments on the existing 30 validation instances. This
phase changes both history distribution and the number of returns; it does
not isolate a distribution-only effect.

The relation update has a positive same-weight effect in the second phase:

| Parent | Next-outcome NLL without update | NLL with update | Native operations without update | With update |
|---|---:|---:|---:|---:|
| 911 | 0.7441 | 0.7241 | 117.38 | 109.95 |
| 912 | 0.7689 | 0.7602 | 119.93 | 111.75 |

The prediction comparison scores the same 2,632 and 2,596 subsequent outcomes.
The update-minus-no-update NLL intervals are −0.0337 to −0.0004 and −0.0210
to −0.0002 nats, resampling whole named recipe groups. Each native condition
finishes all 40 instances. The native operation differences are −7.43
(95% interval −13.52 to −2.53) and −8.18 (−14.15 to −3.37), reductions of
6.3% and 6.8%. These comparisons retain each fitted model's temperature, bias,
core, action head and cost-learning policy while removing only response fitting.

The [support-to-query breakdown](../continual_chemistry/results/response_transfer.json)
checks whether the query mixture was already tested in its support. Every query
with prior evidence uses a different mixture: 2,167 and 2,098 predictions.
Their NLL falls from 0.6115 to 0.5873 and from 0.6276 to 0.6168. The paired
recipe-group intervals for these changes are −0.0387 to −0.0006 and −0.0252
to −0.0002 nats. Cases without prior returns are unchanged within numerical
precision. Thus the predictive effect includes transfer to untested mixtures
within these tasks. It does not establish transfer to new reaction-function
families or retention after a task change.

The second phase's native cohort uses seeds 242000 + 7i, i = 0,...,19, at both
difficulties. It is distinct from the first phase's cohort. Conditions within
each phase share the initial simulator RNG. The two parents share earlier
pretraining; their repetitions are not two independent pretrained models.

An independently fitted calibration-only model uses 113.65 and 110.50 native
operations. The relation method's differences from that reference are −3.70
(−12.57 to +4.68) and +1.25 (−6.76 to +9.89). Thus a useful same-weight update
is established in this comparison, while superiority over the separately
trained reference remains unestablished. Neither experiment demonstrates
autonomous response-compute allocation or knowledge retained across tasks.

[Complete results](../continual_chemistry/results/response_learning.json),
[paired predictions](../continual_chemistry/results/response_learning_predictions.csv)
and [all native outcomes](../continual_chemistry/results/response_learning_instances.csv)
include both phases and every fitted condition.
The [reference model archive](https://github.com/Cherish0925/continual-decision-model/releases/tag/v0.13.0)
contains both shared-core parents and all sixteen response rules. It contains
no unfinished joint-training endpoint.

## Completed joint training

The response update is connected to full physical/chemical core training.
Two continuations each compare relation fitting with the identical fitted
calibration but no within-task response fit. Both train the core, action
interfaces and prediction models on the same external-request budget: 200
windows, 32 streams, 16 physical and 16 chemical requests per window. The
initial learned update parameters are inherited from the completed study.
Response fitting has four fixed iterations; cost-learning activities retain
their existing policy. This phase does not yet learn when to allocate response
computation or introduce a new long-term memory mechanism.

The automatic native comparison includes both initial models, all four final
models, same-weight disabled-response controls and chemical-state resets. It
reuses the existing frozen A–A–B–B–A–A stream benchmark and reports first-return
performance. All four trainings reached update 200; all ten native conditions
finished without execution errors on 31 October 2026. Each condition includes
120 physical and 120 chemical tasks, giving 2,400 executions on the same fixed
instances. Each training took 50–56 minutes with two jobs on each H100.

| Condition | Physics completed, 911 / 912 | Physics operations, 911 / 912 | Chemistry operations, 911 / 912 |
|---|---:|---:|---:|
| Initial fitted model | 119 / 117 | 111.27 / 108.99 | 92.43 / 90.48 |
| Joint relation model | 117 / 117 | 107.33 / 105.13 | 75.22 / 78.99 |
| Joint calibration reference | 116 / 118 | 106.12 / 108.58 | 81.18 / 80.41 |
| Same relation weights, response fitting disabled | 117 / 117 | 107.33 / 105.13 | 82.18 / 87.54 |
| Same relation weights, chemical state reset | 117 / 117 | 107.33 / 105.13 | 75.73 / 80.23 |

Every chemical condition completes 120/120 tasks. Response fitting saves 6.97
and 8.55 operations against its same-weight disabled control: reductions of
8.5% and 9.8%, with whole-stream 95% intervals [−13.62, −1.49] and
[−15.53, −3.11]. The useful update therefore survives full shared-core training.

The independently continued calibration references remain strong. Differences
against them are −5.96 [−13.02, +0.12] and −1.42 [−11.93, +9.69] chemical
operations. Neither interval excludes zero. Physical task completion changes
from 119 to 117 and from 117 to 117 relative to the initial models; these
results do not establish an overall improvement across both domains.

Carrying chemical state saves 0.52 and 1.24 operations on average, with intervals
that include zero. On the first task after the original relationship returns,
carrying state changes cost by +1.80 [−9.50, +13.70] and −3.20 [−15.75, +10.90]
operations relative to resetting it. Long-term recovery is still unestablished.
The response fit remains four fixed iterations. Recorded LEARN and CONSOLIDATE
choices update the separate progress/cost model; they do not demonstrate
autonomous scheduling or consolidation of the new response rule.

[All joint comparisons](results/response_joint.json),
[2,400 individual outcomes](results/response_joint_instances.csv) and
[deployed activities](results/response_joint_activity.json) retain every final
endpoint and control. The next substantive question is whether useful
experimental knowledge can survive task changes and improve returning tasks,
while outperforming independently trained references at comparable resources.

A short initial attempt was stopped to correct per-stream truncation when
histories in a batch have unequal lengths. Its partial logs are retained but
are not counted as results. The completed frozen-prefix and native comparisons
use individual streams and are unaffected. All four formal trainings restarted
from the same declared initial weights after the correction.

```bash
python -m experiments.continual_chemistry.run_response_replay \
  --output runs/response_replay/study
PYTHONHASHSEED=0 python -m experiments.continual_chemistry.confirm_response_replay \
  --study runs/response_replay/study --output runs/response_replay/native_baseline
python -m experiments.continual_chemistry.run_response_replay \
  --active --output runs/response_replay/active_study
PYTHONHASHSEED=0 python -m experiments.continual_chemistry.confirm_response_replay \
  --study runs/response_replay/active_study --output runs/response_replay/active_native \
  --seed-base 242000 \
  --conditions original zero calibration stored refreshed relation relation_no_update
python -m experiments.continual_chemistry.analyze_response_transfer \
  --study runs/response_replay/active_study
python -m experiments.continual_experiment.launch_response_joint \
  --study runs/response_replay/active_study --output runs/response_replay/joint \
  --updates 200 --batch 32 --chemical-steps 16
PYTHONHASHSEED=0 python -m experiments.continual_experiment.shared_confirmation \
  --study response --training-root runs/response_replay/joint \
  --output runs/response_replay/joint_native \
  --manifest experiments/continual_experiment/active_confirmation.json --workers 8
```

The training uses two GPUs independently. Native environments run in a CPU
process pool. Raw histories and model files stay under the selected run root;
the linked result tables retain the finished predictions and task outcomes.
