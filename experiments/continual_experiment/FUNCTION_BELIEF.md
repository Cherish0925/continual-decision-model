# Learning response functions from experiments

This study follows the [response-family comparison](RESPONSE_FAMILIES.md).
Its question is whether learning a joint prediction of several experiments
helps an agent infer an unfamiliar relationship and choose its next experiment.
The inherited CDM and its twelve experimental ledgers remain the practical
reference. A separate integration retains that model's action interface.

## What an observation should change

Two models can assign the same success probability to every individual
experiment and disagree about what one result implies for another experiment.
For example, under two equally likely relationships, experiments A and B may
always succeed together, or exactly one may succeed. Both have marginal success
probability one half. A failed A rules out B in the first case and identifies B
as successful in the second.

We represent this dependence with a learned latent response function. A single
latent index is shared by all proposed experiments. Measurements change the
posterior over those indices and therefore change predictions of experiments
that have not been performed.

## Predictive model

The model learns 512 latent vectors and a shared neural decoder. The decoder
maps a composition and latent vector to probabilities of four intrinsic response
levels. These are probabilistic functions: a component can remain uncertain
about an individual outcome. The prior depends on the public ingredient count.
Neither the generating formula nor the simulator's relationship identity is
an input.

For measured history \(H=\{(x_i,b_i,y_i)\}\), where \(b_i\) is the pre-experiment
response level, the posterior is

\[
w_k(H)\propto \pi_k\prod_i p_\theta(y_i\mid x_i,b_i,z_k).
\]

The observed response is the minimum of the intrinsic level and \(b_i\).
Consequently, no improvement at level one is evidence for intrinsic levels
one, two **or** three. The likelihood sums this censored tail. It does not
label all such measurements as intrinsic level one.

Training maximizes the conditional **joint** likelihood of later measurements
given an earlier context. The same latent explains the sequence. The final
decoder multiplies projected composition features by latent-dependent scales
and adds latent-dependent offsets. An additive decoder is retained as a
comparison. There are 106,500 and 94,020 parameters, respectively.

This is a finite latent-variable response model, related to
[Neural Processes](https://arxiv.org/abs/1807.01622). The factorized references
use a [Conditional Neural Process](https://proceedings.mlr.press/v80/garnelo18a.html)
and an attention-based conditional predictor informed by
[Attentive Neural Processes](https://arxiv.org/abs/1901.05761).
The latter is our deterministic cross-attention implementation, not a claim to
reproduce the original paper's full latent architecture. Function mixtures,
conditioning and experimental information gain are established ideas.

## Shared training observations

All three predictors receive the same measured corpus: 8,192 response laws
with three ingredients and 8,192 with four. Each has 48 distinct measurements,
for 786,432 measured input–output pairs. The five training families are cosine,
metric, interaction, inhibition and multiple solutions. Generating laws are
used by the simulator to return observations; formula labels are not stored in
the training tensors.

A new replica starts after six measurements or an earlier completion. Each
replica is charged preparation, experiment and completion operations. This
produces more informative observations than repeatedly measuring a nearly
finished device. It is a declared digital training protocol, not a claim about
official DiscoveryWorld trajectories. At execution there is one apparatus per
task, with a 1,000-operation limit and no free resets.

The first comparison trains two initializations of each predictor for 6,000
updates. Development results motivate a fixed 24,000-update comparison of the
additive decoder, multiplicative decoder and attentive reference. Each uses
batch 128, identical per-seed sample sequences, AdamW at initial rate 0.0003,
and the same cosine schedule. Context lengths are 0, 2, 4, 8, 16 or 24.
The last update is the evaluated endpoint. The comparison matches observations
and updates; parameter counts and floating-point operations differ.

Both training initializations share a corpus. They measure optimization
variation, not independent data collection. The archived CDM parents have
different earlier training histories; comparisons with them test practical
performance and do not isolate the effect of joint likelihood alone.

## Actions, memory and compute

The standalone policy compares immediate completion per operation, completion
plus information gain, and an approximate two-step rule. The information
coefficient is fixed at 0.5. The first two-step approximation uses the ratio of
cost to success probability for its continuation value. Development results
show it can be poor. That continuation is a repeated-trial approximation,
whereas the executable task excludes already failed recipes.

A subsequent finite-order comparison replaces that approximation with the
expected cost of trying **different** remaining experiments within the remaining
budget. Its first hypothetical observation changes the posterior; its tail
uses a fixed completion order. This is one adaptive lookahead with an ordered
tail, not a complete search tree. A matched `finite_marginal` intervention
retains actual observation updates but disables the hypothetical posterior
update. Both include the four-operation exit and failed-task costs.

The standalone model retains up to twelve actual ledgers. At a task boundary,
each ledger is reconditioned under the same decoder. Their mean posterior
receives prior mass 0.45 and the fresh prior receives 0.55. This tests inference
from retained observations; it does not consolidate knowledge into weights
during deployment.

The CDM integration uses a separate, fixed prior of one half over the inherited
predictor and the new response-function predictor. Observed sequence likelihood
updates that mixture. The old archive reader continues receiving its original
pre-return predictions. The inherited progress/cost ranking, actor residual,
GRU, workspace and physical interface remain intact. The integration adds the
new predictor without updating those inherited weights. It is not a new joint
core training or a demonstration of autonomous compute allocation.

## Evaluation

The existing 192-task development cohort selects a rule for each predictor by
summed completion over both initializations, then mean operations. A separate
576-task test cohort was generated before these comparisons: 48 complete
twelve-task streams, including wider parameters and the withheld
interaction–inhibition combination. Discrete target compositions can recur
across splits. This is a digital extension, distinct from official native
tasks and from broad cross-domain generalization.

All methods receive only actual returned observations during execution.
Costs include failed tasks. Paired intervals resample complete streams, keeping
task-return dependencies together. The known-formula reference has additional
generator knowledge. The same-weight no-memory and hypothetical-update controls
answer different questions and are reported separately.

Native chemical comparisons reuse the saved initial states and twenty
twelve-task streams from the earlier memory study. The integration's physical
weights and shared-core weights are identical to their respective parents;
native chemical executions check the changed interface. The separated-domain
state protocol does not demonstrate transfer through a common evolving state.

## Completed results

![Prediction, memory and planning](results/function_belief.svg)

[Machine-readable comparisons](results/function_belief.json),
[all task rows](results/function_belief_instances.csv),
[native comparisons](results/function_belief_native.json), and
[training curves](results/function_belief_training.csv) are the numeric sources.

Twelve predictor trainings are complete. Their recorded optimization loops
sum to 697.9 seconds on the two H100s; this excludes corpus generation, process
startup and task execution. These are small predictor trainings. The inherited
CDM core is frozen throughout this study.

Each cell below gives **completed tasks / 576; mean operations**, including
failed tasks. The two columns are independent predictor initializations on
the same corpus and task streams.

| Method | Initialization 31 | Initialization 32 |
| --- | ---: | ---: |
| Attentive conditional reference | 554; 205.56 | 560; 203.76 |
| Additive function decoder, information rule | 569; 156.17 | 566; 163.03 |
| Modulated function decoder, information rule | 570; 148.16 | 574; 147.53 |
| Same functions, no task archive | 575; 189.72 | 575; 190.63 |
| Original CDM with twelve ledgers | 572; 144.48 | 571; 146.64 |
| CDM with direct evidence-mixture integration | 568; 165.95 | 571; 154.81 |
| Known-formula reference | 575; 114.75 | Same reference |

Relative to the attentive reference, the modulated model saves 57.40
[32.05, 85.64] and 56.23 [34.16, 78.66] operations, with 16 and 14 additional
completions. Its differences from the original CDM are +3.68 [−14.71, 22.67]
and +0.89 [−20.85, 23.82]. These results establish a competitive learned
response representation; superiority over the existing strong model remains
unestablished.

The archive saves 41.57 [23.24, 61.71] and 43.10 [24.95, 64.49] operations
relative to removing it from the same weights. Completion falls by five and
one tasks, respectively. Retention therefore improves mean search cost while
sometimes transferring an unsuitable prior. Cost and completion must be read
together.

### The continuation model changes the value of lookahead

| Planning rule, same response functions | Initialization 31 | Initialization 32 |
| --- | ---: | ---: |
| Repeated-trial approximation with hypothetical updates | 512; 319.57 | 508; 338.18 |
| Same approximation without hypothetical updates | 540; 229.18 | 542; 234.17 |
| Finite non-repeating completion order with hypothetical updates | 569; 152.94 | 574; 154.80 |
| Same finite order without hypothetical updates | 568; 271.79 | 571; 279.66 |

Under the finite-order model, hypothetical posterior updates save 118.86
[80.03, 159.32] and 124.85 [86.91, 164.06] operations. Both controls still learn
from **actual** returned experiments. The intervention isolates the use of
hypothetical observations in choosing the next experiment.

Under the repeated-trial approximation, the same kind of update instead adds
90.39 [54.15, 125.35] and 104.02 [68.12, 140.67] operations. More detailed
reasoning can optimize an inaccurate continuation model. Correcting that
model changes the sign of this intervention. The finite planner still does
not outperform the cheaper information rule; useful conditional inference
does not establish a benefit from additional computation over the best rule.

### Direct integration does not preserve native efficiency

On the original native chemical cohort, the evidence-mixture integration
completes 239/240 and 240/240 tasks at 91.93 and 91.35 operations. The inherited
parents complete 240/240 at 66.84 and 69.34. The new predictor therefore does
not replace the recommended native model. Its shared core and physical
interface have unchanged weights; the changed chemical predictions and their
use by the inherited ranking cause a different policy.

The current evidence supports learning coherent alternatives and using them
in an appropriate finite-cost decision model. It leaves decision-level
integration, autonomous compute allocation, independent-time operation and
parameter consolidation open. The next integrated training must optimize
subsequent native task cost as well as predictive likelihood. A likelihood
mixture alone has not supplied that connection.

## Reproduce

Run from the repository root in the project's environment:

```bash
python -m experiments.continual_experiment.train_relation_belief \
  --prepare --data runs/continual_physics/function_belief_20261103/data
python -m experiments.continual_experiment.run_relation_training \
  --root runs/continual_physics/function_belief_20261103 --phase initial --updates 6000
python -m experiments.continual_experiment.run_relation_training \
  --root runs/continual_physics/function_belief_20261103 --phase refinement --updates 24000
python -m experiments.continual_experiment.compare_relation_belief \
  --root runs/continual_physics/function_belief_20261103 \
  --models long modulated attentive_long --series refined --candidate modulated
python -m experiments.continual_experiment.compare_relation_belief \
  --root runs/continual_physics/function_belief_20261103 --stage test \
  --models long modulated attentive_long --series refined --candidate modulated
python -m experiments.continual_experiment.compare_relation_belief \
  --root runs/continual_physics/function_belief_20261103 \
  --models modulated --rules finite_plan finite_marginal --series finite
```

The final finite-planning comparisons use the same test manifest and the
`evaluate_relation_belief` entry with `--selection finite_plan` or
`--selection finite_marginal`. The integration entry is:

```bash
python -m experiments.continual_experiment.integrate_relation_belief \
  --parent runs/continual_physics/experimental_memory_20261101/train/memory911/initial.pt \
  --predictor runs/continual_physics/function_belief_20261103/train/modulated31/last.pt \
  --output runs/continual_physics/function_belief_20261103/integrated/cdm31.pt
```

Repeat for predictor 32 and parent 912. Evaluate those checkpoints with
`evaluate_response_families`, then `evaluate_shared_tasks --chemistry-only
--state-sharing separated --memory-mode evidence --memory-slots 12`. The
saved native initial states and full command plans accompany the release.

Training records retain every declared endpoint. Evaluation refuses unfinished
checkpoints. The initial development run that overlapped attentive training was
excluded and repeated on the fixed final checkpoints before rule selection.
