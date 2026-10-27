# Scientific-task learning

These experiments connect the continuing core to visual observations, object-directed
actions, instrument returns and numerical learning. Autonomous execution from
reset and takeover after supplied experimental evidence are measured separately.

## Joint state, action and computation training — 27 October

The successful seed-61 reader now initializes a joint sequence learner. Both
arms update the GRU, shared Transformer workspace, multimodal projections and
action readout; this comparison is no longer a frozen-core reader experiment.
The adaptive arm adds a 299,384-parameter allocator for 0, 1, 2 or 4 workspace
refinements. Both use the original 45 training and eight validation traces,
three fixed epochs and the same update schedule described in
[TRAINING.md](../../TRAINING.md#joint-state-action-and-computation-training--27-october).
Perception and the numerical prior remain fixed.

On eight new native instances (1010–1013, 1034, 1039, 1044, 1049), with paired
initial measurement noise and a 400-action budget:

| Policy | Completed | Mean progress | Mean actions, including failures | Refinements per decision |
| --- | ---: | ---: | ---: | ---: |
| Original successful reader | 3 / 8 | 0.895 | 338.125 | 1 |
| Joint training, fixed computation | 4 / 8 | 0.920 | 311.375 | 1 |
| Joint multi-depth training and learned allocation | 2 / 8 | 0.845 | 357.875 | 1 |
| Same multi-depth weights, forced one refinement | 2 / 8 | 0.845 | 357.875 | 1 |

[Per-instance results](results/joint_compute.csv) retain failures as well as
successes. Four versus three completions on eight instances is not a stable
performance advantage. The adaptive allocator selects one refinement on every
decision, and forcing one produces identical trajectories in this comparison.
Thus multi-depth imitation has not delivered useful adaptive computation; the
performance loss here occurs in the learned model before any change of its
executed compute schedule. These models do not supersede the reference.

Reference replay distinguishes branch capability from allocation. On the
1,382 validation actions, conditional action accuracies at depths 0/1/2/4 are
81.40/87.12/85.02/83.36 percent. If the reference answer is used to choose the
lowest action loss plus compute cost, the respective depth counts are
1,028/252/46/56; the learned allocator chooses depth 1 throughout. This shows
state-dependent differences in the available predictions, not a deployable
oracle or evidence that those choices improve later task completion. Actual
same-prefix continuations are needed to learn their decision value.

Both trainings finish. Fixed computation takes about 431 seconds for three
epochs and multi-depth training about 443 seconds (exact epoch times are in
the run logs). An initial execution attempt was stopped before completing its
first epoch: changing observation lengths repeatedly incurred cuDNN attention
plan construction. Disabling only the cuDNN attention backend, while retaining
cuDNN GRU and flash/efficient/math attention, reduces a representative steady
forward/backward measurement from about 3.2 seconds to 0.063 seconds. This
execution change applies to both arms. BF16 backends can differ numerically;
the sampled sixteen action choices agree, and the native comparison above
uses each final checkpoint directly.

The subsequent [32 native continuations](results/compute_branches.csv) cover
eight training states: an ongoing measurement and the first numerical setting
in each of four previously collected trajectories. Every prefix replays
exactly. Changing the first decision to 0/1/2/4 refinements, then returning to
one, produces **identical executed action sequences within all eight groups**.
Completion, progress and action cost therefore also tie. The decision logs
confirm that the requested internal counts actually ran. These selected states
supply no positive task-return signal for additional refinement. They do not
establish that computation depth is irrelevant in all states or tasks. In the
setting contexts, the first executed setting comes from the explicit numerical
primitive; only the changed neural state could affect subsequent decisions.

The distinction now matters operationally: the earlier workspace-reset result
shows that the policy uses persistent state, whereas these studies have not
shown that additional workspace updates improve experimental reasoning. The
next allocation data should include training states where computation changes
a proposed experiment or prediction, followed by actual matched continuations.
Task-return targets and cross-task reuse remain research work, not capabilities
established by the imitation warm start.

Raw checkpoints, training logs and reference-replay analysis are under
`runs/discoveryworld/joint_compute/`. The new methods do not yet train or
demonstrate cross-task retention.

## Autonomous experiments from reset — 27 October

Observation rendering now uses an in-memory PNG path adapted from
[PR #3](https://github.com/Cherish0925/continual-decision-model/pull/3). On
this server, 24 CPU reads of the same Reactor Lab state have median latency
216.97 ms through the native API, 78.78 ms through in-memory PNG and 72.04 ms
with raw RGB. PNG is the supported trajectory interface and is now the adapter
default; `observation_mode="native"` retains the original API. Pixel, public
feedback and dialogue checks pass. This is a 2.75-fold reduction in observation
read time, not a claim about full training speed. The raw measurements are in
`runs/discoveryworld/observation_local_20261027.json`.

A learned action reader now connects the continuing CDM state to its accumulated
experimental evidence. It learns navigation, instrument use, measurement exits,
calibration access, object placement and progression to the remaining experiment.
The scalar-setting and local-probe primitive is the same explicit component as
in the earlier numerical comparisons. No measurements or teacher actions are
supplied at evaluation reset.

The reader has 442,753 trainable parameters. The source perception, recurrence,
workspace and five-parameter shared numerical prior are frozen. Training uses
45 successful reference trajectories (9,354 actions), with eight separate
validation trajectories (1,382 actions), 30 fixed epochs, batch size 32, learning
rate 0.001 and action-run-weighted imitation. Two independently initialized
readers and their matched controls use this same training budget. Every result
below uses the final checkpoint.

Eight fresh Reactor Lab Normal instances comprise four familiar relation-family
instances (610–613) and four instances from a family excluded from training
(634, 639, 644, 649). All policies receive the same initial native measurement
random streams, then experience the consequences of their own actions. Each has
at most 400 native actions; all numerical adjustments count toward that budget.

| Action policy | Completed | Mean progress | Mean actions, including failures | Newly activated crystals |
| --- | ---: | ---: | ---: | ---: |
| Original source policy with the same numerical primitive | 0 / 8 | 0.395 | 400.0 | 0 |
| Evidence reader, training seed 61 | **8 / 8** | **1.000** | **232.6** | **16** |
| Matched reader with evidence input zeroed, seed 61 | 0 / 8 | 0.400 | 400.0 | 0 |
| Evidence reader, training seed 62 | **7 / 8** | **0.955** | **228.3** | **14** |
| Matched reader with evidence input zeroed, seed 62 | 0 / 8 | 0.395 | 400.0 | 0 |
| Full seed 61, workspace reset before every event | 0 / 8 | 0.080 | 400.0 | 0 |
| Received-facts reader without derived predictions, seed 61 | 7 / 8 | 0.960 | 242.1 | 14 |

The evidence-input control trains an identical head with its 125-dimensional
explicit evidence input zeroed. It still has the original source state,
perception and numerical controller. It therefore tests this explicit readout,
not whether the model has any history at all. The workspace intervention uses
the trained full model, keeps its recurrent state and measurement records, and
performs the same model computation while clearing its workspace each event.
It establishes dependence of this trained policy on that state; it is not a
retrained recurrent baseline.

The received-facts reader keeps measurements, labels, observed object contents,
activation state and recent action feedback. It masks derived predictions and
candidate probabilities. Its 7/8 result shows that much of the gain can be
obtained by connecting explicit experimental facts to action selection. The
8/8 versus 7/8 comparison on this small set does not establish a reliable
additional benefit from exposing the relation posterior to the action reader.
All arms retain the numerical predictor inside the scalar primitive.

The failure in full seed 62 is instance 613: progress reaches 0.64, then the
policy cycles between measurements and dialogue. The received-facts reader
fails on 611 at 0.68. The full seed-61 policy completes all eight but still
spends an average of 130.9 actions taking radiation measurements. Its successful
measurement runs often contain about thirty consecutive samples. Actual task
completion is now demonstrated; efficient experiment selection has not yet
been learned from task return.

The two training seeds use the same eight evaluation instances, not sixteen
independent tasks. Held-out relation families remain within Reactor Lab. The
specified affine hypotheses, exact numerical records and explicit scalar tool
remain part of the method. These results establish autonomous evidence
collection and task completion in this setting, not cross-domain scientific
reasoning or long-term skill retention.

[Per-instance results](results/belief_actions.csv) contain all 56 evaluations.
Full trajectories, returned observations, action timings and fixed final
checkpoints are under `runs/discoveryworld/belief_actions/`. They remain local.
The [task README](README.md#reading-experimental-evidence-to-choose-actions)
gives the training and evaluation commands.

## Further instances and actual-return training

A second fixed evaluation cohort uses instances 810–813 and 834, 839, 844, 849.
The original readers are unchanged. It both extends the autonomous-completion
comparison and tests the first actual-return update on trajectories the model
collected itself. No prefix is supplied, and all policies use paired initial
native random streams and the same 400-action limit.

| Policy | Completed | Mean native actions, including failures | New activations |
| --- | ---: | ---: | ---: |
| Original full reader, seed 61 | 6 / 8 | 270.6 | 13 |
| Original full reader, seed 62 | 6 / 8 | 251.5 | 13 |
| Received-facts reader, seed 61 | 4 / 8 | 306.6 | 9 |
| Full seed 61 after the first PPO update | 5 / 8 | 291.9 | 11 |
| Same proposed update with training-state KL step selection | 6 / 8 | 271.0 | 13 |
| Second actual-return round, continued projected policy | 6 / 8 | 270.6 | 13 |

Across the two evaluation cohorts, the unchanged full readers complete **14/16
and 13/16** instances, compared with **11/16** for the received-facts reader.
These are repeated configurations on the same sixteen instances. The exact
[per-instance second-cohort results](results/active_belief_actions.csv) retain
both successful and unsuccessful cases. The explicit-evidence-zero and
workspace-reset controls were evaluated on the first cohort only.

The PPO data comprise eight new training episodes, four completed, totalling
2,501 actual actions. Policy gradients use the 2,048 actions actually executed
by the learned reader. The remaining numerical-tool operations still contribute
to task reward and cost. Reward is official task-progress change minus 0.001
per action, with discount 0.999 and the finite 400-action horizon. The first
update uses four passes, learning rate 0.0001, clip 0.1 and batch size 64.
All checkpoints and native trajectories are retained.

Replaying 127 sampled decisions under their saved full-precision states
reproduces behaviour log probabilities to a maximum difference of 0.0000148.
After optimization, the full proposed update changes the greedy choice on 9.8%
of the training decisions and has mean KL 1.376 relative to the collecting
policy. The step-length comparison uses those training decisions alone, halves
the parameter change until mean KL is at most 0.02, and selects a fraction of
0.125 (measured KL 0.00247). It restores the parent's six completions but does
not reduce its action cost. The new complete failure of the unconstrained
update is instance 844 at initial progress 0.08.

The second round continues from that projected policy and its learned value
head. It collects eight new 720-series instances (four completed, 2,603 actual
actions, 2,227 reader-controlled choices), then applies the same objective and
step-length rule. The selected fraction is again 0.125, with training-state
KL 0.00469. Native validation remains 6/8, with exactly the original mean of
270.625 actions. Neither reward-training round improves performance.

These rounds establish an
executable learning path from actual experiments and identify a concrete
optimization difficulty; they do not establish efficient experiment selection.
The fixed full readers remain the stronger reference models.

Two failures shared by the full readers, 839 and 849, also expose a numerical
issue. Their local search commits to the most probable affine hypothesis even
when other hypotheses predict substantially different settings. Failed
activation settings are remembered, but do not reweight the candidate
relationships. The next comparison uses received inactive feedback to condition
the full predictive mixture and select reachable settings by probability of
success per native adjustment. It keeps the original action reader and the
sixteen-extra-adjustment allowance fixed. A separate 910-series cohort is
reserved for that method; it does not train on these failures.

## Testing inactive feedback across numerical hypotheses

`MixtureProbe` retains all nine predictive components when the first proposed
setting fails. Inactive feedback excludes the interval within 2 Hz of each
actually observed setting, using the simulator's stated activation rule. The
component probabilities are conditioned on these exclusions. Among settings
reachable through the offered controls, it selects predicted success probability
per shortest native adjustment count and executes the first adjustment. All
returned failures are retained, and the allowance remains sixteen extra
adjustments per reactor/object pair. The original action reader and prior are
unchanged. The method reads no hidden resonance or environment state.

| Cohort | Numerical search | Completed | Mean actions | New activations |
| --- | --- | ---: | ---: | ---: |
| 810-series development instances | Original local search | 6 / 8 | 270.6 | 13 |
| Same instances | Conditional-mixture search | 6 / 8 | 271.1 | 14 |
| New 910-series instances | Original local search | 8 / 8 | 224.9 | 16 |
| Same new instances | Conditional-mixture search | 7 / 8 | 245.5 | 15 |

The paired [raw summary table](results/mixture_probe.csv) is authoritative for
these means. The new method fixes instance 839, finishing in 240 actions, but
loses a former success in 811. It acquires one additional crystal activation in
849 without completing the full task. On the new cohort it loses instance 913.
The unchanged local-search policy therefore remains the reference method.
Official completion in instance 912 is true at progress 0.96; completion and
fractional task-score coverage are recorded separately.

The contrast illustrates a limitation of this experiment-selection objective:
a more explicit uncertainty model does not ensure more useful experiments. In
811, the predicted distribution is broad and the probability-per-action rule
selects successive 10 Hz changes; the original fine local sequence succeeds.
In 839, wider moves instead find a working setting that the original sequence
misses. Uncertainty calibration and the finite sequence of future decisions
both matter. This comparison does not separate their contributions, and no
claim of improved scientific reasoning follows from the additional mathematics.

The native controller experiments and policy-return experiments remain separate:
one changes an explicit numerical action primitive; the other trains the neural
action reader. Their failures are retained beside the successful from-reset
model results. The second reward-training round uses fresh 720-series training
instances, the previous projected policy and its learned value estimator, with
the original scalar tool; its completed result is given above. The fixed
810-series comparison is a development validation set for this continuation,
not an untouched final test set.

## Learning when to end a measurement run

Four paired continuations test a decision directly, using training instances
710–713. For the first repeated measurement run of at least 24 actions, both
arms reproduce its actual history through eight measurements, including native
instrument noise. One takes another reading; the other executes the departure
action the same policy eventually took in that training trajectory. Both then
continue with the unchanged full seed-61 policy and the original scalar tool,
within the same total 400-action budget. Future measurements are not supplied.
All eight prefixes reproduce the recorded public observations and action results.

| Training instance | Continue: completed / actions | Depart: completed / actions | Final progress, both arms |
| --- | --- | --- | ---: |
| 710 | Yes / 224 | Yes / 195 | 1.00 |
| 711 | No / 400 | No / 400 | 0.84 |
| 712 | Yes / 221 | Yes / 197 | 1.00 |
| 713 | No / 400 | No / 400 | 0.80 |

The two successful pairs save 29 and 24 actions without losing completion.
The two failed pairs do not improve. These are conditional interventions on
training histories, not additional autonomous test completions. They measure
the realised value of an earlier departure under this continuation policy;
they do not establish an optimal number of measurements for every instance.
The [per-branch table](results/measurement_branches.csv) keeps all outcomes.

The subsequent training comparison uses these two informative pairs to rank
the actions by actual final progress minus action cost. Both readers start from
full seed 61 and receive the same additional epoch of reference-action replay.
One also receives the pairwise loss; the control gives that loss zero weight.
The final models are evaluated from reset, without interventions, on the same
810-series development instances. This separates extra training from learning
the consequences of the experimental choices.

| Reader | Completed | Mean native actions, including failures |
| --- | ---: | ---: |
| Unchanged full seed 61 | 6 / 8 | 270.625 |
| One additional epoch of reference replay | 6 / 8 | 260.250 |
| Same replay plus paired-return preferences | 5 / 8 | 271.500 |

The paired-return reader learns both training choices: their preferred-action
logit margins become positive. It also completes instances 812, 813 and 844
in 178, 177 and 167 actions, compared with 208, 219 and 213 under matched
replay. However, it loses instance 810 at progress 0.60 and does not resolve
839 or 849. It therefore fails to improve overall completion or action cost.
Reference replay alone reduces mean actions by 10.375 without changing the
six successful instances on this development set; it has not yet been tested
on another cohort.

The [complete per-instance comparison](results/branch_action_learning.csv)
retains all three readers. Two realised benefits from earlier departure were
enough to change actions, but not enough to learn a reliably better experiment
policy. The next training data need decisions where an additional measurement
does improve the subsequent intervention as well as decisions where it wastes
work. Neither lower imitation loss nor a learned preference at the sampled
states establishes that trade-off on new trajectories. The original fixed
readers remain the independently evaluated reference.

## Numerical learning and native experiments — 27 October

The latest method keeps named physical measurements in their original units,
maintains nine candidate single-quantity affine relations, and updates their
coefficient distributions and probabilities from the instance's actual
calibration results. Across training instances it learns either separate
priors for each quantity (54 parameters) or a shared coefficient/noise prior
with uniform candidate weights (5 parameters). The affine hypothesis family is
specified; the method does not discover arbitrary equations. The continuing
source policy retains its perception and recurrent state. An explicit scalar
controller executes predictions; optional local trials use returned activation
feedback within the selected candidate's predictive uncertainty.

All rows below give the **retained-update arm**. Each has sixteen takeovers:
eight native instances, each starting after two or three confirmed calibrations.
A matching arm computes the same type of update on its received feedback and
discards it. Each takeover has at most 48 native actions. Later observations
follow each policy's actual actions. Every discarded-update arm below produced
zero new crystal activations and zero completed tasks.

| Evaluation set | Method | New activation in takeover | Remaining task completed |
| --- | --- | ---: | ---: |
| Development instances | Frozen affine numerical reference | 7 / 16 | 6 / 16 |
| Development instances | Trained typed prior, point control, seed 31 | 10 / 16 | 9 / 16 |
| Development instances | Trained typed prior, point control, seed 32 | 10 / 16 | 9 / 16 |
| Development instances | Same seed 31 with local trials | 13 / 16 | 12 / 16 |
| Fresh instances | Shared prior with local trials, seed 31 | **16 / 16** | **14 / 16** |
| Same fresh instances | Shared prior with local trials, seed 32 | **15 / 16** | **14 / 16** |

The fresh instances were collected after the training choices were fixed.
They include four familiar-family and four held-out-family instances within
Reactor Lab. They are not new scientific domains. The two training seeds use the same eight
instances; the two prefixes per instance are also dependent observations.
Supplied prefixes contain
measurement and calibration work; these counts do not measure autonomous
completion from reset. Both trained models complete the same fourteen
takeovers. In seed 31, the remaining two failures occur after successful activation: one policy resumes repetitive measurements instead of
installing the remaining crystal; another removes a successfully installed
crystal and subsequently adjusts an empty reactor. In seed 32, the latter
failure is identical, while the former fails before activation: a 0.076 Hz
change in the predicted setting changes its discrete starting position and
trial order, exhausting sixteen adjustments before reaching a working setting.
The successful setting remains within its predicted interval. Thus matching
completion counts do not imply identical intervention behaviour.

The numerical comparison explains why both shared structure and new evidence
matter. On the four fresh held-out-family queries with two calibrations, the
untrained uniform prior has MAP mean absolute error 173.95 Hz, the trained
quantity-specific priors 1350.60/1350.63 Hz, and the two shared priors
5.08/5.06 Hz. Familiar-family errors are approximately 0.97 Hz. These examples
show a transfer cost of the quantity-specific configuration and a benefit from
the shared configuration; weight sharing and uniform candidate probabilities
change together, so this comparison does not isolate either alone.

A public working setting is an observed successful setting, not an exact
latent resonance. With three calibrations, the fresh held-out-family prediction
error is about 2.48 Hz relative to such settings. Native feedback and local
trials are therefore measured directly. The probe controller is explicit,
operates within mean ± two predicted standard deviations, and uses at most
sixteen additional adjustments per target inside the unchanged 48-action
budget. It has not been learned by the original action head.

This supplied-evidence comparison motivated the action-reader study above.
Long-term return to earlier tasks remains separate from the within-instance
learning and autonomous completion demonstrated in these experiments.

Machine-readable results: [native comparisons](results/calibration_native_comparison.csv)
and [fresh numerical transfer](results/calibration_transfer.csv).
Full raw records are under `runs/discoveryworld/calibration/`; the training,
point-control, probe-control and fresh-transfer runs retain separate directories.
The earlier experiments below explain the changes in representation and update.

## First image/text policy comparison

Both models start from the released 1,024-wide continual core, use frozen SigLIP 2
features, and train on six Reactor Lab instances (698 actions). Four separate
instances (484 actions) select checkpoints. Training uses 60 epochs, batches of
three episodes, 16-action backpropagation windows and learning rate 0.0001.
The CDM objective includes an immediate-depth imitation term of weight 0.25;
the recurrent baseline omits that term and the workspace. Their comparison is
between training configurations, rather than a single-factor architecture test.

The selected policies then execute 300 actions each in four paired new instances:
310, 311, 312 and 313. No reference policy selects actions during evaluation.

| Policy | Progress by instance | Mean progress | Completed tasks |
| --- | --- | ---: | ---: |
| CDM | 0.36, 0.36, 0.32, 0.32 | 0.34 | 0 / 4 |
| Recurrent policy | 0.28, 0.28, 0.20, 0.20 | 0.24 | 0 / 4 |
| Same CDM, workspace reset before every observation | 0.08, 0.08, 0.08, 0.08 | 0.08 | 0 / 4 |

All four environments start at progress **0.08**. This is an official partial
task score, not a success probability. The reset intervention therefore makes
no progress beyond the initial state. It preserves the recurrent state, public
object registry, weights and number of refinement computations; only the
workspace contents restart. This establishes dependence on the continued
workspace for this trained policy's early task behaviour. It does not establish
that the workspace alone learns a scientific law.

Both trained policies eventually repeat actions instead of finishing the
experiment. The CDM reaches reactor interaction; the recurrent policy often
repeats a measurement. Offline action agreement on reference trajectories was
78.3% and 70.2% at the final epoch, respectively. Those percentages describe
teacher-trajectory prediction and must not be reported as task success.

The adapter repairs an upstream UI-cache inconsistency: a native `USE` operation
can enter a device dialogue without refreshing its rendered options. The repair
synchronizes the current dialogue before image/JSON rendering, without changing
rules, actions or ticks. The table uses this repaired interface for every arm.
Earlier uncorrected runs remain under `runs/discoveryworld/siglip_pilot/`.

Data: [results/pilot.json](results/pilot.json). To render the figure:

```bash
python -m experiments.discoveryworld.plot_results \
  --results experiments/discoveryworld/results/pilot.json \
  --output runs/discoveryworld/figures/pilot
```

## What changed the next experiment

The earlier lexical pilot exposed a specific gap in the demonstration data:
every training action succeeded, while the learned policy repeatedly attempted
to pick up an immovable reactor or repeated an already received measurement.
The public response already contained the error or reading. The outcome head
still predicted success with high confidence on a failed pickup. Adding another
input field would not supply the missing recovery experience.

The next comparison separates two changes:

* More complete demonstrations: 16 additional training instances, 3,596 actions,
  and four validation instances, 898 actions. These demonstrations use a revised
  reference procedure with repeated, isolated radiation measurements. Both
  models continue from their earlier selected checkpoints with the same data.
* Corrections in states reached by the learned policy: query the reference
  before execution, retain its suggestion as an imitation target, and train
  outcome prediction on the action actually executed. The continued neural
  state follows the actual sequence, including mistakes and recovery.

This second direction follows the distribution-of-visited-states motivation of
[DAgger (Ross, Gordon and Bagnell, 2011)](https://proceedings.mlr.press/v15/ross11a.html).
Its value here must be measured by subsequent independent task execution.

The reference is a handwritten scientific procedure with an affine hypothesis
family and public measurements. Its successful completions are demonstration
results, not CDM discoveries. The separate online actor–critic pilot performed
real updates but did not improve its two evaluation instances; it is retained
as an unsuccessful initial learning run.

## Repetition and decisions to stop

After expanded demonstration training, validation action accuracy reaches
823/898 (91.65%) for CDM and 807/898 (89.87%) for the recurrent model. Of these
898 actions, **512 are radiation measurements**. Both models predict all 512
correctly, but both miss **all 16 actions that should leave after a completed
sampling batch**. Excluding radiation, accuracy is 311/386 (80.57%) and 295/386
(76.42%). Frequency-adjustment dialogue accuracy is 17/59 and 11/59, respectively.

In native execution the expanded CDM takes the instruments, performs the first
measurements, isolates a crystal and then repeats radiation measurement for
277–278 actions. It does not yet reach calibration. The current first obstacle
is ending a sampling operation; frequency reasoning is a further obstacle, not
the cause of this earlier stall.

This motivates a comparison recorded before its training: use the same parent,
correction data, optimizer, seed and 12 epochs, but give each contiguous run of
the same demonstrated action one total unit of imitation weight. All observations
still update the continuing state, and each actual outcome retains its original
supervision. The weight rule uses no instrument name or fixed sampling count.
The prediction is improved sampling-exit decisions and advancement to calibration
in independent execution; a higher aggregate imitation accuracy is not required.
The fixed final checkpoints, rather than differently selected validation minima,
will compare the two loss treatments.

That comparison did not resolve independent execution. Four paired instances
give mean progress 0.33 after ordinary correction learning and 0.37 after
action-run weighting; neither completes a task. Additional training on successful
demonstrations gives 0.40, also without completion. Thus weighting alone does not
establish the predicted escape from repeated sampling.

The next comparison connected the core's existing record reader to public
measurement histories. The first digital path carried recurrent state and
workspace contents but supplied no past measurement records to that reader.
The new optional input binds tool, object and unit to received values, sample
count, range and change. It introduces no stopping rule or expert phase label,
and leaves the original model parameter shapes intact. A 2-by-2 comparison
crossed this readable history with the two imitation reductions, using the same
parent, trajectories, seed and twelve epochs. The prospective question was whether
access to accumulated experimental evidence enables a policy to finish sampling
and continue calibration; native task completion remains separately measured.

| Measurement-history input | Event loss | Action-run loss |
| --- | ---: | ---: |
| Absent | 0.33; 0/4 completed | 0.37; 0/4 completed |
| Present | 0.27; 0/4 completed | 0.37; 0/4 completed |

Each cell reports mean official progress after 300 native actions in instances
310–313, followed by completed tasks. All four models use the same expanded
parent, 1,915 training transitions, 946 validation transitions, seed 23, twelve
epochs, batch size two and 64-action backpropagation windows. Evaluation uses
the fixed final checkpoint. History adds input tokens and computation; these
are matched training schedules, not equal-compute runs. The two history models
trained in 573 and 604 seconds on separate H100s.

Neither intervention resolves autonomous completion in this comparison.
The history/event model reaches frequency adjustment in instance 313, then
executes `+1 Hz` 262 times without activating a new reactor. Its instance 312
reaches partial progress 0.48. These are distinct behaviours, but no model has
yet demonstrated learned calibration or hypothesis testing. Aggregate
imitation accuracy cannot stand in for those operations.

The reference procedure's six mixed-control correction episodes all completed;
those successes used expert actions and do not enter this table.

Data and complete progress curves: [results/learning.json](results/learning.json).
The reusable [trajectory analysis](analyze_rollouts.py) measures received
instrument evidence, sampling exits, frequency changes and new activations
without using the reference's phase labels. An exit only means that execution
left a repeated measurement; it does not establish adequate sampling.

## Information needed for scientific inference

Reading the real trajectories identified two representation gaps. The exact
measurement ledger records which crystal a reactor contains when its frequency
is read. The current neural summary drops that containment relation; frequency
and crystal measurements must then be associated indirectly through history
or location. In 2,417 reference/correction transitions, 15 expert navigation
targets also have another object with exactly the same candidate text and
numeric features. Those indistinguishable candidates all refer to tables.
The uncorrected gold validation set has no expert-target collision, so this
does not explain all errors.

Frequency outputs are present: 159 actual adjustments have their returned
frequency in the prediction target. However, signed logarithmic scaling makes
a 1 Hz error near 5,000 Hz contribute only about 4 × 10⁻¹⁰ squared error.
A small aggregate prediction loss therefore does not establish the precision
required for activation. Native actions can express the required frequencies;
the unresolved issue is acquiring and using the numerical relationship.

The task path now preserves object identity and containment as optional relations
between records and action arguments. The pretrained semantic models also
execute through the same native interface, receiving public observations and
exact experimental records. Actual
semantic-model execution, learned CDM execution and expert demonstrations will
continue to have distinct results. Deployed adapter learning and learned
computation allocation remain subsequent training work.

## Evidence use after a supplied experimental prefix

Three trained CDM checkpoints received the same actually replayed measurements
and two calibration anchors in each of instances 310–313. The original expanded
model, action-run correction model and measured-history action-run model each
then controlled 64 native actions. **All twelve runs remained at progress 0.72,
with no new reactor activation.** The 0.72 was achieved during the supplied
prefix and is not autonomous model performance. Full calibration and measurement
access therefore did not resolve these policies' subsequent decisions.

The native object RNGs are saved at initial collection and restored for replay;
every prefix observation and action result was compared in its recorded JSON
representation. Earlier attempts without complete replay stopped before model
takeover and are excluded. Results are in
[results/prefix_takeover.json](results/prefix_takeover.json).

This motivates an object-binding comparison. A candidate action now queries
measurement records through exact public relations between its arguments,
the measuring tool, the measured object, and an object's observed contents.
Identifiers are equality pointers, not numeric features. All original input
tensors remain unchanged. The new reader has zero output at initialization,
preserving the parent model's initial predictions.

The comparison uses the same reader and parameter count in both
conditions: actual object relations versus the same relation matrix set to
zero. Both resume the measured-history action-run parent, use the same 1,915
training transitions, 946 validation transitions, seed 47, twelve epochs,
batch size two, 64-action windows and learning rate 0.00003. The primary question
is whether correct binding produces new native activations after the common
prefix; full 300-action execution additionally measures autonomous collection
and control. Lower training loss without improved executed decisions will not
establish that the binding solved the task.

Both twelve-epoch runs finished. With object relations present, full native
progress was 0.40, 0.40, 0.32 and 0.40 (mean **0.38**). With those flags masked,
it was 0.64, 0.36, 0.28 and 0.40 (mean **0.42**). Neither completed a task.
After the supplied measurement prefix, the identity condition remained at
0.72 on all four instances; the masked condition reached 0.76, 0.80, 0.80 and
0.76. Neither activated a previously inactive crystal. This comparison does
not support the predicted benefit of these identity features under the current
training objective. Both models have 55,360,574 parameters; peak allocated
training memory was 44.71 GB per GPU.

Data: [results/object_binding.json](results/object_binding.json).

To separate collection and navigation from tuning, the same two models also
took over at the first untried crystal's frequency dialogue in instances 310
and 311 (prefix lengths 205 and 206). No earlier action had adjusted an unknown
working frequency. All four runs executed 48 frequency changes, ended at
progress 0.80 from a supplied 0.76, and activated no new crystal. Most choices
were repeated increases of 1 or 10 Hz. This condition reaches the numerical
decision itself, unlike the earlier sampling loops, but does not demonstrate
a useful learned calibration rule. The partial score increase records tuning
activity, not successful calibration.

Data: [results/first_tuning.json](results/first_tuning.json).

## Native semantic-policy probes

The frozen Qwen3.5-2B and Qwen3.6-35B-A3B now execute their own generated
actions through the same public simulator interface. Neither uses the scripted
reference to select or repair an action. Exact measurement records are separate
from the model's optional notes and hypotheses. Model calls retain the supplied
observations, full generated output and actual returned feedback.

The initial short-generation probes on instance 310 used 32 actions and a
128-token greedy output limit. The 2B model executed no successful native
actions; the 35B model executed 18. Both remained at initial progress 0.08.
With a 2,048-token thinking budget, the 2B model exhausted its first call before
producing a final action. The 35B model executed one action, then produced a
valid JSON action inside a Markdown fence that the initial parser rejected.
The parser now accepts a single complete JSON fence, preserving its content;
the old interrupted run is unchanged. These protocol probes have different
budgets and do not rank the models' scientific capabilities. Their actual
configurations and outcomes are in
[results/semantic_protocol_pilots.json](results/semantic_protocol_pilots.json).

An optional research notebook lets the model carry its own hypotheses and
plans between decisions. Notes are committed after the associated real action
returns, and remain identified as model claims rather than measured facts.
The next evaluations use this path and conditional takeover after actual
instrument readings, comparing evidence collection with evidence use.

An action-only probe now restricts generation to the same public native
candidates available to CDM. This prevents malformed actions without supplying
a scientific decision. With the exactly replayed 193-action prefix in instance
310, both frozen models executed all 64 subsequent actions successfully.
Qwen3.5-2B remained at 0.72; Qwen3.6-35B-A3B reached 0.76 by restoring an already
calibrated crystal to its reactor. Neither activated a new crystal. These
single-instance probes use greedy decoding without thinking or a notebook;
they do not measure the models' strongest reasoning configuration.

The complete measurement records were available in these calls. The 2B model
repeated radiation measurements throughout. The 35B model returned to the lab,
restored the calibrated crystal, then navigated and made further measurements;
it did not enter frequency adjustment. Thus these particular executions expose
a gap in stopping sampling and initiating an intervention, before they can
test numerical inference. Results are included in
[results/prefix_takeover.json](results/prefix_takeover.json).

## Additional training

The action-run model without measurement-history inputs completed 24 more
epochs on the same correction data at learning rate 0.00003. Its final
checkpoint reached progress 0.40, 0.52, 0.32 and 0.52 in instances 310–313
(mean **0.44**, **0/4 completed**). The earlier checkpoint scored 0.37 on these
instances. The runs share instance identities, but their noisy instrument
streams were not replay-matched. More training improved observed partial
execution without establishing scientific-task completion.

The planned concurrent history-input continuation exhausted GPU memory before
its first completed epoch. There is no completed two-condition comparison for
this extension. Subsequent training avoids computing unused outcome predictions
for every candidate: only the actually executed action's outcome is supervised.
Output and gradient equivalence are tested. Both current object-binding arms
use this implementation on separate H100s.

Completed results: [results/extended_learning.json](results/extended_learning.json).

## Learning from public calibration outcomes

An episodic numerical bridge now trains the existing CDM scalar predictor and
instance adapters on later public working-frequency returns. The source's
observation projections, recurrent state, workspace and native action path stay
frozen; a quantity encoder, predictor weights and evidence-step parameters are
trained. Queries are frozen at the first adequately measured, untried crystal's
frequency dialogue. Its subsequent working-Hz return supplies the target.
Earlier actual calibration evidence updates A/B before that prediction.

The specified training split contains 48 instances; three historical failed
collections yield no usable query, leaving **45 instances and 90 queries**.
Validation has eight instances and 16 queries; the held-out rule has four
instances and eight queries. The numerical output uses a training-only prior
from 180 distinct calibrated crystals (mean 3674.64 Hz, standard deviation
1703.82 Hz), with a newly initialized final readout. Neither validation nor
held-out labels set this prior. Two training seeds completed ten epochs each.

The fixed final checkpoints below compare retained A/B with computing and
discarding the same updates. Both conditions use identical frozen observation
contexts and exact initial adapter tensors. Two-support queries concern the
first unknown crystal; three-support queries follow its actual confirmation.
Each table entry is **retained / discarded**.

| Training seed | Evaluation | Available calibrations | NLL | MAE (Hz) |
|---|---|---:|---:|---:|
| 31 | Validation | 2 | 8.021 / 8.020 | 1708 / 1599 |
| 31 | Validation | 3 | 7.620 / 7.753 | 1058 / 1023 |
| 32 | Validation | 2 | 8.037 / 8.098 | 1695 / 1658 |
| 32 | Validation | 3 | 7.661 / 7.817 | 1098 / 1113 |
| 31 | Held-out rule | 2 | 7.883 / 8.053 | 1198 / 1510 |
| 31 | Held-out rule | 3 | 7.911 / 8.045 | 1534 / 1727 |
| 32 | Held-out rule | 2 | 7.921 / 8.100 | 1238 / 1451 |
| 32 | Held-out rule | 3 | 7.915 / 8.109 | 1542 / 1890 |

Every condition scored **0/24 predictions within 2 Hz** of the public working
setting. Retained updates change predictive distributions and improve some
paired likelihoods, without establishing precise calibration. Final validation
NLL also worsened from the initialized prior: 7.750 to 7.821 for seed 31 and
7.756 to 7.849 for seed 32. Validation-selected best epochs were 5 and 3; the
table uses epoch 10 consistently. These are numerical predictions, not native
activation results.

A read-only replay of training instance 130 localizes a current limitation.
Its known anchors are 4046.07 and 5526.29 Hz. At the next query, the final models
predict those known crystals as 3925.06/3926.95 Hz and 3717.38/3717.62 Hz.
After a third actual confirmation at 1943 Hz, their known-support MAE remains
1350 and 1456 Hz. All 65 real support updates decrease their immediate support
NLL; the updates are active, but do not fit distinct known settings. The two
anchor quantity embeddings have cosine similarity 0.999936 and 0.999990 despite
different public measurements. Context changes also shift predictions of the
same anchors. This evidence identifies weak numerical discrimination and
support fitting as unresolved, before attributing failure solely to ambiguity
or extrapolation to an unknown crystal.

Metrics, initialization, and support-update evidence:
[results/calibration_learning.json](results/calibration_learning.json).

A subsequent train-only pilot tested whether transient context caused this
failure. `context_mode=object_only` zeros the numerical predictor's h/z inputs;
the source still maintains those states for native actions. Both conditions
used the same eight training instances, initialization, minibatch order and ten
epochs, with no validation or held-out selection. Future-query NLL/MAE were
7.855/1297 Hz with current context and 7.915/1374 Hz with object-only input.
Known-support MAE after two/three calibrations was 1177/1216 Hz versus
1202/1219 Hz. Object-only retained and discarded updates had almost identical
future MAE (1374.3305 versus 1374.3112 Hz); its two known anchors in instance
130 still received nearly equal predictions, 3744.67 and 3744.71 Hz.
Removing state drift alone did not restore numerical discrimination or support
fitting under this fixed pilot budget. These are training-instance measurements;
no new generalization or native-completion result is implied.

Data: [results/calibration_context_pilot.json](results/calibration_context_pilot.json).
An autograd calculation on training instance 130 and the seed-31 final checkpoint
measures the local update geometry across all 32,768 A/B coordinates. At the
two-support query, the prediction Jacobians have cosine **0.999988**; the
eigenvalues of `J Jᵀ` are **21.72 and 3,639,464** (condition number **167,526**).
With three supports the condition number is **637,700**. At the two-support
query, the Gaussian loss's variance and mean gradient components also oppose
each other (cosine **−0.951**). Using only 2-Hz observation noise strengthens the
bounded update but moves both predictions together, to **4019.65/4021.53 Hz**.
A diagnostic Gauss–Newton proposal with damping equal to the weakest eigenvalue
requires **59.4 times** the current adapter norm; its actual nonlinear
predictions become **−17,411/−13,986 Hz**, far from its linear approximation.
Stronger damping primarily changes the common prediction. These copied-state
calculations identify weak object-separating directions and unreliable large
parameter steps; they do not establish an improved learning method. The CPU
calculation took 1.90 seconds including replay of 65 actual updates; no held-out
instance or new training was used. Details and raw-result provenance:
[results/functional_geometry_seed130.json](results/functional_geometry_seed130.json).

A second single-intervention pilot moved the quantity-type embedding before
the numeric encoder's hidden nonlinearity, with unchanged parameters, inputs,
current context and the same eight-instance training budget. Compared with
late binding, typed interaction reduced training-instance future NLL from
7.855 to 7.814 and MAE from 1297 to 1241 Hz. Known-support MAE remained
1160/1176 Hz after two/three calibrations (baseline 1177/1216 Hz). Mean pairwise
embedding distance increased from 0.123 to 0.175 for the two anchors, but their
cosine similarity remained 0.999985. Instance 130's two known frequencies were
still predicted almost identically, as 4072.866 and 4072.830 Hz. All 16 future
predictions missed the 2-Hz criterion. This small training-only gain does not
resolve support fitting or establish improved generalization.

Data: [results/calibration_typed_interaction_pilot.json](results/calibration_typed_interaction_pilot.json).


### Native quantitative control after supplied evidence

The fixed epoch-10 numerical predictors each ran 32 native takeover trials:
eight instances, two or three confirmed calibrations, and matched retained or
computed-discarded updates. All 64 prefixes replayed exactly; no trial errored.
The controller translated a predicted Hz setting into the native additive
dialogue choices while the source policy continued its recurrent state.

One retained-update trial (training seed 31, instance 344, three calibrations)
activated a new crystal and completed the remaining task in 17 actions. The
other 63 trials produced no new activation. This is a **conditional takeover**,
not autonomous completion: its supplied prefix had already collected all
measurements and activated the preceding crystal. On the successful trial,
the model first proposed approximately 3863 Hz, then about 4160 Hz after
leaving and reopening the dialogue, without an additional calibration label.
The matched discarded-update controller crossed from 4154 to 4164 Hz without
activation. A single completion with context-dependent predictions does not
yet demonstrate repeatable numerical inference.

[results/calibration_native.json](results/calibration_native.json) contains
all 64 trial identities, supplied-prefix scores, outcomes and raw output paths.
## Explicit affine hypotheses with learned priors

A new task readout maintains nine single-quantity affine hypotheses and uses
public calibration support to update their Gaussian coefficient posteriors and
model probabilities. The affine basis is an explicit modeling assumption. This
is separate from the original CDM A/B mechanism; the source core and action
policy remain unchanged. Fifty-four shared parameters learn coefficient priors,
noise and hypothesis weights from the same 45 effective training instances.
All nine hypotheses retain nonzero prior mass. Both seeds ran the fixed
100 epochs at learning rate 0.001, without selecting a validation checkpoint.
The initial uniform model uses the same training-only normalization.

The table reports MAP-expert future frequency error in Hz, followed by the
number within 2 Hz of the later public working setting. Mixture means are
reported separately in the data because averaging competing hypotheses can
produce a frequency proposed by none of them.

| Evaluation | Support | Uniform initial | Learned seed 31 | Learned seed 32 |
|---|---:|---:|---:|---:|
| Validation, 8 queries | 2 | 241.39; 7/8 | 38.23; 7/8 | 38.23; 7/8 |
| Validation, 8 queries | 3 | 1.2044; 8/8 | 1.2042; 8/8 | 1.2042; 8/8 |
| Held-out rule, 4 queries | 2 | 1276.85; 3/4 | 1589.50; 0/4 | 1589.50; 0/4 |
| Held-out rule, 4 queries | 3 | 1.5175; 4/4 | 1.5182; 4/4 | 1.5182; 4/4 |

For seed 31, validation mixture NLL changed from 4.293 to 3.250 with two
supports and from 2.340 to 2.183 with three. Held-out-rule NLL changed from
3.934 to 4.427 and from 2.486 to 2.421, respectively; seed 32 was similar.
The learned two-support mixture mean still had validation MAE near 696 Hz.
Computed-and-discarded posterior controls achieved no within-2-Hz predictions
in either split. Known-support fitting now reaches sub-Hz error.

Most of the three-support precision already exists with the explicit affine
prior and Bayesian update before parameter training. Learning improves some
in-distribution likelihoods and errors while harming unseen-rule two-support
choices. Additional real calibration evidence recovers accurate predictions
within this hypothesis class. This does not establish discovery of that class
or autonomous collection and experiment selection. Subsequent variants informed
by these results must treat the reused validation and rule-held-out instances
as development data; a final generalization test needs fresh instances.

Data: [results/calibration_mixture.json](results/calibration_mixture.json).

A subsequent development comparison removed quantity-name-specific
hyperparameters: all nine candidates share two coefficient means, two scales
and one noise parameter, with fixed uniform mixture priors. The five-parameter
model starts with exactly the same predictions as the original initialization.
After a train-only check, it used the same 45 training instances and fixed
100-epoch budget in seeds 31 and 32. No other hyperparameters changed.

Both shared-prior runs achieved development-validation MAP MAE near 38.20 Hz
with two supports (7/8 within 2 Hz) and 1.2055 Hz with three (8/8). On the
reused rule instances, two-support MAE returned to 1276.85 Hz with 3/4 within
2 Hz, compared with 0/4 for learned quantity-specific priors; the three-support
result remained about 1.5178 Hz with 4/4. Shared-prior two-support NLL was
about 2.958 on those rule instances versus 4.427 for quantity-specific priors,
while validation NLL was worse, 3.513 versus 3.250. The shared model learned
normalized coefficient means near [0.075, 0.535], scales near [0.574, 0.596]
and observation noise near 2.333 Hz. All nine prior weights remained 1/9.

This joint sharing-and-uniform-weight intervention avoided an observed
development-set regression; it does not isolate mixture weights alone or
establish fresh-test generalization. The reused validation and rule instances
are development evidence for this comparison.

Data: [results/calibration_shared_prior.json](results/calibration_shared_prior.json).
