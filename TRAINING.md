# Data and training

## First multimodal training route

The architecture is fixed in [DESIGN.md](DESIGN.md); initial dimensions and optimization settings are in [configs/cdm_multimodal_v1.json](configs/cdm_multimodal_v1.json). Pretrained image/text features, native vision-language context and numeric inputs enter one persistent core. Load the released CDM core before training its new connections when the task schema matches. The frontend and semantic reader are implemented. The [DiscoveryWorld task path](experiments/discoveryworld/README.md) now connects public observations, dynamic actions, trajectory imitation, numerical-result prediction and fresh-interaction actor–critic updates. The [joint scientific-task path](experiments/continual_experiment/README.md) now connects activity selection, retrieval and deployment-time prediction updates to the persistent core. It uses structured observations; multimodal joint training and learning that improves physical inference remain separate research tasks. Existing lifetime commands continue to train structured process events.

| Training phase | Updated parameters | Data and objective |
| --- | --- | --- |
| Pretrained initialization | Load weights; no optimization | Frozen SigLIP 2 and Qwen3.5-2B; released CDM checkpoint with its original dimensions and action schema |
| Connection and behavior training | Input projections, semantic reader, input gain and task heads; then the reused core at a lower rate | Observation–action–outcome sequences; action imitation and outcome prediction at zero and nonzero refinement depths |
| Teacher transfer | Connections and student prediction/action parameters | Qwen3.6-35B-A3B candidate trajectories with measured task outcomes; imitate useful actions and distill supported predictions |
| Joint interaction | The same shared core plus activity/value heads | Recurrent PPO over continuing lifetimes; prediction of received outcomes; operation value targets where useful |
| Learning during interaction | Instance adapters during deployment; shared parameters through replayed adapter updates during training | Loss on outcomes after the learning step, using only experience available when that step occurred |
| Quiet-period consolidation | Initially prediction/action parameters; later shared representations with state recomputation | Mixed recent and older experience, and distillation from useful deeper computation into the fast policy |

These are initialization and optimization phases of one model. Do not require each auxiliary task to reach its best isolated score before integration.

### Weight reuse and semantic training

The checkpoint loader preserves every core tensor and the saved configuration. Start with the source observation path and train its new modality connection, then jointly update the reused core. New tasks retain compatible state modules and define new action/outcome heads; process-control actions do not become digital tools merely because their tensor shapes match.

The semantic model remains frozen initially. Train its learned reader through CDM's action and prediction losses. Compress a semantic result to eight context tokens, then train the core using those tokens at different observation ages and with calls still pending. The policy should also perform useful actions when no semantic result is available. Semantic-call cost, elapsed time and resulting task return train its eventual call/target policy.

The offline teacher receives the task's observation history and emits candidate actions or experiment sequences. Execute them in training environments and retain observations, actions, returned outcomes and task scores. Train action imitation and outcome prediction on these trajectories. A teacher's explanation can accompany a trajectory, but environment outcomes supply the prediction targets. Keep collection seeds and task compositions separate from evaluation. Record teacher revision, generation settings and inference time.

Use the same frozen models, trajectories and compute accounting for recurrent, restarted-workspace and persistent-workspace comparisons. Include a direct pretrained vision-language agent and a fixed semantic-call schedule. This separates the contribution of pretrained competence from the contribution of CDM's persistent computation and learned activity.

### Trajectory adapter

Each training item identifies a lifetime and an interval of arrival-ordered events. Store capture/arrival/action times, input assets, observed numeric values and presence fields, instruction chunks, pending requests, the executed action, chosen internal activity, returned outcomes, reward and terminal flags. Preserve the distinction between a training-window boundary, a changed task, and the end of a lifetime.

Cache frozen image/text features once per asset and encoder revision. For the semantic model, key cached final-layer features by model revision, the complete input sequence, processor settings and included images. Recompute the trainable projections and reader during each training pass; their compressed outputs cannot be reused across reader updates. Do not duplicate an unchanged cached image as a new independent measurement. During deployment, recompute perception when an image or instruction changes; numeric/timing events can update the model between expensive perception passes.

Start from available task-native interaction trajectories and reference policies. Report any reference policy's extra simulator information; its labels can initialize behavior without giving that information to CDM. Split environments, generated scientific rules, task compositions and collection seeds before making tensor windows. Public evaluation demonstrations are not training examples for that same evaluation split.

The first image/text task adapter targets DiscoveryWorld's official observation/action API. The existing instrument and thermal families remain the independent-time and adaptation experiments. A benchmark's standard step count and a wall-clock task duration are different quantities.

### Starting optimization settings

Freeze the pretrained image/text towers and semantic model. Start AdamW at **1e-4** for new projections, reader and task heads, and **1e-5** for the reused core when it is unfrozen, with gradient norm clipping at **1.0**; tune on validation trajectories. Use BF16 dense computation and FP32 recurrent state and adaptation reductions. Start with **32-event unrolls**, **256-event rollout windows**, and **128 streams per GPU**, adjusting batch size from measured memory and throughput. These settings are a starting configuration, not measured optimal values.

Retain state, workspace, memory and instance adapters across tensor windows. Recompute representations under the current shared parameter version when replaying a window; include a preceding context segment and report its length. A finite context segment does not exactly reconstruct an arbitrarily long recurrent history.

Train action/prediction readouts at depths 0, 1, 2 and 4. The depth-zero path learns useful immediate behavior; additional computation must improve subsequent decisions enough to cover its time and resource use. Train under delayed observations and input arriving between refinement blocks.

During joint training, optimize action and activity return, value regression, action-conditioned prediction and adaptation on future outcomes. Use duration-aware discounting for operations with unequal durations. Fixed-depth schedules and fixed learning schedules share the same observation encoder and action head for comparison.

### Learning that persists

Prediction adaptation is already implemented in the lifetime model. Its deployed-update objective replays the actual learning choices and their then-available buffers. Action adapters, long-term episode retrieval, general tool/subgoal choices and consolidation still need integration.

The proposed action-adapter update uses recent on-policy action returns. Older experience supports prediction, value learning or explicitly defined teacher distillation; stale action log-probabilities are not treated as fresh PPO data. Training must include revisited conditions and later tasks so that useful retention receives a learning signal.

During quiet intervals, train on both recent and retained episodes. Distill a deeper policy only on tasks where its actions improve measured outcomes, or use a task-native reference. Repeatedly matching a model's own unsupported prediction is not a source of new observations. Evaluate the resulting fast policy on held-out interactions and returning task conditions.

### DiscoveryWorld instance updates

The [public calibration records](experiments/discoveryworld/calibration.py) and [task-specific scalar likelihood](cdm/core.py) are implemented. The quantitative path now deploys the core scalar predictor and A/B updates with causal future-query training. Native evaluation uses an explicit additive-setting controller; the learned action-score route below remains a proposed extension. The completed 64 trials include one successful takeover after supplied evidence, with no established repeatable numerical inference. [Task results](experiments/discoveryworld/RESULTS.md) give the actual comparisons.

Start with Reactor Lab's measured relation between crystal properties and a working resonance frequency. Store received facts as `(action_id, object_handle, quantity, value, unit, received_step, conditions)`, retaining raw messages. Bind instrument readings to the actually measured crystal; bind the current dialogue's Hz and proposed delta to its reactor and currently observed crystal. UUIDs identify these relations, without supplying numerical features. A calibration label is a publicly read frequency paired with that crystal's observed reactor activation; an arbitrary uncalibrated setting is not a correct-frequency label. Preserve measurement conditions, particularly the radiation meter's surrounding sources. Repeated observations of one fact do not add independent evidence.

Build a width-`w` query from the named quantity and that object's available measurements, with presence and unit fields. Concatenate it with `core.state_features(h,z)` to retain the existing `3w` predictor input. For this first integration, adapt one scalar relation—working frequency—from these calibration records. Use invertible Hz scaling and a likelihood whose variance limits respect the required physical precision; the process task's sensor/noise constants must not silently become frequency tolerances. Keep exact current Hz and the public dialogue delta outside compressed text. For each tuning candidate, compute its proposed setting from these numbers and the public range. Its learned action score reads the predicted frequency, uncertainty and proposed-setting residual. Navigation and measurement candidates remain in the same dynamic action set.

The intended interfaces are:

```python
queries = bind_queries(public_facts, native_candidates)
mean, logvar = core.predict(query_inputs(h, z, queries), state["A"], state["B"])
logits = score_candidates(h, z, native_candidates, mean, logvar, bound_controls)
# Only after a native action returns new, correctly associated evidence:
A, B, loss = adapter_update(core, A, B, support_x, support_y,
                            precision, present, learning_rate, differentiable=replay)
```

Online, update only the instance's `A/B`; freeze shared perception, GRU/workspace, query encoding and action readout. Begin with one fixed update after newly completed calibration evidence becomes available, effective at the next decision. Reuse the existing low-rank update and enable its evidence step rule, `eta / (gradient_norm + kappa)`; the outer objective trains these shared step parameters along with query/prediction/readout weights. During outer training, replay every update on its then-available buffer, preserving the adapter version used for each prediction/action, as in [lifetime objectives](experiments/lifetime/objectives.py). Optimize later received-reading likelihood and native-action return through those actual updates. Unlike `predictions_per_tick`'s current detached policy path, this proposed action path must retain gradients through `A/B` and their prediction readout; detach the return/advantage, not the learning trajectory. Keep shared weights fixed during collection and this replay, then take the outer step. The update-to-payoff horizon must reach the new crystal's tuning outcome; truncating `h/z` gradients need not detach `A/B` every 16 actions. First check replayed adapters and action probabilities against collection before claiming this objective works.

The deciding experiment uses a common public measurement/calibration prefix on held-out instances, then actual tuning of a crystal whose working frequency has not been observed. Its other physical properties may have been measured. Compare **updates retained** with **the same updates computed and discarded**, holding shared weights, prefix, `h/z`, facts, refinement budget and native-action budget equal. Score unseen-frequency error against a later, independently observed public working setting, and report it separately from native activation rate and actions required; lower offline prediction loss alone does not establish the route. Two calibrated crystals fit an affine relation for several candidate physical quantities, so they do not identify the correct law. Preserve that ambiguity: use native trials on the third crystal to reject hypotheses, admit a new calibration only after observed activation, and test the remaining crystal. Neither the reference procedure nor the model receives the hidden generating law. A useful result requires retained updates to improve later native actions; if only prediction improves, repair the prediction-to-action connection before scaling training.

Report prediction after two confirmed calibrations separately from prediction
after the next confirmed calibration. Freeze each query at its first fully
measured, uncalibrated dialogue, before any frequency trial. Its eventual
working setting is a later evaluation target. The outer objective differentiates
future prediction loss through the updates actually made from earlier evidence,
following the adaptation objective of
[gradient-based meta-learning](https://proceedings.mlr.press/v70/finn17a.html).
The research question here is whether those updates improve subsequent physical
decisions in a continuing model. When several relations remain compatible with
the evidence, uncertainty and the value of another experiment matter; a single
mean estimate need not be a useful setting to try. Probabilistic predictions
over such alternatives are an extension, related to
[prior-data fitted inference](https://arxiv.org/abs/2112.10510), rather than a
capability already established by this Gaussian predictor.

### Named numerical relations and executable experiment feedback

`train_calibration_mixture.py` learns coefficient and observation-noise priors
through future-query likelihood after actual public calibrations. The typed
variant learns 54 parameters; `--prior-mode shared` ties coefficient and noise
priors across the nine named quantities and fixes their prior weights uniformly,
leaving five learned parameters. The affine basis is specified. Measurements
retain their native units and condition groups; a vector reads the latest
complete condition for each quantity. Normalization uses training instances only.

`mixture_policy.py` preserves the frozen source policy's perception and ongoing
state, fits the numerical posterior on the received support, and sends its
highest-probability candidate setting to explicit native control. With
`--local-probes`, a failed point prediction can initiate trials within two
candidate standard deviations, at most sixteen extra adjustments per target and
within the same total native budget. Only actual returned frequency/activation
feedback advances that probe. External prefix priming updates source state and
received evidence without initiating unexecuted trials.

These components have completed numerical and native comparisons. They do not
train the original action head to invent hypotheses or choose experiments.
The next objective is to train that head and its continuing state through the
use of posterior beliefs, actual intervention outcomes and successful subsequent
actions, then evaluate autonomous evidence collection from reset.

### Reading numerical beliefs when choosing actions — 27 October

The next comparison connects the same continued CDM state to a learned residual
action reader. It receives public object-bound measurement summaries, confirmed
calibrations, observed activation/containment, candidate relation uncertainty,
and actual recent action feedback. No feature selects or removes an action.
The source recurrence, perception and shared-prior numerical model remain fixed
in this first comparison; the new action reader alone is trained on the 45
successful reference training instances, with eight separate validation
instances. Unsuccessful raw traces remain available for outcome and recovery
learning. Its zero initial output preserves the parent policy.

The paired control has identical parameters and training, with this explicit
evidence vector zeroed. Both retain the same source state and the same numerical
control primitive. Thirty epochs of action-run-weighted imitation, seed 61,
batch size 32 and learning rate 0.001 precede fixed-final native evaluation.
The question is whether reading the accumulated evidence lets the model finish
sampling, acquire calibrations and complete an experiment from reset. Compare
whole-task success, new activations and actions; offline accuracy is secondary.
Further reward training will use states actually visited by these policies.

### Joint state, action and computation training — 27 October

`experiments.discoveryworld.train_joint_policy` starts from the successful
seed-61 evidence reader. Unlike reader-only training, it replays received
observations through the current GRU, Transformer workspace, multimodal
projections and action head, and updates them together. Frozen SigLIP features
and the five-parameter numerical prior are reused. Old cached recurrent states
are not training inputs.

Each observation advances the GRU once. A compute allocator reads the updated
state and candidate-pooled public evidence, and chooses 0, 1, 2 or 4 shared
workspace refinements. These refinements neither obtain another measurement
nor advance the native world. Instrument actions acquire new information.
The existing numerical primitive and received-support updates remain fixed.

The first comparison uses the same 45 successful training traces and eight
validation traces as the established reader. It trains three epochs, batch
size four, truncated sequences of 16, core learning rate 0.00001, reader and
allocator rate 0.0001, gradient clipping 1, and BF16 linear/attention operations
with FP32 GRU. The fixed-compute arm always uses one refinement. The adaptive
arm trains all four depths with half uniformly weighted imitation loss and
half allocator-weighted imitation loss plus 0.01 per refinement. A sampled
depth carries the state to the next training observation. Greedy allocation
is used when acting. This objective is a warm start from executable behaviour;
it does not yet learn compute value from later task return.

The adaptive arm has additional training computation. We report that cost and
also execute its *same final weights* with a fixed one-refinement schedule,
which isolates allocation at execution from the different learned parameters.
All final checkpoints are evaluated from reset on instances 1010–1013 and
1034, 1039, 1044, 1049, using the parent's saved initial native noise streams
and the unchanged 400-action budget. No checkpoint is selected on that set.
Report task completion, actions including failures, refinement counts and
separate perception/policy/environment time. Cross-task experience retention
requires a separate task-return experiment; per-instance posterior fitting
does not establish it.

After that completed comparison, a conditional computation experiment uses
four existing *training* trajectories, 710–713. In each it selects the prior
measurement decision (after eight readings in a long run) and the first
numerical-setting decision. These eight public prefixes are replayed exactly,
including their initial native noise. Each is continued four ways: 0, 1, 2 or
4 refinements at the first decision, then the same fixed-one policy. The
trained multi-depth model, numerical primitive and 400-action total budget are
held fixed. Compare subsequent completion and progress minus 0.001 per native
action; record refinement counts separately. A supplied prefix is a controlled
training-state experiment, not another autonomous task completion. These 32
branches ask whether internal computation changes later return before using
such outcomes as allocation targets.

### Action-consequence prediction and workspace feedback — 27 October

The next comparison trains action-conditioned result and behavior-return
forecasts, then lets the persistent workspace read and revise those forecasts
before acting. Both successes and failures supply prediction targets. The
GRU, workspace, projections and action reader train jointly; frozen perception
and numerical tools remain shared with the reference. The data, loss, matched
comparison and executable commands are specified in
[the prediction experiment](experiments/discoveryworld/PREDICTION.md).

### Training the cost of actual experimental decisions

After the reset-based comparison, the seed-61 evidence reader initializes a
policy-gradient phase. The continuing source and numerical prior remain fixed.
The policy collects native Reactor Lab trajectories from fresh training seeds
710, 711, 712, 713, 715, 716, 717 and 718 with temperature 0.5. No reference
trajectory or supplied measurement prefix enters these rollouts.

The finite 400-action objective is received task-progress change minus 0.001
per native action, discounted by 0.999. Every scalar adjustment consumes this
same budget. Only actions actually chosen and executed by the learned reader
receive its policy gradient; a proposal overridden by the numerical primitive
has no actor gradient. Its time, cost and returned task reward still enter the
return of preceding learned choices. Cached frozen-core states are retained in
full precision so action probabilities can be recomputed under the exact state
seen at collection. The value head reads that state and public evidence.

The initial update uses four PPO passes, learning rate 0.0001, clip 0.1,
batches of 64, entropy coefficient 0.001 and value-loss coefficient 0.5.
Evaluation uses new instances 810–813 and 834, 839, 844, 849, with fixed-final
weights, 400 actions and paired initial native noise. The existing 610-series
comparison is not used to select this update. Success and total native actions
are reported together. This phase trains the action reader on task return;
the internal refinement schedule and cross-task retention remain separate work.

The first eight native rollouts finish four tasks. The unconstrained PPO update
changes the greedy action on 9.8% of their 2,048 reader-controlled decisions,
with mean behaviour-policy KL 1.376. A subsequent step-length comparison uses
only these training decisions: halve the parameter-update fraction until their
mean KL is at most 0.02. It selects one eighth of the proposed update (measured
KL 0.00247). This is a policy optimization choice, not additional task knowledge;
the fixed parent, full update and projected update are all evaluated and retained.

A second round continues the projected policy and value estimator on fresh
training instances 720, 721, 722, 723, 725, 726, 727 and 728. It retains the same
objective and optimizer settings. The 810-series set is a development validation
set for this continuation. Both rounds and the unchanged parent remain in the
[completed results](experiments/discoveryworld/RESULTS.md#further-instances-and-actual-return-training);
there is no efficiency gain to report from these two rounds.

The next comparison measures the consequence of ending a measurement run.
It uses the first repeated measurement sequence of at least 24 actions in each
of training trajectories 710–713. Both branches replay the same actual prefix
through eight measurements, restoring the initial native RNG streams and
checking every returned public observation. One branch takes another reading;
the other executes the different action that this same training policy took
when it eventually left that run. No later reading is supplied. Both then use
the unchanged full seed-61 policy greedily, with a total budget of 400 actions.
The action intervention, proposals, actual feedback and later task results are
recorded separately. These are conditional training-state comparisons, not
additional autonomous completions or test-set results.

Those four pairs yield two equal-completion improvements (29 and 24 fewer
actions) and two tied failures. The next fixed comparison starts both readers
from the original full seed-61 checkpoint and replays the same 9,354 reference
actions for one epoch, batch size 32, learning rate 0.0001 and order seed 81.
One reader additionally learns the two informative action pairs with a
pairwise ranking loss, weighted by their actual return differences and scaled
by 0.25 relative to reference replay. The matched control computes the same
loss with coefficient zero. Source states, features, numerical prior and
controller remain fixed. Tied branches do not supply preferred-action labels.
Both final checkpoints are evaluated from reset on the paired 810-series
development instances. This tests whether an actual improvement from a
training-state action intervention can be learned and used elsewhere; the
conditional branch successes themselves are not the endpoint.

The completed comparison yields 6/8 tasks in 260.250 mean actions for replay
alone and 5/8 in 271.500 for paired-return training. The latter learns the two
sampled preferences but loses a previously successful instance. No additional
coefficient search or rollout training has been started from this model.
Future paired data must cover both useful and unnecessary measurements, then
test the resulting choice policy on new instances.

### Option-value gradient correction, 19 October 2026

The previous activity-value term centered estimated values with the same differentiable probabilities used to weight the loss. Its gradient canceled. The implemented function now uses **detached centering weights and detached estimates**, so the activity head receives the expected-value gradient while the value estimator keeps its separate regression objective.

The regression test compares this gradient with direct expected-value optimization, checks both activity factors and masked options, and checks that the value estimator receives no gradient from this policy term. This fixes an auxiliary policy-learning term; it does not establish a new performance result or change the meaning of already published process-prediction experiments.

### Evaluation and scaling

Report official task completion and reward; adaptation speed after a change; performance when an earlier condition returns; task transfer; and the contribution of state, retrieval and parameter updates. Measure capture-to-applied-action latency, its median/p95/p99, action age, observation/compute counts, GPU memory, and total training/inference time. Measure energy directly before claiming power savings.

Use the same encoder and training data for recurrent, restarted-workspace, persistent-workspace and learning comparisons. Include task-native reference agents separately. Account for all failed and completed training runs in total compute.

Use the two H100s for core training and semantic feature/trajectory production. The large offline teacher has a separate collection budget; it is not resident in every training worker. Additional A100 machines can run independent seeds, fixed comparisons and data collection without forcing a heterogeneous synchronous group. Increase throughput after a complete observation–action–outcome–update path works; a longer run of an ineffective objective does not replace that integration.

## Existing studies and detailed objectives

The [supervised pilot](experiments/neural_pilot/README.md) trains the older [sequence core](cdm/model.py) on contextual recall. The [lifetime experiment](experiments/lifetime/README.md) trains the current [continual core](cdm/core.py) on structured process events. The sections below retain their data contracts and training mechanisms. [RUNNING.md](RUNNING.md) separates executable commands from the new integration work.

## Interaction data

The training unit is a lifetime containing changes of goals and operating conditions, delayed results, idle periods, and returns to earlier conditions. It contains time-ordered received observations, computation operations, applied actions, and outcomes. A tensor window used for backpropagation does not end that lifetime or clear its state, experience, or adaptive parameters. Include useful work between observations and observations arriving before a calculation finishes.

| Field group | Stored information |
| --- | --- |
| Identity | Run, trajectory, task family, simulator version, seed |
| Timing | Observation capture and arrival; operation start and finish; action application |
| Input | Modality, source, units, payload reference, goal update |
| Activity | Observation request, refinement, memory read, tool call, learning update, subgoal, or wait; selected target and budget |
| Action | Proposed and applied action, policy probability, pending request identifier |
| Outcome | Measurement, reward, elapsed time, resource use, related action |
| Computation record | Generating context, observation scope, goal, parameter version, depth, retained payload |
| Training state | Policy version, memory capacity, reset locations, adaptation steps, instance-owned adapters and optimizer state |

Sequence batches follow arrival order while retaining capture time. Numeric events can use columnar storage; larger media use indexed assets. Received observations and imagined trajectories have distinct data types.

Split collection runs, layouts, dynamics, goal combinations, and random seeds before constructing training windows. Evaluate new factor combinations separately from longer sequences and changed timing. Repeated contexts must occur within trajectories so reuse can be learned; held-out trajectories and task combinations remain separate.

## Task distributions

The existing grid study contains changing doors, goals, and obstacle positions. Each operation consumes a specified interval. Use three regimes: full context return, return of only some conditions, and continued change without exact repetition. Every method receives the current observations at its action head.

Reference planners initialize behavior from the information assigned to them. Record whether a reference uses the agent's observation history, a reconstructed map, or full simulator state. A planner with extra information can be a teacher or performance reference without becoming a student input.

The main distribution includes instrument tasks with nonlinear dynamics, noisy and delayed measurements, pending requests, and actions that affect future observations, alongside navigation and executable digital tool tasks. These families use the same event/action interface and shared core. Learned activity and adaptation are trained across these families, rather than postponed until navigation is complete. Existing digital-research logs can support prediction and imitation after import; evaluating a new action policy requires an environment in which its consequences can be measured.

## 1. Useful actions at every computation depth

Train the context encoder, workspace initializer, shared refinement, and action head before introducing computation memory. Use reference actions or action values at depths \(k\in\{0,1,2,4,8\}\), with the depth-zero head reading current context.

An initial supervised objective is

\[
\mathcal L_{\mathrm{init}}=
\sum_{k\in\mathcal K}w_k\,
\mathcal L_{\mathrm{action}}\bigl(\pi_\theta(c_t,Z_t^k),y_t^k\bigr)
+\lambda_p\mathcal L_{\mathrm{prediction}}.
\]

Prediction targets are future received observations, rewards, and outcomes conditioned on action and elapsed time. Labels must reflect the information and time available to the corresponding computation branch. Depth weights prevent the model from training only its final readout.

Measure action quality against refinement depth and elapsed time. This establishes whether additional internal computation helps before learning to allocate it.

## 2. Learned reuse under changing conditions

Add bounded computation memory while holding observation schedules and compute budgets fixed. Train the record encoder, context-factor projections, compatibility features, payload revision, and attention reader through the action and prediction losses.

Generate training pairs that change goals, dynamics, or observation content separately and jointly. For each branch, retain the actual generating conditions of its stored computation. Include exact returns, partial returns, and changes for which restarting is useful. These interventions teach the reader which parts of earlier work transfer.

A fixed retention rule is the initial baseline. Compare learned conditional reading with restart, ordinary state carry, exact snapshot lookup, and unconditioned memory attention at matched capacity and read budget. The action-only correction baseline receives the same new context while leaving the stored computation unchanged.

Train for action quality and later return. Hidden-state matching is an optional auxiliary loss with a specified teacher; matching a shallow state can reproduce that state's unfinished error. A more deeply computed teacher is compared with outcome-only learning rather than assumed to be the correct latent representation.

For the environment-training extension, collapse exact duplicate records before retrieval and compare with the current ring-buffer reader. Evaluate repeated refinement on unchanged observations and repeated retrieval of the same computation. Prediction quality should improve only when computation improves the estimate; adding a copied record must not act as another measurement.

## 3. Value of the next operation

For an interaction history \(\mathcal H_t\), create continuations beginning from the same simulator state, observation history, workspace, memory, and pending operations. One continuation performs candidate operation \(b\); another performs comparison operation \(b_0\). Both environments advance during their operations and then use a specified continuation policy.

\[
\widehat{\Delta Q}(b,b_0\mid\mathcal H_t)
=\widehat G_b-\widehat G_{b_0}.
\]

Returns use a common physical-time horizon or discount convention and include operation costs. Pair exogenous randomness when possible while allowing action-dependent transitions to differ. Reference computation is recorded as training cost.

Use these targets to initialize predicted operation values. Improvement divided by operation duration is an optional scheduling heuristic; optimize duration-aware return for the learned policy. Differences between performance gaps at unrelated times do not supply this target.

For tasks in which approximate planning references are available, report information-limited and computation-limited performance separately. Their magnitudes describe the current shortfall, whereas operation values estimate what a particular next activity can achieve.

The lifetime trainer implements these continuations as copies of a process that share its future disturbances, readings and condition changes, with the continuation policy sampling under shared noise ([`runtime.branch_values`](experiments/lifetime/runtime.py)); their returns train an activity-value head. Whether such targets carry signal is measured, not assumed: [`branches.py`](experiments/lifetime/branches.py) values the same states twice with independent randomness. One sampled continuation per option repeats with a correlation below 0.1 at horizons of 4 to 32 ticks, so a single choice's consequence is buried in the variability of what follows; continuations on the policy's most probable choices over 4 ticks, averaged over four independent sets, repeat with a correlation of 0.75. The corrected policy term follows the head in proportion to the share of fresh continuations' variance it predicts; the earlier implementation's gradient canceled as described above.

## 4. Joint activity and action learning

Train recurrent actor–critic policies, initially PPO, from fresh rollouts. Let an operation last \(\Delta t_i\), use \(\gamma_i=\exp(-\beta\Delta t_i)\), and accumulate reward during the interval. Bootstrap from the state after that duration.

\[
\mathcal L=
\mathcal L_{\mathrm{policy}}
+\lambda_v\mathcal L_{\mathrm{value}}
+\lambda_p\mathcal L_{\mathrm{prediction}}
+\lambda_q\mathcal L_{\mathrm{operation}}
-\lambda_h\mathcal H(\pi).
\]

Choose coefficients on validation trajectories and record their gradient scales. The policy selects observation, retrieval, computation, action, learning, subgoal, and wait operations. Learning choices include an experience source, an adapter group, and a bounded update budget. Separate action and activity heads allow continued control while slower work runs. The fast action readout remains available between longer activities.

The distribution needs situations where sensing is useful, where further computation is useful, and where immediate action is best. Include irrelevant or noisy inputs so surprise alone does not determine the activity. Compare fixed budgets, simple benefit-per-time scheduling, and the learned policy with equal training and tuning budgets.

## 5. Learning to acquire and retain abilities

Use shared parameters \(\theta\) across training lifetimes and local adaptation state \(\eta^{(j)}\) for lifetime \(j\). A learning action selects a batch \(B_t\) from that lifetime's received experience and an update budget. A first implementation uses ordinary gradient updates on selected prediction/action adapters:

\[
\eta_{t+1}=\eta_t-\alpha_t\nabla_\eta
\mathcal L_{\mathrm{received}}(\theta,\eta_t;B_t).
\]

This is a starting update rule, not a novelty claim. Its targets are received observations, executed-action outcomes, and available task feedback. Model-generated rollouts and fixed-teacher distillation are separate auxiliary sources. Retain the source of each target in the training data.

The outer actor–critic training optimizes the shared context, refinement, prediction, action, and activity parameters across lifetimes that actually contain these inner updates. PPO estimates the return of choosing learning without requiring gradients through the optimizer. A differentiable inner update is an explicit later comparison, not an unstated property of PPO. The next transition includes the changed adapters, resulting predictions, pending operations, and elapsed time. When recomputing policy probabilities, restore or reconstruct the adapter state that existed at that decision; the final adapters of a lifetime cannot be substituted for earlier ones. Store the adapter/optimizer state at training-window boundaries and the intervening update data. The discrete activity selector uses policy-gradient return targets, while action and prediction losses also train the shared representation; this does not claim an exact gradient through all lifetime adaptation.

Compute value targets across task changes and learning effects. A backpropagation-window boundary uses the next decision state's value; only the specified end of a lifetime is terminal. Use duration-aware discounting and continuation values at horizons long enough to include later tasks. If an update only pays off after the training horizon, that training setup cannot teach its value.

For a subset of simulated histories, branch from the same external state, model state, experience, and optimizer state: learn now versus spend the interval on another operation. Apply the same continuation policy and compare subsequent utility over a common elapsed-time horizon. This can initialize or train operation-value estimates. The deployed selector uses its own received history, not simulator parameters or the branch comparison.

Experience selection initially chooses among recent, returning-task, and task-relevant retained records. Sample examples across all selected streams and records; [the corrected replay sampler](experiments/continual_agent/replay.py) addresses the prefix restriction in the existing operation experiment. A learned retrieval query or mixture may then optimize which experiences improve subsequent return.

The lifetime trainer implements the differentiable comparison (`--objective deployed`): within each training window it replays the learning steps the agent took, at the ticks it took them, each on the buffer it read then, and trains the shared parameters and the step size through every step by the loss of each tick's prediction under the adapter then in effect. The same replay restores the adapter at every decision, so recomputed policy probabilities read the adapter that was in effect. Precise readings, when they return, replace the noisy reading of the transition they measured, and likelihoods weigh each by its sensor's noise.

Compare acquisition with state-only adaptation, retrieved experience, fixed-schedule parameter updates, and learned update selection. Carry learned adapters across changing conditions and revisit earlier conditions without resetting them. Test transfer on held-out combinations. Separate reset experiments identify whether improvement resides in transient activations, retrieved records, or adapted parameters.

### Per-instance and collective learning

Parallel training can aggregate experience to update \(\theta\). In single-instance deployment evaluation, each agent's \(\eta\), optimizer, replay, and computation memory use only that agent's received interactions. Vectorizing these independent copies is valid; pooling their experience into one adaptive model answers a different question.

The current operation script uses fleet experience and shared parameter updates. Retain those results as collective adaptation. The integrated trainer must represent instance-specific adapters explicitly for the single-agent claim. A collaborative mode can later share experience and report the information and communication it receives.

During quiet intervals, choices include waiting, gathering useful observations, replay, and deeper-to-shallower distillation. Evaluate their contribution on subsequent tasks and their measured device time and energy. Repeated updates, a running process, or a falling replay loss are not substitutes for acquired capability.

## Sequence optimization and GPU execution

Use truncated backpropagation through time with numerical state carried between windows. Sweep unroll length and test longer dependencies. Reconstruct training state with a preceding burn-in segment when replay or parameter changes require it.

Keep workspace size, record capacity, and retrieval count fixed within a batch. Group sequences by length and operation count. BF16, activation checkpointing, compilation, and CUDA graphs are evaluated against an FP32, uncompiled reference for the relevant model and hardware.

Parallelize independent environments, sequence batches, seeds, and variants. For a model that fits on one GPU, use one process per device and [DistributedDataParallel](https://docs.pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html), explicitly sharding trajectories. Evaluate [FSDP](https://docs.pytorch.org/docs/stable/distributed.fsdp.fully_shard.html) when model and optimizer storage require sharding.

The current scientific-task core and its strongest matched reference are the first scale comparison. The integrated training objective and application are specified in [RESEARCH.md](RESEARCH.md#current-execution-priority). Complete the already running fixed shared-core study before interpreting its continuing-versus-separated state comparison. Its 102,400 physical requests and 25,600 recorded chemical returns per run measure different forms of exposure; do not combine them into a count of autonomous experimental decisions.

For the subsequent integrated campaign, measure rollout and optimizer throughput once, then fix the complete training budget, validation times and final comparison before launching the paired runs. Report physical and chemical environment decisions, primitive operations, unique trajectories, internal computation and GPU-hours separately. Use learning curves to choose data and duration extensions. A proposed transition count or GPU count is not a measured scaling result.

Vectorize fixed-size state, bounded retrieval and adapter updates. Increase the collection batch while measuring useful transitions per second; choose the backpropagation window from memory use and the delayed learning objective. Carry state across windows, bootstrap future return, and include losses after the next relevant task returns. A longer unroll is one way to carry gradients, not a substitute for future-use training examples.

On the two H100s, run the integrated model and its strongest matched reference with equal data and comparable tuning budgets. Feed GPU batches from CPU environment workers, and run native evaluation alongside training when it does not reduce useful training throughput. Keep the independently timed actor/environment experiment distinct from synthetic receipt delays. If additional A100s are available, assign independent seeds, feature production and reference runs before introducing mixed-device synchronous training. Increase model width in a controlled comparison after identifying whether the limiting factor is prediction, state organization, optimization or data coverage.

Measure batch-one observation-to-action latency separately from batched throughput. Include encoding, recurrence, memory lookup, record adaptation, refinement, action readout, and transfers. Report median and tail latency, parameter count, persistent-state bytes, peak memory, interactions per second, and GPU-hours. Simulated operation ticks and measured hardware time are separate quantities until a measured timing model connects them.

## Run output

For environment-training runs, record configuration, source commit, dependency and simulator versions, seeds, training curves, checkpoints, per-seed task scores, timing distributions, and evaluation trajectories. The current pilot writes configuration, versions, seeds, curves, and task scores; checkpoint saving is optional. Store large checkpoints and datasets externally with versioned download locations and hashes in GitHub. Use isolated output directories so new runs retain the earlier experiments.
