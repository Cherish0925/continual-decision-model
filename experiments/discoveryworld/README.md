# CDM on digital experiments

The next campaign trains a model to choose experiments, continue useful inference as observations arrive, and retain what it learns. It extends the existing GRU/Transformer core and its deployed-update training. The main architecture remains in [DESIGN.md](../../DESIGN.md).

Measured learned-policy results, the workspace intervention and the next learning comparison are in [RESULTS.md](RESULTS.md).

The current continuation trains [action-consequence prediction and workspace
feedback](PREDICTION.md) on reference and autonomous execution traces, including
failures. It directly tests whether additional internal computation improves
scientific action selection.

## Application and comparison settings

Use **Reactor Lab**, **Combinatorial Chemistry**, and **Plant Nutrients** as the initial application families. They provide instrument use and quantitative relations, multistep interventions, and delayed experimental outcomes. Expand to the remaining official discovery topics after the same model can learn and act across these families.

Two settings answer different questions:

| Setting | Purpose | Environment clock |
| --- | --- | --- |
| Official DiscoveryWorld | Scientific task completion, procedural progress, and discovery knowledge | One native action followed by one native world tick |
| Continuing process experiments, then a separately named digital extension | Decisions during computation, delayed returns, returning conditions and learning between requests | Environment advances independently of model inference |

Do not relabel the standard DiscoveryWorld run as real-time. Its API advances only when the caller invokes tick. Keep official scores comparable, and report any continuing-life or independently timed extension separately.

## Implemented task connection

[adapter.py](adapter.py) runs the actual DiscoveryWorld API. It returns the public UI, images, documented actions, teleport locations and observation times. An action produces a fresh observation after the world tick, a result, task scores, an observed-progress training reward, and separate terminal/time-limit flags. The acting policy receives public observations; detailed scorecards and hidden scientific rules are not its inputs.

[collect.py](collect.py) records these transitions from a local policy or an action file. PNG images are stored once per content hash. Each row retains the observation, action, returned result, next observation, task scores and timing needed by a trajectory learner. State after a time limit can be bootstrapped; it is not marked as a completed task.

The simulator is pinned to **fd591323920be0d3786ef350955de1945aa571e5**. Two integration tests run its real Reactor Lab scenario and cover world advancement, current observations, episode replacement and image persistence. No model weights or GPUs are needed for these tests.

The task path now connects these trajectories to the existing CDM core. [`cdm/digital.py`](../../cdm/digital.py) scores native actions over currently accessible or previously observed objects, retaining their exact handles only for execution. It reuses the GRU and persistent workspace; the process-control heads and released checkpoints remain intact. [`train.py`](train.py) learns actions and returned numerical contents from ordered trajectories. [`evaluate.py`](evaluate.py) executes the learned policy in the native simulator. [`finetune.py`](finetune.py) trains the same policy further from its own fresh interactions with a recurrent actor–critic objective.

The established task results use a fixed refinement count. The new
`train_joint_policy` / `adaptive_compute` path jointly trains continuing state,
action readout and a choice of 0, 1, 2 or 4 workspace refinements; its protocol
and execution comparison are in [TRAINING.md](../../TRAINING.md). Instance-owned
numerical updates are implemented below. Cross-task consolidation and learning
the value of computation from later scientific-task return remain unfinished.
The existing process-task results remain separate.

[`reference.py`](reference.py) is an observation-only, hand-written Reactor Lab reference for collecting initial experience. It measures crystals, fits competing measurement–frequency relationships and tests them through native actions. It has a task-specific affine hypothesis class; it is not the learned CDM and does not read the simulator's hidden rule. Returned radiation can include spatial background, so the current reference takes isolated repeated readings. The collector retains unsuccessful hypotheses and failed runs along with successful trajectories.

## Model work that directly enables training

| Existing component | Required integration |
| --- | --- |
| Frozen perception and semantic model | Encode actual task observations; preserve exact instrument readings, units and object references in experience records |
| GRU and persistent Transformer workspace | Retain state through each task; supply new observations between single refinement blocks |
| Process-control action head | Add a task head for operation type, observed-object arguments, direction/location arguments and dialog choices |
| Scalar process predictor | Keep the existing process branch; add prediction of received tool results, measured quantities or frozen observation features for digital tasks |
| Activity policy | Add semantic-call and retrieval choices, using task return and measured operation duration |
| Recent experience and instance adapters | Connect retained episodes to retrieval and local updates; train on later use, including returning conditions |

Use a shared operation/object representation or a factorized action head. Object UUIDs identify observed objects; their integer magnitudes have no physical meaning. The same decoder must support a changing set of visible objects. A fixed table of 81 process-control actions cannot express this interface.

The width-1024 released core can initialize compatible state modules. New task heads need training. Compare its transfer with a random-core initialization using the same dimensions and trajectories before attributing a benefit to the old process weights. Both remain the same architectural family.

Keep numeric observations available alongside compressed semantic context. Eight latent tokens are a bounded semantic connection; they are not a replacement for exact experimental records.

## Data and optimization

1. **Collect executable trajectories.** Use the local pretrained teacher and task-native agents on training instances. Retain successful and unsuccessful experiments with their received results. Use useful actions for imitation; use both successes and failures for outcome learning. Record teacher/model revision and collection cost.
2. **Train the actual action path.** Freeze pretrained perception and semantics initially. Train the new connections, task heads and persistent core on action imitation plus action-conditioned outcome prediction. Include immediate action and deeper refinement readouts. Evaluate by executing the resulting policy in the simulator.
3. **Train action and activity together.** Continue with recurrent actor–critic learning on fresh interactions. Optimize task return with measured observation/computation costs. Preserve state across training windows and reconstruct the local parameter state that existed at each collected decision.
4. **Train learning on its later use.** Replay the instance updates actually taken, on then-available observations, and optimize predictions/actions after those updates. Include situations that recur after intervening tasks. The future-loss and return horizon must reach those recurrences; increasing an unroll length alone does not fix a discount that removes their value.
5. **Train and measure quiet-period work.** Replay old and recent received experience, then distill useful prediction/action behavior. Initially keep state-producing shared parameters fixed during live local adaptation. Recompute state when a later consolidation phase changes those parameters.

Stages initialize one continuing model. Begin the first complete observation–action–outcome–gradient–new-action loop immediately, then improve it; do not require a collection of isolated modules to be perfected first.

Once a policy completes tasks, train conditional activity choices from that checkpoint.
Use matched continuations from the same model/environment state to compare acting
now, observing again, refining the workspace, and taking a local learning step.
Include both states where another observation helps and states where it is redundant.
Targets use subsequent task outcomes and operation costs; a shorter continuation
that fails the task is not automatically preferable. The existing lifetime
[trainer](../lifetime/train.py) and [runtime](../lifetime/runtime.py) already contain
activity-value learning and matched continuation rollouts. Adapt these mechanisms
to the native scientific-task actions rather than duplicating the learning code.

An initial activity-head fit can use a fixed successful core. Continue with joint
updates of the recurrent core, workspace, trainable connections and action/activity
heads so that the state can represent what those choices need. Keep the pretrained
perception frozen initially. Return to earlier conditions after intervening tasks,
retaining instance updates and experience, so the optimization includes their later
use. Task completion, activity selection and retained learning belong to the same
training run.

Split before generating training windows. Reserve official evaluation instances from training. Different random seeds do not necessarily define different scientific rules: Reactor Lab selects a relevant dimension with seed modulo 5, and Plant Nutrients Challenge selects a logical form with seed modulo 4. Record rule families and parameter combinations for dataset splitting, without exposing those labels to the policy. Report new-instance and held-out-rule performance separately. Public benchmark logs are not a training set for the same scored instances.

Repeated conditions also need distinct treatments. Repeating an identical seeded world measures exact-context return. Recovering a scientific relation with changed objects or appearance measures a stronger kind of retention and needs a separately defined generator. Use the existing process lifetimes for controlled returning dynamics while that extension is built.

## The research comparison

The central mechanism is training the model through the sequence it actually executes: observation, partial inference, action, returned outcome, retrieval and local learning. Three contributions matter:

- **Useful continuation:** after a new observation, continuing the existing workspace improves subsequent actions relative to restarting or leaving it unchanged at matched work.
- **Useful retention:** after intervening tasks, retained experience or learned parameters reduce the observations and interactions needed to recover performance.
- **Useful initiative:** learned observation/computation/learning choices improve task return relative to fixed schedules after including their time and resource costs.

Start with a directly acting pretrained model, a recurrent policy with the same frozen inputs and training data, and the integrated CDM. Use paired changes to memory, refinement and retained updates to locate each contribution. Include independently trained strong alternatives; interventions on one checkpoint are not a substitute for them.

Report task completion and procedural progress, recovery after change and return, and observation-to-applied-action latency with its tail. Report discovery-knowledge scores under the official procedure when that evaluator is available. An alternative evaluator is a separate result. Preserve failed runs and count collection, teacher inference and training in the compute budget.

## Execution on two H100s

Use the available two H100s as the baseline resource. Additional A100s increase independent collection and replication; their availability does not block this campaign.

- Initially allocate one H100 to teacher trajectories and frozen features, and the other to the continual core and task heads.
- Cache frozen features during replay; recompute trainable projections/readers. Batch by input length and refinement depth. Keep recurrent state and adaptation reductions in FP32 while using BF16 dense computation.
- Use CPU workers for independent worlds. The native API renders and writes multiple PNG images per observation, so measure simulation/rendering separately from model time. Do not copy the process simulator's 1,024-stream or 100-million-transition settings into this renderer without a throughput measurement.
- Once data and collection are ready, allocate both GPUs to the integrated model and its strongest comparison, then replicate the selected comparison over at least three training seeds. Larger parameter counts follow a task/data bottleneck study.

The first campaign result is an executed scientific task with learned actions and measured held-out behavior, followed by the matched continuation/learning comparison. Installation, test counts and an idle GPU are not that result.

## Run

The semantic policy also supports `--constrained-actions`: token generation
selects exactly one action from the same public candidate set as CDM. This
supplies native syntax, observed object handles and documented argument domains;
the model still chooses the action and the environment determines its outcome.
It does not use a reference procedure or task solution. This action-only mode
is separate from free-form thinking and the optional research notebook.

Training with `--measurement-memory --object-relations` connects each candidate
action to the records of its tools, targets and publicly observed contents.
Handles define equality relations, never numeric embeddings. The existing
observation features remain unchanged, and the added reader starts with zero
output so that a resumed model initially reproduces its earlier predictions.
The option is saved with the checkpoint and inherited by native evaluation.

Install the pinned simulator in the research environment:

~~~bash
python -m pip install "discoveryworld @ git+https://github.com/allenai/discoveryworld.git@fd591323920be0d3786ef350955de1945aa571e5"
python -m unittest discover -s tests -p 'test_discoveryworld_adapter.py' -v
~~~

For multimodal training, use the repository's extras in a project environment. They pin Transformers 5.4.0 and the simulator; PyTorch and Torchvision must be compatible. Reuse existing model weights and released CDM checkpoints where applicable.

~~~bash
python -m pip install -e '.[multimodal,discoveryworld]'
~~~

Replay the included native measurement sequence to collect observations and returned outcomes:

~~~bash
python -m experiments.discoveryworld.collect --scenario "Reactor Lab" \
  --difficulty Normal --seed 100 --max-steps 4 \
  --actions experiments/discoveryworld/fixtures/reactor_measurement_seed100.jsonl \
  --output runs/discoveryworld/replay-100
~~~

The four actions use object references from the public observation of this specific instance. They acquire a thermometer and measure a crystal. On the pinned simulator they returned 32.22 degrees Celsius. This exercises a real tool/measurement transition and image recording; it is a scripted connection example and does not complete the scientific task.

Each input line is a native action dictionary, for example:

~~~json
{"action": "ROTATE_DIRECTION", "arg1": "north"}
~~~

For a live local policy, supply its factory as module:factory and optional JSON configuration:

~~~bash
python -m experiments.discoveryworld.collect --scenario "Reactor Lab" \
  --difficulty Normal --seed 100 --policy your_local_policy:create \
  --policy-config path/to/model.json --output runs/discoveryworld/policy-100
~~~

The factory receives that configuration and returns a callable. The callable receives the public observation plus action descriptions and teleport locations, and returns a native action dictionary or dialog choice. Its recurrent state remains in that callable across decisions. Both reference policies and trained CDM policies use this interface.

Read trajectory.jsonl for transitions, images/ for assets, run.json for the episode configuration, and summary.json for task results. Use a new output directory per run.

## Frozen semantic-model baseline

[`semantic_policy.py`](semantic_policy.py) asks the local frozen
[`SemanticExpert`](../../cdm/semantic.py) to generate one native JSON action per
decision. It uses Qwen3.5-2B's [official processor and generation interface](https://huggingface.co/Qwen/Qwen3.5-2B),
with thinking disabled and greedy decoding by default. Inputs are the current
public UI/image, documented actions, previously observed object handles, actual
action results and exact measurement records, including units, conditions and
contained-object bindings. No reference controller selects or repairs actions.
Measurement rows share column names, identical metadata and condition indices;
every repeated sample keeps its exact value and source event index. Condition views keep
location, visible object handles/distances and historical descriptions of actual
measurement participants; the disk ledger retains all original descriptions and
returned text.

~~~bash
python -m experiments.discoveryworld.semantic_policy \
  --model-path data/cache/qwen3.5-2b --device cuda:0 --dtype bfloat16 \
  --scenario "Reactor Lab" --difficulty Normal --seeds 110 \
  --max-steps 150 --max-input-tokens 32768 --max-new-tokens 128 \
  --temperature 0 --output runs/discoveryworld/semantic-baseline
~~~

Weights and processor files must already be local; loading performs no download.
Use `--thinking --max-new-tokens 2048` to enable the model's native thinking
mode; `--no-thinking` remains the default. The trace retains all generated tokens
and reasoning text. Thinking-mode actions are parsed only after `</think>`;
unfinished reasoning and generation-length limits are recorded explicitly, with
no replacement action or retry.
`--top-p` and `--top-k` pass sampling controls directly to native generation;
omitting them preserves the model's existing defaults. For example, a thinking
pilot can explicitly set `--temperature .6 --top-p .95 --top-k 20` alongside
its thinking and output-length settings. All settings are recorded per run.
`--research-notebook` adds `memory`, `running_hypotheses` and `explanation` to the
same generated action JSON, following the notebook pattern in the pinned
[official Hypothesizer agent](https://github.com/allenai/discoveryworld/blob/fd591323920be0d3786ef350955de1945aa571e5/agents/HypothesizerAgent.py#L356).
After that action receives real feedback, these model-written claims enter the
next input as `model_authored_notes`, separately from measured evidence. Updates
are saved in `decisions.notes.jsonl`; an unexecuted proposal or replayed teacher
annotation cannot update this notebook. The default remains off. This uses one
generation per decision, with no extra knowledge-update call or automatic
correction; allow enough output tokens for both reasoning and notes.
`--no-vision` selects a text-only ablation. `--history-steps` bounds recent action
history. If the token budget is exceeded, whole oldest entries are omitted in
order: action history, measurement summaries, remembered objects, then raw
measurements. The current UI/action schema stays complete; the full measurement
ledger remains on disk. Each decision records its exact message text, image
references, token IDs, omitted counts, raw output and timing in `decisions.jsonl`.
Native action errors are returned as feedback. Invalid JSON stops the episode
and preserves the failed generation and executed prefix without retrying.
The final response may be plain JSON or one complete Markdown JSON code block,
with optional surrounding whitespace. Mixed prose, multiple blocks and invalid
JSON are rejected; raw output is retained without repairing the action.

The policy also supports `collect --policy experiments.discoveryworld.semantic_policy:create`;
its JSON config uses the same argument names with underscores, plus `trace_path`
to save generation records. Both paths call `observe` only after real feedback.
This is a directly acting frozen-model baseline; it does not jointly train CDM
or establish that the model has learned the task's scientific relation.

## Replay an externally supplied prefix

For exact public-prefix takeover, collect a new reference trajectory with its
initial native RNG snapshot. The upstream scenario seed does not seed every
object's `random.Random`; radiation readings therefore differ across resets.

```bash
python -m experiments.discoveryworld.reference --seeds 310,311,312,313 \
  --capture-initial-rng --max-steps 500 --workers 2 \
  --output runs/discoveryworld/reference_with_rng

python -m experiments.discoveryworld.prefix_evaluate \
  --source runs/discoveryworld/reference_with_rng/validation/seed310 \
  --prefix-steps 28 --takeover-steps 100 \
  --policy experiments.discoveryworld.evaluate:create \
  --policy-config path/to/model.json --output runs/discoveryworld/prefix-310
```

`initial_rng.json.gz` is a separate evaluator artifact: native world, API,
UUID-generator, Python-global and object RNG states (including contained objects
and parts, keyed by UUID). Its relative path and hash live in `run.json`.
The evaluator restores it immediately after a matching reset, before replay;
it never enters policy observations, features or training records. This neither
reseeds measurement noise nor changes scientific rules. It is not a complete
world snapshot. Public UI and action-result comparisons remain exact, including
every received reading. Legacy sources without a snapshot are explicitly marked
as uncontrolled RNG replay; existing readings cannot reconstruct the missing
initial state. Any mismatch stops takeover. Prefix actions remain externally
supplied, and conditional completion is reported separately from end-to-end
success. The example's 28 actions demonstrate noisy-prefix replay, not a claim
that calibration is complete.

## Train and execute a learned policy

Choose completed trajectory directories before creating training windows. Keep collection procedure, rule split and preprocessing consistent across training and validation. Different seeds alone do not establish a held-out scientific rule. The native world tick is the model's time input; rendering and inference time remain in the logs for resource measurement.

~~~bash
python -m experiments.discoveryworld.train \
  --train-trajectories runs/discoveryworld/train \
  --validation-trajectories runs/discoveryworld/validation \
  --features siglip --feature-cache runs/discoveryworld/features \
  --feature-device cuda:0 --device cuda:0 \
  --initial-checkpoint path/to/released/model.pt \
  --output runs/discoveryworld/cdm --epochs 60

python -m experiments.discoveryworld.evaluate \
  --checkpoint runs/discoveryworld/cdm/best.pt \
  --feature-cache runs/discoveryworld/features --device cuda:0 \
  --seeds 300 301 --max-steps 250 --output runs/discoveryworld/evaluation

python -m experiments.discoveryworld.finetune \
  --checkpoint runs/discoveryworld/cdm/best.pt \
  --feature-cache runs/discoveryworld/features --device cuda:0 \
  --seeds 100 101 102 105 --episodes 12 --max-steps 250 \
  --output runs/discoveryworld/on-policy
~~~

`--encoder-path` can point to a local copy of the pinned SigLIP model. Frozen features are shared by the comparisons. All text chunks are encoded, including measurements late in an observation. `--features lexical` is an explicit text/numeric, image-masked ablation. `--mode gru` trains a recurrent comparison with the same candidate actions; evaluation supports `--restart-workspace` independently of `--restart-state`. Offline action agreement is reported separately from executed task completion.

`--measurement-memory` adds public measurement history to the policy input.
[`measurements.py`](measurements.py) retains returned values and source text with
their tool, object, unit and observed conditions. Its summaries supply counts,
minima, maxima, means and recent values through the existing `core.record` and
text projections into the GRU and workspace. Every decision sees only feedback
already received from actual actions; expert suggestions add no measurements.

Enable this input in `train`, including when resuming as below. Omitting the flag
inherits the checkpoint setting; `--no-measurement-memory` disables it.
`evaluate` and `finetune` inherit the saved setting automatically. Compatible
checkpoint weights keep their shapes; the record reader requires eight numeric
columns. Checkpoints without this setting remain off. Online updates of the
original per-instance A/B adapters remain unimplemented in this task path;
the trainers update shared policy parameters.

## Learn from actual departures and corrections

[`recovery.py`](recovery.py) maintains a public experiment record: observed objects,
actual measurements, calibration readings and interventions. Recommendations do
not advance that record; only executed actions and received results do.
[`dagger.py`](dagger.py) queries both the learner and this scripted expert before
executing either suggestion. It records `expert_action` as the imitation target
and `action` as the operation whose outcome was actually observed. A completed
mixed-control trajectory is training data, not independent learner success.

```bash
python -m experiments.discoveryworld.dagger \
  --checkpoint runs/discoveryworld/cdm/best.pt \
  --feature-cache runs/discoveryworld/features --device cuda:0 \
  --seed 110 --max-steps 400 --student-prefix-steps 16 \
  --teacher-probability .8 --output runs/discoveryworld/corrections/train/seed110

python -m experiments.discoveryworld.train \
  --resume-checkpoint runs/discoveryworld/cdm/best.pt \
  --train-trajectories runs/discoveryworld/corrections/train \
  --validation-trajectories runs/discoveryworld/corrections/validation \
  --feature-cache runs/discoveryworld/features --device cuda:0 \
  --feature-device cuda:0 --epochs 12 --measurement-memory \
  --output runs/discoveryworld/recovery-trained
```

Collect separate validation instances before the training command. Resume keeps
the model and feature configuration and initializes a new optimizer. The expert
still uses the stated affine hypothesis family; its measurements and corrections
are explicit supervision. The learned policy uses no expert at evaluation.

`--imitation-reduction action_run` gives each contiguous run of one expert action
equal total imitation weight, normalized over its complete episode. It retains
every observation and every outcome loss. The default `event` keeps equal
per-event imitation. Logs report raw action accuracy, action-run-weighted accuracy
and accuracy at the first action after a repeated run; native task execution is
still the outcome comparison. No sampling count or task-specific action rule is
encoded by this loss.

## Train numerical learning from experimental evidence

`calibration.py` joins public instrument readings to observed working settings.
A label requires an activated reactor, its observed crystal and an actual
frequency return. Uncalibrated settings and repeated restorations do not create
new calibration examples. Each prediction query is frozen before that crystal's
first frequency trial; its later observed working setting supplies the target.

The initial numerical learner uses the existing core predictor and instance `A/B`
adapters. Frozen perception and continuing `h/z` contexts are prepared once.
The outer loss differentiates through updates on earlier evidence to predict
later working settings. A training-only frequency prior initializes the new
quantity readout; the source observation and action paths remain fixed.

```bash
python -m experiments.discoveryworld.train_calibration \
  --source-checkpoint runs/discoveryworld/measurement_memory/action_run/last.pt \
  --train-trajectories runs/discoveryworld/reference/train \
    runs/discoveryworld/reference/campaign2/train \
    runs/discoveryworld/reference/calibration_training/train \
  --validation-trajectories runs/discoveryworld/reference/validation \
    runs/discoveryworld/reference/campaign2/validation \
  --heldout-trajectories runs/discoveryworld/reference/heldout_rule \
    runs/discoveryworld/reference/campaign2/heldout_rule \
  --feature-cache runs/discoveryworld/features \
  --context-cache runs/discoveryworld/calibration/contexts \
  --encoder-path data/cache/siglip2 --device cuda:0 \
  --epochs 10 --seed 31 --output runs/discoveryworld/calibration/seed31
```

`--prepare-only --prepare-trajectories ...` prepares disjoint context shards on
separate devices. The numerical comparison retains updates or computes and
discards them, using identical observations, continuing contexts and initial
adapters. Results separate queries after two and three confirmed calibrations.

For native execution, `quantitative_policy:create` loads the calibration
checkpoint through a policy JSON with `checkpoint`, `feature_cache`,
`encoder_path`, `device` and `retain_updates`. It uses the learned frequency
prediction with an **explicit scalar controller** over public frequency-change
options. Other actions come from the existing policy. This controller is a
declared execution primitive, not a learned scientific law or a trained action
head. `prefix_evaluate` records the supplied experiment, real subsequent
actions, new crystal activations, predictions and each instance update. Native
activation and prediction error are separate outcomes.

### Learned priors over named numerical relations

The current numerical path keeps nine explicit single-quantity affine hypotheses
and fits their posterior from the instance's received calibrations. The outer
training loss predicts later queries through that update. `--prior-mode shared`
learns five coefficient/noise parameters shared across quantities with uniform
candidate weights; `typed` learns separate priors and weights (54 parameters).
Both use the source action policy unchanged. The affine basis and native scalar
controller are explicit; [RESULTS.md](RESULTS.md) distinguishes their contribution
from that of the learned policy.

```bash
python -m experiments.discoveryworld.train_calibration_mixture \
  --source-checkpoint runs/discoveryworld/measurement_memory/action_run/last.pt \
  --train-trajectories runs/discoveryworld/reference/train \
    runs/discoveryworld/reference/campaign2/train \
    runs/discoveryworld/reference/calibration_training/train \
  --validation-trajectories runs/discoveryworld/reference/validation \
    runs/discoveryworld/reference/campaign2/validation \
  --heldout-trajectories runs/discoveryworld/reference/heldout_rule \
    runs/discoveryworld/reference/campaign2/heldout_rule \
  --prior-mode shared --epochs 100 --lr 0.001 --batch-size 16 --seed 31 \
  --output runs/discoveryworld/calibration/mixture/shared_seed31

python -m experiments.discoveryworld.evaluate_calibration \
  --prediction-type mixture --local-probes \
  --checkpoint runs/discoveryworld/calibration/mixture/shared_seed31/last.pt \
  --prefixes runs/discoveryworld/calibration/transfer_check_prefixes.json \
  --feature-cache runs/discoveryworld/features --encoder-path data/cache/siglip2 \
  --device cuda:0 --workers 12 --steps 48 \
  --output runs/discoveryworld/calibration/native_transfer_shared_seed31_probes
```

Use a new output directory when repeating a run. Mixture fitting itself is small
and runs on CPU by default; native execution uses the GPU for perception and the
continuing policy. The evaluator pairs retained and computed-but-discarded
updates, restores the supplied public prefix, and executes actual subsequent
actions. `--local-probes` enables feedback-driven scalar trials inside the same
48-action allowance. Omitting it executes point predictions. `reference` selects
the train-only frozen affine reference; `core` selects the original A/B bridge.

The fresh transfer set was collected after training choices were fixed. Its
prefix manifest and raw observations/checkpoints are available in the
[scientific-task release](RELEASE.md), under relative `runs/` paths. The compact
[native results](results/calibration_native_comparison.csv) and
[numerical results](results/calibration_transfer.csv) are tracked with the code.
`summarize_calibration.py` rebuilds these tables from native run directories and
the frozen-model transfer comparison. The release guide explains extraction and reconstruction of excluded feature caches.

## Reading experimental evidence to choose actions

`belief_actions.py` learns action ranking from the continuing core, candidate
text and exact public experimental facts. It changes no candidate action set.
Numerical hypothesis fitting and scalar execution retain their previous roles.
The [results](RESULTS.md#autonomous-experiments-from-reset--27-october) report
native completion from reset, matched evidence-input controls and state
interventions. All training uses the fixed final checkpoint.

Prepare each trajectory partition once, then reuse it across readers:

~~~bash
python -m experiments.discoveryworld.train_belief_actions --prepare \
  --trajectories runs/discoveryworld/reference/train \
    runs/discoveryworld/reference/campaign2/train \
    runs/discoveryworld/reference/calibration_training/train \
  --mixture-checkpoint runs/discoveryworld/calibration/mixture/shared_seed31/last.pt \
  --prepared runs/discoveryworld/belief_actions/prepared/train --device cuda:0

python -m experiments.discoveryworld.train_belief_actions \
  --mixture-checkpoint runs/discoveryworld/calibration/mixture/shared_seed31/last.pt \
  --prepared runs/discoveryworld/belief_actions/prepared/train \
  --validation runs/discoveryworld/belief_actions/prepared/validation \
  --output runs/discoveryworld/belief_actions/full_seed61 \
  --belief-mode full --seed 61 --epochs 30 --batch-size 32 --device cuda:0

python -m experiments.discoveryworld.evaluate --policy-kind belief --local-probes \
  --checkpoint runs/discoveryworld/belief_actions/full_seed61/last.pt \
  --seeds 610 611 612 613 634 639 644 649 --max-steps 400 \
  --device cuda:0 --workers 8 --output runs/discoveryworld/belief_actions/native_full_seed61
~~~

Prepare validation from `runs/discoveryworld/reference/validation` and
`runs/discoveryworld/reference/campaign2/validation`, using the same preparation
command and `--prepared runs/discoveryworld/belief_actions/prepared/validation`.
`--part`/`--parts` shard preparation without changing an episode's continuing
state. `--belief-mode none` trains the matched zero-evidence reader;
`observations` retains received facts but masks derived posterior predictions.
`--initial-rng-root` restores per-instance native random streams saved by another
from-reset evaluation. It does not supply actions, measurements or model state.
`--restart-workspace` intervenes on workspace persistence independently of
measurement records and recurrent state. `analyze_rollouts` produces detailed
behaviour records and a compact per-instance CSV from the actual traces.

Actual-return training uses the same reader and tools:

~~~bash
python -m experiments.discoveryworld.train_active_belief_actions --collect \
  --checkpoint runs/discoveryworld/belief_actions/full_seed61/last.pt \
  --seed 710 --device cuda:0 \
  --output runs/discoveryworld/belief_actions/active_round1/collect/seed710

python -m experiments.discoveryworld.train_active_belief_actions \
  --checkpoint runs/discoveryworld/belief_actions/full_seed61/last.pt \
  --rollouts runs/discoveryworld/belief_actions/active_round1/collect \
  --device cuda:1 --output runs/discoveryworld/belief_actions/active_round1/update
~~~

Each collection process owns its environment and output; independent instances
can run concurrently across devices. The updater uses only completed native
rollouts from its exact behaviour checkpoint. Task progress and all native
action costs train the reader; explicit scalar overrides are excluded from its
policy gradient. The source core stays frozen, so replayed reader inputs retain
the state used when the action was taken. This is action-policy learning, not a
claim that the numerical primitive or internal computation schedule is learned.

Paired continuation training uses an actual action intervention at a replayed
training state. `prefix_evaluate --first-action` overrides only the first action
after the prefix; subsequent actions remain model choices. It logs the executed
action separately from the proposal, and gives the model only actual returned
feedback. `measurement_branches/comparison.json` lists the source trajectory,
prefix length and two actions. Completed branch returns train the reader:

~~~bash
python -m experiments.discoveryworld.train_branch_actions \
  --checkpoint runs/discoveryworld/belief_actions/full_seed61/last.pt \
  --branches runs/discoveryworld/belief_actions/measurement_branches \
  --prepared runs/discoveryworld/belief_actions/prepared/train \
  --validation runs/discoveryworld/belief_actions/prepared/validation \
  --branch-weight 0.25 --device cuda:0 \
  --output runs/discoveryworld/belief_actions/branch_learning/preference
~~~

Use `--branch-weight 0` with a separate output for the matched replay control.
Both use one additional epoch with the same replay order and learning rate.
Only training-state pairs supply preferences; native evaluation starts from
reset and receives neither action interventions nor future returns. The
[paired results](RESULTS.md#learning-when-to-end-a-measurement-run) distinguish
the action-intervention benefit from the trained model's later behaviour.

`--mixture-probes` is an explicit numerical-controller comparison. It uses the
full predicted mixture and received inactive settings, with a known 2 Hz task
tolerance, to choose reachable experiments by predicted success per native
action. It keeps the original sixteen-extra-adjustment allowance. The
[completed comparison](RESULTS.md#testing-inactive-feedback-across-numerical-hypotheses)
did not improve completion, so `--local-probes` remains the reference. It is not
a learned policy update or a substitute for training scientific decisions.

## Observation throughput

[observation.py](observation.py) uses the native world and UI renderers while avoiding
the two debug-frame writes per observation. Its PNG mode preserves both image keys
and decoded pixels, with lighter lossless PNG compression. The image transport is
larger; this mode is intended for local collection. RGB mode returns HWC uint8
arrays directly to a local image processor and skips PNG/base64 encoding.

PNG is the default. Use `--observation-mode native` for the original reader, or construct
`DiscoveryWorldEnv(output, observation_mode="png")`. Existing native observation
behavior remains available. For direct model integration:

~~~python
from experiments.discoveryworld.observation import get_observation

observation = get_observation(api)            # native-compatible PNG data URLs
observation = get_observation(api, rgb=True)  # arrays in vision["rgb_no_grid"] and vision["rgb_with_grid"]
~~~

Call only one reader per environment decision: reading the native UI consumes a
queued tool message. Save trajectory images separately when collecting training
data. In RGB mode, send the array directly to the existing image processor; the
PNG trajectory writer expects PNG mode.

In the contributor CPU benchmark, 24 observation reads per mode in Reactor Lab
Normal seed 100 gave the following medians. These timings exclude world ticks,
model inference, training, and trajectory serialization.

| Reader | Time per observation | Relative speed |
| --- | ---: | ---: |
| Native API | 206.8 ms | 1.00× |
| In-memory PNG | 66.8 ms | 3.10× |
| In-memory RGB | 58.6 ms | 3.53× |

[The recorded run](benchmarks/observation_cpu_20261027.json) also includes payload
sizes. RGB arrays use more transport memory and should stay inside the local
inference process or use shared memory when collected by separate workers.
The native-simulator comparison checks both image views pixel for pixel, public
UI, action results and step counters through the four-action instrument sequence.
This is an observation-reader speedup; whole-training throughput depends on the
remaining simulator, inference and training time.

~~~bash
python -m unittest discover -s tests -p 'test_discoveryworld_observation.py' -v
python -m experiments.discoveryworld.benchmark_observation --steps 24 \
  --output runs/discoveryworld/observation-performance.json
~~~

## Sources and continuation

- [Official project and evaluation definitions](https://allenai.github.io/discoveryworld/)
- [Pinned API](https://github.com/allenai/discoveryworld/blob/fd591323920be0d3786ef350955de1945aa571e5/discoveryworld/DiscoveryWorldAPI.py)
- [Reactor Lab generator](https://github.com/allenai/discoveryworld/blob/fd591323920be0d3786ef350955de1945aa571e5/discoveryworld/scenarios/reactor_lab.py)
- [Plant Nutrients Challenge generator](https://github.com/allenai/discoveryworld/blob/fd591323920be0d3786ef350955de1945aa571e5/discoveryworld/scenarios/plant_growing_challenge.py)
- [Previous CDM results](../lifetime/RESULTS.md#second-round-training-through-deployment)

