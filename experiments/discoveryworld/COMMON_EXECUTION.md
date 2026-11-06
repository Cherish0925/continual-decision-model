# Scientific decisions with shared execution

The current study compares a language planner, CDM and a GRU planner through
the same object-request interface. Each planner chooses the operation,
instrument, sample, recipe component and control-menu answer. A deterministic
executor reaches and acquires the declared objects. Every underlying operation
consumes the native action budget and remains in the recorded trajectory.

**Status, 6 November 2026:** all 32 student runs and their fixed-endpoint
evaluations are complete. Demonstration-initialized interaction learning gives
CDM 6/160 and GRU 2/160 greedy completions. Resetting CDM's neural state gives
8/160. Reliable task execution and a completion benefit from persistent state
have not emerged. The [dated run snapshot](results/common_execution_status.json)
separates completed student comparisons from teacher attempts, some of which
ended with provider errors.

## Execution and learning

[The executor](request_executor.py) preserves all requested arguments. It does
not substitute instruments, select experiments, infer a response law or choose
a control setting. Unachievable requests return their actual failure. Each
request can consume at most twelve native operations. The shared registry
contains previously visible objects; scientific hypotheses and neural state
belong to the planners.

Inputs are public text, structured observations, measurements and operation
feedback. The native `TALK` documentation is clarified for every planner: it
opens an object's dialogue or control menu, including interactive devices. No
device identity or correct menu choice is supplied.

[Learned policies](request_policy.py) select an operation, then its arguments,
avoiding a quadratic list of object pairs. The core assimilates the external
observation **once per complete request**. Successive argument scores read the
same continuing state.

Supervised learning imitates fulfilled requests, excluding failed terminal
answers. Failed operations and unsuccessful episodes remain outcome targets and
history. Autonomous updates use the complete request's joint log probability.
The objective rewards completion and charges actual native operations. Progress
provides potential-based shaping, with zero potential at every episode ending,
including failure and exhausted budget. A request taking `k` operations
discounts subsequent value by `gamma ** k`. Partial progress therefore cannot
leave a terminal bonus for an incorrect final answer. Raw simulator rewards
remain in the records alongside the optimization objective.

## Fixed comparison

The [configuration](common_execution.json) fixes the study before evaluation:

| Item | Setting |
| --- | --- |
| Themes | Chemistry, archaeology, plant nutrients, reactor, proteomics |
| Demonstration and evaluation difficulties | Easy and Normal |
| Demonstrations | 80 training instances; 20 separate development instances |
| Learned arms | CDM and GRU, each with and without demonstrations |
| Initializations | Eight per arm; 32 runs |
| Core | Width 512, four attention heads, eight workspace slots |
| Demonstration stage | Four epochs on the first Easy records, eight on the full collection, then autonomous interaction |
| Autonomous stage | The same 120 fresh training instances per run: Easy, Normal and Challenge |
| Evaluation | The same 20 fresh instances per fixed endpoint |
| Native budget | 100 operations for Easy; 300 for Normal and Challenge |

Each demonstration arm loads its paired self-learning arm's **initial** weights,
before interaction updates. CDM and GRU share experience, initialization seeds
and native interaction limits. Their computation differs; equal interaction
budgets do not establish equal FLOPs.

The initial course uses all forty training and ten development Easy trajectories,
including recorded unsuccessful or interrupted prefixes. Selection is by
difficulty, not success. This fixed early snapshot permits training while Normal
collection continues. The complete data subsequently includes resumed prefixes.

The full demonstration stage uses a fixed snapshot containing 5,542 requests
from eighty training instances and 1,420 requests from twenty development
instances. It retains five training and three development trajectories whose
teacher calls stopped early. Their executed operations and returned observations
remain valid training examples. Later teacher continuations do not change this
snapshot or any model's training data.

Data delivery initially failed while restarting a removed temporary service,
leaving the demonstration jobs waiting. Collection and delivery are now
independent: the same snapshot was delivered to all four training nodes before
the common ready marker was created on each node.

The submitted interaction command selected all three difficulties. Its
120-instance schedule is retained for every arm. The earlier description of
this stage as eighty instances was incorrect; difficulty selection is now
explicit in the configuration and command.

## Completed comparisons

The sixteen self-learning runs completed 1,920 training episodes, 446,727 native
operations and 8,075 optimizer updates. Seven training episodes completed their
scientific task. Every training took 47.6–51.0 minutes; the sixteen runs were
parallel. All fixed-weight evaluations are complete:

| Model without demonstrations | Greedy | Sampled | Greedy, state reset |
| --- | ---: | ---: | ---: |
| CDM | 0/160 | 0/160 | 0/160 |
| GRU | 0/160 | 0/160 | 1/160 |

Each column repeats twenty fresh Easy/Normal instances across eight
initializations. This autonomous curriculum has not produced reliable task
completion within its fixed interaction budget.

All sixteen demonstration models subsequently completed eight epochs on the
fixed full collection. Independent greedy execution gives CDM 0/160 and GRU
0/160 on the same twenty evaluation instances. CDM's mean task progress is
0.0617; GRU's is 0.0371. Task completion remains absent at this supervised
endpoint.

The subsequent interaction stage completed another 1,920 training episodes,
421,667 native actions and 8,595 optimizer updates. Seventy-four training
episodes completed their task. Training took 36.4–47.3 minutes per parallel run.
Its fixed endpoints give:

| Model initialized with demonstrations | Greedy | Sampled | Greedy, state reset |
| --- | ---: | ---: | ---: |
| CDM | 6/160 | 3/160 | 8/160 |
| GRU | 2/160 | 0/160 | 2/160 |

CDM's six greedy successes are repeated executions of three Easy instances:
one plant-nutrient instance and two archaeology instances. No Normal instance
is solved. CDM mean task progress rises from 0.0617 at the supervised endpoint
to 0.2357 after interaction learning; GRU rises from 0.0371 to 0.1583. This
progress measure is not a task completion rate. Neither the low completion
counts nor the state-reset comparison establish a persistent-state advantage.
The fixed curriculum is complete; another unchanged sweep is not supported by
these results.

All sixteen demonstration models finished four epochs on the same 678 training requests.
Each then executed the same ten Easy development instances with two decoding
methods. Each table entry aggregates eight initializations; it is eighty
executions of ten worlds.

| Model | Greedy completion | Sampled completion |
| --- | ---: | ---: |
| CDM | 0/80 | 7/80 |
| GRU | 0/80 | 3/80 |

These starting policies did not execute the tasks reliably. Their initial
weights were fixed, and these development outcomes did not select a model
or replace the planned final endpoints. See the [per-instance results](results/common_execution.json).

Final interaction policies use both greedy and sampled decoding. Resetting
neural state between requests is a same-weight intervention. Demonstration arms
also evaluate the fixed final supervised checkpoint. These repeated executions
do not increase the count of unique tasks. Language-planner inference usage is
recorded separately from student inference and shared execution cost.

The supervised endpoint is evaluated as soon as its fixed checkpoint is saved,
in parallel with interaction training on separate devices. Two seed-401
endpoints also execute the same instances on H100 to measure the complete
runtime path. These hardware repetitions do not add model initializations or
test instances to the scientific comparison.

The completed H100 repetitions took 271.6 seconds for CDM and 253.3 seconds for
GRU across twenty instances each, both completing 0/20. Environment execution
accounted for 138.2 and 143.2 seconds; the decision path, including input
processing, accounted for 115.6 and 94.3 seconds. These measurements explain
why GPU capacity alone does not determine end-to-end throughput. See the
[runtime records](results/common_execution_runtime.json) for the corresponding
A100 repetitions and conditions. Concurrent host work differs, so these are
not controlled hardware speed comparisons.

Instance identities are disjoint; scientific response laws may repeat across
seeds. Unseen-law transfer, returning-rule memory, real-time operation, adaptive
computation and recursive self-improvement require their own experiments in the
[continual model program](../../research/CONTINUAL_AGENT.md). This five-theme
study also leaves the official eight-theme native benchmark as a separate goal.

## Learning the request representation

The completed demonstration models also fit their training requests poorly.
Across eight initializations, last-epoch complete-request imitation accuracy averages
17.47% for CDM and 14.83% for GRU; development accuracy averages 14.40% and
11.43%. Freezing all sixteen supervised checkpoints and replaying the recorded
states gives the following [complete measurements](results/common_execution_fit.json):

| Frozen policy | Training request agreement | Development request agreement |
| --- | ---: | ---: |
| CDM, eight initializations | 15.90% | 14.28% |
| GRU, eight initializations | 12.95% | 11.44% |

All request fields must agree with the demonstration; argument scores receive
the demonstrated preceding fields. Each training score uses 5,296 labeled
requests among 5,542 observed requests. The earlier trajectories remain the
inputs, so the low training fit precedes autonomous rollout distribution shift.
Agreement with one demonstration does not identify every valid alternative.
[imitation_fit.py](imitation_fit.py) reproduces this measurement without changing
weights. These metrics are distinct from independent task completion. The
observations already have separate lexical vectors per text line and explicit
measurement-to-object links. The new comparison tests how these observations
are represented and learned.

The running comparison changes the learned text representation while keeping the
same frozen 80-world demonstration collection and shared executor:

| Representation | Initialization | Text parameters during training |
| --- | --- | --- |
| Lexical control | Existing signed word and word-pair features | Fixed features |
| Frozen text | Pretrained MiniLM-L6 | Frozen |
| Adapted text | The same pretrained MiniLM-L6 | Jointly updated |
| Random text | The same MiniLM-L6 architecture | Jointly updated from random weights |

The [text encoder](../../cdm/text_encoder.py) reads complete tokenized text,
splitting long lines into windows without discarding later feedback. Candidate
operations and arguments use the same encoder. Decision and outcome gradients
reach the adaptable encoder, the continuous core and the action readout.
The pretrained source is
[all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2).
Its weights are retained inside each trained policy checkpoint. This is a
pretrained language representation, not a language planner used at evaluation.

Each representation trains CDM and GRU with four independent seeds (501–504):
32 runs. All use width 512, candidate-specific reading, the same measurement
records and eight recent action-feedback records. Thirty-two supervised epochs
are followed by the same 120-world interaction schedule as the preceding study.
The core/action learning rate is 0.0003 during imitation; adaptable text uses
0.00002. Interaction learning uses 0.00003 and 0.00001 respectively. This
comparison matches experience and update schedules, not parameter count or
FLOPs. Candidate-specific reading and the longer training schedule are present
in the new lexical control too.

Development worlds remain 6300–6301. Fixed final endpoints execute new instance
seeds 6800–6801 in all five themes at Easy and Normal difficulty, with the same
100/300 native-operation budgets and a common object-initialization seed.
Each cell therefore contains four executions of twenty worlds. The final
interaction policy is evaluated greedily, by sampling, and with neural state
reset between requests. No test score selects a checkpoint. New instance seeds
do not guarantee new scientific laws.

[The study driver](language_study.py) runs training, native execution,
interaction learning and final comparisons in sequence:

```sh
python -m experiments.discoveryworld.language_study \
  --root runs/semantic_requests --condition adapted_text --mode cdm --seed 501 \
  --encoder-path models/all-MiniLM-L6-v2
```

The run root contains the fixed `train/` and `development/` trajectory trees.
The other conditions are `lexical`, `frozen_text` and `random_text`; `--mode gru`
uses the same request interface and training data. This study evaluates whether
better learned input representations enable the existing decision process. It
does not yet test deployment-time parameter adaptation or recursive improvement.

The [dated training record](results/language_learning_status.json) contains
completed epochs and losses. Source, data and models have been delivered to
four training nodes; all 32 runs are executing. Full supervised, interaction
and independent-execution results remain pending.

Teacher recovery also uses the same request interface. The
[collector](collect_recovery.py) lets a frozen student make four or twelve
requests, then gives its actual state and operation feedback to the teacher.
Student requests remain history and outcome targets and receive no imitation
label. The teacher attempts the remaining task using public observations and
the same executor. Successful and unsuccessful continuations are retained.
CDM and GRU both supply starting states; their resulting records form a common
collection for a subsequent learning comparison. They never change the frozen
data used by the 32 representation runs. These assisted trajectories measure
training-data collection, not independent student completion.

### Training throughput

The token policy in this comparison has 38,343,587 parameters. Its text encoder
is called for observation lines, measurement records and each request field.
Small recurrent batches therefore produce many encoder calls; parameter count
alone does not describe the amount of text processed per decision.

Training now assembles candidate fields on the CPU before transferring each
field once, and accumulates epoch metrics on the GPU without reading every
scalar after each event. Text microbatch size and activation recomputation are
explicit settings, saved in the model configuration and inherited by evaluation
and interaction training. Existing checkpoints retain their original defaults:
64 text chunks per call and activation recomputation for trainable text.

A local H100 measurement used the same four recorded training episodes, eight
events per episode, width 512, bfloat16, and one optimizer update. Weights were
initialized with seed 501; these timings do not measure trained task performance.
Each row reports the median of three timed repetitions after a warmup:

| Execution | Seconds per 32-event window | Peak allocated GiB |
| --- | ---: | ---: |
| Original, 64 text chunks, recomputation | 2.161 | 6.62 |
| Batched transfers and deferred metrics, 64 chunks, recomputation | 2.168 | 6.62 |
| Batched transfers and deferred metrics, 256 chunks, recomputation | 1.254 | 6.66 |
| Batched transfers and deferred metrics, 64 chunks, no recomputation | 1.118 | 48.06 |

The 256-chunk setting improves this window's throughput by 1.72 times while
retaining recomputation's memory benefit. Larger text microbatches can assign
dropout draws differently; they keep the episode batch, targets and update
count unchanged. Tests compare text outputs and gradients with dropout disabled
and verify that execution settings survive checkpoint loading. Removing
recomputation preserved this window's loss, but its memory use makes a single
window insufficient to choose that setting for every episode. Full-epoch
measurements are required before changing the end-to-end duration estimate.

Use `--text-batch-size 256` with either the training command or the study driver
to select the larger encoder batches. `--no-text-checkpointing` trades memory
for compute explicitly. The already-running comparison retains its frozen source
and settings; it is not silently changed by these implementation updates.

## Run the preceding shared-executor comparison

With DiscoveryWorld installed and hosted credentials in environment variables:

```sh
python -m experiments.discoveryworld.collect_suite \
  --bailian-flash --object-requests --output runs/common/train \
  --seeds 6200 6201 6202 6203 6204 6205 6206 6207 \
  --difficulties Easy Normal \
  --themes 'Combinatorial Chemistry' 'Archaeology Dating' \
    'Plant Nutrients' 'Reactor Lab' 'Proteomics' \
  --workers 32 --max-steps 300 --max-input-tokens 131072 \
  --max-new-tokens 16384 --reasoning-effort low --action-corrections 2
```

Collect development seeds 6300 and 6301 with identical settings. Copy the first
finished-attempt Easy records to `runs/common/initial_data/{train,development}`
and mark the snapshot with `runs/common/initial-data-ready.json`. Mark the
fixed full collection with `runs/common/data-ready.json`. All model jobs must
receive identical copied trajectories before that marker appears; subsequent
collection continues outside this snapshot. For each mode
and seed in the configuration, launch on separate devices:

```sh
CUDA_VISIBLE_DEVICES=0 python -m experiments.discoveryworld.request_study \
  --root runs/common --arm self --mode cdm --seed 401
CUDA_VISIBLE_DEVICES=1 python -m experiments.discoveryworld.request_study \
  --root runs/common --arm demonstrations --mode cdm --seed 401
```

The demonstration command waits for each common data stage and paired initial
weights, then runs imitation, interaction learning and fixed evaluations. Commands, optimizer
curves, checkpoints, requests and native operations remain separate. A failed
stage stops that job and preserves its files.

The first collection attempt redundantly included full scene conditions with
each scalar measurement, causing API context-budget errors. The corrected
projection retains measurements, bindings, units and feedback without repeating
the scene. Repeated object domains in the action schema also share one enum;
all previously available objects and requests remain available.
Continuation permits up to 131,072 input tokens for large public object
registries; the first attempts used 49,152. Provider-reported usage is retained.
Unfinished worlds continue under a recorded new prompt version;
completed instances and earlier attempts are retained. They are teacher records,
not completed student evaluations.

Summarize collected model directories and teacher records with:

```sh
python -m experiments.discoveryworld.summarize_request_study \
  --records runs/common/models --collections runs/common --output runs/common/results
```
