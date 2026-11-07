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

The [final campaign record](results/language_learning_status.json) has thirty
complete pipelines, one failed frozen-text GRU, and one adapted-text GRU paused
during its last state-reset evaluation. All 32 supervised stages reached their
scheduled endpoint; the failed pipeline subsequently stopped on nonfinite
values. Thirty-one interaction stages completed 120 worlds each: 3,720 episodes,
823,644 native actions and 18,386 optimizer updates.

| Representation | CDM greedy after interaction | GRU greedy after interaction |
| --- | ---: | ---: |
| Lexical | 1/80 | 0/80 |
| Frozen MiniLM | 1/80 | 1/60 |
| Adapted MiniLM | 0/80 | 0/80 |
| Random MiniLM | 0/80 | 2/80 |

Each four-seed group repeats the same twenty Easy/Normal instances; the
frozen-text GRU row has three evaluated initializations because one failed.
Missing endpoints are not silently scored as zero. All successful greedy
executions are Easy. Sampling and state-reset results, along with every
instance, are retained in [the complete recorded comparisons](results/language_learning.json).
The result does not establish improved scientific execution from this text
representation change.

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

The token policy has 38,343,587 parameters. Observation lines, measurement
records and request arguments previously invoked the text encoder separately
at each event. The new `--window-text` path prepares an observed window on the
CPU, groups text by length, encodes it in larger batches, and restores each
field's embedding before the causal state loop. Inputs are transferred by dtype;
epoch statistics remain on the GPU until their final read.

Each text row attends only within its own string. A later event cannot enter an
earlier decision. Text occurrences in different original encoder calls retain
independent dropout, although batching changes random-number assignment.
Learned embeddings retain gradients and are recomputed after every parameter
update. Token tensors are cached; this is not a cross-update embedding cache.
Tests cover output/gradient agreement with dropout disabled, all-padding inputs,
future-event independence and state-gradient truncation without a state reset.

The following H100 measurement uses the same four episodes, eight events per
episode, bfloat16 and one optimizer update. The median is taken over three
repetitions after warmup, starting from random weights. It measures execution,
not trained task capability.

| Execution | Seconds per 32-event window | Peak allocated GiB |
| --- | ---: | ---: |
| Original, 64 text chunks, recomputation | 2.161 | 6.62 |
| Batched transfers and deferred metrics, 64 chunks | 2.168 | 6.62 |
| Batched transfers and deferred metrics, 256 chunks | 1.254 | 6.66 |
| Window text encoding, 256 chunks | **0.611** | **2.67** |
| 64 chunks, no activation recomputation | 1.118 | 48.06 |

The first three rows and window encoding retain activation recomputation.
A longer eight-episode, 32-step window takes 3.859 seconds and 15.52 GiB; it
contains 144 real events and padded slots, so it is not a 256-event measurement.

A complete first epoch uses all 5,542 training requests and 1,420 validation
requests, the same seed, four-episode batch, eight-step optimizer window, and
fixed update count. No score selects a timing endpoint.

| Execution | Training plus validation | Input/model preparation | Reported elapsed |
| --- | ---: | ---: | ---: |
| Original | 931.83 s | 329.08 s | 1260.92 s |
| Larger encoder batches and transfers | 534.08 s | 328.54 s | 862.62 s |
| Window encoding | **338.14 s** | 183.50 s | 521.64 s |

Window encoding reduces the training-plus-validation phase by **2.76 times**.
Its training portion is 317.76 seconds and validation is 20.37 seconds; peak
allocated memory is 5.21 GiB. Preparation was measured on different warm-cache
runs and is unchanged by window encoding. Its shorter measured time must not be
attributed to this implementation. The one-epoch training losses are 1.9825,
1.9808 and 1.9802; these are timing checks, not evidence of improved task quality.
The logs now separate training/validation duration, actual event throughput,
optimizer updates and peak allocation. A full first epoch at batch eight and
sequence length thirty-two also finishes: 371.17 seconds of training, 28.76
seconds of validation and 31.21 GiB peak allocation. It has seventy optimizer
updates and a different training batch, so it is a capacity measurement rather
than the matched short-window timing comparison.

[Recorded measurements](results/request_training_throughput.json) include the
execution revisions and PyTorch version. The window experiment is reproducible
with the [benchmark command](benchmark_requests.py):

```sh
python -m experiments.discoveryworld.benchmark_requests \
  --root runs/semantic_requests --encoder-path models/all-MiniLM-L6-v2 \
  --output runs/throughput/window --text-batch-size 256 --window-text
```

Use `--no-window-text --text-batch-size 64` for the current stepwise path.
`--no-text-checkpointing` trades memory for computation. Existing checkpoints
retain their original execution defaults. The earlier 32-run study continues
on its frozen source, so no trained endpoint silently changes implementation.

CPU preparation can now process independent trajectories in separate workers;
events inside a trajectory retain their original order. On the same eighty
training and twenty validation trajectories with warm feature files,
`--prepare-workers 8` reduces preparation from **184.99 to 43.57 seconds**
(4.25 times). A parallel-versus-serial check preserves all prepared history,
labels, outcome targets and episode ordering. The already-running A100 jobs
keep their original source. This preparation measurement is separate from the
full-epoch timing above; their speedups should not be multiplied.

### Recovery data and matched gradient spans

The student-state collection has finished 160 attempted continuations over forty
training instances: CDM/GRU starting policies, each handed over after four or
twelve requests. The teacher-assisted continuations complete 124 attempts; nine
end with provider errors. Their valid prefixes are retained. This collection
contains 11,657 requests, including 9,698 fulfilled teacher requests and 1,243
student requests. Student requests have no imitation labels. Terminal failed
requests are also excluded from imitation by the existing preparation rule.
Teacher-assisted completions are not student completion scores.

A separate learning-rate comparison reran four divergent lexical initializations
at 1e-4 and 3e-5 with the original data and fixed 32 epochs. All eight are finite.
At 1e-4, the two CDM runs complete 1/20 and 6/20 development instances; GRU
completes 5/20 and 4/20. The initializations were selected for divergent training
loss, and the CDM/GRU seed subsets differ. These results motivate the lower rate;
they do not establish a recurrent-architecture advantage.
[All eight runs and case outcomes](results/request_stability.json).

The next fixed comparison uses the union of the original 80 trajectories and
160 recoveries: **240 trajectories, 120 distinct instance identities and 17,199
requests**. Every condition receives this exact snapshot. Development fitting
retains the original twenty instances. Native development uses seeds 7000/7001,
and fixed final evaluation uses 7200/7201, over the same five themes at Easy and
Normal difficulty. Seeds alone are not a claim of held-out scientific laws.

| Variable | Values |
| --- | --- |
| Input | Lexical; jointly adapted pretrained text |
| Decision model | CDM; GRU |
| Initialization | 501; 502 |
| State gradient span | 8; 32 events |
| Optimizer window | 32 events in both span conditions |
| Episode batch | 8 |
| Supervised endpoint | 16 complete epochs; main learning rate 1e-4, text rate 2e-5 |
| Text execution | Window encoding, microbatch 256, activation recomputation |

The eight lexical pipelines completed all stages. The eight adapted-text
pipelines were saved at epoch 9/16 when compute was reallocated. Additional
text CDM repetitions at initialization 503 were saved at 10/16; their two GRU
controls were queued but never started. Their final task evaluations are absent,
not failed. The startup cache-path error preceded all parameter updates and
is not a model result. [The final status](results/request_window_status.json)
records all twenty planned configurations and each completed training stage.

The state is detached after eight or thirty-two events **without resetting its
value**. Both conditions accumulate losses over the same 32-step optimizer
window, use the same batches and perform the same number of parameter updates.
This comparison tests the gradient horizon rather than conflating it with a
change in update count. The dataset and learning rate also differ from the old
study, so an old-versus-new gain cannot be assigned solely to a longer horizon.

```sh
python -m experiments.discoveryworld.language_study \
  --root runs/window_learning --output runs/window_learning/models/adapted_cdm_s501_g32 \
  --condition adapted_text --mode cdm --seed 501 \
  --encoder-path models/all-MiniLM-L6-v2 \
  --epochs 16 --batch-size 8 --sequence-length 32 --state-gradient-length 32 \
  --window-text --text-batch-size 256 --lr 0.0001 \
  --development-seeds 7000 7001 --evaluation-seeds 7200 7201
```

Replace the span by eight for its matched control. Each pipeline proceeds from
the supervised endpoint through native evaluation, the existing 120-world
on-policy interaction schedule, and greedy/sampled/state-reset evaluations.
Only the supervised phase changes gradient span; the subsequent interaction
schedule is held fixed. The pipeline runs independently of the terminal, with
small progress records collected separately from large artifacts.

This work changes offline learning execution and its temporal gradient horizon.
It does not yet implement asynchronous off-policy learning, deployment-time
weight consolidation, or a learned incremental-inference schedule.

### Completed window comparison

Every lexical model completed sixteen supervised epochs and 120 subsequent
interaction worlds. The following scores repeat twenty fixed test instances
(seeds 7200/7201, five themes, Easy/Normal) across two initializations per row.
They are not official 120-instance scores or forty distinct scientific tasks.

| Model / gradient events | Supervised greedy | Interaction greedy | Interaction sampled | Interaction greedy, neural state reset |
| --- | ---: | ---: | ---: | ---: |
| CDM / 8 | 4/40 | 0/40 | 2/40 | 1/40 |
| CDM / 32 | 0/40 | 1/40 | 1/40 | 1/40 |
| GRU / 8 | 3/40 | 0/40 | 2/40 | 0/40 |
| GRU / 32 | 2/40 | 0/40 | 2/40 | 0/40 |

The four CDM models average 50.63% complete-request agreement on training
states and 44.03% on validation states; GRU averages 50.17% and 43.80%.
These are demonstration-state metrics. Supervised greedy CDM success totals
4/80, including Normal Proteomics at instance 7200; GRU totals 5/80, including
the same Normal instance. After interaction training the totals are 1/80 and
0/80. Mean progress drops from 0.3742 to 0.1503 for CDM and from 0.4038 to
0.1612 for GRU. These comparisons use the same instances before and after
interaction. Greedy state reset reaches 2/80 for CDM and 0/80 for GRU, leaving
no established completion benefit from retained neural state.

The eight interaction runs contribute 960 training episodes, 214,532 native
actions and 4,104 optimizer updates. More interaction under the present loss
and schedule erodes the supervised behavior. The experiment does not isolate
why: it changes the policy through actor-critic updates, with a different
state distribution and task objective. The results support revising this
continuation before another unchanged training expansion.

New training inputs, the lower learning rate and the longer optimizer window
all differ from the preceding representation study. Their combined fitting
improvement cannot be attributed to one change. Within the matched window
study, a 32-event state gradient has no stable completion advantage over eight.
The uncompleted text arms remain open comparisons.

[Per-instance results](results/request_window.json) and
[training status](results/request_window_status.json) can be rebuilt from the
released records without running a model:

```sh
python -m experiments.discoveryworld.summarize_request_study \
  --study window --stopped \
  --records runs/discoveryworld/window_learning_20261106/models \
  --additional-records runs/discoveryworld/window_learning_20261106/local_models \
  --collections runs/discoveryworld/window_learning_20261106 \
  --output runs/recomputed-window-results
```

Use `--study language` with the corresponding `semantic_requests_20261106`
model and collection directories for the earlier representation comparison.
The [release guide](RELEASE.md#request-learning-and-sequence-training-v0190)
identifies every completed and partial model, the input data, and archive paths.

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
