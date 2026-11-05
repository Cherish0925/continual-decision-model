# Scientific decisions with shared execution

The current study compares a language planner, CDM and a GRU planner through
the same object-request interface. Each planner chooses the operation,
instrument, sample, recipe component and control-menu answer. A deterministic
executor reaches and acquires the declared objects. Every underlying operation
consumes the native action budget and remains in the recorded trajectory.

**Status, 6 November 2026:** the autonomous comparison and full-data supervised
endpoints are complete; demonstration-initialized interaction training is running.
It resumed after a data-delivery interruption. Supervised
learning, autonomous gradient updates and independent learned-policy execution
have run in the native simulator. All sixteen self-learning runs finished
their interaction schedules on four compute nodes. All sixteen paired demonstration runs completed
four initial epochs on the fixed Easy collection while longer tasks continue.
Full comparative scores are pending.
The [dated run snapshot](results/common_execution_status.json) separates actual
optimizer updates, unfinished teacher attempts and pending model comparisons.

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

## Completed autonomous comparison and initial demonstration execution

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
endpoint. Autonomous interaction training from these weights is running;
its final evaluations have not completed.

All sixteen demonstration models finished four epochs on the same 678 training requests.
Each then executed the same ten Easy development instances with two decoding
methods. Each table entry aggregates eight initializations; it is eighty
executions of ten worlds.

| Model | Greedy completion | Sampled completion |
| --- | ---: | ---: |
| CDM | 0/80 | 7/80 |
| GRU | 0/80 | 3/80 |

These starting policies do not yet execute the tasks reliably. Full imitation,
autonomous learning and the fresh-instance comparison remain pending. The
initial weights are fixed, and these development outcomes do not select a model
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

## Run

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
