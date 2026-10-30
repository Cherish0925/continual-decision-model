# Learning across physical and chemical experiments

This study trains one core through alternating physical decisions and chemical
experiment returns. It asks whether a single continuing neural state can retain
useful information across these changes while maintaining each task's capability.
A matched model shares all weights and learning objectives but preserves a
separate neural state for each task interface. This separates shared-parameter
learning from the effect of continuing the same state across domains.

## Training sequence

Each optimizer window contains eight physical requests per stream, four
recorded chemical experiments, then eight more physical requests. Gradients
cross both domain switches; they stop at the optimizer-window boundary. There
are 32 streams and 200 fixed windows per training: 102,400 physical requests
and 25,600 chemical returns. Task-specific numerical memories remain separate
because they encode different quantities. The GRU and Transformer workspace
are a single parameter object in both conditions.

Physical requests execute the native-cost training environment, with actual
device operations, public feedback and a 400-operation task budget. Their loss
combines action/activity return, value prediction, physical prediction and the
inherited-policy anchor used in the preceding study. The physical interface
preserves the parent's candidate representation and failure masks; its new
physical-action residual starts at zero, and pending feedback initially favors
WAIT over unsolicited refinement.

Chemical examples are the 10,610 recorded training returns from 320 native
instances. Each lane traverses complete recorded histories, then starts another
training episode. Consistent ingredient renaming changes neither outcomes nor
candidate identity within an episode. The loss is the observed, censored rust
likelihood plus train-scaled native-cost regression. Only the executed recipe
receives a label. No hidden recipe or counterfactual result is supplied.

The physical interface learns ACT, REFINE, RETRIEVE, LEARN, CONSOLIDATE and WAIT.
The chemical interface currently receives observations once and predicts
experimental outcomes; its actual task choices use predicted rust reduction
per predicted cost. Its activity schedule is not learned in this experiment.
Chemical replay pauses the simulated physical scheduler, so these experiments
do not establish asynchronous-world responsiveness.

## Matched conditions

Seeds 831 and 832 use the same physical parent, with chemical warmups 753 and
754 respectively. These warmups retain exactly the same inherited core. They
are two continuations and chemical-interface initializations, not independent
pretrained cores. Within each pair the only training-condition change is:

| Condition | Parameters | Neural state |
|---|---|---|
| Continuing | One shared core and the two task interfaces | One h/z crosses both domain switches |
| Separated | Identical initial parameters, optimizer groups and budgets | One continuing h/z per task interface |

All four runs have fixed final endpoints. The physical request counts match;
primitive operation totals depend on the chosen actions and are recorded.
Chemical return counts and sequence construction match within each pair.
Core learning rate is 3e-6; physical inherited heads use 3e-5, new/physical-belief
parameters 1e-4, and chemical interface parameters 1e-4. One optimizer owns each
parameter exactly once.

```bash
python -m experiments.continual_experiment.launch_shared_tasks \
  --output runs/continual_physics/shared_tasks_joint \
  --updates 200 --batch 32 --external-per-stream 16 --chemical-steps 4
```

## Native task sequences

The [fixed manifest](shared_confirmation.json) contains twenty streams, each
with six Reactor Lab instances in an A–B–A relation sequence and six alternating
Normal/Challenge chemistry instances. After every complete physical task the
same loaded model performs a chemical task and then returns to physical work.
This uses longer within-domain intervals than the training windows.

The 120 physical and 120 chemical seeds do not overlap the prior experiment
manifests or the chemical training collection. They use familiar simulator
function families; this is a new-instance and domain-switch test, not a test of
previously unseen physical or chemical laws. Each condition uses the same
saved initial simulator randomness and task budgets.

Eight conditions include all four trained endpoints, both untrained paired
interfaces with separated state, and each continuing model executed with
separated state at unchanged weights. Each condition runs 240 native tasks,
for 1,920 executions. Results are reported separately by domain and paired by
whole twelve-task stream. Averages retain unsuccessful tasks. The same-weight
intervention isolates cross-domain h/z continuation; numerical memories and
within-task online updates remain enabled.

```bash
PYTHONHASHSEED=0 python -m experiments.continual_experiment.shared_confirmation \
  --training-root runs/continual_physics/shared_tasks_joint \
  --output runs/continual_physics/shared_tasks_confirmation
```

## Completed comparison

All four 200-update trainings and all 1,920 native executions completed. The
following costs include failed tasks. Each row contains the same 120 physical
and 120 chemical instances, interleaved in twenty ordered streams.

| Initialization | Training / execution state | Physics completed | Physics actions | Chemistry completed | Chemistry actions |
|---|---|---:|---:|---:|---:|
| 831 | Initial / separated | 119/120 | 105.40 | 120/120 | 110.33 |
| 831 | Continuing / continuing | 119/120 | 107.61 | 120/120 | 115.28 |
| 831 | Separated / separated | 120/120 | 89.25 | 120/120 | 109.51 |
| 831 | Continuing / separated | 119/120 | 110.00 | 120/120 | 104.68 |
| 832 | Initial / separated | 119/120 | 105.40 | 120/120 | 183.38 |
| 832 | Continuing / continuing | 119/120 | 101.08 | 120/120 | 150.46 |
| 832 | Separated / separated | 119/120 | 106.12 | 118/120 | 198.23 |
| 832 | Continuing / separated | 119/120 | 101.83 | 120/120 | 152.96 |

Continuing one h/z has no consistent advantage. In repetition 831 it costs
18.36 more physical operations than separately maintained state (paired
whole-stream 95% interval 9.49 to 27.99). In repetition 832 it saves 47.78
chemical operations (14.33 to 84.17). At unchanged weights, separating state
saves 10.61 chemical operations for model 831 (1.21 to 20.76), while the other
same-weight differences include zero. These are domain- and
initialization-dependent effects, not a general preference for either state
layout. Shared parameters do not require a single undifferentiated task state.

The result also leaves the principal learning gap visible: chemical predictions
were trained on recorded behavior, while deployment chooses experiments using
those predictions. Learning on the policy's own returned experiments is the
next training change. State organization alone does not address that shift.

The [machine-readable summary](results/shared_tasks.json) and
[all instance results](results/shared_tasks_instances.csv) retain every endpoint,
paired intervals and timing. These intervals resample twenty complete streams;
they do not measure uncertainty across a population of pretrained models.
Training took 2,837–2,969 seconds per run on one H100. The final endpoints were
fixed at update 200; no favorable intermediate checkpoint replaced them.
