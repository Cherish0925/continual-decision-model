# Scientific-task models and experiments

[Downloads](https://github.com/Cherish0925/continual-decision-model/releases) · [Results](RESULTS.md) · [Joint training](../continual_experiment/README.md)

## Learning chemical experiment responses: v0.12.0

[Release](https://github.com/Cherish0925/continual-decision-model/releases/tag/v0.12.0)
· [Training and complete results](../continual_chemistry/README.md)
· [360 native executions](../continual_chemistry/results/reaction_confirmation_instances.csv)

This release supplies the actual experimental experience and four fixed
checkpoints for Combinatorial Chemistry. Both joint training repetitions and
their frozen-core warmup models are included. The six-condition confirmation
uses 60 new instances, common tools, candidate recipes and operation budgets.
All conditions complete 60/60; the two joint models use 143.35 and 278.37 mean
operations, compared with 367.38 for primitive-recipe enumeration. The linked
results include history/state interventions and recipe-group intervals.

| Archive | Contents | Download |
| --- | --- | ---: |
| `chemistry-experience.tar.gz` | 320 native training instances, 10,610 experimental returns, public action records, split definitions and the upstream license | 106 MB |
| `chemistry-models.tar.gz` | Warmup and joint endpoints for training seeds 753 and 754, with their inherited core and task interfaces | 638 MB |

The checkpoints are self-contained and use structured public observations.
Running a released model requires no earlier weight archive:

```bash
python -m pip install -e '.[neural,discoveryworld]'
gh release download v0.12.0 --repo Cherish0925/continual-decision-model \
  --pattern 'chemistry-*.tar.gz' --dir artifacts/v0.12.0
tar -xzf artifacts/v0.12.0/chemistry-models.tar.gz -C artifacts/v0.12.0
tar -xzf artifacts/v0.12.0/chemistry-experience.tar.gz -C artifacts/v0.12.0
python -m experiments.continual_chemistry.native \
  --checkpoint artifacts/v0.12.0/chemistry-models/joint753.pt \
  --policy model --seed 7401 --primitive-candidates \
  --output runs/chemistry_example
```

To repeat the two joint epochs, pass a released warmup checkpoint to
`experiments.continual_chemistry.joint_reaction`, together with
`--data-root artifacts/v0.12.0/chemistry-experience`. Repeating warmup itself
starts from the seed-221 physical policy in v0.11.0. The experiment README gives
the full schedules and distinguishes the original sampling split from
ingredient-renaming augmentation. Chemistry and physical control remain
separate trained task adaptations.

## Physical prediction and action: v0.11.0

[Release](https://github.com/Cherish0925/continual-decision-model/releases/tag/v0.11.0)
· [Method and complete results](../physical_belief/README.md)
· [Native instance table](../physical_belief/results/native_instances.csv)

Four physical pretrainings and six joint action trainings connect acquired
relationship knowledge to measurement and setting decisions. Two independent
parents each receive ordinary continuation, task-reset physical training and
training through physical memory. All final checkpoints use the fixed
400-update budget. Thirteen native conditions share 120 new task instances,
including same-weight memory reset and computed-and-discarded controls.

| Archive | Contents | Approximate download |
| --- | --- | ---: |
| `cdm-physical-seed221-inference.tar.zst` | Three final policies, their physical beliefs, seed-211 predictive pretrainings and numerical prior; optimizer omitted | 351 MB |
| `cdm-physical-seed222-inference.tar.zst` | Corresponding policies and seed-212 predictive pretrainings | 351 MB |
| `cdm-physical-seed221-models.tar.zst` | Same trained checkpoints with optimizer state | 945 MB |
| `cdm-physical-seed222-models.tar.zst` | Same trained checkpoints with optimizer state | 942 MB |
| `cdm-physical-records.tar.zst` | Training curves, unique-source prediction comparisons, development and all confirmation executions, shared simulator randomness and failure-feedback replay | 637 MB |

Archives restore repository-relative paths and include separate file lists.
These policies use structured observations; no visual encoder or language model
is required. Download one inference archive and the records to reproduce the
complete first repetition:

```bash
python -m pip install -e '.[neural,discoveryworld]'
gh release download v0.11.0 --repo Cherish0925/continual-decision-model \
  --pattern 'cdm-physical-seed221-inference.tar.zst' \
  --pattern 'cdm-physical-records.tar.zst' --dir artifacts/v0.11.0
for group in seed221-inference records; do
  zstd -dc --long=30 artifacts/v0.11.0/cdm-physical-${group}.tar.zst | tar -x
done
python -m experiments.physical_belief.evaluate_native \
  --checkpoint runs/physical_belief/joint_20261029/carry_seed221/policy.pt \
  --belief-checkpoint runs/physical_belief/joint_20261029/carry_seed221/belief.pt \
  --belief-mode carry --state-mode reset \
  --manifest experiments/physical_belief/native_confirmation.json \
  --mixture-checkpoint runs/discoveryworld/calibration/mixture/shared_train_only_seed31/last.pt \
  --initial-rng-root runs/physical_belief/native_confirmation/initial_rng \
  --output runs/reproduced-physical-221 --workers 6 --threads 1
```

Use `--belief-mode reset` or `--belief-mode discard` for the same-weight
interventions. The second repetition uses
`runs/physical_belief/joint_repeat_20261029/carry_seed222/`. Each output directory
must be distinct. The separately trained `reset` models use `--belief-mode reset`;
ordinary continuations omit `--belief-checkpoint`.

To repeat training, additionally extract `cdm-active-inference.tar.zst` from
[v0.8.0](https://github.com/Cherish0925/continual-decision-model/releases/tag/v0.8.0)
and `cdm-joint-activity-data-records.tar.zst` from
[v0.10.0](https://github.com/Cherish0925/continual-decision-model/releases/tag/v0.10.0).
They provide the two parent policies and the fixed measured training data.
The [dependency table](../physical_belief/results/reproduction_dependencies.csv)
identifies the paths. Training starts a fresh optimizer from each parent;
the model archives preserve final optimizer state for further work.

```bash
python -m experiments.physical_belief.launch_policy \
  --output runs/reproduced-physical-joint --updates 400 --batch 128 --unroll 16
```

The launcher uses three simultaneous jobs on each of two GPUs. The recorded
H100 trainings each occupied 5.30–5.90 GiB of allocated GPU memory and took
20–22 minutes while sharing their device. For a single device, invoke
[train_policy.py](../physical_belief/train_policy.py) for each condition directly.
Physical pretraining commands and the exact observation/learning protocols are
in the [experiment README](../physical_belief/README.md).

## Joint activity and learning: v0.10.0

[Release](https://github.com/Cherish0925/continual-decision-model/releases/tag/v0.10.0)
· [Results](../continual_experiment/results/README.md)
· [Model and training](../continual_experiment/README.md)

Eight trainings connect the scientific-task core, internal activity policy,
experience retrieval and fast/slow prediction updates. Four conditions are
repeated from the same v0.9 parent: learned activities, no adaptation,
task-boundary resets and a fixed complete activity schedule. Native comparisons
use 184 instances in 20 changing-relation streams, with three earlier policies
as references and same-weight state/update interventions. Exploratory deployment
checks are reported separately.

| Archive | Contents |
| --- | --- |
| `cdm-joint-activity-seed201-inference.tar.zst` | Four final policies from repetition 201 and the numerical prior; optimizer state omitted |
| `cdm-joint-activity-seed202-inference.tar.zst` | The corresponding four policies from repetition 202 and the numerical prior |
| `cdm-joint-activity-seed201-models.tar.zst` | The four repetition-201 checkpoints with optimizer state |
| `cdm-joint-activity-seed202-models.tar.zst` | The four repetition-202 checkpoints with optimizer state |
| `cdm-joint-activity-records.tar.zst` | All eight training histories, primary and exploratory native executions, comparison plans and shared simulator randomness |
| `cdm-joint-activity-data-records.tar.zst` | Prepared measurement data and calibration worlds used for training |

Each inference archive is approximately 467 MB and restores project-relative
paths. The structured-observation policy needs PyTorch and the pinned simulator;
visual-encoder and language-model weights are not used in this study. Start with
one inference archive and the records to reproduce a trained policy:

```bash
python -m pip install -e '.[neural,discoveryworld]'
gh release download v0.10.0 --repo Cherish0925/continual-decision-model \
  --pattern 'cdm-joint-activity-seed201-inference.tar.zst' \
  --pattern 'cdm-joint-activity-records.tar.zst*' --dir artifacts/v0.10.0
zstd -dc --long=30 artifacts/v0.10.0/cdm-joint-activity-seed201-inference.tar.zst | tar -x
cat artifacts/v0.10.0/cdm-joint-activity-records.tar.zst* | zstd -dc --long=30 | tar -x
python -m experiments.discoveryworld.evaluate_active_stream \
  --continual \
  --checkpoint runs/continual-experiment/joint_activity_20261028/full_seed201/last.pt \
  --manifest experiments/continual_experiment/native_comparison.json \
  --mixture-checkpoint runs/discoveryworld/calibration/mixture/shared_train_only_seed31/last.pt \
  --initial-rng-root runs/discoveryworld/continual_experiment/native/initial_rng \
  --state-mode carry --feedback-cost native --device cpu \
  --max-actions 400 --max-decisions 160 --workers 6 --threads 1 \
  --output runs/reproduced-joint-201
```

Use `--state-mode reset` for the same-weight state intervention or
`--discard-adaptation` to compute updates without retaining their changes.
`--activity-selection sample --activity-seed 233` reproduces the exploratory
internal-activity sampling condition; external actions remain greedy.
Use a separate output directory for each condition. Repetition 202 uses its
corresponding archive and checkpoint paths.

The [training commands](../continual_experiment/README.md#run-the-comparison)
start from `active_lifetime/carry_seed191/update1200.pt` in the v0.9.0 model
release, with the prepared data above. Model archives also preserve the outer
optimizer for further research; the present training CLI loads checkpoint
weights and starts a fresh optimizer. The records retain the training budget
and actual interactions separately. These repetitions share one parent and the
same native test instances.

## Experience across experimental instances: v0.9.0

[Release](https://github.com/Cherish0925/continual-decision-model/releases/tag/v0.9.0)
· [Results](results/active_stream.md) · [Method](ACTIVE_STREAM.md)

Four trainings compare continuing state with task-boundary resets, from two
independent parents. Native comparisons use 184 instances in 20 changing-relation
streams. Retaining state reduces actions relative to resetting the same weights
in both repetitions. The release also preserves the absence of a consistent
overall gain over the frozen parents and the explicit relation-memory reference.

| Archive | Contents |
|---|---|
| `cdm-task-stream-inference.tar.zst` | Four final policies and their numerical prior, without optimizer state; about 449 MB |
| `cdm-task-stream-models.tar.zst` | The same four final policies with optimizer state for further training |
| `cdm-task-stream-records.tar.zst` | Training configurations, curves and complete task histories; all ten native comparisons, initial RNG states, paired summaries and action-cost decomposition |
| `cdm-chemistry-records.tar.zst` | Public reference recipe experiments for the new chemistry interface, including unsuccessful trials and early interface failures; not a trained-policy result |

Each archive includes a separate file list and restores paths relative to this
repository. The scientific policy uses structured observations; this execution
does not require downloading a visual encoder or language model.

```bash
gh release download v0.9.0 --repo Cherish0925/continual-decision-model \
  --pattern 'cdm-task-stream-inference.tar.zst' \
  --pattern 'cdm-task-stream-records.tar.zst' --dir artifacts/v0.9.0
for group in inference records; do
  zstd -dc --long=30 artifacts/v0.9.0/cdm-task-stream-${group}.tar.zst | tar -x
done
python -m experiments.discoveryworld.evaluate_active_stream \
  --manifest experiments/discoveryworld/active_lifetime_comparison.json \
  --checkpoint runs/discoveryworld/active_lifetime/carry_seed191/update1200.pt \
  --mixture-checkpoint runs/discoveryworld/calibration/mixture/shared_train_only_seed31/last.pt \
  --state-mode carry \
  --initial-rng-root runs/discoveryworld/active_experiments/lifetime_native/initial_rng \
  --output runs/reproduced-task-stream-191 --workers 4 --threads 2
```

Use `--state-mode reset` with the same checkpoint and a new output directory for
the task-boundary intervention. The separately trained control is
`active_lifetime/task_reset_seed191/update1200.pt`; repetition 192 uses the
corresponding `seed192` paths. Frozen parent comparisons use the two final
calibration-rehearsal models in v0.8.0. Reproducing the training from those
parents requires their optimizer states from the v0.8.0 model archives, the
prepared measurement data, and the commands in [ACTIVE_STREAM.md](ACTIVE_STREAM.md).

## Active experimental learning: v0.8.0

The persistent core now learns measurement and setting decisions jointly from
incomplete evidence. Calibration rehearsal generates alternative training tasks
from measured quantities and two public calibrations. Two independently trained
parents, each with matched ordinary and rehearsal continuations, provide the
comparison. Both rehearsal policies complete 62/64 fresh native instances,
matching full measurement with 53.1% and 54.6% fewer native actions. Ordinary
continuations complete 54/64 and 60/64. The [full comparison](results/calibration_rehearsal.md)
reports the additional exploration costs and state-reset interventions.

The active policy trains 26.65 million parameters, including the inherited GRU
and Transformer workspace. This comparison uses structured public observations,
nine explicit numerical hypotheses and a scalar setting tool. It does not require
visual-encoder or language-model weights.

| Download group | Contents |
| --- | --- |
| `cdm-active-inference.tar.zst` | Both final rehearsal policies and their numerical prior, with optimizer state omitted; a 253 MB download for native execution |
| `cdm-active-models.tar.zst.part*` | Two persistent and two ledger trainings, first-parent rehearsal and ordinary continuations, plus their original core, numerical prior and selector dependencies |
| `cdm-active-repeat-models.tar.zst` | Matched rehearsal and ordinary continuations from the independently trained second parent |
| `cdm-active-records.tar.zst*` | Prepared measurement datasets, training configurations and curves, all native evaluations, shared random states, paired summaries and feasibility measurements |
| `cdm-active-evidence-records.tar.zst.part*` | Original collected public measurement histories for re-extracting the expanded training data |

To try either final policy, download only the compact inference archive. It
restores the original checkpoint paths and preserves every model tensor. Python
3.12, PyTorch and the pinned simulator are sufficient:

```bash
python -m pip install -e '.[neural,discoveryworld]'
gh release download v0.8.0 --repo Cherish0925/continual-decision-model \
  --pattern 'cdm-active-inference.tar.zst' --dir artifacts/v0.8.0
zstd -dc --long=30 artifacts/v0.8.0/cdm-active-inference.tar.zst | tar -x
python -m experiments.discoveryworld.evaluate_active_native \
  --strategy learned \
  --checkpoint runs/discoveryworld/active_experiments/rehearsal/counterfactual_seed183/update1200.pt \
  --mixture-checkpoint runs/discoveryworld/calibration/mixture/shared_train_only_seed31/last.pt \
  --seeds 3400 --initial-rng-root runs/first-active-rng \
  --output runs/first-active-experiment \
  --device cpu --workers 1 --max-actions 400 --max-decisions 160
```

This command uses a new native noise realization. Exact paired reproduction
uses the recorded initial randomness and full comparison below. Continued
training requires the model archives, which also contain optimizer state.

The accompanying `*-files.txt` lists identify the contents. Archives restore
relative paths below `runs/discoveryworld/`; extract into a fresh checkout.
The prepared data and model packages suffice for training and native evaluation.
The larger evidence archive is optional for those uses. The original source
histories preceding this collection remain in v0.5.0 and v0.6.0.

```bash
git clone --branch v0.8.0 https://github.com/Cherish0925/continual-decision-model.git
cd continual-decision-model
python -m pip install -e '.[neural,discoveryworld]'
gh release download v0.8.0 --repo Cherish0925/continual-decision-model \
  --pattern 'cdm-active-models.tar.zst*' \
  --pattern 'cdm-active-repeat-models.tar.zst' \
  --pattern 'cdm-active-records.tar.zst*' --dir artifacts/v0.8.0
for group in models repeat-models records; do
  cat artifacts/v0.8.0/cdm-active-${group}.tar.zst* | zstd -dc --long=30 | tar -x
done
python -m experiments.discoveryworld.evaluate_active_native \
  --strategy learned \
  --checkpoint runs/discoveryworld/active_experiments/rehearsal/counterfactual_seed183/update1200.pt \
  --mixture-checkpoint runs/discoveryworld/calibration/mixture/shared_train_only_seed31/last.pt \
  --manifest experiments/discoveryworld/active_rehearsal_comparison.json \
  --initial-rng-root runs/discoveryworld/active_experiments/rehearsal_native/initial_rng \
  --output runs/reproduced-native-rehearsal \
  --device cpu --workers 8 --max-actions 400 --max-decisions 160
```

For the independent repeat, use
`runs/discoveryworld/active_experiments/rehearsal_repeat/counterfactual_seed184/update1200.pt`.
The [method](ACTIVE_LEARNING.md#reproduce) gives both training phases, the matched
continuations and the full-measurement control. CPU workers run native evaluation;
GPU batching accelerates core training. Release tables report actual native
completion, separately from the training replay's activation proxy.

## Complete experimental choices: v0.7.0

This study learns the success and cost of complete experimental plans. Four
317,314-parameter selectors train on 237 native branches; 65 additional branches
provide validation. The subsequent 264 from-reset evaluations compare complete
execution, learned choice, neural-state inputs and transfer to the original
action reader.

On the first 24-instance cohort, the predictive parent completes 14/24,
posterior-ordered plans 16/24, and numerical selectors from two trainings
17/24 and 18/24. With the original reader on a fresh cohort, the same selectors
both complete 23/24, compared with 19/24 for the reader and 21/24 for
posterior-ordered plans. No transfer training is performed. Adding neural state
to the selector does not give a consistent gain. [Full results](RESULTS.md#learning-complete-experimental-choices--28-october)
retain individual gains, losses and action costs.

| Archive | Contents |
| --- | --- |
| `cdm-experimental-plans-models.tar.zst` | All four final selectors: two training seeds, with and without continuing neural-state inputs |
| `cdm-experimental-plans-records.tar.zst` | Collection choices, matched-prefix outcomes, training curves, experimental designs, all native executions and initial random states |
| `cdm-experimental-plans-observations.tar.zst` | Simulator images accompanying collection and evaluation |

Each archive has a corresponding `*-files.txt` list. The selectors reuse the
original reader and numerical prior from v0.5.0. The first comparison also uses
the predictive parent from v0.6.0; the transferred policy below needs only the
v0.5.0 reference package and the new selector.

```bash
git clone --branch v0.7.0 https://github.com/Cherish0925/continual-decision-model.git
cd continual-decision-model
gh release download v0.5.0 --repo Cherish0925/continual-decision-model \
  --pattern 'cdm-scientific-reference.tar.zst' --dir artifacts/v0.5.0
gh release download v0.7.0 --repo Cherish0925/continual-decision-model \
  --pattern 'cdm-experimental-plans-models.tar.zst' \
  --pattern 'cdm-experimental-plans-records.tar.zst' --dir artifacts/v0.7.0
for archive in artifacts/v0.5.0/*.tar.zst artifacts/v0.7.0/*.tar.zst; do
  zstd -dc --long=30 "$archive" | tar -x
done
```

Install the dependencies and visual encoder using the instructions below, then:

```bash
python -m experiments.discoveryworld.evaluate \
  --policy-kind experimental_belief --local-probes \
  --checkpoint runs/discoveryworld/belief_actions/full_seed61/last.pt \
  --plan-checkpoint runs/discoveryworld/experimental_plans/numeric/last.pt \
  --feature-cache runs/discoveryworld/features \
  --encoder-path data/cache/siglip2 --device cuda:0 \
  --seeds 3000 --max-steps 400 --workers 1 \
  --initial-rng-root runs/discoveryworld/experimental_plans/transfer/reference \
  --output runs/reproduced-experimental-selection
```

The records also contain the actual incoming features and measured alternatives
needed to train a selector. Retraining this head does not require replaying
image encoding or downloading the observation archive:

```bash
python -m experiments.discoveryworld.train_plans \
  --train runs/discoveryworld/experimental_plans/train \
  --validation runs/discoveryworld/experimental_plans/validation \
  --no-state --seed 171 --epochs 100 --device cuda:0 \
  --output runs/retrained-experimental-selector
python -m experiments.discoveryworld.summarize_plans --include-transfer
```

Use seed 172 for the second numerical selector, or omit `--no-state` for the
state-conditioned comparison. To recollect the experiments, also download the
v0.6.0 predictive models and records. The optional v0.7.0 observation archive
preserves the rendered views of the original runs. The
[protocol](PREDICTION.md#learning-choices-between-complete-experiments) identifies
the source histories, held-out instances, fixed budgets and outcome definitions.

## Prediction and action consequences: v0.6.0

The prediction study and its paired-action follow-up include model weights,
training curves, all native executions, and their observations. The original
successful reader remains the reference; these experiments have not established
an overall improvement. The [results](RESULTS.md) distinguish task completion,
prediction accuracy and reliance on additional workspace computation.

| Archive group | Contents |
| --- | --- |
| `cdm-predictive-models.tar.zst` | Both joint prediction trainings, including their epoch checkpoints |
| `cdm-predictive-records.tar.zst` | Collected training/validation histories, training logs, five-policy native comparison and computation branches |
| `cdm-predictive-observations.tar.zst` | Images accompanying those histories and executions |
| `cdm-consequences-models.tar.zst` | Paired-consequence learning and matched replay checkpoints |
| `cdm-consequences-records.tar.zst` | All 96 alternative-action continuations, both training runs and the four-policy native comparison |
| `cdm-consequences-observations.tar.zst` | Images accompanying the alternative-action and follow-up executions |

Every group has a corresponding `*-files.txt` content list. Extract the archives
in a fresh checkout; paths are relative to the repository. The shared source
models and original demonstrations come from v0.5.0. Download its compact
reference package to run the new policies; add its records and observations to
retrain from all trajectories.

```bash
git clone --branch v0.6.0 https://github.com/Cherish0925/continual-decision-model.git
cd continual-decision-model
gh release download v0.5.0 --repo Cherish0925/continual-decision-model \
  --pattern 'cdm-scientific-reference.tar.zst' --dir artifacts/v0.5.0
gh release download v0.6.0 --repo Cherish0925/continual-decision-model \
  --pattern '*.tar.zst' --dir artifacts/v0.6.0
for archive in artifacts/v0.5.0/*.tar.zst artifacts/v0.6.0/*.tar.zst; do
  zstd -dc --long=30 "$archive" | tar -x
done
```

After installing the dependencies and visual encoder described below:

```bash
python -m experiments.discoveryworld.evaluate \
  --policy-kind predictive --local-probes \
  --checkpoint runs/discoveryworld/predictive_actions/feedback/last.pt \
  --feature-cache runs/discoveryworld/features \
  --encoder-path data/cache/siglip2 --device cuda:0 \
  --seeds 2400 --max-steps 400 --workers 1 \
  --initial-rng-root runs/discoveryworld/predictive_actions/native/reference \
  --output runs/reproduced-prediction-policy
```

[PREDICTION.md](PREDICTION.md) gives the complete data split, feature preparation,
training objectives and paired comparisons. Native evaluation supports `--resume`
to retain completed instances and restart an interrupted instance from its saved
initial randomness. Prepared features and cached training states are rebuilt
from the released records rather than distributed as duplicate data.

## Scientific-task foundation: v0.5.0

[Download v0.5.0](https://github.com/Cherish0925/continual-decision-model/releases/tag/v0.5.0)

This release collects the scientific-task work completed on 26–27 October 2026:
native environment integration, multimodal policies, numerical learning,
autonomous experimental action, feedback-based updates and joint state/action/
computation training. Successful and unsuccessful comparisons are retained.
The earlier process and navigation studies remain in releases v0.2.0–v0.4.0.

### Download

| Archive | Contents |
| --- | --- |
| `cdm-scientific-reference.tar.zst` | Compact runnable package: the two reference readers, evidence controls and their shared model dependencies |
| `cdm-scientific-models.tar.zst*` | Saved policy and numerical checkpoints, including earlier alternatives, matched controls and the latest joint-training models |
| `cdm-scientific-records.tar.zst*` | Run configurations, training curves, summaries, action/decision trajectories, initial random-state records, numerical analyses and experiment scripts saved with the runs |
| `cdm-scientific-observations.tar.zst*` | Collected simulator images for reference demonstrations and policy evaluations; identical images share storage inside the archive |

Each archive restores relative paths below `runs/discoveryworld/`. Large
archives are split into numbered parts; join every part before extraction. Install the standard `zstd` command for decompression.
The accompanying `models-files.txt`, `records-files.txt` and
`observations-files.txt` list the exact contents. Existing local runs should be
kept separately; extract into a fresh checkout.

```bash
git clone --branch v0.5.0 https://github.com/Cherish0925/continual-decision-model.git
cd continual-decision-model
gh release download v0.5.0 --repo Cherish0925/continual-decision-model \
  --pattern 'cdm-scientific-reference.tar.zst' \
  --pattern 'cdm-scientific-records.tar.zst' --dir artifacts/scientific-v0.5.0
for group in reference records; do
  cat artifacts/scientific-v0.5.0/cdm-scientific-${group}.tar.zst* | zstd -dc --long=30 | tar -x
done
```

The reference package and records suffice for examining results and running
the reference policies. The full model archive adds historical alternatives
and joint-training checkpoints. Download observations as well to rebuild frozen features and train on the
recorded trajectories. Feature caches, prepared feature tensors, third-party
pretrained weights and Python environments are excluded. They can be rebuilt
with the supplied code. To obtain all materials, repeat the download with
`--pattern 'cdm-scientific-*'`, then extract each remaining archive in the same
way. Checkpoint metadata and text records use relative
project paths; trained tensor values are unchanged. The original run files
remain preserved separately.

### Run a released policy

Install Python 3.12 and a CUDA-compatible PyTorch/Torchvision pair, then:

```bash
python -m pip install -e '.[multimodal,discoveryworld]'
python - <<'PY'
from huggingface_hub import snapshot_download
from cdm.multimodal import ENCODER_ID, ENCODER_REVISION
snapshot_download(ENCODER_ID, revision=ENCODER_REVISION,
                  local_dir='data/cache/siglip2')
PY
python -m experiments.discoveryworld.evaluate \
  --policy-kind belief --local-probes \
  --checkpoint runs/discoveryworld/belief_actions/full_seed61/last.pt \
  --feature-cache runs/discoveryworld/features \
  --encoder-path data/cache/siglip2 --device cuda:0 \
  --seeds 610 --max-steps 400 --workers 1 \
  --initial-rng-root runs/discoveryworld/belief_actions/native_full_seed61 \
  --output runs/reproduced-scientific-policy
```

The initial random-state record reproduces the original measurement noise; it
supplies no actions or experimental evidence to the policy. Omit that argument
for a new native noise realization. The reported two-cohort completion counts
use all sixteen instances, not the single-instance command above.

The reference policy needs three checkpoints, all included: its action reader,
`calibration/mixture/shared_seed31/last.pt` and
`measurement_memory/action_run/last.pt`. Paths are relative to
`runs/discoveryworld/`. SigLIP is loaded separately from its upstream release.
The successful policy does not require Qwen model weights.

### Reproduce the comparisons

Use the [task README](README.md#reading-experimental-evidence-to-choose-actions)
for feature preparation, independent reader seeds, evidence controls and native
evaluation. The [joint-training protocol](../../TRAINING.md#joint-state-action-and-computation-training--27-october)
covers the latest state and computation comparison. [Results](RESULTS.md) link
each study to its per-instance tables and identify its run directory.

The reference readers complete 14/16 and 13/16 on the two reported cohorts.
The later joint-training comparison uses a separate cohort and does not
establish adaptive-computation gains. The explicit affine hypothesis class
and numerical control primitive remain part of the method. Cross-task skill
retention has not yet been demonstrated by these scientific-task experiments.

To rebuild the release archives from original runs, use
`python -m experiments.discoveryworld.package_release --kind models --output artifacts/scientific-v0.5.0`
and repeat for `records` and `observations`. Packaging and extraction use `zstd`; compression is lossless.

Simulator observations are generated with
[DiscoveryWorld](https://github.com/allenai/discoveryworld); its code and visual
assets retain their upstream licenses. Pretrained models retain their own
licenses and are downloaded from their original publishers. CDM code and
documentation use the repository's MIT license.

