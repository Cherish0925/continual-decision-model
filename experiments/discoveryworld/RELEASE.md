# Scientific-task models and experiments

[v0.6.0 downloads](https://github.com/Cherish0925/continual-decision-model/releases/tag/v0.6.0) · [Results](RESULTS.md) · [Training commands](README.md)

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

