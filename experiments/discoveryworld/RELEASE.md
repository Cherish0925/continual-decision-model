# Scientific-task models and experiments

[v0.5.0 downloads](https://github.com/Cherish0925/continual-decision-model/releases/tag/v0.5.0) · [Results](RESULTS.md) · [Training commands](README.md)

This release collects the scientific-task work completed on 26–27 October 2026:
native environment integration, multimodal policies, numerical learning,
autonomous experimental action, feedback-based updates and joint state/action/
computation training. Successful and unsuccessful comparisons are retained.
The earlier process and navigation studies remain in releases v0.2.0–v0.4.0.

## Download

| Archive | Contents |
| --- | --- |
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
  --pattern 'cdm-scientific-*' --dir artifacts/scientific-v0.5.0
for group in models records observations; do
  cat artifacts/scientific-v0.5.0/cdm-scientific-${group}.tar.zst* | zstd -dc --long=30 | tar -x
done
```

Models and records suffice for examining results and running fresh evaluations.
Download observations as well to rebuild frozen features and train on the
recorded trajectories. Feature caches, prepared feature tensors, third-party
pretrained weights and Python environments are excluded. They can be rebuilt
with the supplied code. Checkpoint metadata and text records use relative
project paths; trained tensor values are unchanged. The original run files
remain preserved separately.

## Run a released policy

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

## Reproduce the comparisons

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

