# Measured-reaction policies

[v0.18.0 downloads](https://github.com/Cherish0925/continual-decision-model/releases/tag/v0.18.0)
provide all **80 new endpoints** and their complete learning and evaluation
records. They supplement the earlier 40 width-32 models linked in the
[study](../../experiments/measured_reactions/README.md).

| Archive | Contents |
| --- | --- |
| `measured-reactions-models.tar.gz` | `scale/training/` and `relations/training/`: every model and training configuration, eight seeds per policy |
| `measured-reactions-records.tar.gz` | Learning curves, paired per-screen evaluations, aggregate results and analysis figures for both campaigns |

Extract both archives into the same directory. Prepare the published source
table with the study's `prepare` command, then run `report`, `external` and
`compare` against the extracted directories. The scripts use relative paths;
the raw laboratory workbooks remain with their original publishers.

Each campaign used width 128, eight state slots, batches of 64 screens,
1,000 updates and an eight-measurement budget. Training seeds are 101–108.
The original named conditional policy contains 867,345 parameters. The
relation-only models retain the same core implementation; the kernel-only
control uses ten trainable parameters and skips the recurrent scorer at
inference. Its checkpoint also retains inactive core weights for loader
compatibility.

These models choose among recorded experimental conditions. They do not
control laboratory hardware. They are independent of the larger native-action
DiscoveryWorld checkpoints and are not a single model trained across all CDM
applications. No endpoint was selected by its final-test score; the seven
final Roche reaction types have not been evaluated.

Model weights use the repository's [MIT license](../../LICENSE). Cite the
original experimental studies as well as this repository when using their
measurements; the model release does not change source-data licenses.
