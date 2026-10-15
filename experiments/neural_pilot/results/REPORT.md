# Sequential contextual recall: trained neural pilot

5 initialization/data seeds; 120 AdamW updates per run; 32 episodes per update; 512 held-out episodes per run. Each episode has four cue events and 8 query events. Each event uses 2 shared Transformer refinements.

Values are mean ± sample standard deviation across training seeds. Accuracy is measured after the final refinement; cross-entropy averages both refinement readouts.

| Method | Visible-cue accuracy | Query accuracy | Cross-entropy | Train seconds |
|---|---:|---:|---:|---:|
| restart_workspace | 99.81 ± 0.24% | 38.44 ± 3.31% | 0.8865 ± 0.0214 | 4.38 |
| persistent_workspace | 83.50 ± 6.56% | 36.03 ± 1.27% | 1.0391 ± 0.0273 | 4.18 |
| computational_memory | 97.30 ± 4.16% | 56.20 ± 3.63% | 0.7720 ± 0.0740 | 7.18 |

Memory minus persistent-workspace query accuracy: **+20.18 percentage points** (paired sample SD 3.52 points).

Each method instantiates the same 26,764-parameter model. Bypassed memory parameters receive no action-loss gradients in the two memory-free ablations. The action head is trained; activity and value heads are untrained. All methods have a recurrent context and receive identical current observations and goals.

The comparison matches training episodes, optimizer updates and refinement count. Memory reading and writing add computation; wall time is reported rather than matched. Elapsed times are measurements on `cpu` from this run, not real-time environment performance.

This pilot measures trainability and cue-conditioned recall. It does not measure moving-environment control, learned activity allocation, continued parameter adaptation, or a reduction in computation needed to resume an interrupted plan.

Raw measurements and learning curves are in `results.json` and the per-run JSON files. The exact arguments and software versions are in `config.json`.
