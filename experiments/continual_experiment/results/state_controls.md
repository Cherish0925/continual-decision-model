# What persists between scientific tasks

This comparison separates the continuing neural state (`h`, `z`) from the outcome replay buffer and fast/slow prediction adapters in the released v0.10 model. The numerical physical posterior is recomputed from each instance; it is not part of the carried state.

Both trained full models use their original ordered 184-instance cohort, initial simulator RNG, public observations, tools, and 400-action limit. Four new conditions execute 736 instances. Full carry and full reset reuse the original results. These are post-hoc mechanism controls, without training or checkpoint selection.

| Repetition | Boundary state | Completed | Actions / instance | First return: completed | First return: actions |
| --- | --- | ---: | ---: | ---: | ---: |
| 201 | Reset all | 179/184 | 197.12 | 19/20 | 212.95 |
| 201 | Carry h/z only | 179/184 | 148.58 | 19/20 | 236.15 |
| 201 | Carry replay and adapters only | 179/184 | 197.12 | 19/20 | 212.95 |
| 201 | Carry all | 179/184 | 148.58 | 19/20 | 236.15 |
| 202 | Reset all | 178/184 | 201.35 | 18/20 | 216.25 |
| 202 | Carry h/z only | 174/184 | 193.80 | 16/20 | 248.85 |
| 202 | Carry replay and adapters only | 178/184 | 201.35 | 18/20 | 216.25 |
| 202 | Carry all | 174/184 | 193.76 | 16/20 | 248.85 |

Keeping only h/z reproduces the complete first-return penalty: +23.20 and +32.60 native actions per instance in repetitions 201 and 202. Carrying replay and adapters while resetting h/z produces no first-return difference from resetting everything. The additional carry effects and their interaction are exactly zero on these first-return outcomes.

Across all 184 instances, full carry and h/z-only have identical outcomes in repetition 201. In repetition 202, carrying replay and adapters in addition to h/z saves eight actions in total, without changing any completion. With h/z reset, carrying replay and adapters adds one action in repetition 201 and changes no action count in repetition 202. Thus these components do not explain the observed first-return penalty in this cohort.

The neural state also accounts for most of the benefit during continued work on a relation. Its effect depends on the task transition: useful recent state can coexist with costly return to an earlier relation. These interventions locate the effect in the carried h/z pathway. They do not distinguish recurrent state from workspace state, stale object bindings from other stored information, or the downstream computations that convert that information into actions. Those are separate hypotheses.

## First-return contrasts

Differences are added state minus the corresponding reset control. Negative action differences mean fewer native operations. Intervals resample complete streams; the two training repetitions are reported separately.

| Repetition | Added state and existing state | Action difference [95% interval] | Completion difference, pp [95% interval] |
| --- | --- | ---: | ---: |
| 201 | h/z, replay/adapters reset | +23.20 [+1.65, +47.90] | +0.0 [+0.0, +0.0] |
| 201 | Replay/adapters, h/z reset | +0.00 [+0.00, +0.00] | +0.0 [+0.0, +0.0] |
| 201 | h/z, replay/adapters carried | +23.20 [+1.65, +47.90] | +0.0 [+0.0, +0.0] |
| 201 | Replay/adapters, h/z carried | +0.00 [+0.00, +0.00] | +0.0 [+0.0, +0.0] |
| 201 | Interaction of the two carried groups | +0.00 [+0.00, +0.00] | +0.0 [+0.0, +0.0] |
| 202 | h/z, replay/adapters reset | +32.60 [+3.05, +66.75] | -10.0 [-25.0, +0.0] |
| 202 | Replay/adapters, h/z reset | +0.00 [+0.00, +0.00] | +0.0 [+0.0, +0.0] |
| 202 | h/z, replay/adapters carried | +32.60 [+3.05, +66.75] | -10.0 [-25.0, +0.0] |
| 202 | Replay/adapters, h/z carried | +0.00 [+0.00, +0.00] | +0.0 [+0.0, +0.0] |
| 202 | Interaction of the two carried groups | +0.00 [+0.00, +0.00] | +0.0 [+0.0, +0.0] |

The interaction is full − neural-only − learning-only + reset. A group intervention measures its total effect through subsequent decisions and observations; it does not identify an individual hidden variable or internal computation.

Pending requests, terminal feedback, activity counts, and observation counts follow the original carry protocol. Only the stated neural tensors or replay/adaptation tensors are reset. Each task starts with a fresh public measurement ledger. All 20 untreated first episodes per condition reproduce the original completion, action, measurement and test counts. An unchanged-carry pilot also reproduced all seven episodes in the first stream for both models.

[Per-instance data](state_controls_instances.csv) · [Phase summaries and paired intervals](state_controls_summary.json)

Reproduce with `python -m experiments.continual_experiment.state_controls --output runs/continual-experiment/state_controls`, then `python -m experiments.continual_experiment.summarize_state_controls --input runs/continual-experiment/state_controls`. The original release checkpoints and initial RNG records are required.
