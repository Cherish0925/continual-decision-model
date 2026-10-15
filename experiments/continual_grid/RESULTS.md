# Dynamic-grid results

The five-seed CPU run completed 180 episodes and 34,560 decisions in 10.88 seconds. Retaining ongoing computation substantially improved the low-budget continuous-change condition. Local factorization did not improve on nearest-context retrieval.

## Return

Mean undiscounted return across five seeds; higher is better. The theoretical maximum is eight successful goal arrivals with no travel cost. The current-map reference uses additional planning work.

| Regime | Backups/tick | Restart | Carry | Snapshot | Nearest | Factorized | Current-map reference |
|---|---:|---:|---:|---:|---:|---:|---:|
| Full return | 20 | 6.596 | 6.596 | 6.720 | 6.720 | 6.720 | 6.840 |
| Full return | 81 | 6.784 | 6.784 | 6.816 | 6.816 | 6.816 | 6.840 |
| Partial return | 20 | 6.516 | 6.516 | 6.516 | **6.624** | 6.524 | 6.840 |
| Partial return | 81 | 6.752 | 6.752 | 6.752 | **6.808** | 6.760 | 6.840 |
| Continuous | 20 | 0.852 | 6.576 | 6.576 | **6.700** | 6.668 | 6.832 |
| Continuous | 81 | 6.776 | 6.776 | 6.776 | **6.792** | 6.780 | 6.832 |

At 20 backups per tick in the continuous regime, restart completed 42.5% of goal phases; every other tested method completed 100%. Every method completed all phases in the other five regime/budget combinations. Return differences there reflect travel time.

## Computational cost

Each tested method performed 3,840 cell backups per episode at the small budget and 15,552 at the large budget. Memory lookup was additional work and is included in the following CPU decision times.

| Continuous regime | Backups/tick | Mean decision time (ms) |
|---|---:|---:|
| Carry | 20 | 0.105 |
| Nearest | 20 | 0.124 |
| Factorized | 20 | 0.260 |
| Carry | 81 | 0.115 |
| Nearest | 81 | 0.138 |
| Factorized | 81 | 0.251 |

The local-context reader was approximately twice as slow as nearest-context retrieval in this implementation. These are CPU timings, not GPU measurements; CPU time does not drive the exogenous environment trace.

## Implication for the next model

Persistent computation is useful under restricted update budgets. Complete-context snapshots account for the gains under full restoration. Whole-workspace nearest-context retrieval outperformed assembling values from local-map matches in both partial-return and continuous conditions. Local transition similarity alone is therefore insufficient for the proposed computational-memory representation in this experiment.

The next neural comparison should learn which contextual factors determine reusable computation and compare directly with whole-workspace retrieval. It should retain fresh observations in every action head and include memory latency. These results establish the current baseline behavior; they contain no trained neural CDM results.

See [configuration](results/cpu_20261015/config.json), [per-seed episodes](results/cpu_20261015/episodes.csv), [goal phases](results/cpu_20261015/phases.csv), [decision traces](results/cpu_20261015/decisions.csv.gz), and [full summaries](results/cpu_20261015/summary.csv) for all values and sample standard deviations.
