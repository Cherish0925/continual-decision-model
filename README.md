<p>
  <img src="assets/readme-header.png" alt="Continual Decision Model — Persistent state, continual computation." width="1280">
</p>

[Research](RESEARCH.md) · [Model](DESIGN.md) · [Experiments](EXPERIMENTS.md) · [Run](RUNNING.md) · [References](ref/README.md)

**CDM studies how a model can keep perceiving, reasoning, acting, and learning while the world changes.** Internal state persists across interactions. The model learns external actions and the choice of its next observation or computation. A conversation is one event in that continuing process.

An observation can become outdated while inference is still running. The research question is how to preserve useful computation, incorporate new observations, and spend additional computation where it improves subsequent decisions.

## Model

The implemented neural core combines:

- **Recurrent context:** observations, goals, environment descriptors, previous actions, and elapsed time update a GRU state.
- **Persistent workspace:** a fixed number of latent slots undergo repeated computation through one shared Transformer block.
- **Computational memory:** a bounded record bank stores intermediate work with its generating context, goal, observation-prefix index, parameter version, and depth.
- **Conditioned reading:** learned goal, environment, and context scores select records; slotwise gates combine reuse, revision, and initialization.
- **Action and activity outputs:** the action head always reads current context; separate heads represent activity selection and value.

The current reader uses one record-selection distribution shared across slots. Learning finer-grained reuse, its contribution to changing-world control, and the choice of computation are the next experiments. [DESIGN.md](DESIGN.md) gives the equations and executable API; [research/CONTINUAL_COMPUTATION.md](research/CONTINUAL_COMPUTATION.md) develops the central hypothesis.

## Implemented and measured

| Study | Implementation | Result |
| --- | --- | --- |
| [Neural pilot](experiments/neural_pilot/results/REPORT.md) | Five seeds, three model variants, supervised sequential contextual recall | Query accuracy: memory **56.20%**, persistent workspace **36.03%**, restarted workspace **38.44%** |
| [Dynamic grid](experiments/continual_grid/RESULTS.md) | Five seeds, three change regimes, two computation budgets; world advances during planning | Reuse helps at low budgets; this local factorization does not outperform nearest-context retrieval |
| [Numerical transport](experiments/transport/RESULTS.md) | Changing quadratic objectives and finite-depth state reconstruction | Tested correction loses to strong solvers; reconstruction quality and task quality differ |

The neural pilot trains action prediction with a fixed memory-write schedule. It does not yet train activity selection. The dynamic grid compares explicit planning algorithms, separately from the neural pilot. Their integration is the next development stage.

## Run

Python 3.12 is the recorded environment. The numerical experiments require NumPy; the neural core uses PyTorch. Installation, CPU reproduction, CUDA commands, and profiling are in [RUNNING.md](RUNNING.md).

```bash
python experiments/continual_grid/run_experiment.py --output-dir runs/continual-grid
python experiments/neural_pilot/train.py --device cpu --output-dir runs/neural-pilot
```

Each command writes configuration and raw measurements to its run directory. Archived measurements remain alongside their experiment.

## Research map

| Document | Contents |
| --- | --- |
| [RESEARCH.md](RESEARCH.md) | Objectives, proactivity, continuous operation, learning, applications |
| [DESIGN.md](DESIGN.md) | State, model mechanisms, interfaces, and execution |
| [TRAINING.md](TRAINING.md) | Data, losses, activity learning, adaptation, GPU training |
| [EXPERIMENTS.md](EXPERIMENTS.md) | Current studies, primary comparisons, instrument task, scaling |
| [RUNNING.md](RUNNING.md) | Executable commands and next implementation steps |
| [ref/](ref/README.md) | 118 source records, reading depth, and closest related mechanisms |

Target applications include robotics, electromechanical control, autonomous experiments, edge devices, and digital research. They require timely decisions, persistent context, and learning across continued interaction. [Eidolve](https://github.com/Cherish0925/eidolve) is a related knowledge and memory project; CDM can also connect to other tool and memory environments.

Original code and documentation use the [MIT License](LICENSE). External papers and implementations retain their own licenses. [CITATION.cff](CITATION.cff) identifies this project; include the commit used when reporting an experiment.
