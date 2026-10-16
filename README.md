<p>
  <img src="assets/readme-header.png" alt="Continual Decision Model — Continual operation, decision, and evolution." width="1280">
</p>

[Research](RESEARCH.md) · [Model](DESIGN.md) · [Experiments](EXPERIMENTS.md) · [Run](RUNNING.md) · [References](ref/README.md)

**CDM studies how a model can keep perceiving, reasoning, acting, and learning while the world changes.** Internal state persists across interactions. The model learns external actions and the choice of its next observation or computation. A conversation is one event in that continuing process.

An observation can become outdated while inference is still running. The research question is how to preserve useful computation, incorporate new observations, and spend additional computation where it improves subsequent decisions.

## Model

Two implementations share the CDM state and operations.

- **Sequence core** ([cdm/model.py](cdm/model.py)): a GRU context over observations, goals, environment descriptors, previous actions and elapsed time; a workspace of latent slots refined by one shared Transformer block; a bounded record bank that stores intermediate work with its generating context, goal, observation-prefix index, parameter version and depth; conditioned reading with slotwise reuse, revision and initialization gates; action, activity and value heads.
- **Spatial planner** ([cdm/planner.py](cdm/planner.py)): a convolutional workspace over a grid, refined by one gated block and read out as an action and a distance for every cell. After a change it can restart, carry, retrieve stored workspaces, or read them with learned attention; it can read its workspace at the goal cell, rebuild channels through a learned keep-or-rebuild gate, split its workspace into map and goal parts, and read the age of every cell of a partial map.
- **Continual agent** ([experiments/continual_agent](experiments/continual_agent/README.md)): the spatial planner with an information state (the last seen map and the age of each cell) and an activity head that chooses each tick how many refinements to run, whether to request a full observation and whether to move, trained on reward minus computation and observation costs. Computation it leaves unused pays for online learning of the planner.

[DESIGN.md](DESIGN.md) gives the equations and interfaces; [research/CONTINUAL_COMPUTATION.md](research/CONTINUAL_COMPUTATION.md) develops the central hypothesis and summarises what has been measured.

## Implemented and measured

| Study | Implementation | Result |
| --- | --- | --- |
| [Neural pilot](experiments/neural_pilot/results/REPORT.md) | Five seeds, three model variants, supervised sequential contextual recall | Query accuracy: memory **56.20%**, persistent workspace **36.03%**, restarted workspace **38.44%** |
| [Dynamic grid](experiments/continual_grid/RESULTS.md) | Five seeds, three change regimes, two computation budgets; world advances during planning | Reuse helps at low budgets; this local factorization does not outperform nearest-context retrieval |
| [Numerical transport](experiments/transport/RESULTS.md) | Changing quadratic objectives and finite-depth state reconstruction | Tested correction loses to strong solvers; reconstruction quality and task quality differ |
| [Changing maze](experiments/dynamic_maze/RESULTS.md) | Neural latent planners act every tick while doors, obstacles and goals change; 14 variants with five seeds and 3 partial-view variants with three; shared budgets | A read of the workspace at the goal cell lets a goal change reach every cell in one refinement (affected cells right: 92.6% against 43.0% for carry); a learned keep-or-rebuild gate makes 83% right before any refinement without a hand-made split; at the training size the best planner earns the optimal policy's return within 0.01 with one refinement per tick |
| [Continual agent](experiments/continual_agent/RESULTS.md) | One running process over a stream of changing sites: partial view, asynchronous surveys, requests with idle periods; PPO-trained choice of refinements, surveys and moves; learning from unused compute | The learned activity head matches the best hand-written schedule with 24–54% fewer refinements, bursting computation when a request arrives and surveying more as its map ages; anchored online learning improves a deployed planner on larger mazes, learning toward its own live output collapses |

The neural pilot trains action prediction with a fixed memory-write schedule. The changing-maze planners and the continual agent run on GPUs; their training, evaluation and analysis commands are in the [maze](experiments/dynamic_maze/README.md#run) and [agent](experiments/continual_agent/README.md#run) READMEs.

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
| [DESIGN.md](DESIGN.md) | State, model operations, interfaces, and execution |
| [TRAINING.md](TRAINING.md) | Data, losses, activity learning, adaptation, GPU training |
| [EXPERIMENTS.md](EXPERIMENTS.md) | Current studies, primary comparisons, instrument task, scaling |
| [RUNNING.md](RUNNING.md) | Executable commands and next implementation steps |
| [ref/](ref/README.md) | 122 source records, reading depth, and closest related methods |

Target applications include robotics, electromechanical control, autonomous experiments, edge devices, and digital research. They require timely decisions, persistent context, and learning across continued interaction. [Eidolve](https://github.com/Cherish0925/eidolve) is a related knowledge and memory project; CDM can also connect to other tool and memory environments.

Original code and documentation use the [MIT License](LICENSE). External papers and implementations retain their own licenses. [CITATION.cff](CITATION.cff) identifies this project; include the commit used when reporting an experiment.
