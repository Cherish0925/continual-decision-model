# Continue on a research server

## Repository state

| Component | State |
| --- | --- |
| Research, architecture, training, and experiment specifications | Written |
| Literature collection | 108 sources with recorded reading depth |
| Quadratic transport experiment | Implemented and run on CPU |
| Neural CDM and instrument environment | To implement |
| GPU kernels and distributed training | To implement and profile |
| Multimodal, tool-use, and long-term learning experiments | Planned |

Start with [RESEARCH.md](RESEARCH.md), [DESIGN.md](DESIGN.md), [TRAINING.md](TRAINING.md), and [EXPERIMENTS.md](EXPERIMENTS.md). [research/COMPUTATION_TRANSPORT.md](research/COMPUTATION_TRANSPORT.md) derives the current revision candidate.

## Run the available experiment

Use Python 3.12 for the recorded dependency configuration.

```bash
git clone https://github.com/Cherish0925/continual-decision-model.git
cd continual-decision-model
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r experiments/transport/requirements.txt
python experiments/transport/run_experiment.py --output-dir runs/transport-local
```

The output directory contains a generated `RESULTS.md` and a `results/` directory with fitted coefficients, raw measurements, configuration, and runtime metadata. Choose a different directory for another retained run.

The original reference data remains under `experiments/transport/results/`. Fixed-work numerical results should agree within floating-point tolerance. CPU timings and the latency-calibrated iteration budget vary with the machine.

## Build the neural experiment

Implement this sequence, keeping each stage runnable:

1. **Environment and time.** Add the nonlinear instrument, observation delays, action application, and computation-duration handling. Run simple controllers and save trajectories.
2. **Persistent baseline.** Implement observation encoding, a recurrent policy, prediction heads, fixed-depth inference, and recurrent training.
3. **Workspace and revision.** Add shared Transformer refinement and the candidate revision alongside restart, initializer, ordinary recurrence, and action-only correction.
4. **Active observation and computation.** Train activity selection under the same task and compute budgets.
5. **Continual learning.** Add retained experience, replay, parameter updates, and evaluation across task sequences.
6. **Transfer and scale.** Add a second domain, multimodal encoders, longer runs, and distributed training.

A compact source layout for those implementations is `cdm/` for model and learner, `environments/` for interaction, `configs/` for run definitions, and `experiments/` for evaluation. Add modules as they become executable.

The first complete neural run should contain a trained baseline and candidate, five independent seeds, held-out dynamics, action-dependent observations, and measured observation-to-action timing. The cluster plan in [TRAINING.md](TRAINING.md) then distributes training, collection, and evaluation.

## Retained research decisions

- Keep the full target: persistent perception, proactive activity, continuing reasoning, and learning across tasks.
- Treat the recurrent backbone as an implementation choice to compare.
- Train ongoing computation for subsequent task outcomes. Same-depth state reconstruction is auxiliary.
- Compare persistent revision with ordinary recurrence, learned initialization, and action-only correction.
- Preserve negative numerical results and source attribution alongside new experiments.
- Record each implemented model and measured result at the commit that produced it.

The research specification includes offline consolidation and learning-operation selection. Their implementation follows the fixed-parameter and ordinary-update comparisons so their effects can be measured.

## Hardware and run records

At setup, record GPU models and count, device memory, interconnect, CPU/RAM, driver, CUDA, Python, framework version, and available simulator throughput. Fit sequence length and environment batch size to measured memory. Use the distributed strategy described in [TRAINING.md](TRAINING.md) after profiling the actual implementation.

Retain source, configurations, analysis, and result summaries in GitHub. Store large datasets and checkpoints with versioned locations and hashes referenced from the corresponding experiment.
