# Run and extend CDM

## Install

```bash
git clone https://github.com/Cherish0925/continual-decision-model.git
cd continual-decision-model
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

This installs the NumPy dependency for the analytical experiments. For the recorded CPU neural environment:

```bash
python -m pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e '.[neural]'
```

For GPU work, first install the PyTorch build that matches the local CUDA setup using the [official installer](https://pytorch.org/get-started/locally/), then install the neural extra. The changing-maze figures also need `python -m pip install -e '.[plots]'`. The CPU studies used Python 3.12, NumPy 2.3.5, and PyTorch 2.14.1+cpu. The changing-maze study used Python 3.12.13, NumPy 2.5.3, PyTorch 2.11.0+cu128, and matplotlib 3.10.8 on two H100 GPUs (driver 595.71.05).

## Reproduce the studies

### Dynamic planning

```bash
python experiments/continual_grid/run_experiment.py --output-dir runs/continual-grid
```

Defaults: five seeds, three change regimes, two cell-backup budgets, 192 decisions per episode. Each computation block advances the world; actions use fresh observations. The run contains 150 tested-planner episodes plus 30 current-map reference episodes. Configuration, episode/phase rows, decision traces, and summaries are written under the selected output directory. Read the [experiment description](experiments/continual_grid/README.md) and [recorded results](experiments/continual_grid/RESULTS.md).

### Neural training

```bash
python experiments/neural_pilot/train.py --device cpu --output-dir runs/neural-pilot
```

Defaults: five seeds, three methods, 120 optimizer updates per run, 32 episodes per batch, and 512 held-out episodes. The full twelve-event sequence participates in backpropagation. Action readouts are supervised after both shared Transformer refinements. Activity and value heads are not trained by this loss.

To run on one GPU and retain weights:

```bash
python experiments/neural_pilot/train.py --device cuda --save-checkpoints --output-dir runs/neural-pilot-cuda
```

Independent seeds can run on separate GPUs with distinct output directories. For example, assign `CUDA_VISIBLE_DEVICES=0` and `--seeds 11` to one process, and another device and seed to another. This is independent-seed parallelism. A distributed optimizer is not part of the current pilot.

The [pilot report](experiments/neural_pilot/results/REPORT.md) contains the actual CPU results. This small task is a training baseline; increase task difficulty and connect the environment clock before allocating a large training run.

### Changing maze on GPUs

```bash
python experiments/dynamic_maze/launch.py --root runs/dynamic_maze/main --per-gpu 8 --min-free-gb 16 --stagger 120 \
    --extra --steps 3000
python experiments/dynamic_maze/evaluate_all.py --root runs/dynamic_maze/main --out runs/dynamic_maze/eval --per-gpu 4 \
    --extra --rooms 5 7 9 --think 1 2 4 8 --episodes 1024 --batch 1024
```

The launcher trains every variant for five seeds, starts a run only when its GPU has the requested free memory, and skips finished runs and runs already in progress. Each run writes `config.json`, `train_log.jsonl`, and `model.pt` under `runs/dynamic_maze/main/<variant>/seed<k>/`. Evaluation, latency, the controlled-change probe, shared-budget allocation, action fields, and the report are listed in the [study README](experiments/dynamic_maze/README.md#run); the measurements are in [RESULTS.md](experiments/dynamic_maze/RESULTS.md).

### Numerical state revision

```bash
python experiments/transport/run_experiment.py --output-dir runs/transport-local
```

This preserves the earlier polynomial transport study. Fixed-work numerical outputs should agree within floating-point tolerance; CPU timings and latency-calibrated iteration counts depend on the machine. Its original raw data and interpretation remain under [experiments/transport/](experiments/transport/README.md).

## Measure the complete neural step

```bash
python experiments/profile_core.py --device cpu --output runs/core-profile.json
python experiments/profile_core.py --device cuda --batch-size 32 --hidden-dim 128 --output runs/core-profile-cuda.json
```

The profiler measures assimilation, workspace initialization or memory reading, shared refinement, output, and memory writing. It compares restart, persistent workspace, and memory modes with one model. It records per-step times, p50/p95/p99, parameter count, persistent-state bytes, software versions, and CUDA allocation where available.

Inputs are already on the device and weights are untrained. These measurements isolate model execution; sensor capture, transfer, environment execution, and actuator latency must be added in a deployment experiment. CUDA measurements synchronize around each complete step. The [recorded CPU profile](experiments/profiles/cpu_20261015.json) is a small-batch implementation measurement.

## Check model and environment behavior

```bash
python -m unittest discover -s tests -p 'test_*.py'
python -m unittest discover -s experiments/continual_grid -p 'test_*.py'
python -m unittest discover -s experiments/neural_pilot -p 'test_*.py'
```

These checks cover persistent computation, fresh-context actions, batched memory and gradients, observation-prefix metadata, parameter-version selection, world-time advancement, planning budgets, and labels reconstructed from observed cue history.

## Next implementations

| Next implementation | Result to produce |
| --- | --- |
| Learned premises beyond map and goal | The keep-or-rebuild gate of the maze planner applied to workspaces whose inputs have no fixed spatial split, and an account of which channels depend on which input |
| Learning as an activity | The continual agent's activity head also decides when to spend compute on learning, with the planner and the head trained together |
| Learning under a partial view | Anchored online learning that improves the partially observing agent's reward: more recorded experience, hindsight from every survey, replay across sites |
| Activity policy in the sequence core | The continual agent's learned choice of computation, observation and action applied to the GRU/Transformer core on non-spatial tasks |
| Richer computational reuse | Per-slot reading and revision compared with the existing shared record reader |
| Continual learning across tasks | Online updates and background replay across task sequences, measured as learning speed, retention and transfer |
| Second domain | Nonlinear instrument with delayed, action-dependent observations |
| Scale | Measured GPU execution, larger tasks, longer sequences, distributed training |

Record GPU model/count, memory, interconnect, driver, CUDA, PyTorch version, environment throughput, and model-step timing at setup. Profile state length and environment batch size before choosing a distributed layout. [TRAINING.md](TRAINING.md) specifies the training stages and data fields; [EXPERIMENTS.md](EXPERIMENTS.md) defines the comparisons.

Retain source, configurations, raw small-run measurements, and analysis in GitHub. Large datasets and checkpoints should have versioned locations and hashes linked from their experiment. Use a separate output directory for each retained run.
