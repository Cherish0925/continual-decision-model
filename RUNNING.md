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

For GPU work, first install the PyTorch 2.14 build appropriate to the server's CUDA setup using the [official installer](https://pytorch.org/get-started/locally/), then install the neural extra. The completed runs used Python 3.12, NumPy 2.3.5, and PyTorch 2.14.1+cpu. GPU latency and distributed throughput have yet to be measured.

## Reproduce the three studies

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

## Continue on the server

| Next implementation | Result to produce |
| --- | --- |
| Neural dynamic navigation | Trained recurrent, workspace, snapshot, and conditional-memory policies on the same changing environment |
| Operation-value learning | Paired rollouts that measure the return from another observation or computation |
| Activity policy | Learned observation/computation/action selection under elapsed time |
| Richer computational reuse | Per-slot reading and revision compared with the existing shared record reader |
| Continual learning | Online parameter updates and background replay across task sequences |
| Second domain | Nonlinear instrument with delayed, action-dependent observations |
| Scale | Measured GPU execution, larger tasks, longer sequences, distributed training |

Record GPU model/count, memory, interconnect, driver, CUDA, framework version, environment throughput, and model-step timing at setup. Profile state length and environment batch size before choosing a distributed layout. [TRAINING.md](TRAINING.md) specifies the training stages and data fields; [EXPERIMENTS.md](EXPERIMENTS.md) defines the comparisons.

Retain source, configurations, raw small-run measurements, and analysis in GitHub. Large datasets and checkpoints should have versioned locations and hashes linked from their experiment. Use a separate output directory for each retained run.
