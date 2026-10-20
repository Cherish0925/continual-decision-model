# Run and extend CDM

## Multimodal input and state path

Install the optional perception and vision-language dependencies (Transformers 5.4.0; PyTorch and Torchvision must be a compatible pair):

~~~bash
python -m pip install -e '.[multimodal]'
~~~

Run the selected pretrained image/text towers through the new CDM state path:

~~~bash
python examples/multimodal_forward.py --image path/to/observation.png --instruction "Measure the temperature before changing the heater." --device cuda
~~~

Omit the image argument for a text/numeric event. The example loads the pinned SigLIP checkpoint. Add a lifetime checkpoint and the native semantic model to reuse trained weights:

~~~bash
python examples/multimodal_forward.py --core-checkpoint path/to/model.pt --semantic --image path/to/observation.png --instruction "Read the instrument and propose the next measurement." --device cuda
~~~

Replace the checkpoint path with a model extracted from [v0.4.0](https://github.com/Cherish0925/continual-decision-model/releases/tag/v0.4.0). The loader retains its original dimensions and action schema. Without that argument the reference core is randomly initialized. The semantic flag uses the pinned Qwen3.5-2B processor and weights; it encodes the supplied image/text and compresses its final hidden states to eight tokens. The example runs sequentially; asynchronous scheduling is separate work.

New modality projections and the semantic reader need task training. Numeric values, goals and experience records in this example are placeholders; output reports tensor shapes, not task performance. The 4,096-token semantic input budget includes image tokens. For GPU throughput, install compatible flash-linear-attention and causal-conv1d kernels following the [model implementation documentation](https://huggingface.co/docs/transformers/model_doc/qwen3_5); the CPU implementation also runs without them.

Use [TRAINING.md](TRAINING.md#first-multimodal-training-route) for trajectories, bridge training, teacher transfer and joint optimization.

The tests below use small actual Siglip and Qwen3.5 modules and the released checkpoint structure; they do not download model weights:

~~~bash
python -m unittest discover -s tests -p 'test_multimodal.py' -v
python -m unittest discover -s tests -p 'test_pretrained_integration.py' -v
python -m unittest discover -s tests -p 'test_activity_value_gradient.py' -v
python -m unittest discover -s tests -p 'test_lifetime*.py' -v
~~~

The existing lifetime training commands below continue to use structured events. The multimodal environment/trajectory adapter and joint trainer are not connected by the forward example.



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

### Continual model over process lifetimes

```bash
python experiments/lifetime/launch.py --output-root runs/lifetime/c4 --seeds 0 1 2 3 4 \
    --variants full "full_meta=full --meta 0.5" frozen fixed_learn world_model predict_only recurrent -- \
    --streams 1024 --width 1024 --hidden 1024 --interactions 2e7 --lifetime 16384
python experiments/lifetime/evaluate.py --model runs/lifetime/c4/full/seed0/model.pt --also frozen discard \
    --devices 512 --ticks 16384 --output runs/lifetime/eval/full_seed0.json
python experiments/lifetime/realtime.py --model runs/lifetime/c4/full/seed0/model.pt --output runs/lifetime/realtime.json
```

One model, [`cdm/core.py`](cdm/core.py), acts, measures, refines and learns over instrument and thermal process lifetimes; settings switch its learned activities and its learning on and off. The trainer, evaluation settings, two-family runs, timing measurements and the controller against an independently advancing process are described in the [experiment README](experiments/lifetime/README.md); results are in [RESULTS.md](experiments/lifetime/RESULTS.md). One training run of 2·10⁷ transitions at 35.9 M parameters saturates an H100 at about 11,000 transitions per second.

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

## Next integrated implementation

The continual core acts and learns throughout changing process lifetimes ([experiment](experiments/lifetime/README.md)). The selected multimodal extension is specified in [DESIGN.md](DESIGN.md), training budgets and objectives in [TRAINING.md](TRAINING.md), and comparisons in [EXPERIMENTS.md](EXPERIMENTS.md#integrated-model-campaign).

| Implementation | State |
| --- | --- |
| Persistent model and event interface | GRU fast state, a persistent Transformer workspace over experience records, action-conditioned prediction with instance adapters, control and activity heads ([`cdm/core.py`](cdm/core.py)). Retrieval of computation records, subgoals and language remain open |
| Lifetime trainer | Recurrent PPO over lifetimes whose state, experience and adapters persist across windows; trained choice of measurement, refinement and learning; optional training through the learning step |
| Execution | Activities with simulated durations in training; [`realtime.py`](experiments/lifetime/realtime.py) runs control against processes on their own clock with learning on a separate stream |
| Shared task adapters | Instrument and thermal families on one event and action interface, trained jointly; navigation and delayed digital tools remain |
| Main campaign | Seven settings with five seeds each, held-out dynamics, acquisition and retention curves, and two-family runs; external baselines and scale comparisons remain |

Develop these against one common state/event/action interface. Task adapters and timed execution can be implemented while the model and trainer are developed. Existing maze retraining is a comparison workload, not a prerequisite for the core integration. Implement one reference recurrence first; additional backbones should answer a measured limitation of that reference.

Record GPU model/count, memory, interconnect, driver, CUDA, PyTorch version, environment throughput, and model-step timing at setup. Separate collection parallelism from backpropagation batch size and single-instance deployment latency. The research campaign uses measured throughput to allocate the available devices.

Retain source, configurations, raw measurements, and analysis in GitHub. Large datasets and checkpoints have versioned locations and hashes linked from their experiment. Use a separate output directory for each run and retain unsuccessful training attempts alongside completed ones.
