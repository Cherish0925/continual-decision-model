# Sequential contextual recall

This experiment trains the neural model in [`cdm/`](../../cdm/) on a small sequential action-prediction task. It measures whether observations can be retained and used across events, and whether conditional computational memory helps under a fixed number of training updates.

## Task

Every episode independently assigns one of four actions to each of four contexts. Its first four events reveal those assignments in a random context order. Eight subsequent events request actions in randomly selected contexts without showing the cue again. A binary goal selects either the revealed action or its cyclic successor. Thus the label at every event can be computed entirely from observations already received in that episode.

The model receives a five-dimensional observation (four cue channels and a visibility bit), a four-dimensional context descriptor, and a two-dimensional goal descriptor. During query events all five observation channels are zero. It must use its recurrent state or computational memory to recover the appropriate cue. Random-choice query accuracy is 25%.

Training and evaluation use separate pseudorandom streams. Evaluation uses the same 512 held-out episodes for every method and seed. These episodes contain newly sampled sequences; this is an in-distribution recall task, not a new-context or new-goal generalization experiment.

## Three ablations

| Method | Recurrent context | Workspace between events | Computational memory |
|---|---|---|---|
| `restart_workspace` | Retained | Initialized from current context | Bypassed |
| `persistent_workspace` | Retained | Retained | Bypassed |
| `computational_memory` | Retained | Retained and contextually revised | Read each event; written on four cue events |

All methods receive fresh observations and goals before their action readout. They instantiate the same architecture and start from identical parameters for a given seed. Bypassed parameters are inactive, so equal total parameter counts do not imply equal active computation. Memory writing follows the visible-cue schedule supplied by this experiment. The activity policy does not select that schedule.

The model uses a 32-dimensional GRU context, four persistent slots, two calls to one shared Transformer refinement block per event, and eight memory records. Cross-entropy supervises the action readout after each refinement. Gradients cover the complete twelve-event episode, including memory writes and later reads. Parameters update once per batch; model state resets between episodes. Activity and value heads remain untrained.

## Run

From the repository root, install the neural package extra. The CPU-only PyTorch installation is described in [`RUNNING.md`](../../RUNNING.md).

```bash
pip install -e '.[neural]'
python experiments/neural_pilot/train.py --device cpu --output-dir runs/neural-pilot
```

The defaults run five seeds, three methods, 120 updates per run, batch size 32, and 512 evaluation episodes. CPU training uses one PyTorch thread. With a CUDA-enabled PyTorch installation:

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 python experiments/neural_pilot/train.py --device cuda --output-dir runs/neural-pilot-cuda --save-checkpoints
```

The script sets this deterministic cuBLAS configuration if it is absent. `--save-checkpoints` writes trained weights alongside measurements. The command creates its own configuration, per-run metrics, aggregate results, and a generated Markdown report. It does not overwrite archived measurements unless their directory is explicitly selected.

```bash
python -m unittest discover -s experiments/neural_pilot -p 'test_*.py'
```

The data test reconstructs every requested action from the observed cue history and checks that query observations contain no action cue.

## Results and scope

See the archived [report](results/REPORT.md), [configuration](results/config.json), and [per-seed results](results/results.json). Training updates, episode batches, and refinement counts are matched. Reading and writing memory add work, so the experiment reports training time and does not claim an equal-wall-time comparison.

A recall improvement would establish that this implementation can train and use stored latent records on this task. The dynamic-world experiment must separately measure action quality while the world advances, the work needed to resume a useful computation, and the complete observation-to-action latency. Online parameter learning and background consolidation require their own training runs.
