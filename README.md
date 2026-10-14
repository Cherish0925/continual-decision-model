# Continual Decision Model

**CDM studies how a model can keep perceiving, reasoning, acting, and learning while the world changes.** It maintains internal state across interactions and learns both external actions and the choice of its next observation or computation. A conversation is one event in that continuing process.

The central problem is time: an observation can become outdated while the model is still computing. CDM investigates how to update ongoing inference, act on current observations, and spend additional computation where it improves subsequent decisions.

## Current work

The repository contains the research specification, a 108-source literature collection, a concrete state-update proposal, and a reproducible CPU numerical experiment. The neural CDM, closed-loop environments, and distributed training are specified for implementation.

The numerical experiment found that its polynomial state correction underperformed strong optimization baselines. It also showed why reproducing an intermediate state is a poor primary training target: that state can still be far from a good decision. The next model will train continuing computation for task outcomes.

## Model

A recurrent encoder maintains the current context. A persistent latent workspace supports repeated Transformer computation. Incoming observations update both; a fast action policy reads current context and completed workspace updates. An activity policy selects observations, computation, retrieval, subgoals, and actions. Prediction and learning connect those choices to their consequences.

The first mechanism under study jointly learns the workspace representation, its refinement, and a small observation-conditioned revision. [DESIGN.md](DESIGN.md) specifies the complete architecture; [the mechanism note](research/COMPUTATION_TRANSPORT.md) derives this particular update.

## Read and run

| Document | Content |
| --- | --- |
| [RESEARCH.md](RESEARCH.md) | Objectives, proactivity, continuous operation, learning, and applications |
| [DESIGN.md](DESIGN.md) | State, computation, model interfaces, and execution |
| [TRAINING.md](TRAINING.md) | Trajectory data, learning objectives, continual learning, and multi-GPU training |
| [EXPERIMENTS.md](EXPERIMENTS.md) | Environments, comparison models, measurements, and scaling studies |
| [RUNNING.md](RUNNING.md) | Server setup, runnable experiment, and implementation order |
| [ref/](ref/README.md) | Related work and source records |
| [Experiment results](experiments/transport/RESULTS.md) | Recorded numerical results and their interpretation |

Reproduce the existing numerical experiment with Python 3.12:

```bash
python -m pip install -r experiments/transport/requirements.txt
python experiments/transport/run_experiment.py --output-dir runs/transport-local
```

The command fits small numerical models and writes a new run directory. GPU model training is the next implementation stage.

## Applications and connections

Target applications include robotics, electromechanical control, autonomous experiments, edge devices, and digital research. They share a need for timely decisions, persistent context, and learning across continued interaction.

[Eidolve](https://github.com/Cherish0925/eidolve) is a related knowledge and memory project. CDM's model interfaces allow other tool and memory environments as well.

Original code and documentation use the [MIT License](LICENSE). External papers and implementations retain their own licenses. [CITATION.cff](CITATION.cff) identifies this project; include the commit used when reporting an experiment.
