# Lifetime: one model that keeps acting, measuring, computing and learning

A single model runs each process for a whole lifetime. It keeps one state throughout, chooses at every tick how to act, whether to request a precise measurement, and how to spend the compute until the next tick: on refinements of its latent workspace or on a learning step that updates its own adapter. These activities have simulated durations and control is produced every tick; [`realtime.py`](realtime.py) also runs the controller against processes that advance on their own wall clock. This is the integrated campaign of [EXPERIMENTS.md](../../EXPERIMENTS.md#primary-comparisons): the same model under settings that switch its learned activities and its learning on and off, then one set of weights across two process families. A second round trains the model through the process it is deployed in: its learning steps as they were taken, its workspace across events, its activities by their consequences on copies of the same process ([Training through deployment](#training-through-deployment)), on instruments that also ask for their largest output and, with a third family of vehicles behind walls, on three families at once.

## Processes

[`cdm/instrument.py`](../../cdm/instrument.py) holds what every family shares (`ProcessFleet`): a lifetime has four operating conditions; the active one changes with probability 1/400 per tick, one change in five brings a condition never seen before and the others return to an earlier one, and nothing announces a change. Requests start with probability 1/40 per idle tick and stay active for 60 ticks; a setpoint is the steady output of one of the actions under the current condition. A cheap reading arrives every tick with noise 0.25; a precise measurement can be requested, one at a time, and returns the output at the request time after three ticks with noise 0.02. Two actuators with nine levels each give 81 actions. Every family advances in parallel on the GPU.

**Instruments** (`InstrumentFleet`) relax toward a peaked response of the actuator positions, as when aligning a beam onto a detector; the actuators move with backlash:

\[
r(v)=a\,\exp\!\Bigl(-\frac{\lVert v-c\rVert^2}{2w^2}\Bigr)+b+\delta_t,\qquad
y_{t+1}=y_t+\frac{r(v_t)-y_t}{\tau}+\epsilon_t .
\]

| Quantity | Range |
| --- | --- |
| Height \(a\), peak position \(c\), width \(w\), floor \(b\) | 0.8–1.5; [−0.7, 0.7]²; 0.25–0.6; −0.3–0.1 |
| Time constant \(\tau\), actuator backlash | 2–8 ticks; up to 0.15 |
| Drift \(\delta_t\) | random walk, 0.004 per tick, reset when the condition changes |

Setpoints come from actions where the peak is visible.

**Thermal processes** ([`cdm/thermal.py`](../../cdm/thermal.py), `ThermalFleet`) heat a block that exchanges heat with a body; the body loses heat to its surroundings, faster while coolant flows, and its temperature is the output:

\[
C_1\dot T_1=\eta h^\gamma-k_{12}(T_1-T_2)-k_1(T_1-T_a),\qquad
C_2\dot T_2=k_{12}(T_1-T_2)-k_2(1+\kappa c)(T_2-T_a).
\]

The first actuator sets the heater command \(h\), which acts after a transport delay of up to two ticks; the second sets the coolant flow \(c\). A condition draws \(k_{12}\in[0.1,0.3]\), \(k_1\in[0.02,0.06]\), \(k_2\in[0.06,0.15]\), \(\kappa\in[0.5,2]\), \(\gamma\in[0.5,2]\), \(T_a\in[-0.4,0.2]\), block and body time constants of 1–3 and 3–8 ticks, the delay, and a steady rise of 0.9–1.6 at full heat without coolant, which fixes \(\eta\). The ambient drifts like the instrument offset. From ambient, full heat reaches 90 % of its rise in a median of 37 ticks.

**Requests for the largest output** (`--search-share`, 0.5 in the second round): a share of instrument and vehicle requests do not give a setpoint but ask for the largest output the process can hold, without saying what it is; the setpoint reads zero and a flag marks the request. The error is the shortfall from the largest steady output over the actions. The goal term of the control then measures each action's predicted reading against the largest predicted reading.

**Vehicles** ([`cdm/navigation.py`](../../cdm/navigation.py), `NavigationFleet`) move in the square \([-1,1]^2\). An action names one of the 81 grid points; the vehicle heads for it in a straight line at the speed of its condition and stops just short of the first wall in its way, so reaching a point behind a wall takes several waypoints. Its position is the actuator state it reports. A beacon emits a field that falls with the length \(d\) of the shortest path around the walls, computed on a 25 × 25 grid with moves to the eight neighbours:

\[
y = a\,\exp\!\Bigl(-\frac{d(x)^2}{2w^2}\Bigr) + b + \delta_t .
\]

| Quantity | Range |
| --- | --- |
| Height \(a\), width \(w\), floor \(b\) | 0.8–1.5; 0.5–1.0; −0.3–0.1 |
| Beacon | [−0.85, 0.85]² |
| Walls | two axis-parallel segments of length 0.8–1.4 at [−0.6, 0.6] |
| Speed | 0.06–0.15 per tick; position noise 0.005 |

Setpoints are field values at grid points where the field stands clear of its floor; holding there would read them.

`MixedFleet` runs families in fixed blocks of streams; the model is not told which family a stream belongs to. `--family mixed` gives half the streams to instruments and half to thermal processes, `--family all` a third each to instruments, thermal processes and vehicles.

Reward per tick is minus the absolute tracking error while a request is active, minus 0.002 per refinement, 0.01 per precise measurement and 0.02 per learning step.

[`references.py`](references.py) gives two reference policies per family ([results](results/references.json)): random actions, and a controller that knows the true steady response and takes the action whose steady output is closest to the setpoint. The second ignores the dynamics, so a policy that anticipates relaxation, backlash or delay can do better; it is a reference point, not a bound.

| Family | Random actions | True-steady greedy |
| --- | ---: | ---: |
| Instrument | 0.305 | 0.115 |
| Thermal | 0.269 | 0.091 |
| Instrument, time constant 8–16, width 0.15–0.25 | 0.389 | 0.224 |
| Instrument, cheap noise 0.4 | 0.305 | 0.115 |
| Instrument, a change every 150 ticks, half of them new | 0.311 | 0.113 |

## Model

[`cdm/core.py`](../../cdm/core.py), one network for every setting:

| Part | Content |
| --- | --- |
| Fast state | Event features (readings, setpoint, request flags, previous control and activities, pending measurement, time since the last measurement, last prediction error) → MLP → GRU |
| Workspace | 8 latent slots; a refinement applies one shared Transformer block (slot self-attention, attention to the fast state and the 32 most recent experience records, MLP) and normalises each slot, so the workspace keeps one scale over any number of refinements |
| Predictor | Mean and log-variance of the next reading for every action, from the fast state, the slot mean and learned action features. Its first layer carries an instance-owned rank-8 adapter |
| Control | Scores every action from the shared trunk, the action's features and the action's own predicted reading relative to the setpoint, with weights shared across actions, minus a learned multiple of the action's predicted distance from the setpoint |
| Activities | Precise measurement (yes, no); compute (0, 1, 2 or 4 refinements, or a learning step) |
| Value | Critic over the shared trunk |

The campaign uses width 1,024 and 35.9 M parameters.

Options added for the second round:

| Option | Content |
| --- | --- |
| `gated_workspace` | Each refinement mixes, per slot, the slot as it was, its revision and the learned initial slot, with weights from a softmax over the slot and its revision; the mixture is then normalised. The gate starts close to always revising |
| `sensor_noise` | Predictions describe the noise-free output; a cheap reading adds a learned noise initialised at 0.25, a precise reading one initialised at 0.02, so a precise reading weighs about 150 times more in a likelihood |
| `step_rule="evidence"` | A learning step moves the adapter by \(-\eta\,g/(\lVert g\rVert+\kappa)\), with \(\eta\) and \(\kappa\) learned: proportional to the gradient while it is small against \(\kappa\), of bounded size once it is large |
| `adapter_blocks` | The adapter is a set of rank-8 blocks; a softmax over blocks read from the state weighs each block's projection, for every prediction and every buffer entry, so a block learns from the experience it was selected for and can hold an operating condition until it returns |
| `activity_values` | A head that estimates from the trunk how much each activity option is worth against the others |

### Time

Control is produced every tick from the fast state, the workspace completed so far and the predictions under the current adapter. The compute between one tick and the next goes to the chosen option. Refinements are read from the next tick on. A learning step samples 32 of the instance's last 64 ticks (state features, action, following reading), takes one normalised gradient step of size 0.05 on the adapter, and occupies the compute for two ticks: during the second the instance can neither refine nor start another step, and the updated adapter is used from the tick after it completes. A precise measurement arrives three ticks after the request while the process keeps moving.

[`realtime.py`](realtime.py) removes the simulated tick: the fleet steps on a wall-clock schedule in its own thread and CUDA stream, applies the latest action it has received, and queues every observation with the action it applied. Each actor cycle absorbs the queued observations in order, so the fast state, the experience records and the learning buffer follow every process step with the action actually applied, then publishes an action as soon as the control path finishes; refinements follow on its stream, and learning steps run on a learner thread and third stream whose adapter is installed when it returns. Because nothing on the control path waits, the workspace reads experience records one step older than in the simulated runtime.

## Training

Recurrent PPO over lifetimes ([`train.py`](train.py), [`runtime.py`](runtime.py)). 1,024 streams run lifetimes of 16,384 ticks with staggered starts. Fast state, workspace, experience records, adapters and learning buffers persist across 128-tick training windows; only the end of a lifetime resets them, and only that end is terminal for the value targets, which otherwise bootstrap from the next decision state.

For each window: four epochs over minibatches of 64 streams; backpropagation through the fast state over the window from its stored starting state, with each stream reset at its collected lifetime boundaries; every tick's refinement recomputed from the stored workspace before it, with workspace gradients extending to the next tick only; γ = 0.995, λ = 0.95, clip 0.2, entropy 0.003, rewards scaled by 0.1 for the critic, Adam at 3·10⁻⁴. The shared predictor is trained on the readings that followed each action (Gaussian likelihood, weight 0.5). Policy probabilities use the predictions cached during collection, which include the instance's adapter at that tick; PPO does not differentiate through those cached predictions. Each window logs its collection and update time; each run records the source revision.

| Setting | Measurement | Compute | Predictions |
| --- | --- | --- | --- |
| `recurrent` | learned | none | not supplied, no prediction loss |
| `predict_only` | learned | none | yes |
| `frozen` | learned | learned refinements; no learning | yes |
| `fixed_learn` | learned | a learning step every 8 ticks; refinements learned otherwise | yes |
| `full` | learned | learned: refinements or learning | yes |
| `world_model` | every 8 ticks | a learning step every 8 ticks, one refinement otherwise | yes |
| `full_meta` | as `full`, with a loss through the learning step | | yes |

Read top to bottom, each setting adds one ability: world prediction, learned workspace computation, online learning on a fixed schedule, and the choice of when to learn; `world_model` against `fixed_learn` separates learned from fixed allocation of observation and compute. `full_meta` trains the shared network through the learning step itself: from the adapter each stream held when the window began, one step on the first half of the window, then the prediction loss on the second half, with gradients through the step (weight 0.5, `--meta 0.5`). All settings share the network and the training procedure; settings that do not refine or predict leave those parts unused. Each setting has five seeds of 2·10⁷ interactions, 10⁸ per setting.

The two-family experiment trains `full`, `frozen` and `recurrent` on mixed lifetimes, half instrument and half thermal (`--family mixed`), and `full` on thermal lifetimes alone (`--family thermal`), three seeds each. `predict_hidden` trains the recurrent policy with the prediction loss of the predicting settings while the policy reads no prediction, which separates prediction supervision from prediction readout.

### Training through deployment

The first round left three gaps between what was trained and what ran: the predictor was trained without the adapters it ran with and the meta loss took one step per window where the agent took one every few ticks; the workspace's gradients stopped after one tick and the policy read cached predictions; activities were learned only from sampled returns. The second round closes each ([`objectives.py`](objectives.py)), with the configuration of the first round otherwise unchanged except minibatches of 128 streams.

- **Learning as deployed** (`--objective deployed`, `--support buffer`). A learning step reads every filled entry of the 64-entry buffer instead of 32 samples. In the PPO update the window's learning steps are replayed at the ticks they were chosen: from the adapter each stream held when the window began, including a step still running, each step reads the buffer as it stood then, rebuilt from the buffer at the window's start and the window's own transitions, and takes effect two ticks later. The loss of the prediction of every tick under the adapter then in effect trains the shared network, the action features and the step size through every step. Replayed adapters equal those of collection (test in [`tests/test_lifetime_objectives.py`](../../tests/test_lifetime_objectives.py)). `--objective step` keeps the first round's single step per window; `shared` trains the shared predictor alone.
- **Precise targets** (`--precise-targets`, `--sensor-noise`). A precise reading, when it returns, replaces the cheap reading of the transition it measured in the buffer, from the tick it returns, and in the prediction loss; likelihoods add the noise of each sensor.
- **Evidence-scaled steps** (`--step-rule evidence`).
- **Workspace across events** (`--segment 16`, `--gated-workspace`). PPO recomputes the workspace one tick after another from its stored value at the start of each 16-tick segment, so its gradients span up to 16 external events; segments run in parallel and each tick's refinement is checkpointed.
- **Predictions recomputed** (`--recompute-predictions`). The heads read the predictions of every action recomputed by the current network under the adapter in effect at each tick, with the predictor and adapters held fixed, so the policy loss reaches the fast state and workspace through the predictions it acts on.
- **Activities by their consequences** (`--branch-streams 128 --branch-horizon 32 --activity-values`). Once per window at a random tick, 128 streams are copied once per activity option: not measuring, measuring, and each compute option. Copies of one process draw the same disturbances, readings and condition changes, and the policy samples with the same noise in every copy, so copies differ only through the option and what follows from it. Each copy takes its option, with the other activity and the control chosen by the policy, then the policy runs for 32 ticks; the scaled discounted reward plus the discounted value after them is the option's value. Values are centred within each activity and scaled by their spread in the window. They train the activity-value head; at every tick the activity heads follow the expected advantage of every option under the head's estimates, with weight `--branch-weight` times the running share of variance of fresh copies the head predicts, so a head that predicts nothing has no influence. `--branch-weight 0` keeps the copies and the head but leaves the policy to PPO alone, at the same compute. [`branches.py`](branches.py) measures the same states' option values twice with independent randomness: values from one set of sampled 32-tick copies repeat with a correlation below 0.1, so the first configuration gives the head nothing to learn and its weight stays near zero. With the copies on the policy's most probable choices (`--branch-greedy`), a 4-tick horizon and four independent sets averaged (`--branch-replicates 4`), the values repeat with a correlation of 0.75; `cdm_b6` and `cdm_b6_ppo` compare following and not following them, at 17,920 simulated transitions per window.
- **Measured costs.** A learning step on the 64-entry buffer takes 1.82 times a refinement at 1,024 instances ([`latency.py`](latency.py)), so it costs 0.0036 where a refinement costs 0.002 (`--cost-learning 0.0036`); a precise measurement still costs 0.01.

| Variant | Differs from the first |
| --- | --- |
| `cdm` | `full` with every item above |
| `cdm_b6`, `cdm_b6_ppo` | greedy 4-tick copies, four sets averaged; followed, or kept but not followed |
| `cdm_truncated` | one-tick workspace gradients, no gate, cached predictions |
| `cdm_nolearn` | no learning (`frozen`) |
| `cdm_step_meta` | `--objective step` |
| `cdm_cheap_targets` | cheap targets only, no sensor noise |
| `cdm_bank` | four adapter blocks |
| `cdm_all`, `cdm_navigation`, `cdm_thermal` | three families; vehicles only; thermal processes only |
| `predict_only`, `predict_only_all` | predictions only, with recomputed predictions; on one or three families |
| `predict_hidden`, `recurrent` | as in the first round |

Every variant trains on instruments with half the requests for the largest output, with precise targets and sensor noise unless stated, 2·10⁷ interactions per seed; the sampled 32-tick copies add 29,568 simulated transitions per window.

## Evaluation

[`evaluate.py`](evaluate.py) runs each trained model through one uninterrupted lifetime per process on newly sampled processes, with nothing reset. It reports tracking error, reward, response time to a new setpoint, prediction error with the instance adapter and, on the same states, of the shared predictor alone, and activity rates; tracking error, prediction error, learning and refinement as functions of time since the last condition change, separately for new and returning conditions; and error across the lifetime. Tracking error and condition statistics use the request and condition under which the action was applied. Response time counts executed controls, starting at one; requests that expire without reaching the tolerance are assigned their duration, and requests still open when the evaluation ends are counted separately.

Each learning model is also evaluated under other settings with `--also`, every setting starting from the same random state so the comparisons use common random numbers:

- `discard`: chosen learning steps keep their time and cost but their adapter updates are thrown away; this isolates what the updates contribute.
- `frozen`: learning is switched off and its compute goes to the remaining options, so this measures removing learning together with the change of computation. A model trained on fixed schedules has untrained activity heads and is evaluated with `discard` only.
- `norefine`: the refinement options are removed, so compute goes to learning or nothing; this isolates what refinement contributes.
- `noidle`: refinement and learning are removed while no request is active; this isolates what computation between requests contributes to later requests.

Discarded learning steps are computed before their updates are thrown away, so `discard` matches the compute of the model's own setting. Errors are also reported separately for setpoint requests and requests for the largest output, activity rates separately for ticks with and without an active request, and for a bank of adapter blocks, the share of ticks on which each condition used its main block and the share of instances in which two conditions shared their main block.

`--family` evaluates a model on either family; `--instrument` changes fields of the process specification, for example `time_constant=8,16 width=0.15,0.25` for dynamics outside the training ranges. [`report.py`](report.py) aggregates seeds, converts refinements and learning steps into measured GPU time per instance and tick, and gives per-seed paired differences between settings.

[`probe.py`](probe.py) `readout` compares the policy with a greedy readout of the model's own predictions, with learning on and off. `decisions` follows one trajectory and compares the adapted predictor and the shared predictor alone with the noise-free next output of every action (`expected_next`): residual error for the action taken and over all actions, how often their greedy choices differ, and the one-step regret and steady-response gap of each choice. [`latency.py`](latency.py) times the control path, a refinement and a learning step for several numbers of instances, and response time from scheduled starts with a learner on a second stream; it repeats fixed inputs, so it measures contention for the GPU. [`realtime.py`](realtime.py) measures control of the advancing processes under the model's own setting, without learning, and without computation between requests: response time from the newest published observation to the published action, observations absorbed per decision, steps run on an action that did not answer the observation before them, tracking error, refinement and learning throughput over measured wall time, and the GPU's mean power and energy per process step.

The corrections of [issue #1](https://github.com/Cherish0925/continual-decision-model/issues/1) changed PPO state reconstruction at lifetime resets and evaluation accounting. Checkpoints trained before them are re-evaluated with the corrected metrics and reported separately ([results/prefix_trainer](results/prefix_trainer)).

## Run

```bash
python experiments/lifetime/launch.py --output-root runs/lifetime/c4 --seeds 0 1 2 3 4 \
    --variants full "full_meta=full --meta 0.5" frozen fixed_learn world_model predict_only recurrent -- \
    --streams 1024 --width 1024 --hidden 1024 --interactions 2e7 --lifetime 16384
python experiments/lifetime/launch.py --output-root runs/lifetime/x1 --seeds 0 1 2 \
    --variants "mixed_full=full --family mixed" "mixed_frozen=frozen --family mixed" \
    "mixed_recurrent=recurrent --family mixed" "thermal_full=full --family thermal" -- \
    --streams 1024 --width 1024 --hidden 1024 --interactions 2e7 --lifetime 16384
python experiments/lifetime/evaluate.py --model runs/lifetime/c4/full/seed0/model.pt --also frozen discard \
    --devices 512 --ticks 16384 --output runs/lifetime/eval/full_seed0.json      # one file per model
python experiments/lifetime/report.py --evaluations runs/lifetime/eval/*.json --runs runs/lifetime/c4 \
    --latency experiments/lifetime/results/latency.json --output experiments/lifetime/results/main
python experiments/lifetime/probe.py --model runs/lifetime/c4/full/seed0/model.pt --output experiments/lifetime/results/probe.json
python experiments/lifetime/latency.py --model runs/lifetime/c4/full/seed0/model.pt --output experiments/lifetime/results/latency.json
python experiments/lifetime/realtime.py --model runs/lifetime/c4/full/seed0/model.pt --output experiments/lifetime/results/realtime.json
python experiments/lifetime/references.py --output experiments/lifetime/results/references.json \
    --family slow_narrow instrument "time_constant=8,16 width=0.15,0.25"
```

Second round:

```bash
BASE="--precise-targets --sensor-noise --support buffer"
DEP="--objective deployed --step-rule evidence"
WS="--segment 16 --gated-workspace --recompute-predictions"
BR="--branch-streams 128 --branch-horizon 32 --activity-values"
B6="--branch-streams 128 --branch-horizon 4 --branch-replicates 4 --branch-greedy --activity-values"
python experiments/lifetime/launch.py --output-root runs/lifetime/c6 --seeds 0 1 2 --variants \
    "cdm=full $BASE $DEP $WS $BR" "cdm_b6=full $BASE $DEP $WS $B6" \
    "cdm_b6_ppo=full $BASE $DEP $WS $B6 --branch-weight 0" "cdm_truncated=full $BASE $DEP $BR" \
    "cdm_nolearn=frozen $BASE $WS $BR" "predict_only=predict_only $BASE --recompute-predictions" \
    "cdm_all=full --family all $BASE $DEP $WS $BR @0,1" "cdm_bank=full $BASE $DEP $WS $BR --adapter-blocks 4 @0,1" \
    "cdm_step_meta=full $BASE --objective step --meta 0.5 --step-rule evidence $WS $BR @0,1" \
    "cdm_cheap_targets=full --support buffer $DEP $WS $BR @0,1" "predict_hidden=predict_hidden $BASE @0,1" \
    "recurrent=recurrent $BASE @0,1" "predict_only_all=predict_only --family all $BASE --recompute-predictions @0,1" \
    "cdm_navigation=full --family navigation $BASE $DEP $WS $BR @0,1" \
    "cdm_thermal=full --family thermal $BASE $DEP $WS $BR @0" -- \
    --streams 1024 --width 1024 --hidden 1024 --interactions 2e7 --lifetime 16384 --minibatch 128 \
    --search-share 0.5 --cost-learning 0.0036
python experiments/lifetime/evaluate.py --model runs/lifetime/c6/cdm/seed0/model.pt \
    --also frozen discard norefine noidle --devices 512 --ticks 16384 --output runs/lifetime/eval_c6/cdm_seed0.json
python experiments/lifetime/branches.py --model runs/lifetime/c6/cdm/seed0/model.pt --greedy --replicates 4 \
    --output experiments/lifetime/results/branches_cdm_seed0.json
python experiments/lifetime/latency.py --model runs/lifetime/c6/cdm/seed0/model.pt --output experiments/lifetime/results/latency_c6.json
```

`launch.py` starts a run on a GPU only when enough memory is free and skips runs that have finished or are in progress.
