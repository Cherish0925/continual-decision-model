# Learning to acquire experimental evidence

The previous selector chose among numerical settings after most measurements
had already been collected. This study moves the learned decision earlier:
which object and physical quantity should be measured next, and when is the
available evidence sufficient to try a setting? The GRU and Transformer
workspace are trained together with those decisions.

## Task and available information

The environment is DiscoveryWorld Reactor Lab, Normal difficulty. Each instance
starts with two working reactors and two uncalibrated crystals. Public navigation
and dialogue locate the objects and read the working settings. The model then
chooses among twenty object–instrument requests and nine candidate settings for
the current target. After the first target works, the same model state continues
to the second target.

Five instruments measure density, temperature, radiation, quantum gap and a
five-channel spectrum. A spectrum is one physical measurement, not five actions.
Each radiation sample is requested separately. Its running minimum contains only
values actually returned; the policy never receives a thirty-two-sample estimate
for the cost of one reading.

The numerical tool maintains nine affine hypotheses. Each is fitted only to
measurements and working settings already observed. A setting becomes available
after the corresponding quantity has been measured on the target and at least
two calibrated objects. A fixed tool performs navigation, instrument use and
precise scalar adjustment. Those operations all count toward the native action
budget.

The policy reads structured public measurements, object roles, received feedback
and predictions computed from that evidence. It does not receive the scenario
seed, hidden scientific rule, future readings or the evaluator's score. This
study trains the experimental core with structured observations; the earlier
image encoders are not part of this input path.

## Model and training

The model inherits the released 1,024-dimensional recurrent core and eight-slot
Transformer workspace. Event and evidence projections, candidate scoring, value
and outcome heads connect it to experimental decisions. Training updates
26,653,443 parameters, including 23,108,608 parameters in the continuing core.

Two matched models start from the same weights. The **persistent** model carries
its GRU and workspace across decisions. The **ledger** control is trained
separately under the same initialization and schedule but resets these states for
each decision. It receives the same accumulated measurement ledger. This tests
the contribution of continuing neural computation beyond explicit records.

Training first replays actual instrument returns under the policy's chosen
requests. Hindsight targets use training-world outcomes to estimate promising
measurements and settings; these targets enter the loss, never the policy input.
An actor–critic phase then trains on the consequences of the model's own choices,
with costs for measurements and unsuccessful setting attempts. Gradients pass
through sequences of decisions into both the GRU and Transformer.

Replay success uses an eight-hertz prediction tolerance to approximate a setting
followed by local numerical adjustment. It is a training reward, not the
simulator's success definition. Native evaluation executes every adjustment and
uses the simulator's official completion flag.

The initial data contain 451 usable training worlds and 900 labeled targets.
The expanded collection contains 1,663 worlds, 3,319 labeled targets and 239,004
actual instrument returns. Seven worlds have only one valid target; missing
labels are masked, not replaced with fictitious values. Sixty-four further
worlds supply validation data. The fixed native comparison uses different
instances, including a relation family excluded from training.

## Native comparison

Every method starts from the same saved initial environment randomness and has
the same 400-action budget. The learned policies are compared with full
measurement followed by posterior selection, and full measurement followed by
the earlier numerical selector. All use the same public discovery procedure,
measurement operator and scalar adjustment tool.

Results report official task completion, actual native actions, requested
measurements and setting attempts separately. Optional intermediate objectives
can leave progress below one after both target reactors are working; progress
does not substitute for the completion flag. Paired comparisons reuse the same
instances. They estimate variation between instances, not between independent
training runs.

## Implementation

- [Core and experimental readouts](../../cdm/active_experiment.py)
- [Native measurement extraction](active_evidence_data.py)
- [Replay environment and joint trainer](train_active_core.py)
- [Public measurement and setting execution](active_measurement_policy.py)
- [From-reset episode interface](active_native.py)
- [Native evaluation](evaluate_active_native.py)
- [Paired result summary](summarize_active.py)

Training data, configuration, intermediate checkpoints and complete native
trajectories are stored under `runs/discoveryworld/active_experiments/`. The
recorded commands select the data directory, numerical prior, source core and
training seed explicitly. Native evaluation reads the fixed instance list in
[active_native_comparison.json](active_native_comparison.json).
