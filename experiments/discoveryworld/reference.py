"""Public-observation Reactor Lab reference and CPU trajectory collection.

This is a hand-authored scientific procedure, not a learned CDM result. It
measures every available crystal coordinate, reads the already working reactor
frequencies, fits competing one-coordinate affine hypotheses, and tests their
predictions by actually tuning reactors. Two calibration crystals cannot
identify an arbitrary affine relation: activation of a third crystal provides
the additional experimental evidence. The surviving hypothesis predicts the
fourth. Neither policy construction nor action selection receives a seed,
scorecard, simulator object, or hidden rule.

Collection splits are outside the policy. Official seeds 0--4 are reserved;
training uses seeds >=100 and <200 with seed % 5 in {0,1,2,3}; new-instance
validation uses seeds >=200 with the same families, and held-out-rule validation
uses seed % 5 == 4. The modulo label comes from the documented generator split,
and is stored only in collection metadata.
"""
from __future__ import annotations

import argparse
from contextlib import redirect_stdout
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import math
from pathlib import Path
import re
import time

from .adapter import DiscoveryWorldEnv, TrajectoryWriter


INSTRUMENTS = ("densitometer", "thermometer", "radiation meter", "microscope", "spectrometer")
NUMBER = r"([-+]?\d+(?:\.\d+)?)"
# Capture once when this worker imports the policy: later development edits
# must not change the claimed source of an already running collector.
REFERENCE_SOURCE = Path(__file__).read_bytes()


def observed_objects(observation: dict) -> dict[int, dict]:
    """Only object descriptions the API has delivered to this observer."""
    ui = observation["ui"]
    objects = {}
    for group in ui.get("nearbyObjects", {}).get("objects", {}).values():
        objects.update((o["uuid"], o) for o in group)
    for field in ("accessibleEnvironmentObjects", "inventoryObjects"):
        objects.update((o["uuid"], o) for o in ui.get(field, []))
    return objects


def parse_measurement(instrument: str, message: str) -> dict[str, float]:
    """Extract numeric readings with their physical coordinate, never UUIDs."""
    patterns = {
        "densitometer": ("density_g_cm3", rf"approximately {NUMBER} grams per cubic centimeter"),
        "thermometer": ("temperature_c", rf"temperature of {NUMBER} degrees Celsius"),
        "radiation meter": ("radiation_usv_h", rf"level of {NUMBER} micro Seiverts per hour"),
        "microscope": ("quantum_gap_nm", rf"quantum gap.*?{NUMBER} nm"),
    }
    if instrument == "spectrometer":
        values = {f"spectrum_channel_{channel}": float(value)
                  for channel, value in re.findall(rf"Channel (\d+): {NUMBER}", message)}
    else:
        feature, pattern = patterns[instrument]
        match = re.search(pattern, message)
        values = {} if match is None else {feature: float(match.group(1))}
    if not values:
        raise ValueError(f"No {instrument} numeric reading in returned public message: {message!r}")
    return values


@dataclass(frozen=True)
class AffineHypothesis:
    feature: str
    slope: float
    intercept: float
    calibration_rmse: float

    def predict(self, measurements: dict[str, float]) -> float:
        return self.slope * measurements[self.feature] + self.intercept


def fit_hypotheses(calibration: list[tuple[dict[str, float], float]]) -> list[AffineHypothesis]:
    """Fit hypotheses from measured coordinates and publicly read frequencies.

    Near-integer coefficients are a simplicity prior used for ordering trials,
    not an assumption that any particular coordinate or coefficient is correct.
    Every accepted prediction still requires observed reactor activation.
    """
    if len(calibration) < 2:
        raise ValueError("At least two calibrated crystals are required")
    features = set.intersection(*(set(readings) for readings, _ in calibration))
    hypotheses = []
    for feature in sorted(features):
        xs = [row[feature] for row, _ in calibration]
        ys = [frequency for _, frequency in calibration]
        xbar, ybar = sum(xs) / len(xs), sum(ys) / len(ys)
        denominator = sum((x - xbar) ** 2 for x in xs)
        if denominator < 1e-12:
            continue
        slope = sum((x - xbar) * (y - ybar) for x, y in zip(xs, ys)) / denominator
        intercept = ybar - slope * xbar
        rmse = math.sqrt(sum((slope * x + intercept - y) ** 2 for x, y in zip(xs, ys)) / len(xs))
        hypotheses.append(AffineHypothesis(feature, slope, intercept, rmse))
    return sorted(hypotheses, key=lambda h: (
        round(h.calibration_rmse, 6),
        abs(h.slope - round(h.slope)) + abs(h.intercept - round(h.intercept)),
        abs(h.slope) + abs(h.intercept), h.feature,
    ))


def current_frequency(dialog: dict) -> float:
    match = re.search(rf"current resonance frequenc[ey] is: {NUMBER} Hertz", dialog["dialogIn"])
    if match is None:
        raise ValueError(f"No public frequency in reactor dialog: {dialog!r}")
    return float(match.group(1))


def tuning_choice(dialog: dict, target: float) -> int | None:
    """Choose a displayed increment that strictly reduces prediction error."""
    frequency = current_frequency(dialog)
    error = abs(frequency - target)
    options = []
    for key, label in dialog["dialogOptions"].items():
        match = re.fullmatch(rf"(Increase|Decrease) frequency by {NUMBER} Hz", label)
        if match:
            delta = float(match.group(2)) * (1 if match.group(1) == "Increase" else -1)
            # Clamp only to the allowable range explicitly shown by the dialog.
            bounds = re.search(r"allowable range is (\d+) to ([\d,]+) Hz", dialog["dialogIn"])
            proposed = frequency + delta
            if bounds:
                proposed = min(float(bounds.group(2).replace(",", "")), max(float(bounds.group(1)), proposed))
            remaining = abs(proposed - target)
            if remaining < error - 1e-9:
                options.append((remaining, int(key)))
    return min(options)[1] if options else None


class ReactorLabReference:
    """Stateful coroutine implementing experiments through native actions only."""

    def __init__(self):
        self.observation = None
        self.program = None
        self.phase = "initial"
        self.measurements = {}
        self.evidence = []
        self.hypotheses = []
        self.selected = None
        self.finished = False

    def __call__(self, observation: dict) -> dict:
        self.observation = observation
        if self.program is None:
            self.program = self._run()
        try:
            return next(self.program)
        except StopIteration as exc:
            self.finished = True
            raise RuntimeError("Reference procedure ended before environment termination; inspect evidence") from exc

    @property
    def objects(self):
        return observed_objects(self.observation)

    def _act(self, action: str, arg1=None, arg2=None):
        if action not in self.observation["actions"]:
            raise ValueError(f"Action {action} is not documented by this environment")
        payload = {"action": action}
        if arg1 is not None:
            payload["arg1"] = arg1
        if arg2 is not None:
            payload["arg2"] = arg2
        yield payload

    def _location(self, contains: str):
        choices = [name for name in self.observation["teleport_locations"] if contains in name]
        if len(choices) != 1:
            raise ValueError(f"Expected one public teleport location containing {contains!r}: {choices}")
        yield from self._act("TELEPORT_TO_LOCATION", choices[0])

    def _approach(self, uuid: int):
        if uuid not in self.objects:
            raise ValueError(f"Target {uuid} is not in the current public observation")
        accessible = {o["uuid"] for o in self.observation["ui"]["accessibleEnvironmentObjects"]}
        if uuid not in accessible:
            yield from self._act("TELEPORT_TO_OBJECT", uuid)

    def _exit_dialog(self):
        dialog = self.observation["ui"]["dialog_box"]
        if dialog["is_in_dialog"]:
            option = next(int(key) for key, label in dialog["dialogOptions"].items() if label.lower() == "exit")
            yield {"chosen_dialog_option_int": option}

    def _record_measurement(self, crystal, name):
        message = self.observation["ui"]["lastActionMessage"]
        values = parse_measurement(name, message)
        self.measurements[crystal].update(values)
        self.evidence.append({"kind": "measurement", "crystal": crystal,
                              "instrument": name, "readings": values, "message": message})

    def _measure(self, crystal: int, instruments: dict[str, int], return_location: str, container: int):
        self.phase = "measure_crystal"
        self.measurements[crystal] = {}
        for name, instrument in instruments.items():
            if name == "radiation meter":
                continue
            yield from self._act("USE", instrument, crystal)
            self._record_measurement(crystal, name)

        # The radiation instrument responds to nearby emitters as well as its
        # target. Carry only this crystal to a clearing before measuring it;
        # restore the original container afterward, including calibrated ones.
        # All navigation uses delivered objects, locations, and movement options.
        self.phase = "isolate_radiation_measurement"
        yield from self._act("PICKUP", crystal)
        yield from self._location("instruments")
        nearby = self.observation["ui"]["nearbyObjects"]["objects"]
        grasses = [(o["distance"], direction, o["uuid"])
                   for direction, group in nearby.items() for o in group
                   if o["name"] == "grass" and direction != "same_location"]
        if not grasses:
            raise ValueError("No observed clearing for an isolated radiation measurement")
        _, _, grass = min(grasses, key=lambda item: item[:2])
        yield from self._act("TELEPORT_TO_OBJECT", grass)
        lab_locations = list(self.observation["teleport_locations"].values())
        visited_grass = {grass}
        for _ in range(24):
            position = self.observation["ui"]["agentLocation"]
            distance = min(max(abs(position["x"] - place["gridX"]),
                               abs(position["y"] - place["gridY"])) for place in lab_locations)
            if distance >= 9:
                break
            choices = []
            for direction, group in self.observation["ui"]["nearbyObjects"]["objects"].items():
                if direction == "same_location":
                    continue
                axes = direction.split("-")
                for obj in group:
                    if obj["name"] != "grass" or obj["uuid"] in visited_grass:
                        continue
                    # Nearby observations give a compass direction and Manhattan
                    # distance. For diagonal groups this estimate only orders
                    # exploration; the next delivered position verifies progress.
                    span = obj["distance"] / len(axes)
                    x = position["x"] + span * (("east" in axes) - ("west" in axes))
                    y = position["y"] + span * (("south" in axes) - ("north" in axes))
                    separation = min(max(abs(x - p["gridX"]), abs(y - p["gridY"])) for p in lab_locations)
                    choices.append((separation, obj["distance"], obj["uuid"]))
            if not choices:
                raise ValueError("No unvisited observed clearing for the radiation sample")
            # UUIDs identify the chosen object; their magnitude is not a policy
            # feature or a geometric tie-breaker.
            grass = max(choices, key=lambda item: item[:2])[2]
            visited_grass.add(grass)
            yield from self._act("TELEPORT_TO_OBJECT", grass)
        else:
            raise ValueError("Could not separate the radiation sample from the public lab locations")

        # Repeated instrument returns fluctuate even at the isolated location.
        # Their lower envelope estimates the source with nonnegative ambient
        # contamination removed. Retain every reading in transitions and the
        # aggregate's provenance; rounding or residual noise can still cause a
        # hypothesis to fail its subsequent intervention test.
        samples = []
        messages = []
        for _ in range(32):
            yield from self._act("USE", instruments["radiation meter"], crystal)
            message = self.observation["ui"]["lastActionMessage"]
            samples.append(parse_measurement("radiation meter", message)["radiation_usv_h"])
            messages.append(message)
        value = min(samples)
        self.measurements[crystal]["radiation_usv_h"] = value
        self.evidence.append({"kind": "measurement", "crystal": crystal,
                              "instrument": "radiation meter", "readings": {"radiation_usv_h": value},
                              "aggregation": "minimum of 32 repeated isolated measurements",
                              "samples": samples, "message": messages[samples.index(value)]})
        yield from self._location(return_location)
        yield from self._approach(container)
        yield from self._act("PUT", crystal, container)

    def _activated(self, reactor):
        return self.objects[reactor]["name"] == "crystal reactor (activated)"

    def _run(self):
        self.phase = "acquire_instruments"
        yield from self._location("instruments")
        instruments = {}
        for name in INSTRUMENTS:
            if not any(o["name"] == name for o in self.objects.values()):
                yield from self._location("instruments")
            instrument = next(o["uuid"] for o in self.objects.values() if o["name"] == name)
            yield from self._approach(instrument)
            yield from self._act("PICKUP", instrument)
            instruments[name] = instrument

        yield from self._location("crystal bench")
        loose = [o["uuid"] for o in self.objects.values()
                 if o["name"].startswith("quantum crystal") and "(on table" in o["description"]]
        if not loose:
            raise ValueError("No uninstalled crystals in the public crystal bench observation")
        for crystal in loose:
            yield from self._approach(crystal)
            container = int(re.search(r"\[uuid: (\d+)\]", self.objects[crystal]["description"]).group(1))
            yield from self._measure(crystal, instruments, "crystal bench", container)

        self.phase = "read_calibration"
        yield from self._location("reactor lab")
        reactors = [o["uuid"] for o in self.objects.values() if o["name"].startswith("crystal reactor")]
        working = [r for r in reactors if self._activated(r)]
        empty = [r for r in reactors if "no crystal present" in self.objects[r]["name"]]
        calibration = []
        for reactor in working:
            yield from self._approach(reactor)
            crystal = next(o["uuid"] for o in self.objects.values()
                           if o["name"].startswith("quantum crystal") and f"[uuid: {reactor}]" in o["description"])
            yield from self._act("TALK", reactor)
            frequency = current_frequency(self.observation["ui"]["dialog_box"])
            self.evidence.append({"kind": "calibration", "reactor": reactor,
                                  "crystal": crystal, "frequency_hz": frequency})
            yield from self._exit_dialog()
            yield from self._measure(crystal, instruments, "reactor lab", reactor)
            calibration.append((self.measurements[crystal], frequency))
        self.hypotheses = fit_hypotheses(calibration)
        if len(empty) != len(loose):
            raise ValueError("Observed empty reactor and loose crystal counts differ")

        yield from self._location("crystal bench")
        for crystal in loose:
            yield from self._approach(crystal)
            yield from self._act("PICKUP", crystal)
        yield from self._location("reactor lab")

        for crystal, reactor in zip(loose, empty):
            self.phase = "test_prediction"
            yield from self._approach(reactor)
            yield from self._act("PUT", crystal, reactor)
            yield from self._act("TALK", reactor)
            hypotheses = self.hypotheses if self.selected is None else [self.selected] + [
                h for h in self.hypotheses if h != self.selected]
            for hypothesis in hypotheses:
                target = hypothesis.predict(self.measurements[crystal])
                if not 0 <= target <= 10000:
                    continue
                while True:
                    # A tuning path might cross a working frequency en route to
                    # a wrong prediction. Finish testing the prediction unless
                    # the public task message says the episode has completed.
                    # The terminal case is recorded at its actual applied value.
                    if self._activated(reactor) and all(
                        task["completed"] for task in self.observation["ui"]["taskProgress"]
                    ):
                        break
                    option = tuning_choice(self.observation["ui"]["dialog_box"], target)
                    if option is None:
                        break
                    yield {"chosen_dialog_option_int": option}
                activated = self._activated(reactor)
                self.evidence.append({"kind": "hypothesis_test", "crystal": crystal,
                                      "reactor": reactor, "hypothesis": asdict(hypothesis),
                                      "prediction_hz": target,
                                      "applied_hz": current_frequency(self.observation["ui"]["dialog_box"]),
                                      "prediction_reached": tuning_choice(
                                          self.observation["ui"]["dialog_box"], target) is None,
                                      "activated": activated})
                if activated:
                    self.selected = hypothesis
                    break
            else:
                raise RuntimeError("No measured-coordinate affine hypothesis activated the reactor")
            yield from self._exit_dialog()
        self.phase = "complete"


def create(config: dict | None = None) -> ReactorLabReference:
    if config:
        raise ValueError("The reference accepts no seed, rule, or simulator configuration")
    return ReactorLabReference()


def split_for_seed(seed: int) -> str:
    if seed < 100:
        raise ValueError("Reference collection reserves official seeds and starts at 100")
    if seed % 5 == 4:
        return "heldout_rule"
    return "train" if seed < 200 else "validation"


def collect_episode(output: Path, seed: int, max_steps: int = 500, *, capture_initial_rng: bool = False) -> dict:
    """Seed and evaluator remain in collector scope, never reference inputs."""
    split = split_for_seed(seed)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "trajectory.jsonl").exists():
        raise FileExistsError(f"Preserving existing trajectory {output}")
    started = time.perf_counter()
    reference = create()
    with (output / "simulator.log").open("x") as log, redirect_stdout(log):
        env = DiscoveryWorldEnv(output, max_steps)
        observation = env.reset("Reactor Lab", "Normal", seed)
        metadata = {**env.configuration, "policy": "scripted_public_reactor_reference_v1",
                    "is_learned_policy": False, "split": split, "rule_family": seed % 5,
                    "evaluation_kind": "held_out_rule" if split == "heldout_rule" else
                                       "new_instance" if split == "validation" else "training",
                    "action_descriptions": env.actions, "teleport_locations": env.teleports,
                    "initial_task_metrics": env.metrics,
                    "hypothesis_family": "affine in one measured coordinate; intervention validates"}
        if capture_initial_rng:
            filename = "initial_rng.json.gz"
            metadata["initial_rng_snapshot"] = {"path": filename, **env.save_initial_rng(output / filename)}
        source = REFERENCE_SOURCE
        (output / "reference_source.py").write_bytes(source)
        metadata["reference_source_sha256"] = sha256(source).hexdigest()
        error = None
        with TrajectoryWriter(output, metadata) as writer:
            try:
                while not env.ended:
                    action = reference(observation)
                    transition = env.step(action)
                    transition["reference_phase"] = reference.phase
                    writer.append(transition)
                    observation = env.policy_input()
            except (RuntimeError, ValueError, StopIteration) as exc:
                error = f"{type(exc).__name__}: {exc}"
        # Resume the coroutine once to record the last activation observation;
        # never execute its proposed action after the environment is terminal.
        if env.ended and error is None:
            try:
                reference(observation)
            except RuntimeError:
                pass
        summary = {"transitions": env.steps, "task_metrics": env.metrics,
                   "ended": env.ended, "error": error, "split": split,
                   "collection_cutoff": "reference_error" if error else
                                        "task_terminal" if all(m["completed"] for m in env.metrics) else "time_limit",
                   "elapsed_seconds": time.perf_counter() - started,
                   "is_learned_policy": False, "reference_evidence": reference.evidence,
                   "hypotheses": [asdict(h) for h in reference.hypotheses]}
        (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return {k: v for k, v in summary.items() if k not in ("reference_evidence", "hypotheses")}


def _collect_seed(job):
    seed, root, max_steps, capture_initial_rng = job
    output = root / split_for_seed(seed) / f"seed{seed}"
    result = collect_episode(output, seed, max_steps, capture_initial_rng=capture_initial_rng)
    return {"seed": seed, "output": str(output), **result}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", default="100,101,102,103,105,106,107,108")
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--capture-initial-rng", action="store_true",
                        help="Save evaluator-only native RNG states for exact action-prefix replay")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("workers must be positive")
    jobs = [(int(s), args.output, args.max_steps, args.capture_initial_rng) for s in args.seeds.split(",")]
    if args.workers == 1:
        for result in map(_collect_seed, jobs):
            print(json.dumps(result), flush=True)
    else:
        from concurrent.futures import ProcessPoolExecutor
        from multiprocessing import get_context
        with ProcessPoolExecutor(args.workers, mp_context=get_context("spawn")) as pool:
            for result in pool.map(_collect_seed, jobs):
                print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
