"""Compare physical predictions on identical held-out task streams."""
from __future__ import annotations

import argparse
import csv
import gzip
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from cdm.experimental_belief import load_experimental_belief_checkpoint
from experiments.discoveryworld.calibration_mixture import load_mixture_checkpoint
from experiments.discoveryworld.train_active_core import visible_beliefs
from .data import RecordedWorlds, episode, target_order
from .train import stream_loss


@torch.no_grad()
def reference_rows(mixture, stream):
    batch, length = stream["values"].shape[:2]
    rows, index = [], torch.arange(batch, device=mixture.device)
    for position in range(length):
        current = episode(stream, position)
        known, targets = current["support"].clone(), target_order(current["targets"])
        for target_number in range(targets.shape[1]):
            query = targets[:, target_number]
            valid = current["targets"][index, query]
            prediction = visible_beliefs(current["values"], current["seen"],
                                         torch.where(known, current["settings"], 0.), known, query, mixture)
            target = current["settings"][index, query]
            std, means, weights = (prediction[k].double() for k in ("std", "means", "weights"))
            likelihood = -.5 * ((target[:, None] - means) / std).square() - std.log() - .5 * np.log(2 * np.pi)
            nll = -(weights.clamp_min(1e-300).log() + likelihood).logsumexp(-1)
            choice = weights.argmax(-1)
            rows.append({"episode": position, "target_number": target_number, "valid": valid,
                         "nll": nll, "target_hz": target,
                         "mixture_mean_hz": (weights * means).sum(-1),
                         "map_mean_hz": means.gather(1, choice[:, None])[:, 0], "map_quantity": choice,
                         "weights": weights, "means_hz": means, "std_hz": std})
            known = known.scatter(1, query[:, None], valid[:, None] | known.gather(1, query[:, None]))
    return rows


def flat_rows(rows, stream, method):
    output = []
    length = stream["segment_length"]
    for block in rows:
        arrays = {key: value.detach().cpu().numpy() for key, value in block.items() if isinstance(value, torch.Tensor)}
        episode_index = block["episode"]
        for i in np.flatnonzero(arrays["valid"]):
            record = {"method": method, "stream": int(i), "kind": stream["kind"],
                      "episode": episode_index, "phase": ("initial", "switch", "return")[episode_index // length],
                      "position_in_phase": episode_index % length,
                      "target_number": block["target_number"],
                      "source_seed": int(stream["seeds"][i, episode_index]),
                      "environment_quantity": int(stream["quantities"][i, episode_index]),
                      "generated": bool(stream["generated"][i, episode_index])}
            for key, value in arrays.items():
                if key == "valid": continue
                record[key] = value[i].tolist()
            output.append(record)
    return output


def metrics(rows):
    nll = np.asarray([r["nll"] for r in rows])
    output = {"queries": len(rows), "mean_nll_hz": float(nll.mean())}
    target = np.asarray([r["target_hz"] for r in rows])
    for field in ("map_mean_hz", "mixture_mean_hz"):
        error = np.asarray([r[field] for r in rows]) - target
        prefix = "map" if field.startswith("map") else "mixture"
        output.update({prefix + "_mae_hz": float(np.abs(error).mean()),
                       prefix + "_median_error_hz": float(np.median(np.abs(error))),
                       prefix + "_rmse_hz": float(np.sqrt((error ** 2).mean())),
                       prefix + "_within_2hz": int((np.abs(error) < 2).sum())})
    output["relation_accuracy"] = float(np.mean([r["map_quantity"] == r["environment_quantity"] for r in rows]))
    return output


@torch.no_grad()
def evaluate(args):
    torch.set_num_threads(args.threads)
    model, saved = load_experimental_belief_checkpoint(args.checkpoint, args.device)
    model.eval()
    initial = load_experimental_belief_checkpoint(args.initial_checkpoint, args.device)[0] if args.initial_checkpoint else None
    reference = load_mixture_checkpoint(args.reference_mixture, args.device)[0] if args.reference_mixture else None
    worlds = RecordedWorlds(args.data / "validation.npz", args.device, args.radiation_readings)
    args.output.mkdir(parents=True, exist_ok=True)
    summary = {"protocol": "offline all-selected-readings prediction; target labels revealed after scoring",
               "distinct_source_worlds": worlds.size, "streams_per_kind": args.streams,
               "sampling": "no source world repeats within a stream; worlds recur across streams, so stream count is not independent-instance count",
               "wide": "generated affine alternatives, episodewise scale 0.35..2.0 and offset -1000..1000 Hz; prediction stress test",
               "training": saved.get("training"), "groups": []}
    all_rows, started = [], perf_counter()
    for kind in args.kinds:
        generator = torch.Generator(device=args.device).manual_seed(args.seed)
        stream = worlds.sample_stream(args.streams, generator=generator, segment_length=args.segment_length,
                                      generated_fraction=1., kind=kind, unique_worlds=True)
        methods = [(mode, model, mode) for mode in args.modes]
        if initial is not None: methods.append(("initial_retain", initial, "retain"))
        for name, current, mode in methods:
            _, rows = stream_loss(current, stream, mode=mode, return_rows=True)
            flat = flat_rows(rows, stream, name)
            all_rows.extend(flat)
            entry = {"kind": kind, "method": name, **metrics(flat)}
            summary["groups"].append(entry)
            print(json.dumps(entry), flush=True)
        if reference is not None:
            flat = flat_rows(reference_rows(reference, stream), stream, "original_numerical_prior")
            all_rows.extend(flat)
            summary["groups"].append({"kind": kind, "method": "original_numerical_prior", **metrics(flat)})
    summary["phase_groups"] = []
    for kind in args.kinds:
        for method in sorted({r["method"] for r in all_rows}):
            for phase in ("initial", "switch", "return"):
                rows = [r for r in all_rows if r["kind"] == kind and r["method"] == method and r["phase"] == phase]
                summary["phase_groups"].append({"kind": kind, "method": method, "phase": phase, **metrics(rows)})
                first = [r for r in rows if r["position_in_phase"] == 0 and r["target_number"] == 0]
                summary["phase_groups"].append({"kind": kind, "method": method, "phase": phase + "_first_prediction", **metrics(first)})
    summary["elapsed_seconds"] = perf_counter() - started
    with gzip.open(args.output / "predictions.jsonl.gz", "wt") as out:
        for row in all_rows: out.write(json.dumps(row) + "\n")
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (args.output / "methods.csv").open("w") as out:
        writer = csv.DictWriter(out, fieldnames=list(summary["groups"][0]))
        writer.writeheader(); writer.writerows(summary["groups"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--initial-checkpoint", type=Path)
    parser.add_argument("--reference-mixture", type=Path)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=1029)
    parser.add_argument("--streams", type=int, default=128)
    parser.add_argument("--segment-length", type=int, default=2)
    parser.add_argument("--radiation-readings", type=int, default=32)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--kinds", nargs="+", default=["native", "rehearsal", "wide"])
    parser.add_argument("--modes", nargs="+", default=["retain", "reset", "discard", "last_context", "no_retrieval"])
    evaluate(parser.parse_args())


if __name__ == "__main__":
    main()
