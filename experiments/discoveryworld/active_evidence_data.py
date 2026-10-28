"""Extract partial-observation tasks from measurements actually made in native worlds.

The arrays are an environment-side store. A policy receives a measurement only
after choosing its object and instrument. Working frequencies are observed
successful settings, not the simulator's hidden exact resonance frequencies.
Instrument events remain separate so repeated readings retain their true cost.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

import numpy as np

from .calibration import CalibrationLog, FEATURE_QUANTITIES


INSTRUMENT_GROUPS = ((0,), (1,), (2,), (3,), (4, 5, 6, 7, 8))
INSTRUMENT_NAMES = ("densitometer", "thermometer", "radiation meter", "microscope", "spectrometer")


def extract(job):
    project, split, entry = job
    source = Path(entry["source"])
    log = CalibrationLog()
    with (project / source).open() as stream:
        for line in stream:
            row = json.loads(line)
            # Only public observations, executed actions and their actual returns.
            log.append_transition({key: row[key] for key in
                                   ("index", "observation", "action", "action_result", "next_observation")})
    records = [r for r in log.measurements.records if r.kind == "instrument"
               and r.quantity in FEATURE_QUANTITIES and r.target_handle is not None]
    labels = {}
    for label in log.labels:
        labels.setdefault(label.object_handle, []).append(label)
    first_seen = {}
    for record in records:
        first_seen.setdefault(record.target_handle, record.sequence)
    handles = sorted(first_seen, key=lambda h: (
        not (h in labels and labels[h][0].activation_sequence is None), first_seen[h]))
    indices = {handle: index for index, handle in enumerate(handles)}
    objects = []
    for handle in handles:
        readings = [r for r in records if r.target_handle == handle]
        first = {}
        for reading in readings:
            first.setdefault(reading.quantity, reading)
        working = labels.get(handle, [])
        objects.append({"source_handle": handle,
                        "first_readings": [first[q].value if q in first else 0. for q in FEATURE_QUANTITIES],
                        "reading_present": [q in first for q in FEATURE_QUANTITIES],
                        "first_reading_sequences": [first[q].sequence if q in first else -1 for q in FEATURE_QUANTITIES],
                        "working_frequency_hz": working[0].frequency_hz if working else 0.,
                        "label_present": bool(working),
                        "initial_support": bool(working and working[0].activation_sequence is None),
                        "first_working_sequence": working[0].received_sequence if working else -1,
                        "activation_sequence": working[0].activation_sequence if working else None,
                        "observed_working_settings": [r.frequency_hz for r in working]})
    grouped = {}
    quantity_group = {FEATURE_QUANTITIES[q]: g for g, columns in enumerate(INSTRUMENT_GROUPS) for q in columns}
    for record in records:
        group = quantity_group[record.quantity]
        key = (record.sequence, indices[record.target_handle], group)
        event = grouped.setdefault(key, {"source_sequence": record.sequence,
                                         "object_index": indices[record.target_handle], "instrument_group": group,
                                         "values": [0.] * 9, "present": [False] * 9,
                                         "instrument_handle": record.instrument_handle,
                                         "conditions": record.conditions})
        column = FEATURE_QUANTITIES.index(record.quantity)
        event["values"][column], event["present"][column] = record.value, True
    for obj in objects:
        obj["target"] = bool(obj["label_present"] and not obj["initial_support"] and all(obj["reading_present"]))
    usable = (sum(o["initial_support"] and all(o["reading_present"]) for o in objects) >= 2
              and any(o["target"] for o in objects))
    return {"split": split, "identity": entry["identity"], "source": str(source),
            "objects": objects, "measurement_events": list(grouped.values()), "usable": usable}


def pack(rows):
    worlds = len(rows)
    objects = max(len(row["objects"]) for row in rows)
    grouped = []
    for row in rows:
        groups = {}
        for event in row["measurement_events"]:
            groups.setdefault((event["object_index"], event["instrument_group"]), []).append(event)
        grouped.append(groups)
    repeats = max(len(events) for groups in grouped for events in groups.values())
    arrays = {"x": np.zeros((worlds, objects, 9), np.float32),
              "reading_present": np.zeros((worlds, objects, 9), bool),
              "working_frequency_hz": np.zeros((worlds, objects), np.float32),
              "label_present": np.zeros((worlds, objects), bool),
              "initial_support_mask": np.zeros((worlds, objects), bool),
              "target_mask": np.zeros((worlds, objects), bool),
              "object_present": np.zeros((worlds, objects), bool),
              "sample_values": np.zeros((worlds, objects, 5, repeats, 9), np.float32),
              "sample_present": np.zeros((worlds, objects, 5, repeats, 9), bool),
              "sample_count": np.zeros((worlds, objects, 5), np.int32),
              "source": np.asarray([row["source"] for row in rows]),
              "scenario_seed": np.asarray([row["identity"][2] for row in rows], np.int64)}
    for i, row in enumerate(rows):
        for j, obj in enumerate(row["objects"]):
            arrays["x"][i, j] = obj["first_readings"]
            arrays["reading_present"][i, j] = obj["reading_present"]
            for key in ("working_frequency_hz", "label_present"):
                arrays[key][i, j] = obj[key]
            arrays["initial_support_mask"][i, j] = obj["initial_support"]
            arrays["target_mask"][i, j] = obj["target"]
            arrays["object_present"][i, j] = True
        for (j, group), events in grouped[i].items():
            arrays["sample_count"][i, j, group] = len(events)
            for k, event in enumerate(events):
                arrays["sample_values"][i, j, group, k] = event["values"]
                arrays["sample_present"][i, j, group, k] = event["present"]
    return arrays


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=Path.cwd())
    parser.add_argument("--manifest", type=Path, default=Path("experiments/discoveryworld/results/prediction_data.json"))
    parser.add_argument("--extra-train", type=Path, action="append", default=[],
                        help="Additional native training collection, relative to project; may be repeated")
    parser.add_argument("--validation-collection", type=Path,
                        help="Use this separate native collection instead of the manifest validation instances")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a new output directory")
    manifest = json.loads((args.project / args.manifest).read_text())
    collections = [("train", collection) for collection in args.extra_train]
    if args.validation_collection is not None:
        manifest["validation"] = []
        collections.append(("validation", args.validation_collection))
    for split, collection in collections:
        for source in sorted((args.project / collection).glob("seed*/trajectory.jsonl")):
            run = json.loads((source.parent / "run.json").read_text())
            if run.get("split") != split:
                raise ValueError(f"Collection must identify {split} data: {source}")
            manifest[split].append({"source": str(source.relative_to(args.project)),
                                    "identity": [run["scenario"], run["difficulty"], run["seed"]]})
    identities = [{tuple(row["identity"]) for row in manifest[split]} for split in ("train", "validation")]
    for split, keys in zip(("train", "validation"), identities):
        if len(keys) != len(manifest[split]):
            raise ValueError(f"Repeated native instance identity in {split}")
    if identities[0] & identities[1]:
        raise ValueError("Training and validation must use different native instances")
    args.output.mkdir(parents=True)
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    jobs = [(args.project, split, entry) for split in ("train", "validation") for entry in manifest[split]]
    with ProcessPoolExecutor(args.workers) as pool:
        rows = list(pool.map(extract, jobs))
    (args.output / "worlds.json").write_text(json.dumps(rows, indent=2) + "\n")
    summary = {"source_manifest": str(args.manifest), "additional_train_collections": [str(p) for p in args.extra_train],
               "validation_collection": str(args.validation_collection) if args.validation_collection else None,
               "quantities": FEATURE_QUANTITIES,
               "instrument_names": INSTRUMENT_NAMES, "instrument_groups": INSTRUMENT_GROUPS,
               "value_definition": "first actual instrument return; repeated returns remain separate events",
               "label_definition": "first public setting observed while the device was activated",
               "policy_exclusions": ["scenario_seed", "source", "source_handle", "unrequested measurements", "future working frequencies"],
               "setting": "offline partial-observation measurement replay; native task success evaluated separately",
               "splits": {}}
    for split in ("train", "validation"):
        selected = [r for r in rows if r["split"] == split and r["usable"]]
        arrays = pack(selected)
        np.savez_compressed(args.output / f"{split}.npz", **arrays)
        summary["splits"][split] = {"source_instances": sum(r["split"] == split for r in rows),
                                    "usable_instances": len(selected),
                                    "objects": int(arrays["object_present"].sum()),
                                    "targets": int(arrays["target_mask"].sum()),
                                    "measurement_events": sum(len(r["measurement_events"]) for r in selected),
                                    "max_objects": arrays["x"].shape[1],
                                    "max_repeated_measurements": arrays["sample_values"].shape[3],
                                    "excluded_sources": [r["source"] for r in rows if r["split"] == split and not r["usable"]]}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
