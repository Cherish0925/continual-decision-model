"""Inspect changed completions using saved public evidence, without new execution."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch

from cdm.experimental_belief import load_experimental_belief_checkpoint
from experiments.discoveryworld.calibration import FEATURE_QUANTITIES


def ledger(row):
    return {"values": torch.zeros(1, 4, 9, dtype=torch.float64),
            "counts": torch.zeros(1, 4, 9, dtype=torch.long),
            "known_hz": torch.tensor([row["bootstrap"]["known_support_hz"] + [0., 0.]], dtype=torch.float64),
            "known": torch.tensor([[True, True, False, False]])}


def ingest(state, row, decision):
    handles = {obj["public_handle"]: obj["slot"] for obj in row["bootstrap"]["public_object_order"]}
    for record in decision["result"].get("measurements", []):
        if record["kind"] != "instrument" or record["target_handle"] not in handles:
            continue
        if record["quantity"] not in FEATURE_QUANTITIES:
            continue
        obj, quantity = handles[record["target_handle"]], FEATURE_QUANTITIES.index(record["quantity"])
        count = state["counts"][0, obj, quantity]
        old = state["values"][0, obj, quantity]
        value = record["value"]
        state["values"][0, obj, quantity] = (min(float(old), value) if quantity == 2 and count > 0
                                          else (old * count + value) / (count + 1))
        state["counts"][0, obj, quantity] += 1
    if decision["success"]:
        result = decision["result"]
        if (result.get("setting") or {}).get("activated"):
            value = result["setting"]["frequency_hz"]
        else:
            value = next(attempt["frequency_hz"] for attempt in reversed(result["attempts"])
                         if attempt["activated"])
        query = decision["query_index"]
        state["known_hz"][0, query] = value
        state["known"][0, query] = True


def predict(model, state, query, memory):
    return model.observe_predict(state["values"], state["counts"] > 0, state["known_hz"],
                                 state["known"], torch.tensor([query]), memory)


def read_rows(path):
    return {row["seed"]: row for stream in json.loads((path / 'summary.json').read_text())
            for row in stream['episodes']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--csv', type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    cases = []
    with torch.inference_mode():
        for seed in (221, 222):
            arm = args.root / f'carry_seed{seed}'
            control = args.root / f'carry_seed{seed}_memory_discard'
            reference = read_rows(control)
            configuration = json.loads((arm / 'configuration.json').read_text())
            model, _ = load_experimental_belief_checkpoint(configuration['belief_checkpoint'])
            initial = model.initial(1)
            for stream in sorted(arm.glob('confirmation_*')):
                if not stream.is_dir():
                    continue
                memory = model.initial(1)
                for episode in sorted(stream.glob('episode*/summary.json')):
                    row = json.loads(episode.read_text())
                    other = reference[row['seed']]
                    changed = row['completed_successfully'] != other['completed_successfully']
                    current = ledger(row)
                    tests = []
                    max_reconstruction_error = 0.
                    for decision in row['decisions']:
                        prediction, _ = predict(model, current, decision['query_index'], memory)
                        saved = torch.tensor(decision['physical_prediction']['means'], dtype=torch.float64)
                        max_reconstruction_error = max(max_reconstruction_error,
                                                       float((prediction['means'][0]-saved).abs().max()))
                        if changed and decision['action_index'] >= 20:
                            base, _ = predict(model, current, decision['query_index'], initial)
                            quantity = decision['action_index'] - 20
                            tests.append({'decision': decision['decision'], 'query': decision['query_index'],
                                          'quantity': FEATURE_QUANTITIES[quantity],
                                          'prediction_hz': float(prediction['means'][0, quantity]),
                                          'without_memory_same_ledger_hz': float(base['means'][0, quantity]),
                                          'prediction_std_hz': float(prediction['std'][0, quantity]),
                                          'relation_probability': float(prediction['weights'][0, quantity]),
                                          'fresh_prior_mass': float(prediction['context_weights'][0, 0]),
                                          'received_quantity_values': current['values'][0, :, quantity].tolist(),
                                          'received_quantity_counts': current['counts'][0, :, quantity].tolist(),
                                          'known_hz': current['known_hz'][0].tolist(),
                                          'known': current['known'][0].tolist(),
                                          'native_actions': decision['native_actions'],
                                          'success': decision['success']})
                        ingest(current, row, decision)
                    _, final = predict(model, current, 3, memory)
                    memory = model.remember(memory, final)
                    if changed:
                        cases.append({'training_seed': seed, 'native_seed': row['seed'],
                                      'stream': row['stream'], 'segment_role': row['segment_role'],
                                      'position_in_segment': row['position_in_segment'],
                                      'carry_completed': row['completed_successfully'],
                                      'discard_completed': other['completed_successfully'],
                                      'carry_actions': row['native_actions'], 'discard_actions': other['native_actions'],
                                      'max_prediction_reconstruction_error_hz': max_reconstruction_error,
                                      'tests': tests})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({'method': 'Replay only saved public evidence; no new model training or native task execution.',
                                     'same_ledger_comparison': 'Recompute predictions with retained physical memory and an empty memory on the identical received ledger.',
                                     'cases': cases}, indent=2) + '\n')
    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        rows = []
        for case in cases:
            for test in case['tests']:
                row = {key: value for key, value in case.items() if key != 'tests'}
                row.update(test)
                rows.append({key: json.dumps(value) if isinstance(value, list) else value
                             for key, value in row.items()})
        with args.csv.open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator='\n')
            writer.writeheader()
            writer.writerows(rows)
    print(json.dumps([{key: row[key] for key in ('training_seed', 'native_seed',
        'carry_completed', 'max_prediction_reconstruction_error_hz')} for row in cases], indent=2))


if __name__ == '__main__':
    main()
