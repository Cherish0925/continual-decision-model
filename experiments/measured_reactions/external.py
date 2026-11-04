"""Evaluate frozen Roche-trained policies on two separately published reaction screens."""
import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from statistics import median

import numpy as np
import torch

from .prepare import ROLES, digest
from .study import LEARNED, REFERENCES, FEEDBACK_CONTROLS, Screens, evaluation, load_policy, summarize


SOURCES = {
    'ahneman': ('d7f318abfbb42eaadca482b1e9aa54540eeed60e0089b6a0eb4e75366c31ee2e', 'FullCV_01',
                ['Aryl halide'], {'Base': 'Base', 'Ligand': 'Ligand', 'Additive': 'Additive'}, 'Output'),
    'perera': ('a869e020ba31bd5676c67a4791c3b7384711b5216de6af444b8cd0a24c284640', 'Sheet1',
               ['Reactant_1_Name', 'Reactant_2_Name'], {'Base': 'Reagent_1_Short_Hand',
               'Ligand': 'Ligand_Short_Hand', 'Catalyst': 'Catalyst_1_Short_Hand',
               'Solvent': 'Solvent_1_Short_Hand'}, 'Product_Yield_PCT_Area_UV'),
}


def external_table(name, path, source):
    from openpyxl import load_workbook
    sha, sheet, substrate, reagents, target = SOURCES[name]
    if hashlib.sha256(path.read_bytes()).hexdigest() != sha:
        raise ValueError(f'Unexpected source workbook: {name}')
    book = load_workbook(path, read_only=True, data_only=True)
    values = iter(book[sheet].values)
    header = next(values)
    groups = defaultdict(lambda: defaultdict(list))
    for row in values:
        fields = {col: ('' if v is None else str(v).strip()) for col, v in zip(header, row)}
        key = tuple(fields[k] for k in substrate)
        x = tuple(fields[reagents[r]] if r in reagents else '' for r in ROLES)
        y = float(fields[target])
        if not 0 <= y <= 100:
            raise ValueError(f'Invalid experimental measurement: {y}')
        groups[key][x].append(y)
    book.close()
    vocabs = [sorted({x[j] for g in groups.values() for x in g}) for j in range(len(ROLES))]
    encode = [{v: i for i, v in enumerate(names)} for names in vocabs]
    screens, offsets, codes, outcomes = [], [0], [], []
    for key, options in sorted(groups.items()):
        sid = digest(json.dumps([name, list(key)], ensure_ascii=False))
        screens.append(dict(id=sid, family=name, notebook=sid, split='external'))
        for action, obs in sorted(options.items()):
            codes.append([idx[v] for idx, v in zip(encode, action)])
            outcomes.append(median(obs) / 100.)
        offsets.append(len(outcomes))
    data = Screens.external(dict(screens=screens, roles=ROLES, vocab=vocabs), np.array(offsets),
                            np.array(codes, np.int16), np.array(outcomes, np.float32), 'cpu', source)
    return data, {'screens': len(screens), 'observations': sum(len(o) for g in groups.values() for o in g.values()),
                  'distinct_conditions_within_screens': len(outcomes), 'workbook_sha256': sha}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-table', type=Path, required=True)
    p.add_argument('--ahneman', type=Path, required=True)
    p.add_argument('--perera', type=Path, required=True)
    p.add_argument('--runs', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--seeds', default='41,73,109,127,149,173,197,223')
    p.add_argument('--modes', default=','.join(LEARNED))
    args = p.parse_args()
    torch.set_num_threads(1)
    source = Screens(args.source_table, 'cpu')
    seeds = [int(s) for s in args.seeds.split(',')]
    tables = {k: external_table(k, getattr(args, k), source) for k in SOURCES}
    report = {k: dict(**meta, runs={}) for k, (_, meta) in tables.items()}
    jobs = [(m, None) for m in REFERENCES] + [(m, s) for s in seeds for m in args.modes.split(',')]
    curves = defaultdict(list)
    keys, labels = {}, []
    for mode, seed in jobs:
        name = mode if seed is None else f'{mode}{seed}'
        model = None if seed is None else load_policy(args.runs / name / 'model.pt', mode, 'cpu', 8)
        modes = [(mode, name)]
        if mode in FEEDBACK_CONTROLS:
            modes.append((FEEDBACK_CONTROLS[mode], name + '_without_readings'))
        for actual_mode, label in modes:
            labels.append(label)
            for k, (data, _) in tables.items():
                rows = sorted(evaluation(data, model, actual_mode, 8, 'external', 16, 4),
                              key=lambda r: (r['screen'], r['start']))
                ids = [(r['screen'], r['start']) for r in rows]
                arr = np.array([[r[f'best_{i}'] for i in range(1, 9)] for r in rows], np.float32)
                if k in keys and (keys[k] != ids or not np.array_equal(curves[k][0][:, 0], arr[:, 0])):
                    raise ValueError('Methods do not share an initial measurement')
                keys[k] = ids
                curves[k].append(arr)
                report[k]['runs'][label] = summarize(rows, 8)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / 'external.json').write_text(json.dumps(report, sort_keys=True, indent=2) + '\n')
    arrays = {'conditions': labels}
    for k in tables:
        screen, start = zip(*keys[k])
        arrays.update({f'{k}_screen': screen, f'{k}_start': start, f'{k}_best': np.stack(curves[k])})
    np.savez_compressed(args.out / 'external_curves.npz', **arrays)


if __name__ == '__main__':
    main()
