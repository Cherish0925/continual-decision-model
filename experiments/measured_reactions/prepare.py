"""Prepare sequential condition selection from the published Roche HTE v1 table."""
import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median

import numpy as np


SOURCE = 'georg-wuitschik-Roche/zScore-App'
REVISION = 'a7fa5d0311218cfa7e4a743be92be05bb6fb8824'
SHA256 = '50ff92f35ab983cfe211613bcfc420a0eeac75befdea7d15a6d09f5023abdac0'
ROLES = ('Base', 'Catalyst', 'Solvent', 'Ligand', 'Additive', 'Coupling Reagent',
         'Secondary Solvent', 'Tertiary Solvent')
GROUP = ('ELN_ID', 'PLATENUMBER', 'Reaction Type', 'FG A', 'FG B', 'output_column')
SPLITS = ('train', 'dev', 'test')


def digest(s):
    return hashlib.sha256(s.encode()).hexdigest()


def prepare(source, output):
    import pyarrow.parquet as pq
    if hashlib.sha256(source.read_bytes()).hexdigest() != SHA256:
        raise ValueError('Expected the pinned upstream v1.parquet')
    rows = pq.read_table(source, columns=[*GROUP, *ROLES, 'AREA_TOTAL_REDUCED']).to_pylist()
    groups = defaultdict(lambda: defaultdict(list))
    eln_family = {}
    for row in rows:
        eln, family = row['ELN_ID'], row['Reaction Type']
        if eln_family.setdefault(eln, family) != family:
            raise ValueError('Cannot separate families by notebook')
        key = tuple(str(row[k]) if row[k] is not None else '' for k in GROUP)
        option = tuple(str(row[k]) if row[k] is not None else '' for k in ROLES)
        y = float(row['AREA_TOTAL_REDUCED'])
        if not np.isfinite(y) or not 0 <= y <= 100:
            raise ValueError('Invalid measured peak area')
        groups[key][option].append(y)
    groups = {k: v for k, v in groups.items() if len(v) >= 12}
    notebooks = defaultdict(set)
    for eln, _, family, *_ in groups:
        notebooks[family].add(eln)
    # Balance family sizes using notebook counts alone; neither yields nor coordinates choose a split.
    families = sorted(notebooks, key=lambda f: (-len(notebooks[f]), f))
    split = {}
    for start in range(0, len(families), 5):
        block = sorted(families[start:start + 5], key=lambda f: digest('CDM-HTE-v1|' + f))
        split.update(zip(block, ('train', 'dev', 'train', 'test', 'train')))
    vocabs = [sorted({o[j] for options in groups.values() for o in options}) for j in range(len(ROLES))]
    # Integers index literal reagent names only. No rank or outcome is encoded in them.
    indices = [{name: i for i, name in enumerate(v)} for v in vocabs]
    screens, conditions, values, offsets, used_rows = [], [], [], [0], 0
    for key, options in sorted(groups.items()):
        eln, _, family, *_ = key
        screen_id = digest(json.dumps(key, ensure_ascii=False))
        screens.append(dict(id=screen_id, notebook=eln, family=family, split=split[family]))
        for o, measures in sorted(options.items()):
            conditions.append([idx[val] for idx, val in zip(indices, o)])
            values.append(median(measures))
            used_rows += len(measures)
        offsets.append(len(values))
    meta = dict(source=SOURCE, revision=REVISION, sha256=SHA256, roles=ROLES, vocab=vocabs,
                screens=screens, raw_rows=len(rows), retained_rows=used_rows, conditions=len(values))
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, meta=json.dumps(meta, ensure_ascii=False), offsets=offsets,
                        conditions=np.asarray(conditions, dtype=np.int16),
                        peak_area=np.asarray(values, dtype=np.float32))
    print(json.dumps(dict(raw_rows=len(rows), retained_rows=used_rows, conditions=len(values),
                          screens=dict(Counter(s['split'] for s in screens)),
                          families=dict(Counter(split.values())))))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('source', type=Path)
    p.add_argument('output', type=Path)
    args = p.parse_args()
    prepare(args.source, args.output)
