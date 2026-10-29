"""Prepare the selected chemistry data and four fixed model artifacts locally."""
from __future__ import annotations
import argparse,importlib.metadata,json,shutil,tarfile
from pathlib import Path


def archive(directory):
    target=directory.with_suffix('.tar.gz')
    with tarfile.open(target,'w:gz',compresslevel=3) as handle:
        for path in sorted(directory.rglob('*')):
            if path.is_file():handle.add(path,arcname=str(path.relative_to(directory.parent)))
    return {'file':target.name,'bytes':target.stat().st_size,
            'unpacked_bytes':sum(p.stat().st_size for p in directory.rglob('*') if p.is_file())}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=Path('runs/release_staging/chemistry_reaction'))
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    source=Path('runs/continual_physics/chemistry_exploration')
    data=args.output/'chemistry-experience';models=args.output/'chemistry-models'
    data.mkdir();models.mkdir()
    for name in ('manifest.json','plan.json'):shutil.copyfile(source/name,data/name)
    manifest=json.loads((source/'manifest.json').read_text());trials=actions=0
    for row in manifest['episodes']:
        relative=Path(row['directory']);src=source/relative;dst=data/relative;dst.mkdir(parents=True)
        shutil.copyfile(src/'experiments.jsonl',dst/'experiments.jsonl')
        shutil.copyfile(src/'summary.json',dst/'summary.json')
        trials+=sum(1 for line in (src/'experiments.jsonl').open() if line.strip())
        with (src/'trajectory.jsonl').open() as inp,(dst/'public_actions.jsonl').open('w') as out:
            for line in inp:
                record=json.loads(line)
                projected={key:record[key] for key in ('index','action','action_result','terminated','truncated')}
                for key in ('observation','next_observation'):
                    projected[key]={k:v for k,v in record[key]['ui'].items() if k!='taskProgress'}
                out.write(json.dumps(projected,separators=(',',':'))+'\n');actions+=1
    distribution=importlib.metadata.distribution('discoveryworld')
    license_path=next(distribution.locate_file(f) for f in distribution.files if str(f).endswith('licenses/LICENSE.txt'))
    shutil.copyfile(license_path,data/'DiscoveryWorld-LICENSE.txt')
    (data/'README.md').write_text('''# Native chemical experiment experience

320 DiscoveryWorld Combinatorial Chemistry instances provide 10,610 returned
experiments. `experiments.jsonl` preserves each actual public prefix, selected
recipe, rust response and operation cost. `public_actions.jsonl` contains the
corresponding primitive actions and public UI observations, excluding task-score
fields. No simulator stdout, hidden world snapshots or local paths are included.
`manifest.json` supplies the training episodes; `plan.json` records the separate
development/confirmation groups. Evaluator group labels must not enter model inputs.

DiscoveryWorld is from https://github.com/allenai/discoveryworld and its included
Apache-2.0 license governs the upstream task text and simulator-derived material.
Use the matching continual-decision-model source and pass this directory to
`experiments.continual_chemistry.train_reaction --data-root` or the fixed-epoch
warmup/joint training commands documented in that source repository.
''')
    selected=[]
    model_root=Path('runs/continual_physics/chemistry_reaction')
    for seed in (753,754):
        for phase in ('warmup','joint'):
            name=f'{phase}{seed}.pt';shutil.copyfile(model_root/f'{phase}{seed}'/f'{phase}.pt',models/name)
            selected.append({'file':name,'seed':seed,'phase':phase,'bytes':(models/name).stat().st_size})
    (models/'README.md').write_text('''# Fixed chemical response models

The four self-contained checkpoints are the predetermined 12-epoch frozen-core
warmup and two-epoch joint endpoints for seeds 753 and 754. They include the
inherited recurrent core, task interfaces and conditional response model.
No initial, intermediate or duplicate candidate weights are included.

Load with `experiments.continual_chemistry.observations.load_chemistry_policy`.
Native execution uses `python -m experiments.continual_chemistry.native
--checkpoint chemistry-models/joint753.pt --policy model --seed 7401
--primitive-candidates --output runs/chemistry_example` (on one command line).
Refer to the matching source repository for results, scope and model license.
''')
    record={'episodes':len(manifest['episodes']),'returned_experiments':trials,'native_actions':actions,
        'models':selected,'archives':[archive(data),archive(models)],
        'status':'Prepared locally; not published','source':'experiments/continual_chemistry/README.md'}
    (args.output/'package.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record,indent=2))


if __name__=='__main__':main()
