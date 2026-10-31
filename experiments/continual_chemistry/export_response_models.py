"""Package the two frozen parents and fitted response rules for reproduction."""
import argparse
from pathlib import Path
import tarfile

import torch

from experiments.continual_experiment.shared_tasks import load_shared_checkpoint


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    args = p.parse_args()
    torch.set_num_threads(1)
    package = args.output / 'response-learning-models'
    package.mkdir(parents=True, exist_ok=False)
    (package / 'parents').mkdir()
    for seed in (911, 912):
        source = torch.load(args.root / 'active_study' / f'relation{seed}' / 'update.pt', map_location='cpu', weights_only=True)
        model, record = load_shared_checkpoint(source['parent'])
        training = {k: record['training'][k] for k in ('state_sharing', 'chemical_feedback_statistics')}
        training['source'] = f'completed active-feedback continuation {seed}'
        torch.save({**model.checkpoint(), 'training': training}, package / 'parents' / f'parent{seed}.pt')
    for name, directory in [('recorded', 'study'), ('selected', 'active_study')]:
        destination = package / name; destination.mkdir()
        for source in sorted((args.root / directory).glob('*/update.pt')):
            data = torch.load(source, map_location='cpu', weights_only=True)
            seed = source.parent.name[-3:]
            data['parent'] = f'parents/parent{seed}.pt'
            torch.save(data, destination / (source.parent.name + '.pt'))
    (package / 'README.md').write_text('''# Response-learning models

Two complete shared-core parents and sixteen fitted response rules from the
completed study. The physical and chemical core weights remain those of each
parent; the rules change chemical prediction. This archive does not contain the
joint continuations that were still training when the study was released.

Use the repository environment and run from its root. After unpacking:

```python
import torch
from cdm.response_replay import ResponseReplay, ResponseReplayConfig
from experiments.continual_experiment.shared_tasks import load_shared_checkpoint

root = "response-learning-models"
model, _ = load_shared_checkpoint(f"{root}/parents/parent911.pt")
rule = torch.load(f"{root}/selected/relation911.pt", map_location="cpu", weights_only=True)
chemical = model.tasks["chemistry"]
chemical.response_replay = ResponseReplay(config=ResponseReplayConfig(**rule["config"]))
chemical.response_replay.load_state_dict(rule["model"])
model.eval()
```

`recorded/` rules fit old exploration trajectories. `selected/` rules fit
parent-selected experiments and returned outcomes. Calibration, stored-feature,
refreshed-feature and relation-kernel fits are all included for both parents.
See `experiments/continual_experiment/RESPONSE_LEARNING.md` in the repository
for training data, native tasks, comparisons and the unfinished joint study.
''')
    archive = args.output / 'response-learning-models.tar.gz'
    with tarfile.open(archive, 'w:gz') as file:
        file.add(package, arcname=package.name)
    print(f'{archive}: {archive.stat().st_size / 2**20:.1f} MiB')


if __name__ == '__main__':
    main()
