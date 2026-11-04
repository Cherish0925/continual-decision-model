"""Train on a fixed set of completed demonstrations, then learn by acting."""
import argparse
import json
from pathlib import Path
import subprocess
import sys


def completed_traces(roots):
    files, rows = [], []
    for root in roots:
        for summary in sorted(root.glob('*/summary.json')):
            path = summary.parent / 'trajectory.jsonl'
            if not path.exists() or not path.stat().st_size:
                continue
            row = json.loads(summary.read_text())
            files.append(path)
            rows.append({key: row[key] for key in ('theme', 'difficulty', 'seed', 'steps', 'success', 'status')})
    if not files:
        raise ValueError('No completed demonstrations are available')
    return files, rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--initial-checkpoint', type=Path, required=True)
    p.add_argument('--train-collections', type=Path, nargs='+', required=True)
    p.add_argument('--development-collections', type=Path, nargs='+', required=True)
    p.add_argument('--scope', choices=('basic_operations','complete_solutions'), required=True)
    p.add_argument('--mode', choices=('cdm','gru'), required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--feature-cache', type=Path, required=True)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--epochs', type=int, default=8)
    p.add_argument('--seed', type=int, default=301)
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--threads', type=int, default=1)
    p.add_argument('--wait-for-collections', action='store_true')
    p.add_argument('--evaluate', action='store_true',
                   help='Execute the final imitation and reinforcement endpoints without a teacher')
    p.add_argument('--evaluation-workers', type=int, default=2)
    p.add_argument('--evaluation-rng-from', type=Path,
                   help='Use common initial simulator random streams for endpoint comparisons')
    p.add_argument('--learning-seeds', nargs='+', type=int, default=[1000,1001,1002,1003])
    args = p.parse_args()
    if args.wait_for_collections:
        from .train_suite import wait_for_collections
        wait_for_collections([*args.train_collections, *args.development_collections])
    train, train_rows = completed_traces(args.train_collections)
    development, dev_rows = completed_traces(args.development_collections)
    if args.scope == 'basic_operations' and any(not row['theme'].startswith('Small Skills:')
                                              for row in train_rows + dev_rows):
        raise ValueError('Basic-operation supervision cannot include scientific task solutions')
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = dict(scope=args.scope, training=train_rows, development=dev_rows,
                    initial_checkpoint=str(args.initial_checkpoint),
                    teacher_actions=sum(row['steps'] for row in train_rows),
                    selected_after_completion=True, imitation='successful episodes only',
                    random_seed=args.seed, mode=args.mode)
    (args.output/'data.json').write_text(json.dumps(manifest,indent=2)+'\n')
    common = [sys.executable,'-u','-m']
    train_command = common + ['experiments.discoveryworld.train',
        '--resume-checkpoint',str(args.initial_checkpoint),
        '--train-trajectories',*map(str,train),'--validation-trajectories',*map(str,development),
        '--output',str(args.output/'imitation'),'--feature-cache',str(args.feature_cache),
        '--device',args.device,'--threads',str(args.threads),'--mode',args.mode,'--epochs',str(args.epochs),
        '--batch-size','4','--sequence-length','8','--precision','bfloat16',
        '--disable-cudnn-attention','--allow-failed-actions','--imitate-successful-episodes',
        '--imitation-reduction','action_run','--outcome-weight','0.5','--immediate-weight','0',
        '--seed',str(args.seed)]
    rl_command = common + ['experiments.discoveryworld.finetune',
        '--checkpoint',str(args.output/'imitation/last.pt'),'--output',str(args.output/'reinforcement'),
        '--feature-cache',str(args.feature_cache),'--device',args.device,
        '--native-suite','all','--small-skills-first','--seeds',*map(str,args.learning_seeds),
        '--workers',str(args.workers),'--threads',str(args.threads),'--sequence-length','8',
        '--mode',args.mode,'--random-seed',str(args.seed),'--lr','0.0001']
    (args.output/'commands.json').write_text(json.dumps([train_command,rl_command],indent=2)+'\n')
    for command in (train_command,rl_command):
        print(json.dumps({'starting':command[3],'scope':args.scope,'mode':args.mode}),flush=True)
        subprocess.run(command,check=True)
    if args.evaluate:
        for stage in ('imitation', 'reinforcement'):
            config = args.output / (stage + '-policy.json')
            config.write_text(json.dumps(dict(checkpoint=str(args.output/stage/'last.pt'),
                feature_cache=str(args.feature_cache), device=args.device, mode=args.mode),indent=2)+'\n')
            command = common + ['experiments.discoveryworld.full_suite','run',
                '--policy','experiments.discoveryworld.evaluate:create',
                '--policy-config',str(config),'--policy-id',args.scope+'-'+args.mode+'-'+stage,
                '--workers',str(args.evaluation_workers),'--threads','1',
                '--output',str(args.output/(stage+'-evaluation'))]
            if args.evaluation_rng_from:
                command.extend(('--rng-from',str(args.evaluation_rng_from)))
            subprocess.run(command,check=True)


if __name__ == '__main__':
    main()
