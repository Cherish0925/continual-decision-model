"""Summarize whole-stream comparisons for the classical calibration reference."""
from __future__ import annotations
import argparse,csv,json
from pathlib import Path
import numpy as np


def rows(root):
    return [json.loads(path.read_text()) for path in sorted(Path(root).glob('*/episode*/summary.json'))]


def paired(left,right,*,draws=10000):
    left={r['seed']:r for r in left};right={r['seed']:r for r in right}
    if left.keys()!=right.keys():raise ValueError('Paired inputs must contain exactly the same instances')
    names=sorted({r['stream'] for r in left.values()});rng=np.random.default_rng(117)
    picks=rng.integers(0,len(names),size=(draws,len(names)))
    report={}
    for metric in ('completed_successfully','native_actions','seconds','inference_seconds','execution_seconds'):
        if not all(metric in r for r in (*left.values(),*right.values())):continue
        values=np.asarray([np.mean([float(left[k][metric])-float(right[k][metric]) for k in left if left[k]['stream']==name]) for name in names])
        sample=values[picks].mean(1)
        report[metric]={'mean_difference':float(values.mean()),'confidence_interval_95':np.quantile(sample,[.025,.975]).tolist()}
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--reset',required=True);p.add_argument('--forgetting',required=True)
    p.add_argument('--output',required=True);p.add_argument('--manifest',required=True);p.add_argument('--phase',choices=('development','confirmation'),required=True)
    p.add_argument('--parent-root');a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    expected={s for stream in json.loads(Path(a.manifest).read_text())['streams'] for s in stream['seeds']}
    groups={name:rows(path) for name,path in (('reset',a.reset),('forgetting',a.forgetting))}
    result={'phase':a.phase,'conditions':{},'comparisons':{},'resampling_unit':'complete ordered six-task stream'}
    flat=[]
    for name,group in groups.items():
        if {r['seed'] for r in group}!=expected or len(group)!=len(expected):raise ValueError('Missing or duplicate instances')
        if any(r['error'] for r in group):raise ValueError('Native execution errors cannot be treated as task failures')
        result['conditions'][name]={'instances':len(group),'completed':sum(r['completed_successfully'] for r in group),
            **{f'mean_{key}':float(np.mean([r[key] for r in group])) for key in ('native_actions','seconds','inference_seconds','execution_seconds')},
            'first_return_mean_actions':float(np.mean([r['native_actions'] for r in group if r['segment_role']=='return' and r['first_in_segment']]))}
        for r in group:flat.append({'method':name,**{k:r[k] for k in ('stream','seed','segment_role','position_in_segment','completed_successfully','native_actions','seconds','inference_seconds','execution_seconds')}})
    result['comparisons']['forgetting:reset']=paired(groups['forgetting'],groups['reset'])
    if a.parent_root:
        for parent in ('parent_seed221','parent_seed222'):
            reference=rows(Path(a.parent_root)/parent)
            for name,group in groups.items():result['comparisons'][f'{name}:{parent}']=paired(group,reference)
    (out/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    with (out/'instances.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(flat[0]),lineterminator="\n");w.writeheader();w.writerows(flat)

if __name__=='__main__':main()
