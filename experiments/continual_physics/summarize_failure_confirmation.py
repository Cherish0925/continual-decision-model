"""Whole-stream paired results for the fixed failure-conditioning intervention."""
from __future__ import annotations
import csv,json
from pathlib import Path
from experiments.physical_belief.summarize_native import load_rows
from .summarize_native import comparison,paired_seconds
from .failure_confirmation import ROOT,MANIFEST


def main():
    plan=json.loads((ROOT/'evaluation_plan.json').read_text());manifest=json.loads(MANIFEST.read_text())
    expected={(s['name'],i,seed) for s in manifest['streams'] for i,seed in enumerate(s['seeds'])}
    rows={job['name']:load_rows(ROOT/job['name']) for job in plan['jobs']}
    for name,values in rows.items():
        actual={(r['stream'],r['episode_index'],r['seed']) for r in values}
        if len(values)!=120 or actual!=expected:raise ValueError(f'Incomplete confirmation: {name}')
    result={'protocol':plan,'conditions':{},'comparisons':{}}
    for name,values in rows.items():
        result['conditions'][name]={'instances':len(values),'completed':sum(r['completed_successfully'] for r in values),
          'mean_actions':sum(r['native_actions'] for r in values)/len(values),
          'mean_episode_seconds':sum(r['seconds'] for r in values)/len(values),
          'execution_errors':sum(bool(r.get('error')) for r in values)}
    for control in ('task_reset_seed232','parent_seed222'):
        left=rows['task_reset_seed232_conditioned'];right=rows[control]
        result['comparisons']['conditioned:'+control]=comparison(left,right)
        result['comparisons']['conditioned:'+control]['timing']={key:paired_seconds(left,right,key) for key in
            ('episode_seconds','inference_seconds','execution_seconds','feedback_seconds')}
    destination=Path('experiments/continual_physics/results');destination.mkdir(exist_ok=True)
    (ROOT/'comparisons.json').write_text(json.dumps(result,indent=2)+'\n')
    (destination/'failure_conditioning_confirmation.json').write_text(json.dumps(result,indent=2)+'\n')
    keys=['stream','episode_index','seed','segment_role','position_in_segment','completed_successfully','native_actions','measurement_requests','test_requests','seconds']
    with (destination/'failure_conditioning_instances.csv').open('w') as file:
        writer=csv.DictWriter(file,fieldnames=['condition',*keys]);writer.writeheader()
        writer.writerows({'condition':name,**{key:row.get(key) for key in keys}} for name,values in rows.items() for row in values)
    print(json.dumps({'conditions':result['conditions'],'comparisons':{name:value['all'] for name,value in result['comparisons'].items()}},indent=2))

if __name__=='__main__':main()
