"""Project completed primary and additional native comparisons into result tables."""
from __future__ import annotations
import csv,json
from pathlib import Path
from cdm.continual_experiment import Activity
from experiments.physical_belief.summarize_native import load_rows


def project(name,row):
    record={'condition':name,**{k:row.get(k) for k in ('stream','episode_index','seed','segment_role','position_in_segment',
        'completed_successfully','native_actions','measurement_requests','test_requests','seconds')}}
    record.update(row.get('timing',{}))
    record.update({'activities_'+kind.name:count for kind,count in zip(Activity,row.get('internal_activity_counts',[]))})
    record.update(row.get('physical_learning',{}))
    return record


def write_csv(path,records):
    keys=list(dict.fromkeys(k for r in records for k in r))
    with path.open('w') as file:
        writer=csv.DictWriter(file,fieldnames=keys);writer.writeheader();writer.writerows(records)


def main():
    root=Path('runs/continual_physics');output=Path('experiments/continual_physics/results');output.mkdir(exist_ok=True)
    primary=root/'native_confirmation/comparisons';result=json.loads((primary/'summary.json').read_text())
    with (primary/'instances.csv').open() as file:records=list(csv.DictReader(file))
    reference=root/'recurrent_reference_native';additional=json.loads((reference/'summary.json').read_text())
    for name,condition in additional['conditions'].items():
        rows=load_rows(reference/name)
        if len(rows)!=120:raise ValueError(name)
        condition['mean_episode_seconds']=sum(r['seconds'] for r in rows)/120
        result['conditions'][name]=condition;records.extend(project(name,r) for r in rows)
    result['comparisons'].update(additional['comparisons'])
    result['additional_references']={'status':'complete','conditions':list(additional['conditions']),
        'scope':'Additional fixed recurrent comparison; original twelve primary conditions unchanged'}
    (output/'native_confirmation.json').write_text(json.dumps(result,indent=2)+'\n')
    write_csv(output/'native_instances.csv',records)
    keys=('instances','completed','mean_actions','mean_episode_seconds')
    write_csv(output/'native_confirmation.csv',[{'condition':name,**{k:values[k] for k in keys}} for name,values in result['conditions'].items()])
    initial=root/'initial_native'
    if (initial/'summary.json').exists():
        data=json.loads((initial/'summary.json').read_text())
        (output/'initial_native.json').write_text(json.dumps(data,indent=2)+'\n')
        write_csv(output/'initial_native_instances.csv',[project(name,row) for name in data['conditions'] for row in load_rows(initial/name)])
    print(json.dumps({'completed_conditions':len(result['conditions']),'native_executions':len(records),'initial_complete':(initial/'summary.json').exists()}))


if __name__=='__main__':main()
