"""A waiting checkpoint must not occupy the only available evaluation slot."""
import json
from pathlib import Path
import subprocess
import sys


def test_ready_job_can_release_an_earlier_waiting_job(tmp_path):
    script = r'''
from pathlib import Path
import sys
from experiments.continual_experiment.joint_belief import execute
root = Path(sys.argv[1])
program = "from pathlib import Path; import sys; p=Path(sys.argv[2]); p.mkdir(); (p/'finished.json').write_text('{}'); "
consumer = dict(name='consumer', ready=str(root/'producer/finished.json'),
    arguments=['-c', program, '--output', str(root/'consumer')])
producer = dict(name='producer', arguments=['-c', program, '--output', str(root/'producer')])
execute([consumer, producer], root/'queue', 1)
'''
    subprocess.run([sys.executable, '-c', script, str(tmp_path)],
                   cwd=Path(__file__).resolve().parents[1], check=True, timeout=30)
    rows = json.loads((tmp_path / 'queue/completed.json').read_text())
    assert [row['name'] for row in rows] == ['producer', 'consumer']
    assert all(row['exit_code'] == 0 for row in rows)
