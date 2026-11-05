"""Native simulator subprocess, independent of Torch and the GPU runtime."""
from contextlib import redirect_stdout
import json
from pathlib import Path
import random
import sys
from time import perf_counter
import traceback

import numpy as np

from .adapter import TrajectoryWriter
from .structured_observation import StructuredDiscoveryWorldEnv


class JsonConnection:
    """One request and one reply over a subprocess's dedicated standard streams."""

    def __init__(self, reader, writer):
        self.reader, self.writer = reader, writer

    def send(self, value):
        self.writer.write(json.dumps(value) + '\n')
        self.writer.flush()

    def recv(self):
        line = self.reader.readline()
        if not line:
            raise EOFError('The native simulator connection closed')
        return json.loads(line)

    def close(self):
        self.reader.close()
        self.writer.close()


def serve(pipe):
    writer = log = None
    try:
        while True:
            command, payload = pipe.recv()
            if command == 'close':
                return
            if command == 'reset':
                if writer:
                    writer.close()
                    log.close()
                case, folder, settings = payload
                theme, difficulty, seed, limit = case
                random.seed(settings['environment_rng'])
                np.random.seed(settings['environment_rng'] % (2**32))
                folder = Path(folder)
                folder.mkdir(parents=True, exist_ok=True)
                log = (folder / 'simulator.log').open('w')
                started = perf_counter()
                with redirect_stdout(log):
                    if settings.get('features',{}).get('action_interface') == 'object_requests_v1':
                        from .request_environment import RequestEnvironment
                        env = RequestEnvironment(folder,limit,max_request_actions=settings.get('max_request_actions',12))
                    else:
                        env = StructuredDiscoveryWorldEnv(folder, limit)
                    observation = env.reset(theme, difficulty, seed)
                    env.save_initial_rng(folder / 'initial_rng.json.gz')
                    configuration = dict(env.configuration, policy='batched_on_policy_actor_critic',
                        action_descriptions=env.actions, teleport_locations=env.teleports,
                        initial_task_metrics=env.metrics, **settings)
                    writer = TrajectoryWriter(folder, configuration)
                pipe.send({'observation': observation})
            elif command == 'step':
                action, behavior = payload
                potential_before = sum(m['progress'] for m in env.metrics)/len(env.metrics)
                with redirect_stdout(log):
                    transition = env.step(action)
                    writer.append({**transition, **behavior})
                reply = {key: transition[key] for key in
                         ('action', 'action_result', 'index', 'training_reward', 'terminated', 'truncated')}
                reply['observation'] = env.policy_input()
                reply['done'] = env.ended
                if 'executor' in transition:
                    reply.update(executor=transition['executor'], potential_before=potential_before,
                        potential_after=sum(m['progress'] for m in env.metrics)/len(env.metrics),
                        completed_successfully=all(m['completed_successfully'] for m in env.metrics))
                if env.ended:
                    summary = dict(scenario=case[0], difficulty=case[1], seed=case[2], steps=env.steps,
                        completed_successfully=all(m['completed_successfully'] for m in env.metrics),
                        task_metrics=env.metrics, elapsed_seconds=perf_counter()-started)
                    if 'executor' in transition:
                        summary['requests'] = env.decision_steps
                    (folder / 'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
                    reply['summary'] = summary
                    writer.close(); writer = None
                    log.close(); log = None
                pipe.send(reply)
            else:
                raise ValueError(f'Unknown simulator command: {command}')
    except EOFError:
        return
    except (Exception, SystemExit):
        pipe.send({'error': traceback.format_exc()})
    finally:
        if writer:
            writer.close()
        if log:
            log.close()


if __name__ == '__main__':
    import faulthandler
    faulthandler.enable()
    if sys.argv[1:] == ['--collection']:
        from .collect_suite import environment_worker
        environment_worker(JsonConnection(sys.stdin, sys.stdout))
    else:
        serve(JsonConnection(sys.stdin, sys.stdout))
