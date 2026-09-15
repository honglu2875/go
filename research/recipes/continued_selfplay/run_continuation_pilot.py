#!/usr/bin/env python3
"""Execute the two registered full-state learning-rate continuation arms once."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

sys.dont_write_bytecode = True
SOURCE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(SOURCE / 'packages/gozero/src'))
from gozero.checkpoint_forks import contract
from gozero.checkpoints import sha256
from gozero.snapshots import canonical_json, read_json, verify


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workspace-root', type=Path, required=True); p.add_argument('--spec', type=Path, required=True)
    a = p.parse_args(); root = a.workspace_root.resolve(); verify(SOURCE); spec = read_json(a.spec)
    if spec['runner_snapshot'] != SOURCE.name or spec['maximum_training_attempts'] != 2 or spec['arm_order'] != ['inherit', 'anneal']:
        raise ValueError('Wrong frozen runner or budget')
    for item in spec['prerequisites']:
        if sha256(root / item['path']) != item['sha256'] or read_json(root / item['path'])['status'] != 'passed':
            raise ValueError('Prerequisite changed or failed')
    for arm in spec['arms'].values():
        source = root / '.gozero/snapshots' / arm['snapshot']; verify(source)
        config = read_json(source / 'resolved_config.json'); origin, descriptor, _, _ = contract(root, source, config)
        if (sha256(source / 'resolved_config.json') != arm['config_sha256'] or origin['parent_turn'] != spec['parent_turn']
                or descriptor['parent_group_sha256'] != spec['parent_group_sha256'] or config['selfplay_turns'] != spec['final_turn']
                or config['learner']['learning_rate'] != arm['learning_rate']):
            raise ValueError('Registered arm differs from complete-state contract')
        for path, expected in spec['training_code_sha256'].items():
            if sha256(source / path) != expected:
                raise ValueError('Qualified training code changed: ' + path)
    output = root / spec['training_output']; output.mkdir(parents=True, exist_ok=False)
    (output / 'spec.json').write_bytes(canonical_json(spec))
    report = {'schema_version': 1, 'kind': 'online_annealing_training_execution', 'status': 'running',
              'runner_snapshot': SOURCE.name, 'spec_sha256': sha256(a.spec), 'started_unix': time.time(), 'arms': {}}
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1')
    try:
        for name in spec['arm_order']:
            arm = spec['arms'][name]; command = [sys.executable, '-B', str(SOURCE / 'ops/pod_run.py'), '--workspace-root', str(root),
                '--snapshot', str(root / '.gozero/snapshots' / arm['snapshot']), '--timeout', str(spec['training_timeout_seconds']),
                '--prepare-timeout', str(spec['prepare_timeout_seconds']), '--controller-cpus', '32', '--chips', '16']
            print('Starting registered arm ' + name, flush=True)
            row = {'command': command, 'started_unix': time.time()}; report['arms'][name] = row
            with (output / (name + '.log')).open('xb') as log:
                process = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=False)
            row.update(returncode=process.returncode, finished_unix=time.time(), log_sha256=sha256(output / (name + '.log')))
            launches = [json.loads(line) for line in (output / (name + '.log')).read_text().splitlines()
                        if line.startswith('{') and json.loads(line).get('kind') == 'pod_attempt']
            if len(launches) != 1:
                raise ValueError('One controller attempt is required per arm')
            row['attempt'] = launches[0]['attempt_id']; path = root / 'runs' / row['attempt'] / 'result.json'; result = read_json(path)
            row.update(result_sha256=sha256(path), reserved_chip_hours=result['reserved_chip_hours'])
            if process.returncode != 0 or result['status'] != 'passed' or result['snapshot_id'] != arm['snapshot']:
                raise ValueError('Registered training arm failed')
            (output / 'progress.json').write_bytes(canonical_json(report))
            print(json.dumps({'arm': name, **row}), flush=True)
        report['status'] = 'passed'
    except BaseException as error:
        report.update(status='failed', error=repr(error)); raise
    finally:
        report['finished_unix'] = time.time()
        with (output / 'result.json').open('xb') as stream:
            stream.write(canonical_json(report))
        print(canonical_json(report).decode(), flush=True)


if __name__ == '__main__':
    main()
