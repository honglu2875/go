#!/usr/bin/env python3
"""Run one pre-registered four-host full-state fork/restart qualification."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time

sys.dont_write_bytecode = True
SOURCE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(SOURCE / 'packages/gozero/src'))
from gozero.checkpoints import sha256
from gozero.snapshots import canonical_json, read_json, verify


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workspace-root', type=Path, required=True); p.add_argument('--spec', type=Path, required=True)
    a = p.parse_args(); root = a.workspace_root.resolve(); verify(SOURCE); spec = read_json(a.spec)
    if spec['runner_snapshot'] != SOURCE.name or spec['maximum_pod_attempts'] != 4:
        raise ValueError('Wrong runner or attempt budget')
    for name, expected in spec['qualified_code_sha256'].items():
        if sha256(SOURCE / name) != expected:
            raise ValueError('Qualified code changed: ' + name)
    for item in spec['prerequisites']:
        if sha256(root / item['path']) != item['sha256'] or read_json(root / item['path'])['status'] != 'passed':
            raise ValueError('Prerequisite changed or failed')
    if sha256(SOURCE / 'resolved_config.json') != spec['config_sha256']:
        raise ValueError('Configuration changed')
    output = root / spec['output']; output.mkdir(parents=True, exist_ok=False)
    (output / 'spec.json').write_bytes(canonical_json(spec))
    report = {'schema_version': 1, 'kind': spec['kind'], 'status': 'running', 'analysis_snapshot': SOURCE.name,
              'spec_sha256': sha256(a.spec), 'started_unix': time.time(), 'commands': [], 'attempts': {}}
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1')

    def run(name, arguments):
        command = [sys.executable, '-B', *map(str, arguments)]
        record = {'name': name, 'command': command, 'started_unix': time.time()}
        report['commands'].append(record); print('Starting ' + name, flush=True)
        try:
            with (output / (name + '.log')).open('xb') as log:
                # The pod controller owns bounded stage/prepare/run/collection
                # and peer cancellation; an outer kill must not strand it.
                process = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=False)
            record['returncode'] = process.returncode
            if process.returncode:
                raise RuntimeError(name + ' failed; see retained controller log')
        finally:
            record['finished_unix'] = time.time(); record['log_sha256'] = sha256(output / (name + '.log'))

    def pod(name, sid, extra):
        run(name, [SOURCE / 'ops/pod_run.py', '--workspace-root', root, '--snapshot', root / '.gozero/snapshots' / sid,
                   '--timeout', spec['timeout_seconds'], '--prepare-timeout', spec['prepare_timeout_seconds'],
                   '--controller-cpus', 32, '--chips', 16, *extra])
        import json
        lines = [json.loads(line) for line in (output / (name + '.log')).read_text().splitlines() if line.startswith('{')]
        launch = next(row for row in lines if row.get('kind') == 'pod_attempt')
        attempt = root / 'runs' / launch['attempt_id']; result = read_json(attempt / 'result.json')
        if result['status'] != 'passed' or result['snapshot_id'] != sid:
            raise ValueError('Pod result identity or status differs')
        report['attempts'][name] = {'attempt': attempt.name, 'result_sha256': sha256(attempt / 'result.json'),
                                    'reserved_chip_hours': result['reserved_chip_hours']}
        return attempt

    try:
        reference = pod('reference', spec['parent_snapshot'], ['--native-receipt', root / spec['native_receipt'],
                         '--resume-attempt', spec['parent_attempt'], '--resume-turn', spec['parent_turn'],
                         '--stop-after-turn', spec['final_turn']])
        full = pod('fork', SOURCE.name, [])
        short = pod('short', SOURCE.name, ['--stop-after-turn', spec['resume_turn']])
        resumed = pod('resumed', SOURCE.name, ['--resume-attempt', short.name, '--resume-turn', spec['resume_turn']])
        run('fork_comparison', [SOURCE / 'research/recipes/continued_selfplay/compare_fork.py', '--workspace-root', root,
                               '--reference', reference, '--fork', full, '--hosts', 4, '--final-turn', spec['final_turn'],
                               '--output', output / 'fork_comparison.json'])
        run('resume_comparison', [SOURCE / 'ops/compare_resume.py', '--continuous', full, '--resumed', resumed,
                                 '--resume-origin', short, '--resume-turn', spec['resume_turn'], '--final-turn', spec['final_turn'],
                                 '--hosts', 4, '--output', output / 'resume_comparison.json'])
        report['comparisons'] = {name: {'path': str((output / (name + '.json')).relative_to(root)),
                                       'sha256': sha256(output / (name + '.json'))}
                                 for name in ('fork_comparison', 'resume_comparison')}
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
