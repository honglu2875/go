#!/usr/bin/env python3
"""Execute one registered CPU fork and ordinary-resume qualification."""
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
    p.add_argument('--workspace-root', type=Path, required=True)
    p.add_argument('--spec', type=Path, required=True)
    a = p.parse_args(); root = a.workspace_root.resolve(); verify(SOURCE)
    spec = read_json(a.spec)
    if spec['runner_snapshot'] != SOURCE.name or spec['maximum_attempts'] != 1:
        raise ValueError('Wrong frozen runner or attempt budget')
    for name, expected in spec['qualified_code_sha256'].items():
        if sha256(SOURCE / name) != expected:
            raise ValueError('Qualified code changed: ' + name)
    for item in spec['prerequisites']:
        if sha256(root / item['path']) != item['sha256'] or read_json(root / item['path'])['status'] != 'passed':
            raise ValueError('Qualification prerequisite changed or failed')
    if sha256(SOURCE / 'resolved_config.json') != spec['config_sha256']:
        raise ValueError('Frozen configuration changed')
    output = root / spec['output']; output.mkdir(parents=True, exist_ok=False)
    (output / 'spec.json').write_bytes(canonical_json(spec))
    report = {'schema_version': 1, 'kind': spec['kind'], 'status': 'running', 'analysis_snapshot': SOURCE.name,
              'spec_sha256': sha256(a.spec), 'started_unix': time.time(), 'commands': []}
    env = dict(os.environ, PYTHONPATH=str(SOURCE / 'packages/gozero/src'), PYTHONDONTWRITEBYTECODE='1',
               JAX_PLATFORMS='cpu', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', GOZERO_HOST_RANK='0')
    python = ['taskset', '-c', ','.join(map(str, spec['cpus'])), sys.executable, '-B']
    recipe = SOURCE / 'research/recipes/continued_selfplay'

    def run(name, arguments):
        command = python + list(map(str, arguments)); record = {'name': name, 'command': command, 'started_unix': time.time()}
        report['commands'].append(record); print('Starting ' + name, flush=True)
        try:
            with (output / (name + '.log')).open('xb') as log:
                process = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT,
                                         timeout=spec['timeout_seconds_per_command'], check=False)
            record['returncode'] = process.returncode
            if process.returncode:
                raise RuntimeError(name + ' failed; see retained log')
        finally:
            record['finished_unix'] = time.time()
            record['log_sha256'] = sha256(output / (name + '.log'))

    try:
        common = [recipe / 'train.py', '--config', SOURCE / 'resolved_config.json']
        run('full', common + ['--output', output / 'full'])
        run('short', common + ['--output', output / 'short', '--stop-after-turn', spec['resume_turn']])
        run('resumed', common + ['--output', output / 'resumed', '--resume',
                                output / 'short/checkpoints' / f"turn-{spec['resume_turn']:09d}"])
        run('fork_comparison', [recipe / 'compare_fork.py', '--workspace-root', root, '--reference', root / spec['reference'],
                               '--fork', output / 'full', '--hosts', 1, '--final-turn', spec['final_turn'],
                               '--output', output / 'fork_comparison.json'])
        run('resume_comparison', [SOURCE / 'ops/compare_resume.py', '--continuous', output / 'full',
                                 '--resumed', output / 'resumed', '--resume-origin', output / 'short',
                                 '--resume-turn', spec['resume_turn'], '--final-turn', spec['final_turn'], '--hosts', 1,
                                 '--output', output / 'resume_comparison.json'])
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
