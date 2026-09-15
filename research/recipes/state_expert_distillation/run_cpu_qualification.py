#!/usr/bin/env python3
"""Bounded frozen model/native tests and exact recovery for both expert arms."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

sys.dont_write_bytecode = True
SOURCE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(SOURCE / 'packages/gozero/src'))
from gozero.checkpoints import sha256
from gozero.snapshots import canonical_json, read_json, verify


def require(value, message):
    if not value:
        raise ValueError(message)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workspace-root', type=Path, required=True); p.add_argument('--protocol', type=Path, required=True)
    p.add_argument('--expected-protocol-sha256', required=True); p.add_argument('--output', type=Path, required=True)
    a = p.parse_args(); verify(SOURCE); root = a.workspace_root.resolve()
    require(sha256(a.protocol) == a.expected_protocol_sha256, 'Qualification protocol changed')
    protocol = read_json(a.protocol)
    require(protocol['kind'] == 'state_expert_cpu_qualification' and protocol['maximum_attempts'] == 1
            and protocol['runner_snapshot'] == SOURCE.name and protocol['registered_unix'] < time.time(), 'Wrong qualification identity')
    require(a.output.resolve() == root / protocol['output'] and not a.output.exists(), 'Qualification output differs or exists')
    for mode, arm in protocol['arms'].items():
        source = root / '.gozero/snapshots' / arm['snapshot']; verify(source)
        require(sha256(source / 'resolved_config.json') == arm['config_sha256']
                and read_json(source / 'resolved_config.json')['model']['architecture'] == mode, 'Arm identity differs')
        for name, digest in protocol['qualified_code_sha256'].items():
            require(sha256(source / name) == digest, 'Arm code differs: ' + name)
    native = root / protocol['native']['path']; require(sha256(native) == protocol['native']['sha256'], 'Native binary differs')
    a.output.mkdir(parents=True, exist_ok=False)
    env = os.environ.copy(); env.update(JAX_PLATFORMS='cpu', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
        PYTHONPATH=str(SOURCE / 'packages/gozero/src'), GOZERO_NATIVE_LIBRARY=str(native), GOZERO_NATIVE_SHA256=protocol['native']['sha256'])
    report = {'schema_version': 1, 'kind': protocol['kind'], 'status': 'failed', 'snapshot': SOURCE.name,
              'protocol_sha256': a.expected_protocol_sha256, 'started_unix': time.time(), 'commands': [], 'arms': {}}
    def run(name, command, timeout):
        start = time.time(); entry = {'id': name, 'command': command, 'timeout_seconds': timeout, 'started_unix': start}
        report['commands'].append(entry)
        with (a.output / (name + '.log')).open('x') as log:
            child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True)
            try:
                code = child.wait(timeout=timeout)
            except BaseException:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL); child.wait()
                entry.update(returncode=child.returncode, finished_unix=time.time(), timed_out=True,
                             log_sha256=sha256(a.output / (name + '.log')))
                raise
        entry.update(returncode=code, finished_unix=time.time(), log_sha256=sha256(a.output / (name + '.log')))
        print(json.dumps(entry), flush=True); require(code == 0, 'Qualification phase failed: ' + name)
    try:
        run('tests', [sys.executable, '-B', '-m', 'unittest', 'discover', '-s', str(Path(__file__).parent), '-p', 'test_*.py', '-v'],
            protocol['test_timeout_seconds'])
        for mode in ('history', 'state'):
            arm = protocol['arms'][mode]; source = root / '.gozero/snapshots' / arm['snapshot']
            folder = a.output / mode; folder.mkdir()
            trainer = source / 'research/recipes/state_expert_distillation/train.py'
            command = [sys.executable, '-B', str(trainer), '--config', str(source / 'resolved_config.json')]
            run(mode + '-baseline', command + ['--output', str(folder / 'baseline')], protocol['training_timeout_seconds'])
            checkpoint = folder / 'baseline/checkpoints' / f'turn-{protocol["resume_turn"]:09d}'
            run(mode + '-resumed', command + ['--output', str(folder / 'resumed'), '--resume', str(checkpoint)], protocol['training_timeout_seconds'])
            recovery = folder / 'recovery.json'
            run(mode + '-audit', [sys.executable, '-B', str(Path(__file__).parent / 'qualify_recovery.py'),
                '--workspace-root', str(root), '--baseline', str(folder / 'baseline'), '--resumed', str(folder / 'resumed'),
                '--hosts', '1', '--output', str(recovery)], 60)
            report['arms'][mode] = {'recovery_sha256': sha256(recovery), 'recovery': read_json(recovery)}
        report['status'] = 'passed'
    except BaseException as error:
        report['error'] = repr(error); raise
    finally:
        report['finished_unix'] = time.time()
        with (a.output / 'result.json').open('xb') as f:
            f.write(canonical_json(report))
        verify(SOURCE); require(sha256(a.protocol) == a.expected_protocol_sha256, 'Qualification protocol changed during execution')
        print(json.dumps({k: v for k, v in report.items() if k not in ('commands', 'arms')}), flush=True)


if __name__ == '__main__':
    main()
