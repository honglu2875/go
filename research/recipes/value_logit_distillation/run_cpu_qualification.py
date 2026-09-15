#!/usr/bin/env python3
"""Bounded fresh-process tests and exact learner continuation for BCE2."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

sys.dont_write_bytecode = True
SOURCE = Path(__file__).resolve().parents[3]
RECIPE = Path(__file__).resolve().parent
sys.path.insert(0, str(SOURCE / 'packages/gozero/src'))
from gozero.checkpoints import sha256
from gozero.snapshots import canonical_json, read_json, verify


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workspace-root', type=Path, required=True); p.add_argument('--protocol', type=Path, required=True)
    p.add_argument('--expected-protocol-sha256', required=True); a = p.parse_args(); verify(SOURCE)
    if sha256(a.protocol) != a.expected_protocol_sha256:
        raise ValueError('CPU qualification protocol changed')
    protocol = read_json(a.protocol); root = a.workspace_root.resolve(); c = read_json(SOURCE / 'resolved_config.json')
    if (protocol['snapshot'] != SOURCE.name or protocol['config_sha256'] != sha256(SOURCE / 'resolved_config.json')
            or protocol['kind'] != 'value_logit_cpu_recovery_qualification' or protocol['maximum_training_invocations'] != 2
            or c['platform'] != 'cpu' or c['model']['value_objective'] != 'bce2'
            or c['steps'] != 8 or protocol['resume_turn'] != 4):
        raise ValueError('CPU qualification scope differs')
    output = root / protocol['output']; output.mkdir(parents=True, exist_ok=False)
    report = {'schema_version': 1, 'kind': 'value_logit_cpu_qualification', 'status': 'running', 'snapshot': SOURCE.name,
              'protocol_sha256': a.expected_protocol_sha256, 'started_unix': time.time(), 'commands': []}
    def execute(name, argv):
        command = [sys.executable, '-B', *argv]; before = time.time()
        print(json.dumps({'kind': 'qualification_phase', 'phase': name}), flush=True)
        with (output / (name + '.log')).open('x') as stream:
            result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, timeout=protocol['maximum_seconds_per_segment'])
        report['commands'].append({'phase': name, 'argv': command, 'returncode': result.returncode,
                                   'seconds': time.time() - before, 'log_sha256': sha256(output / (name + '.log'))})
        if result.returncode:
            raise RuntimeError(name + ' failed; retained output is not retried')
    try:
        execute('tests', ['-m', 'unittest', 'discover', '-s', str(RECIPE), '-p', 'test_*.py', '-v'])
        execute('baseline', [str(RECIPE / 'train.py'), '--config', str(SOURCE / 'resolved_config.json'), '--output', str(output / 'baseline')])
        execute('resumed', [str(RECIPE / 'train.py'), '--config', str(SOURCE / 'resolved_config.json'), '--output', str(output / 'resumed'),
                            '--resume', str(output / 'baseline/checkpoints/turn-000000004')])
        execute('recovery_audit', [str(RECIPE / 'qualify_recovery.py'), '--workspace-root', str(root), '--baseline', str(output / 'baseline'),
                                  '--resumed', str(output / 'resumed'), '--hosts', '1', '--output', str(output / 'recovery.json')])
        recovery = read_json(output / 'recovery.json')
        if recovery['status'] != 'passed':
            raise ValueError('Recovery audit did not pass')
        report.update(status='passed', recovery_sha256=sha256(output / 'recovery.json'), recovery=recovery)
    except BaseException as error:
        report.update(status='failed', error=repr(error)); raise
    finally:
        report['finished_unix'] = time.time(); (output / 'result.json').write_bytes(canonical_json(report)); verify(SOURCE)
        print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
