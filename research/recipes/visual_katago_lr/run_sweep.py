#!/usr/bin/env python3
"""Run the registered LR arms serially, auditing each before the next launch.

This driver never retries a failed arm or changes its configuration. A failure
stops the queue for diagnosis while preserving the attempt and every log.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys

SOURCE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(SOURCE / 'packages/gozero/src'))
from gozero.checkpoints import sha256
from gozero.snapshots import canonical_json, read_json, verify


def require(condition, message):
    if not condition:
        raise ValueError(message)


def publish(path, value):
    with path.open('xb') as stream:
        stream.write(canonical_json(value))
        stream.flush()
        os.fchmod(stream.fileno(), 0o444)
        os.fsync(stream.fileno())


def idle(root):
    require(not any(not p.with_name('result.json').exists()
                    for p in (root / 'runs').glob('pod-*/launch.json')),
            'A pod attempt remains open')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workspace-root', type=Path, required=True)
    p.add_argument('--registration', type=Path, required=True)
    p.add_argument('--registration-sha256', required=True)
    p.add_argument('--reference-audit', type=Path, required=True)
    p.add_argument('--reference-audit-sha256', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--check-only', action='store_true')
    a = p.parse_args()
    verify(SOURCE)
    root = a.workspace_root.resolve()
    require(root == Path('/workspace/go'), 'Unexpected workspace')
    require(sha256(a.registration) == a.registration_sha256, 'Registration changed')
    require(sha256(a.reference_audit) == a.reference_audit_sha256, 'Reference audit changed')
    registration = read_json(a.registration)
    reference = read_json(a.reference_audit)
    require(reference['status'] == 'passed' and reference['steps'] == 1024
            and reference['training_snapshot'] == registration['reference']['training_snapshot'],
            'Reference does not match the registered completed model')
    cases = registration['candidates_in_execution_order']
    reference_source = root / '.gozero/snapshots' / reference['training_snapshot']
    verify(reference_source)
    control = read_json(reference_source / 'resolved_config.json')
    def without_lr(c):
        return {**c, 'learner': {k: v for k, v in c['learner'].items()
                               if k not in ('learning_rate', 'end_learning_rate')}}
    for case in cases:
        source = root / '.gozero/snapshots' / case['snapshot']
        manifest = verify(source)
        require(sha256(source / 'resolved_config.json') == case['config_sha256'], 'Candidate configuration changed')
        config = read_json(source / 'resolved_config.json')
        require(without_lr(config) == without_lr(control), 'Non-LR controls differ')
        require(config['learner']['learning_rate'] == case['peak_learning_rate']
                and config['learner']['end_learning_rate'] == case['end_learning_rate'], 'LR differs')
        for name, digest in case['all_parent_python_files_identical'].items():
            require(sha256(source / manifest['recipe'] / name) == digest, 'Candidate source changed')
    if a.check_only:
        print(canonical_json({'status': 'checked', 'rates': [c['peak_learning_rate'] for c in cases],
                              'reference_audit_sha256': a.reference_audit_sha256,
                              'operator_snapshot': SOURCE.name}).decode().strip(), flush=True)
        return
    a.output.mkdir(parents=True, exist_ok=False)
    audits = [('1e-4', str(a.reference_audit), a.reference_audit_sha256)]
    events = []
    started = datetime.now(timezone.utc).isoformat()
    def event(kind, **fields):
        row = {'kind': kind, 'time': datetime.now(timezone.utc).isoformat(), **fields}
        events.append(row)
        with (a.output / 'events.jsonl').open('ab') as stream:
            stream.write(canonical_json(row))
            stream.flush()
            os.fsync(stream.fileno())
        print(canonical_json(row).decode().strip(), flush=True)
    environment = {**os.environ, 'OPENBLAS_NUM_THREADS': '1', 'OMP_NUM_THREADS': '1',
                   'PYTHONDONTWRITEBYTECODE': '1'}
    cpu_environment = {**environment, 'JAX_PLATFORMS': 'cpu'}
    status = 'failed'
    failure = None
    try:
        with (root / 'runs/.registered-lr-sweep.lock').open('a+b') as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            for index, case in enumerate(cases, 1):
                idle(root)
                source = root / '.gozero/snapshots' / case['snapshot']
                require(not any(read_json(p)['snapshot_id'] == source.name
                                for p in (root / 'runs').glob('pod-*/launch.json')),
                        'Candidate already has an attempt; inspect it before continuing')
                disk = os.statvfs(root)
                require(disk.f_bavail * disk.f_frsize >= 3_500_000_000,
                        'Insufficient owner disk space for a complete checkpoint')
                label = format(case['peak_learning_rate'], '.0e')
                log = a.output / f'{index}-{label}-pod.log'
                command = [sys.executable, '-B', str(source / 'ops/pod_run.py'),
                           '--snapshot', str(source), '--workspace-root', str(root),
                           '--timeout', str(registration['maximum_seconds_per_attempt']),
                           '--prepare-timeout', '180', '--controller-cpus', '32']
                event('arm_launch', rate=case['peak_learning_rate'], snapshot=source.name, log=str(log))
                with log.open('xb') as stream:
                    completed = subprocess.run(command, cwd=root, env=environment,
                                               stdout=stream, stderr=subprocess.STDOUT)
                launches = [read_json(p) for p in (root / 'runs').glob('pod-*/launch.json')
                            if read_json(p)['snapshot_id'] == source.name]
                require(len(launches) == 1, 'Cannot identify the unique attempt')
                attempt = root / 'runs' / launches[0]['attempt_id']
                closed = read_json(attempt / 'result.json')
                event('arm_closed', rate=case['peak_learning_rate'], attempt=attempt.name,
                      status=closed['status'], result_sha256=sha256(attempt / 'result.json'))
                require(completed.returncode == 0 and closed['status'] == 'passed',
                        'Arm failed; queue stopped for diagnosis, with no automatic retry')
                audit = a.output / f'{index}-{label}-audit.json'
                with (a.output / f'{index}-{label}-audit.log').open('xb') as stream:
                    subprocess.run(['taskset', '-c', '64-95', sys.executable, '-B',
                        str(SOURCE / 'research/recipes/visual_katago_lr/audit_learning.py'),
                        '--workspace-root', str(root), '--attempt', str(attempt), '--output', str(audit)],
                        cwd=root, env=cpu_environment, stdout=stream, stderr=subprocess.STDOUT, check=True)
                audits.append((label, str(audit), sha256(audit)))
                analysis = a.output / f'{index}-analysis.json'
                command = [sys.executable, '-B', str(Path(__file__).with_name('analyze_lr.py')),
                           '--workspace-root', str(root), '--registration', str(a.registration),
                           '--registration-sha256', a.registration_sha256, '--output', str(analysis)]
                for triple in audits:
                    command.extend(['--audit', *triple])
                with (a.output / f'{index}-analysis.log').open('xb') as stream:
                    subprocess.run(command, cwd=root, env=cpu_environment, stdout=stream,
                                   stderr=subprocess.STDOUT, check=True)
                observed = read_json(analysis)
                event('arm_audited', rate=case['peak_learning_rate'], audit=str(audit),
                      audit_sha256=sha256(audit), analysis=str(analysis), analysis_sha256=sha256(analysis),
                      best_completed_rate=observed['lowest_completed_endpoint_rate'],
                      pending_rates=observed['pending_rates'])
            status = 'passed'
    except BaseException as error:
        failure = repr(error)
        event('queue_stopped', error=failure)
        raise
    finally:
        publish(a.output / 'result.json', {'kind': 'registered_lr_sweep_execution', 'status': status,
            'error': failure, 'operator_snapshot': SOURCE.name, 'started_at': started,
            'finished_at': datetime.now(timezone.utc).isoformat(),
            'registration_sha256': a.registration_sha256, 'audits': audits, 'events': events})


if __name__ == '__main__':
    main()
