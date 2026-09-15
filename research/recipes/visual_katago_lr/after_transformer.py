#!/usr/bin/env python3
"""After the current transformer sweep, retain checkpoints and run CNN LR controls."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time

SOURCE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(SOURCE / 'packages/gozero/src'))
from gozero.checkpoints import sha256
from gozero.snapshots import canonical_json, read_json, verify


def publish(path, value):
    with path.open('xb') as stream:
        stream.write(canonical_json(value))
        stream.flush()
        os.fchmod(stream.fileno(), 0o444)
        os.fsync(stream.fileno())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workspace-root', type=Path, required=True)
    p.add_argument('--registration', type=Path, required=True)
    p.add_argument('--registration-sha256', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--wait-seconds', type=int, default=7200)
    a = p.parse_args(); verify(SOURCE); root = a.workspace_root.resolve()
    if sha256(a.registration) != a.registration_sha256:
        raise ValueError('CNN registration changed')
    registration = read_json(a.registration)
    a.output.mkdir(parents=True, exist_ok=False)
    env = {**os.environ, 'OPENBLAS_NUM_THREADS': '1', 'OMP_NUM_THREADS': '1', 'PYTHONDONTWRITEBYTECODE': '1'}
    recipe = Path(__file__).parent
    reference = registration['reference']
    command = [sys.executable, '-B', str(recipe / 'run_sweep.py'), '--workspace-root', str(root),
               '--registration', str(a.registration), '--registration-sha256', a.registration_sha256,
               '--reference-audit', str(root / reference['audit_path']),
               '--reference-audit-sha256', reference['audit_sha256'], '--output', str(a.output / 'sweep')]
    subprocess.run([*command, '--check-only'], env=env, cwd=root, check=True)
    report = {'kind': 'cnn_lr_followup_execution', 'status': 'running', 'operator_snapshot': SOURCE.name,
              'registration_sha256': a.registration_sha256, 'maintenance': []}
    try:
        prerequisite = registration['wait_for_transformer_publication']
        path = root / prerequisite['manifest_path']
        deadline = time.monotonic() + a.wait_seconds
        print(canonical_json({'phase': 'waiting_for_transformer_sweep', 'manifest': str(path)}).decode().strip(), flush=True)
        while not path.exists() or path.stat().st_mode & 0o222:
            closed = root / prerequisite['runner_result_path']
            if closed.exists() and not closed.stat().st_mode & 0o222 and read_json(closed)['status'] != 'passed':
                raise ValueError('Transformer sweep stopped; inspect it before advancing')
            if time.monotonic() >= deadline:
                raise TimeoutError('Transformer publication not ready; no CNN run launched')
            time.sleep(min(30, max(.1, deadline - time.monotonic())))
        observed = read_json(path)
        if observed['status'] != 'passed' or observed['execution_sha256'] != prerequisite['execution_sha256']:
            raise ValueError('Wrong transformer publication')
        for name, digest in observed['files'].items():
            if sha256(path.parent / name) != digest:
                raise ValueError('Transformer publication file changed')
        report['transformer_publication_sha256'] = sha256(path)
        maintenance = a.output / 'maintenance'; maintenance.mkdir()
        for item in registration['checkpoint_storage_preparation']:
            receipt = maintenance / (item['attempt'] + '.json')
            op = [sys.executable, '-B', str(SOURCE / 'ops/retain_checkpoint_replicas.py'),
                  '--workspace-root', str(root), '--attempt', item['attempt'], '--retained-hosts',
                  *map(str, item['retained_hosts']), '--copy-missing-replicas', '--apply', '--output', str(receipt)]
            if item['single_owner_relocation']:
                op.append('--relocate-single-owner')
            with receipt.with_suffix('.log').open('xb') as stream:
                subprocess.run(op, cwd=root, env=env, stdout=stream, stderr=subprocess.STDOUT, check=True)
            if read_json(receipt)['status'] != 'passed':
                raise ValueError('Checkpoint storage preparation failed')
            report['maintenance'].append({'path': str(receipt), 'sha256': sha256(receipt)})
            print(canonical_json({'phase': 'checkpoint_storage_ready', 'receipt': str(receipt)}).decode().strip(), flush=True)
        with (a.output / 'sweep.log').open('xb') as stream:
            subprocess.run(command, cwd=root, env=env, stdout=stream, stderr=subprocess.STDOUT, check=True)
        subprocess.run([sys.executable, '-B', str(recipe / 'publish_comparison.py'),
            '--workspace-root', str(root), '--registration', str(a.registration),
            '--registration-sha256', a.registration_sha256,
            '--cnn-execution', str(a.output / 'sweep/result.json'),
            '--transformer-analysis', str(path.parent / 'analysis.json'),
            '--output', str(root / registration['publication_directory'])], cwd=root, env=env, check=True)
        ledger = root / 'research/studies/runtime_qualification/reservation_ledger.json'
        subprocess.run([sys.executable, '-B', str(SOURCE / 'ops/update_reservation_ledger.py'),
            '--workspace-root', str(root), '--expected-previous-sha256', sha256(ledger)], cwd=root, env=env, check=True)
        report['ledger_sha256'] = sha256(ledger); report['status'] = 'passed'
    except BaseException as error:
        report['status'] = 'failed'; report['error'] = repr(error)
        raise
    finally:
        verify(SOURCE); publish(a.output / 'result.json', report)
        print(canonical_json({'status': report['status'], 'receipt': str(a.output / 'result.json')}).decode().strip(), flush=True)


if __name__ == '__main__':
    main()
