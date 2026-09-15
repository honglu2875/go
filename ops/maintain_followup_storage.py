"""Wait for the context study to close, then reclaim only fixtures/cache storage."""
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

SOURCE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE / 'packages/gozero/src'))
from gozero.snapshots import canonical_json, read_json, verify
from gozero.checkpoints import sha256


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workspace-root', type=Path, required=True)
    p.add_argument('--gate', type=Path, required=True)
    p.add_argument('--gate-operator', required=True)
    p.add_argument('--runtime-key', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--prior-output', type=Path)
    p.add_argument('--prior-result-sha256')
    a = p.parse_args(); root = a.workspace_root.resolve(); verify(SOURCE)
    a.output.mkdir(parents=True, exist_ok=False)
    result = {'status': 'running'}; started = time.time(); commands = []; receipts = []; reused = []
    if a.prior_output is not None:
        a.prior_output.resolve().relative_to(root)
        if sha256(a.prior_output / 'result.json') != a.prior_result_sha256:
            raise ValueError('Prior maintenance result changed')
    def reuse(label):
        path = a.prior_output / label / 'result.json' if a.prior_output is not None else None
        if path is None or not path.exists() or read_json(path).get('status') != 'passed':
            return False
        receipts.append(path); reused.append({'phase':label,'receipt':str(path),'sha256':sha256(path)})
        print(json.dumps({'phase':label,'status':'reusing_verified_receipt'}),flush=True)
        return True
    env = {**os.environ, 'JAX_PLATFORMS': 'cpu', 'OPENBLAS_NUM_THREADS': '1',
           'OMP_NUM_THREADS': '1', 'PYTHONDONTWRITEBYTECODE': '1'}
    def run(label, argv, timeout=900):
        commands.append({'phase': label, 'argv': list(map(str, argv))})
        print(json.dumps({'phase': label, 'status': 'starting'}), flush=True)
        with (a.output / (label + '.log')).open('xb') as f:
            subprocess.run(list(map(str, argv)), cwd=root, env=env, stdout=f,
                           stderr=subprocess.STDOUT, check=True, timeout=timeout)
        print(json.dumps({'phase': label, 'status': 'passed'}), flush=True)
    try:
        deadline = time.monotonic() + 7200
        while not a.gate.exists():
            if time.monotonic() > deadline:
                raise TimeoutError('Preceding context study has not closed')
            time.sleep(10)
        gate = read_json(a.gate)
        if gate['status'] != 'passed' or gate['operator_snapshot'] != a.gate_operator:
            raise ValueError('Preceding study requires review')
        if any(not (x.parent / 'result.json').exists() for x in (root / 'runs').glob('pod-*/launch.json')):
            raise ValueError('Another TPU attempt is open')
        if not reuse('host0-fixtures'):
            run('host0-fixtures', ['taskset', '-c', '112-119', root / '.venv/bin/python', '-B',
                SOURCE / 'ops/deduplicate_snapshots.py', '--workspace-root', root,
                '--output', a.output / 'host0-fixtures'])
            receipts.append(a.output / 'host0-fixtures/result.json')
        ssh = ['ssh', '-F', '/dev/null', '-o', 'BatchMode=yes']
        for host in (1, 2, 3):
            target = f'go-user@worker-{host}.example.invalid'
            run(f'host{host}-stage', ['rsync', '-a', '--ignore-existing', '--protect-args', '-e',
                shlex.join(ssh), '--', str(SOURCE) + '/', target + ':' + str(SOURCE) + '/'])
            for action, script, extra in [('uv', 'prune_uv_cache.py', ['--runtime-key', a.runtime_key]),
                                           ('fixtures', 'deduplicate_snapshots.py', [])]:
                if reuse(f'host{host}-{action}'):
                    continue
                destination = a.output / f'host{host}-{action}'
                command = ['taskset', '-c', '112-119', root / '.gozero/environments' / a.runtime_key / 'bin/python',
                    '-B', SOURCE / 'ops' / script, '--workspace-root', root, '--output', destination, *extra]
                remote = 'env JAX_PLATFORMS=cpu OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 ' + shlex.join(list(map(str, command)))
                run(f'host{host}-{action}', [*ssh, target, remote])
                run(f'host{host}-{action}-receipt', ['rsync', '-a', '--ignore-existing', '--protect-args', '-e',
                    shlex.join(ssh), '--', target + ':' + str(destination) + '/', str(destination) + '/'])
                receipts.append(destination / 'result.json')
        rows = [read_json(x) for x in receipts]
        if len(rows) != 7 or any(x['status'] != 'passed' for x in rows):
            raise ValueError('Incomplete maintenance receipts')
        released = sum(x.get('released_allocated_bytes', 0) +
            x.get('cache_allocated_bytes_before', 0) - x.get('cache_allocated_bytes_after', 0) for x in rows)
        result = {'status': 'passed', 'gate_sha256': sha256(a.gate),
            'receipts': {str(x.relative_to(root)): sha256(x) for x in receipts},
            'released_allocated_bytes': released,'reused_prior_receipts':reused,
            'scope': 'Identical read-only eval fixtures share inodes; disposable peer uv caches removed after installed-byte verification. Every unique checkpoint and existing replica retained.'}
    except BaseException as error:
        result = {'status': 'failed', 'error': repr(error)}; raise
    finally:
        result.update(kind='followup_storage_maintenance', operator_snapshot=SOURCE.name,
            started_unix=started, ended_unix=time.time(), commands=commands)
        with (a.output / 'result.json').open('xb') as f:
            f.write(canonical_json(result))
        for path in a.output.iterdir():
            if path.is_file(): path.chmod(0o444)
        print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
