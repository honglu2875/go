"""Materialize a locked dependency environment outside the frozen source tree."""

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tomllib

sys.dont_write_bytecode = True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--snapshot', type=Path, required=True)
    parser.add_argument('--environment', type=Path, required=True)
    parser.add_argument('--uv', type=Path, required=True)
    parser.add_argument('--cache', type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.snapshot / 'packages/gozero/src'))
    from gozero.snapshots import canonical_json, verify
    verify(args.snapshot)
    lock = tomllib.loads((args.snapshot / 'uv.lock').read_text())
    expected = {p['name']: p['version'] for p in lock['package'] if 'registry' in p['source']}
    python_version = (args.snapshot / '.python-version').read_text().strip()
    args.environment.parent.mkdir(parents=True, exist_ok=True)
    with (args.environment.parent / (args.environment.name + '.lock')).open('a') as lockfile:
        fcntl.flock(lockfile, fcntl.LOCK_EX)
        receipt_path = args.environment / 'gozero-runtime.json'
        if not receipt_path.exists():
            environment = {**os.environ, 'UV_CACHE_DIR': str(args.cache), 'UV_LINK_MODE': 'copy',
                           'UV_PROJECT_ENVIRONMENT': str(args.environment), 'UV_PYTHON_DOWNLOADS': 'never'}
            command = [str(args.uv), 'sync', '--frozen', '--extra', 'tpu', '--no-install-workspace',
                       '--project', str(args.snapshot), '--python', sys.executable]
            subprocess.run(command, env=environment, check=True)
        script = ('import importlib.metadata as m,json,platform; '
                  'print(json.dumps({"python":platform.python_version(),"packages":'
                  '{name:m.version(name) for name in ' + repr(sorted(expected)) + '}}))')
        result = subprocess.run([str(args.environment / 'bin/python'), '-c', script],
                                capture_output=True, text=True, check=True)
        installed = json.loads(result.stdout)
        if installed['python'] != python_version or installed['packages'] != expected:
            raise RuntimeError('Installed runtime does not match the frozen lock/Python version')
        receipt = {**installed, 'uv_sha256': hashlib.sha256(args.uv.read_bytes()).hexdigest(),
                   'uv_lock_sha256': hashlib.sha256((args.snapshot / 'uv.lock').read_bytes()).hexdigest(),
                   'extras': ['tpu'], 'workspace_installation': 'source supplied by snapshot PYTHONPATH'}
        if receipt_path.exists() and json.loads(receipt_path.read_text()) != receipt:
            raise RuntimeError('Existing runtime receipt differs; use a new environment identity')
        if not receipt_path.exists():
            temporary = receipt_path.with_suffix('.tmp')
            temporary.write_bytes(canonical_json(receipt))
            temporary.replace(receipt_path)
        verify(args.snapshot)
        print(json.dumps({'kind': 'runtime_prepared', 'environment': str(args.environment), **receipt}), flush=True)


if __name__ == '__main__':
    main()
