"""Remove disposable uv cache entries between attempts; verify installed bytes."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

SOURCE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE / 'packages/gozero/src'))
from gozero.snapshots import canonical_json, read_json, verify
from gozero.checkpoints import sha256


def inventory(folder):
    digest = hashlib.sha256(); count = 0; size = 0
    for path in sorted(folder.rglob('*')):
        if path.is_symlink():
            item = [str(path.relative_to(folder)), 'symlink', os.readlink(path)]
        elif path.is_file():
            s = path.stat(); count += 1; size += s.st_size
            item = [str(path.relative_to(folder)), s.st_size, s.st_mode & 0o777, sha256(path)]
        else:
            continue
        digest.update(canonical_json(item))
    return {'sha256': digest.hexdigest(), 'regular_files': count, 'logical_bytes': size}


def allocated(folder):
    seen = set(); total = 0
    for path in folder.rglob('*'):
        if not path.is_symlink() and path.is_file():
            s = path.stat(); key = (s.st_dev, s.st_ino)
            if key not in seen:
                total += s.st_blocks * 512; seen.add(key)
    return total


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workspace-root', type=Path, required=True)
    p.add_argument('--runtime-key', required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args(); root = a.workspace_root.resolve(); verify(SOURCE)
    if root != Path('/workspace/go') or len(a.runtime_key) != 64:
        raise ValueError('Unexpected workspace/runtime')
    if any(not (x.parent / 'result.json').exists() for x in (root / 'runs').glob('pod-*/launch.json')):
        raise ValueError('Only prune caches between TPU attempts')
    environment = root / '.gozero/environments' / a.runtime_key
    cache = root / '.gozero/cache/uv'
    if cache.is_symlink() or not cache.is_dir() or not environment.is_dir():
        raise ValueError('Expected ordinary cache and installed environment')
    receipt = read_json(environment / 'gozero-runtime.json')
    uv = root / '.gozero/tools' / ('uv-' + receipt['uv_sha256'])
    if sha256(uv) != receipt['uv_sha256'] or sha256(SOURCE / 'uv.lock') != receipt['uv_lock_sha256']:
        raise ValueError('Pinned uv/runtime lock differs')
    if "'UV_LINK_MODE': 'copy'" not in (SOURCE / 'ops/prepare_host.py').read_text():
        raise ValueError('Environment materialization contract changed')
    a.output.mkdir(parents=True, exist_ok=False)
    started = time.time(); result = {'status': 'running'}
    with (environment.parent / (a.runtime_key + '.lock')).open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        before = inventory(environment); cache_before = allocated(cache)
        try:
            with (a.output / 'uv.log').open('xb') as log:
                subprocess.run([str(uv), 'cache', 'clean', '--cache-dir', str(cache)],
                    stdout=log, stderr=subprocess.STDOUT, check=True, timeout=180)
            after = inventory(environment)
            if before != after:
                raise ValueError('Installed runtime bytes changed')
            # Metadata lookup exercises the retained interpreter without generating bytecode.
            script = 'import importlib.metadata as m,json,platform;print(json.dumps({"python":platform.python_version(),"packages":{n:m.version(n) for n in ' + repr(sorted(receipt['packages'])) + '}}))'
            check = subprocess.run([str(environment / 'bin/python'), '-B', '-c', script],
                capture_output=True, text=True, check=True,
                env={**os.environ, 'JAX_PLATFORMS': 'cpu', 'PYTHONDONTWRITEBYTECODE': '1'}, timeout=60)
            installed = json.loads(check.stdout)
            if installed != {k: receipt[k] for k in ('python', 'packages')}:
                raise ValueError('Installed packages differ from runtime receipt')
            result = {'status': 'passed', 'runtime_before': before, 'runtime_after': after,
                'cache_allocated_bytes_before': cache_before, 'cache_allocated_bytes_after': allocated(cache),
                'runtime_receipt_sha256': sha256(environment / 'gozero-runtime.json')}
        except BaseException as error:
            result = {'status': 'failed', 'error': repr(error), 'runtime_before': before}; raise
        finally:
            result.update(kind='disposable_uv_cache_pruning', operator_snapshot=SOURCE.name,
                host=os.uname().nodename, runtime_key=a.runtime_key, started_unix=started, ended_unix=time.time(),
                scope='Only the uv download/unpacked-wheel cache. Installed runtime, sources, datasets and checkpoint bytes retained. Cache entries can be downloaded again from the frozen lock.')
            with (a.output / 'result.json').open('xb') as f:
                f.write(canonical_json(result)); f.flush(); os.fchmod(f.fileno(), 0o444); os.fsync(f.fileno())
            print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
