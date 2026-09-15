"""Stage a new checkpoint in RAM and publish it into reserved peer disk space.

The archive contains the ordinary four checkpoint files. A persistent receipt
records its location and exact bytes; the RAM copy is only a readable cache.
Existing checkpoints are never replaced or removed by this module.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import tempfile
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gozero import checkpoints
from gozero.snapshots import canonical_json, read_json

ROOT = Path('/workspace/go')
FILES = {'arrays.npz', 'manifest.json', 'state.json', 'actors.json'}
SSH = ['ssh', '-F', '/dev/null', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15']


def publish(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as f:
        f.write(canonical_json(value)); f.flush(); os.fsync(f.fileno()); os.fchmod(f.fileno(), 0o444)
    checkpoints._sync_directory(path.parent)


def safe_path(path, parent):
    path, parent = Path(path), Path(parent)
    if not path.is_absolute() or '..' in path.parts or not path.is_relative_to(parent):
        raise ValueError('Path is outside the archive contract')
    if any(x.is_symlink() for x in (path, *path.parents)):
        raise ValueError('Archive paths cannot be symlinks')
    return path


def locations(spec, root=ROOT):
    identity = spec['manifest_sha256']; token = spec['token']
    if not re.fullmatch('[0-9a-f]{64}', identity) or not re.fullmatch('[0-9a-f]{32}', token):
        raise ValueError('Invalid immutable archive identity')
    base = Path(root)/'.gozero/checkpoint-archives'
    destination = safe_path(base/identity, base)
    incoming = safe_path(base/('.'+identity+'.incoming-'+token), base)
    return incoming, destination


def verify_files(path, spec):
    if path.is_symlink() or {p.name for p in path.iterdir()} != FILES:
        raise ValueError('Unexpected archive files')
    if set(spec['files']) != FILES:
        raise ValueError('Incomplete archive contract')
    for name, wanted in spec['files'].items():
        p = path/name
        if (not p.is_file() or p.is_symlink() or p.stat().st_mode & 0o222
                or p.stat().st_size != wanted['bytes'] or checkpoints.sha256(p) != wanted['sha256']):
            raise ValueError('Archive bytes differ: '+name)
    if checkpoints.sha256(path/'manifest.json') != spec['manifest_sha256']:
        raise ValueError('Archive manifest differs')


def prepare(spec, root=ROOT):
    incoming, destination = locations(spec, root)
    if destination.exists():
        verify_files(destination, spec)
        return {'already_committed': True, 'path': str(destination)}
    reservation = safe_path(Path(root)/spec['reservation'], Path(root)/'.gozero/checkpoint-reservations')
    if (not reservation.is_file() or reservation.stat().st_nlink != 1
            or reservation.stat().st_size < spec['files']['arrays.npz']['bytes']):
        raise ValueError('Missing or undersized checkpoint reservation')
    if set(spec['metadata']) != FILES-{'arrays.npz'}:
        raise ValueError('Unexpected checkpoint metadata')
    # Validate before consuming this run's allocation.
    for name, raw in spec['metadata'].items():
        data = raw.encode()
        if len(data) != spec['files'][name]['bytes'] or hashlib.sha256(data).hexdigest() != spec['files'][name]['sha256']:
            raise ValueError('Invalid metadata payload')
    incoming.parent.mkdir(parents=True, exist_ok=True); incoming.mkdir()
    os.rename(reservation, incoming/'arrays.npz')
    for name, raw in spec['metadata'].items():
        with (incoming/name).open('xb') as f:
            f.write(raw.encode()); f.flush(); os.fsync(f.fileno()); os.fchmod(f.fileno(), 0o444)
    checkpoints._sync_directory(incoming)
    return {'already_committed': False, 'incoming': str(incoming)}


def stream(spec, source, root=ROOT):
    incoming, _ = locations(spec, root); path = safe_path(incoming/'arrays.npz', incoming)
    wanted = spec['files']['arrays.npz']; remaining = wanted['bytes']; h = hashlib.sha256()
    if not path.is_file() or path.stat().st_size < remaining or not path.stat().st_mode & 0o200:
        raise ValueError('Reserved incoming array is unavailable')
    # r+b preserves the allocation. Only the unused tail is truncated after a
    # complete transfer and checksum check, avoiding a second disk allocation.
    with path.open('r+b') as f:
        while remaining:
            block = source.read(min(8*2**20, remaining))
            if not block: raise ValueError('Truncated checkpoint stream')
            f.write(block); h.update(block); remaining -= len(block)
        if source.read(1) or h.hexdigest() != wanted['sha256']:
            raise ValueError('Checkpoint stream differs')
        f.truncate(wanted['bytes']); f.flush(); os.fsync(f.fileno()); os.fchmod(f.fileno(), 0o444)
    return {'streamed_bytes': wanted['bytes'], 'sha256': h.hexdigest()}


def commit(spec, root=ROOT):
    incoming, destination = locations(spec, root)
    if destination.exists():
        verify_files(destination, spec)
    else:
        verify_files(incoming, spec)
        checkpoints._sync_directory(incoming)
        incoming.rename(destination); checkpoints._sync_directory(destination.parent)
    return {'status': 'committed', 'path': str(destination), 'manifest_sha256': spec['manifest_sha256'], 'files': spec['files']}


def remote(host, action, spec, *, python, source=None):
    if type(host) is not int or host not in (1, 2, 3): raise ValueError('Expected an authorized peer host')
    command = [str(python), '-B', str(Path(__file__).resolve()), action]
    if source is not None:
        command += ['--spec', json.dumps({k: v for k, v in spec.items() if k != 'metadata'}, separators=(',', ':'))]
    argv = [*SSH, f'go-user@worker-{host}.example.invalid', shlex.join(command)]
    if source is None:
        result = subprocess.run(argv, input=canonical_json(spec), capture_output=True, timeout=600)
    else:
        with Path(source).open('rb') as f:
            result = subprocess.run(argv, stdin=f, capture_output=True, timeout=600)
    if result.returncode:
        raise RuntimeError('Checkpoint archive '+action+' failed: '+result.stderr.decode()[-4000:])
    return json.loads(result.stdout)


def write(logical_path, *, state, arrays, actors, archive, python):
    logical_path = safe_path(logical_path, ROOT/'runs')
    rel = logical_path.relative_to(ROOT/'runs')
    if len(rel.parts) != 5 or rel.parts[1:4] != ('rank-0', 'artifacts', 'checkpoints'):
        raise ValueError('Only the owner checkpoint uses peer storage')
    cache = Path('/dev/shm/gozero-archived-checkpoints')/rel
    safe_path(cache, Path('/dev/shm/gozero-archived-checkpoints'))
    identity = checkpoints.write(cache, state=state, arrays=arrays, actors=actors, compress=True)
    spec = {'manifest_sha256': identity, 'token': uuid.uuid4().hex,
            'reservation': archive['reservation'],
            'files': {name: {'sha256': checkpoints.sha256(cache/name), 'bytes': (cache/name).stat().st_size} for name in sorted(FILES)},
            'metadata': {name: (cache/name).read_text() for name in sorted(FILES-{'arrays.npz'})}}
    prepared = remote(archive['host'], 'prepare', spec, python=python)
    if not prepared['already_committed']:
        remote(archive['host'], 'stream', spec, python=python, source=cache/'arrays.npz')
    committed = remote(archive['host'], 'commit', spec, python=python)
    receipt = {'kind': 'peer_archived_checkpoint', 'status': 'committed', 'host': archive['host'],
               'cache_path': str(cache), 'logical_path': str(logical_path), **committed,
               'restore_contract': 'Use gozero.checkpoint_archive.restore to copy the ordinary four-file checkpoint back into its cache path, then use the unchanged checkpoint reader.'}
    receipt_path = logical_path.with_suffix('.archive.json'); publish(receipt_path, receipt)
    return cache, identity, {'receipt': str(receipt_path), 'receipt_sha256': checkpoints.sha256(receipt_path), **receipt}


def restore(receipt, destination):
    if receipt['status'] != 'committed' or receipt['host'] not in (1, 2, 3): raise ValueError('Invalid archive receipt')
    source = safe_path(receipt['path'], ROOT/'.gozero/checkpoint-archives')
    destination = Path(destination)
    if destination.exists() or destination.is_symlink(): raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix='.archive-restore-', dir=destination.parent))/'checkpoint'
    staging.mkdir()
    target = f"go-user@worker-{receipt['host']}:"+str(source)+'/'
    subprocess.run(['rsync', '-a', '--protect-args', '--ignore-existing', '-e', shlex.join(SSH), '--', target, str(staging)+'/'], check=True, timeout=600)
    verify_files(staging, receipt)
    checkpoints.read(staging, expected_manifest_sha256=receipt['manifest_sha256'])
    parent = staging.parent; staging.rename(destination); parent.rmdir(); checkpoints._sync_directory(destination.parent)
    return receipt['manifest_sha256']


def main():
    p = argparse.ArgumentParser(); p.add_argument('action', choices=['prepare', 'stream', 'commit', 'restore'])
    p.add_argument('--spec'); p.add_argument('--receipt', type=Path); p.add_argument('--destination', type=Path)
    a = p.parse_args()
    if a.action == 'restore':
        result = {'restored_manifest_sha256': restore(read_json(a.receipt), a.destination)}
    else:
        spec = json.loads(a.spec) if a.spec is not None else json.load(sys.stdin)
        result = stream(spec, sys.stdin.buffer) if a.action == 'stream' else globals()[a.action](spec)
    print(json.dumps(result), flush=True)


if __name__ == '__main__': main()
