"""Explicit temporary RAM checkpoints for bounded, reproducible experiments.

Only metadata and hashes are persistent. These are not durable checkpoints;
selected results must be promoted separately before releasing the pod.
"""
from pathlib import Path
import os
from gozero import checkpoints
from gozero.checkpoint_archive import ROOT, FILES, publish, safe_path


def write(logical_path, *, state, arrays, actors, cache_root=Path('/dev/shm/gozero-staged-checkpoints'),
          compress=True, minimum_free_bytes=5_000_000_000):
    logical_path = safe_path(logical_path, ROOT/'runs')
    rel = logical_path.relative_to(ROOT/'runs')
    if len(rel.parts) != 5 or rel.parts[1:4] != ('rank-0', 'artifacts', 'checkpoints'):
        raise ValueError('Only the owner checkpoint uses temporary storage')
    cache_root = Path(cache_root)
    safe_path(cache_root, Path('/dev/shm') if cache_root.is_relative_to('/dev/shm') else cache_root)
    cache_root.mkdir(parents=True, exist_ok=True)
    fs = os.statvfs(cache_root)
    raw_bytes = sum(a.nbytes for a in arrays.values())
    if type(compress) is not bool or type(minimum_free_bytes) is not int or minimum_free_bytes<5_000_000_000:
        raise ValueError('Invalid checkpoint encoding or minimum free-space contract')
    if fs.f_bavail * fs.f_frsize < raw_bytes + minimum_free_bytes:
        raise ValueError('Insufficient temporary checkpoint space with the requested reserve')
    cache = safe_path(cache_root/rel, cache_root)
    identity = checkpoints.write(cache, state=state, arrays=arrays, actors=actors, compress=compress)
    protection = None
    if cache_root.is_relative_to('/dev/shm'):
        from gozero.ram_checkpoints import seal
        protection=seal([cache/name for name in sorted(FILES)])
    receipt = {'kind': 'temporary_ram_checkpoint', 'status': 'temporary',
        'cache_path': str(cache), 'logical_path': str(logical_path), 'manifest_sha256': identity,
        'files': {name: {'sha256': checkpoints.sha256(cache/name), 'bytes': (cache/name).stat().st_size}
                  for name in sorted(FILES)},
        'durability': 'Arrays are in volatile RAM. This persistent receipt does not preserve their bytes. Promote a selected checkpoint before releasing the pod.',
        'source_snapshot': state['snapshot_id'], 'compressed':compress, 'minimum_free_bytes':minimum_free_bytes}
    if protection is not None:receipt['logout_cleanup_protection']=protection
    receipt_path = logical_path.with_suffix('.temporary.json')
    publish(receipt_path, receipt)
    # Persist tiny standard checkpoint metadata for inspection and recovery.
    metadata = logical_path.with_suffix('.metadata')
    metadata.mkdir(parents=True, exist_ok=False)
    for name in sorted(FILES-{'arrays.npz'}):
        with (metadata/name).open('xb') as f:
            f.write((cache/name).read_bytes()); f.flush(); os.fsync(f.fileno()); os.fchmod(f.fileno(),0o444)
    checkpoints._sync_directory(metadata)
    return cache, identity, {'receipt': str(receipt_path), 'receipt_sha256': checkpoints.sha256(receipt_path), **receipt}
