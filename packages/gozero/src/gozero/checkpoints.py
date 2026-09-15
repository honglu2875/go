"""Atomic, checksummed, non-pickle checkpoint directories on a local filesystem.

Only a completed directory rename publishes a checkpoint. Multi-host trainers
must additionally publish a group manifest after every rank succeeds.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import tempfile
import zipfile

from .snapshots import canonical_json, read_json


def sha256(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def _sync_directory(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def write(path: Path, *, state: dict, arrays: dict, actors: str, compress: bool = False) -> str:
    import numpy as np
    path = Path(path)
    if type(compress) is not bool:
        raise ValueError('Checkpoint compression must be a boolean')
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    for name, value in arrays.items():
        if not name.isidentifier() or np.asarray(value).dtype.hasobject:
            raise ValueError('Checkpoint arrays require identifier names and non-object dtypes')
    # A failed write remains visibly incomplete and is never a restore candidate.
    temporary = Path(tempfile.mkdtemp(prefix='.' + path.name + '.partial-', dir=path.parent))
    with (temporary / 'arrays.npz').open('wb') as stream:
        if compress:
            # ZIP keeps NumPy's non-pickle NPZ interface. Level 1 trades some
            # ratio for low checkpoint latency; member timestamps are fixed by
            # ZipFile.open's new-member defaults (1980-01-01), not wall time.
            with zipfile.ZipFile(stream, mode='w', compression=zipfile.ZIP_DEFLATED,
                                 compresslevel=1, allowZip64=True) as archive:
                for name, value in arrays.items():
                    with archive.open(name + '.npy', 'w', force_zip64=True) as member:
                        np.lib.format.write_array(member, np.asarray(value), allow_pickle=False)
        else:
            np.savez(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    for name, value in [('state.json', canonical_json(state)), ('actors.json', actors.encode())]:
        with (temporary / name).open('wb') as stream:
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
    manifest = {'schema_version': 1, 'files': {name: {'sha256': sha256(temporary / name),
                'bytes': (temporary / name).stat().st_size} for name in ('arrays.npz', 'state.json', 'actors.json')}}
    with (temporary / 'manifest.json').open('wb') as stream:
        stream.write(canonical_json(manifest))
        stream.flush()
        os.fsync(stream.fileno())
    for entry in temporary.iterdir():
        entry.chmod(0o444)
    _sync_directory(temporary)
    temporary.rename(path)
    _sync_directory(path.parent)
    return sha256(path / 'manifest.json')


def read(path: Path, *, expected_manifest_sha256: str | None = None, array_prefix: str | None = None):
    import numpy as np
    if array_prefix is not None and (not isinstance(array_prefix, str) or not array_prefix):
        raise ValueError('Array prefix must be a nonempty string')
    path = Path(path)
    if path.is_symlink() or path.name.startswith('.'):
        raise ValueError('Cannot restore a symlink or incomplete checkpoint')
    names = {'manifest.json', 'state.json', 'arrays.npz', 'actors.json'}
    if {p.name for p in path.iterdir()} != names or any(not (path / n).is_file() or (path / n).is_symlink() for n in names):
        raise ValueError('Checkpoint files are incomplete or unexpected')
    if expected_manifest_sha256 is not None and sha256(path / 'manifest.json') != expected_manifest_sha256:
        raise ValueError('Checkpoint manifest identity differs')
    manifest = read_json(path / 'manifest.json')
    if set(manifest) != {'schema_version', 'files'} or manifest['schema_version'] != 1 or set(manifest['files']) != names - {'manifest.json'}:
        raise ValueError('Invalid checkpoint manifest')
    for name, expected in manifest['files'].items():
        if (path / name).stat().st_size != expected['bytes'] or sha256(path / name) != expected['sha256']:
            raise ValueError('Checkpoint integrity failure: ' + name)
    with np.load(path / 'arrays.npz', allow_pickle=False) as saved:
        if len(saved.files) != len(set(saved.files)):
            raise ValueError('Duplicate checkpoint array names')
        # All file hashes above are checked even when only parameters are
        # requested. Evaluation need not materialize the much larger replay.
        arrays = {name: saved[name] for name in saved.files if array_prefix is None or name.startswith(array_prefix)}
    return read_json(path / 'state.json'), arrays, (path / 'actors.json').read_text()
