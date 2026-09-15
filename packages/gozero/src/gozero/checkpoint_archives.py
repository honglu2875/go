"""Lossless cold storage of immutable NPZ bytes, with explicit restoration.

Original checkpoint manifests never change. Historical checkpoint readers require
restoration before use. Receipts live beside checkpoint directories, not inside
their strictly validated four-file contents.
"""
import gzip
import hashlib
import os
from pathlib import Path
import shutil
import stat
import tempfile

from .checkpoints import _sync_directory, sha256
from .snapshots import canonical_json, read_json


def _path(root, name):
    root = Path(root).resolve()
    path = root / name
    if Path(name).is_absolute() or '..' in Path(name).parts or not path.resolve().is_relative_to(root):
        raise ValueError('Expected workspace-relative artifact path')
    if any(p.is_symlink() for p in [path, *path.parents] if p.is_relative_to(root)):
        raise ValueError('Archive paths cannot be symlinks')
    return path


def _publish(path, value):
    data = canonical_json(value)
    if path.exists():
        if path.read_bytes() != data:
            raise FileExistsError('Existing receipt differs')
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name('.' + path.name + '.partial')
    with temporary.open('xb') as f:
        f.write(data); f.flush(); os.fsync(f.fileno())
    temporary.chmod(0o444)
    temporary.rename(path)
    _sync_directory(path.parent)


def _uncompressed_identity(path, sink=None):
    digest, size = hashlib.sha256(), 0
    with gzip.open(path, 'rb') as f:
        while data := f.read(8 * 1024 * 1024):
            digest.update(data); size += len(data)
            if sink is not None:
                sink.write(data)
    return digest.hexdigest(), size


def archive(root, name, expected_manifest_sha256, *, remove_original=False):
    root = Path(root).resolve()
    path = _path(root, name)
    if not (name.startswith('runs/pod-') and '/artifacts/checkpoints/turn-' in name and path.name == 'arrays.npz'):
        raise ValueError('Only completed pod checkpoint arrays may be archived')
    manifest = path.with_name('manifest.json')
    if sha256(manifest) != expected_manifest_sha256:
        raise ValueError('Checkpoint manifest changed')
    expected = read_json(manifest)['files']['arrays.npz']
    receipt = path.parent.with_suffix('.arrays-archive.json')
    blob = root / '.gozero/checkpoint-archives' / (expected['sha256'] + '.npz.gz')
    info = path.stat() if path.exists() else None
    if info is not None and (not stat.S_ISREG(info.st_mode) or info.st_mode & 0o222
                             or info.st_size != expected['bytes'] or sha256(path) != expected['sha256']):
        raise ValueError('Original array bytes or immutable mode differ')
    if info is None and not receipt.is_file():
        raise ValueError('Missing original has no published archive receipt')
    blob.parent.mkdir(parents=True, exist_ok=True)
    if not blob.exists():
        if info is None:
            raise ValueError('Original and archive are both missing')
        fd, temporary_name = tempfile.mkstemp(prefix='.arrays-', dir=blob.parent)
        temporary = Path(temporary_name)
        try:
            with os.fdopen(fd, 'wb') as raw:
                with gzip.GzipFile(filename='', mode='wb', compresslevel=1, mtime=0, fileobj=raw) as compressed:
                    with path.open('rb') as source:
                        shutil.copyfileobj(source, compressed, 8 * 1024 * 1024)
                raw.flush(); os.fsync(raw.fileno())
            if _uncompressed_identity(temporary) != (expected['sha256'], expected['bytes']):
                raise ValueError('Compressed archive does not reproduce original bytes')
            temporary.chmod(0o444)
            temporary.rename(blob)
            _sync_directory(blob.parent)
        finally:
            temporary.unlink(missing_ok=True)
    if _uncompressed_identity(blob) != (expected['sha256'], expected['bytes']):
        raise ValueError('Published compressed payload differs')
    record = {'schema_version': 1, 'kind': 'lossless_checkpoint_array_archive',
              'original_path': name, 'original_sha256': expected['sha256'], 'original_bytes': expected['bytes'],
              'manifest_sha256': expected_manifest_sha256,
              'archive_path': str(blob.relative_to(root)), 'archive_sha256': sha256(blob),
              'archive_bytes': blob.stat().st_size, 'compression': 'gzip-level1-mtime0',
              'restoration_required_for_checkpoint_reader': True}
    _publish(receipt, record)
    released = 0
    if remove_original and info is not None:
        now = path.stat()
        if (now.st_ino, now.st_size, now.st_mtime_ns, now.st_mode) != (info.st_ino, info.st_size, info.st_mtime_ns, info.st_mode):
            raise ValueError('Original changed during archival')
        if sha256(path) != expected['sha256'] or sha256(blob) != record['archive_sha256']:
            raise ValueError('Last integrity check failed')
        if now.st_nlink == 1:
            released = now.st_blocks * 512
        path.unlink(); _sync_directory(path.parent)
    return {**record, 'receipt_path': str(receipt.relative_to(root)),
            'receipt_sha256': sha256(receipt), 'released_original_allocated_bytes': released}


def restore(root, receipt_name, expected_receipt_sha256, *, destination=None):
    root = Path(root).resolve()
    receipt = _path(root, receipt_name)
    if sha256(receipt) != expected_receipt_sha256:
        raise ValueError('Archive receipt differs')
    record = read_json(receipt)
    if record['kind'] != 'lossless_checkpoint_array_archive' or record['schema_version'] != 1:
        raise ValueError('Invalid archive record')
    blob = _path(root, record['archive_path'])
    path = _path(root, record['original_path'] if destination is None else destination)
    if sha256(blob) != record['archive_sha256']:
        raise ValueError('Compressed bytes differ')
    if path.exists():
        if sha256(path) != record['original_sha256']:
            raise ValueError('Refusing to overwrite different destination')
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(path.parent).free < record['original_bytes'] + 64 * 1024 * 1024:
        raise ValueError('Insufficient room for exact restoration')
    fd, temporary_name = tempfile.mkstemp(prefix='.restore-', dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, 'wb') as sink:
            observed = _uncompressed_identity(blob, sink)
            sink.flush(); os.fsync(sink.fileno())
        if observed != (record['original_sha256'], record['original_bytes']):
            raise ValueError('Restored bytes differ from original manifest')
        temporary.chmod(0o444)
        temporary.rename(path); _sync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)
    return path
