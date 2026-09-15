"""Verified, reversible compression of closed training-game directories.

Source, checkpoints and result records retain their original paths and bytes.
The archive manifest is content addressed, including every JSON/SGF byte hash.
Eviction and restoration are resumable; neither operation overwrites a game.
"""
from __future__ import annotations

from contextlib import contextmanager
import fcntl
import hashlib
import io
import os
from pathlib import Path
import re
import stat
import tempfile
import zipfile

from .checkpoints import sha256, _sync_directory
from .snapshots import canonical_json, read_json

HEX = re.compile(r'[0-9a-f]{64}')
RECORD = re.compile(r'[0-9a-f]{16}\.(json|sgf)')
POINTER = 'games.archive.json'
MAX_MEMBER_BYTES = 64 * 2**20
MAX_TOTAL_BYTES = 128 * 2**30


def _regular(path):
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode):
        raise ValueError('Expected a regular file: ' + str(path))
    return info


def _signature(info):
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def _digest_stream(stream, maximum, destination=None):
    digest = hashlib.sha256()
    size = 0
    while data := stream.read(2**20):
        size += len(data)
        if size > maximum:
            raise ValueError('Game record exceeds its declared size bound')
        digest.update(data)
        if destination is not None:
            destination.write(data)
    return size, digest.hexdigest()


def _record_names(directory):
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError('Expected a real game directory')
    names = sorted(p.name for p in directory.iterdir())
    if any(RECORD.fullmatch(name) is None for name in names):
        raise ValueError('Game directory has unexpected entries')
    for name in names:
        _regular(directory / name)
    return names


def _closed_result(artifacts):
    if artifacts.is_symlink() or not artifacts.is_dir():
        raise ValueError('Expected a real artifacts directory')
    path = artifacts / 'result.json'
    _regular(path)
    result = read_json(path)
    if (result.get('status') != 'passed' or not HEX.fullmatch(result.get('snapshot_id', ''))
            or type(result.get('turn')) is not int or result['turn'] <= 0):
        raise ValueError('Archival requires a completed successful training result')
    return result, sha256(path)


@contextmanager
def _lock(artifacts):
    # Completed trainers never write more games. This lock serializes archive
    # operators, including partial eviction/restoration after a process failure.
    _closed_result(artifacts)
    fd = os.open(artifacts / 'games.archive.lock', os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield
    finally:
        os.close(fd)


def verify_archive(path, *, expected_id=None):
    path = Path(path)
    if path.is_symlink() or not path.is_dir() or {x.name for x in path.iterdir()} != {'manifest.json', 'games.zip'}:
        raise ValueError('Archive files are incomplete or unexpected')
    for name in ('manifest.json', 'games.zip'):
        _regular(path / name)
    identity = sha256(path / 'manifest.json')
    if expected_id is not None and identity != expected_id:
        raise ValueError('Archive manifest identity differs')
    manifest = read_json(path / 'manifest.json')
    fields = {'schema_version', 'kind', 'training_result_sha256', 'training_snapshot', 'turn',
              'compression', 'archive_sha256', 'archive_bytes', 'original_bytes', 'files'}
    if (set(manifest) != fields or manifest['schema_version'] != 1
            or manifest['kind'] != 'closed_training_games' or manifest['compression'] != 'zip-deflate-1'
            or any(not isinstance(manifest[k], str) or not HEX.fullmatch(manifest[k])
                   for k in ('training_result_sha256', 'training_snapshot', 'archive_sha256'))):
        raise ValueError('Invalid game archive manifest')
    files = manifest['files']
    if not isinstance(files, dict) or not files:
        raise ValueError('Archive must contain paired game records')
    total = 0
    for name, entry in files.items():
        if (not RECORD.fullmatch(name) or set(entry) != {'bytes', 'sha256', 'mode', 'mtime_ns'}
                or type(entry['bytes']) is not int or not 0 < entry['bytes'] <= MAX_MEMBER_BYTES
                or not isinstance(entry['sha256'], str) or not HEX.fullmatch(entry['sha256'])
                or entry['mode'] not in (0o444, 0o644, 0o664, 0o600, 0o640)
                or type(entry['mtime_ns']) is not int or not -(2**63) <= entry['mtime_ns'] < 2**63):
            raise ValueError('Invalid game archive member')
        paired = str(Path(name).with_suffix('.sgf' if name.endswith('.json') else '.json'))
        if paired not in files:
            raise ValueError('Unpaired game archive member')
        total += entry['bytes']
    if total != manifest['original_bytes'] or total > MAX_TOTAL_BYTES:
        raise ValueError('Invalid game archive total size')
    archive = path / 'games.zip'
    if archive.stat().st_size != manifest['archive_bytes'] or sha256(archive) != manifest['archive_sha256']:
        raise ValueError('Archive integrity failure')
    with zipfile.ZipFile(archive) as saved:
        infos = saved.infolist()
        if len(infos) != len(files) or {info.filename for info in infos} != set(files):
            raise ValueError('Archive member set differs or contains duplicates')
        for info in infos:
            expected = files[info.filename]
            if info.file_size != expected['bytes'] or info.compress_type != zipfile.ZIP_DEFLATED or info.flag_bits & 1:
                raise ValueError('Archive member dimensions or encoding differ')
            with saved.open(info) as stream:
                size, digest = _digest_stream(stream, expected['bytes'])
            if size != expected['bytes'] or digest != expected['sha256']:
                raise ValueError('Archive member integrity failure: ' + info.filename)
    return identity, manifest


def _pointer(artifacts, store):
    path = artifacts / POINTER
    _regular(path)
    pointer = read_json(path)
    if set(pointer) != {'schema_version', 'archive_id'} or pointer['schema_version'] != 1 or not HEX.fullmatch(pointer['archive_id']):
        raise ValueError('Invalid archive pointer')
    archive = store / pointer['archive_id']
    identity, manifest = verify_archive(archive, expected_id=pointer['archive_id'])
    result, digest = _closed_result(artifacts)
    if (digest != manifest['training_result_sha256'] or result['snapshot_id'] != manifest['training_snapshot']
            or result['turn'] != manifest['turn']):
        raise ValueError('Archive belongs to a different training result')
    return archive, identity, manifest


class RecordReader:
    """Read checked member bytes during an ``open_records`` context.

    The archive has been fully verified before the context is entered. Each
    selected member is checked again as it is read. No game is materialized.
    """
    def __init__(self, archive, identity, manifest):
        self._archive = archive
        self._files = manifest['files']
        self._closed = False
        self.archive_id = identity
        self.training_snapshot = manifest['training_snapshot']
        self.training_result_sha256 = manifest['training_result_sha256']
        self.turn = manifest['turn']

    @property
    def names(self):
        return tuple(self._files)

    def member_sha256(self, name):
        if self._closed:
            raise ValueError('Game archive reader is closed')
        if not isinstance(name, str) or not RECORD.fullmatch(name):
            raise ValueError('Invalid game archive member name')
        return self._files[name]['sha256']

    def read(self, name):
        expected_hash = self.member_sha256(name)
        expected_size = self._files[name]['bytes']
        destination = io.BytesIO()
        with self._archive.open(name) as stream:
            size, digest = _digest_stream(stream, expected_size, destination)
        if size != expected_size or digest != expected_hash:
            raise ValueError('Selected game record integrity failure: ' + name)
        return destination.getvalue()


@contextmanager
def open_records(artifacts: Path, store: Path):
    """Verify a closed archive, hold its operator lock, and stream members.

    Publish downstream artifacts only after leaving the context successfully:
    the final check also rejects a changed training result or archive file.
    """
    artifacts, store = Path(artifacts), Path(store)
    with _lock(artifacts):
        path, identity, manifest = _pointer(artifacts, store)
        archive_path = path / 'games.zip'
        before = _signature(_regular(archive_path))
        fd = os.open(archive_path, os.O_RDONLY | os.O_NOFOLLOW)
        with os.fdopen(fd, 'rb') as file:
            if _signature(os.fstat(file.fileno())) != before:
                raise ValueError('Archive changed before opening')
            with zipfile.ZipFile(file) as archive:
                reader = RecordReader(archive, identity, manifest)
                try:
                    yield reader
                    if (_closed_result(artifacts)[1] != manifest['training_result_sha256']
                            or _signature(os.fstat(file.fileno())) != before
                            or _signature(_regular(archive_path)) != before):
                        raise ValueError('Archive or training result changed during reading')
                finally:
                    reader._closed = True


def _existing(artifacts, manifest):
    """Validate every existing member before any destructive action."""
    games = artifacts / 'games'
    signatures = {}
    for name in _record_names(games):
        if name not in manifest['files']:
            raise ValueError('Game directory has an unarchived member')
        path = games / name
        before = _regular(path)
        expected = manifest['files'][name]
        if before.st_size != expected['bytes'] or sha256(path) != expected['sha256']:
            raise ValueError('Original game differs from archive: ' + name)
        after = _regular(path)
        if _signature(before) != _signature(after):
            raise ValueError('Original game changed during verification')
        signatures[name] = _signature(after)
    return signatures


def pack(artifacts: Path, store: Path):
    artifacts, store = Path(artifacts), Path(store)
    with _lock(artifacts):
        if (artifacts / POINTER).exists():
            return _pointer(artifacts, store)[1]
        result, result_sha = _closed_result(artifacts)
        names = _record_names(artifacts / 'games')
        name_set = set(names)
        if not names or any(str(Path(n).with_suffix('.sgf' if n.endswith('.json') else '.json')) not in name_set for n in names):
            raise ValueError('Complete JSON/SGF pairs are required')
        store.mkdir(parents=True, exist_ok=True)
        if store.is_symlink():
            raise ValueError('Archive store cannot be a symlink')
        with tempfile.TemporaryDirectory(prefix='.partial-', dir=store) as temporary:
            temporary = Path(temporary)
            entries = {}
            with (temporary / 'games.zip').open('xb') as stream:
                with zipfile.ZipFile(stream, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=1, allowZip64=True) as archive:
                    for name in names:
                        path = artifacts / 'games' / name
                        before = _regular(path)
                        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
                        with os.fdopen(fd, 'rb') as original, archive.open(name, 'w', force_zip64=True) as member:
                            if _signature(os.fstat(original.fileno())) != _signature(before):
                                raise ValueError('Game changed before archival')
                            size, digest = _digest_stream(original, MAX_MEMBER_BYTES, member)
                            if _signature(os.fstat(original.fileno())) != _signature(before):
                                raise ValueError('Game changed during archival')
                        entries[name] = {'bytes': size, 'sha256': digest,
                                         'mode': stat.S_IMODE(before.st_mode), 'mtime_ns': before.st_mtime_ns}
                stream.flush()
                os.fsync(stream.fileno())
            if _record_names(artifacts / 'games') != names or sha256(artifacts / 'result.json') != result_sha:
                raise ValueError('Training artifacts changed during archival')
            manifest = {'schema_version': 1, 'kind': 'closed_training_games',
                        'training_result_sha256': result_sha, 'training_snapshot': result['snapshot_id'], 'turn': result['turn'],
                        'compression': 'zip-deflate-1', 'archive_sha256': sha256(temporary / 'games.zip'),
                        'archive_bytes': (temporary / 'games.zip').stat().st_size,
                        'original_bytes': sum(e['bytes'] for e in entries.values()), 'files': entries}
            with (temporary / 'manifest.json').open('xb') as stream:
                stream.write(canonical_json(manifest)); stream.flush(); os.fsync(stream.fileno())
            identity, _ = verify_archive(temporary)
            destination = store / identity
            for path in temporary.iterdir():
                path.chmod(0o444)
            _sync_directory(temporary)
            if destination.exists():
                verify_archive(destination, expected_id=identity)
            else:
                temporary.rename(destination)
                _sync_directory(store)
            pointer = canonical_json({'schema_version': 1, 'archive_id': identity})
            with tempfile.NamedTemporaryFile(prefix='.archive-pointer-', dir=artifacts) as stream:
                stream.write(pointer); stream.flush(); os.fsync(stream.fileno())
                os.chmod(stream.name, 0o444)
                os.link(stream.name, artifacts / POINTER)
            _sync_directory(artifacts)
            return identity


def evict(artifacts: Path, store: Path):
    artifacts, store = Path(artifacts), Path(store)
    with _lock(artifacts):
        _, identity, manifest = _pointer(artifacts, store)
        signatures = _existing(artifacts, manifest)
        for name, signature in signatures.items():
            path = artifacts / 'games' / name
            if _signature(_regular(path)) != signature:
                raise ValueError('Original game changed before eviction')
            path.unlink()
        _sync_directory(artifacts / 'games')
        return {'archive_id': identity, 'evicted_files': len(signatures),
                'evicted_logical_bytes': sum(manifest['files'][n]['bytes'] for n in signatures)}


def restore(artifacts: Path, store: Path):
    artifacts, store = Path(artifacts), Path(store)
    with _lock(artifacts):
        archive, identity, manifest = _pointer(artifacts, store)
        existing = _existing(artifacts, manifest)
        missing = sorted(set(manifest['files']) - set(existing))
        with tempfile.TemporaryDirectory(prefix='.games-restore-', dir=artifacts) as temporary:
            with zipfile.ZipFile(archive / 'games.zip') as saved:
                for name in missing:
                    path = Path(temporary) / name
                    expected = manifest['files'][name]
                    with saved.open(name) as source, path.open('xb') as target:
                        size, digest = _digest_stream(source, expected['bytes'], target)
                        target.flush(); os.fsync(target.fileno())
                    if size != expected['bytes'] or digest != expected['sha256']:
                        raise ValueError('Archive changed during restoration')
                    path.chmod(expected['mode'])
                    os.utime(path, ns=(expected['mtime_ns'], expected['mtime_ns']))
                    # Atomic publication without replacing an existing name.
                    os.link(path, artifacts / 'games' / name)
                    path.unlink()
        _sync_directory(artifacts / 'games')
        return {'archive_id': identity, 'restored_files': len(missing)}


def verify(artifacts: Path, store: Path):
    with _lock(Path(artifacts)):
        _, identity, manifest = _pointer(Path(artifacts), Path(store))
        existing = _existing(Path(artifacts), manifest)
        return {'archive_id': identity, 'total_files': len(manifest['files']),
                'materialized_files': len(existing), 'archive_bytes': manifest['archive_bytes'],
                'original_bytes': manifest['original_bytes']}
