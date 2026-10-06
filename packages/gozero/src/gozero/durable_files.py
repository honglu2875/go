"""Atomic disk publication shared by collection and recovery tooling."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile


def sha256(path: Path) -> str:
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def sync_directory(path: Path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def mkdir(path: Path):
    path = Path(path)
    if path.is_symlink():
        raise ValueError('A durable directory must not be a symlink')
    if path.is_dir():
        return
    mkdir(path.parent)
    try:
        path.mkdir()
    except FileExistsError:
        if not path.is_dir() or path.is_symlink():
            raise
    sync_directory(path.parent)


def require_disk(path: Path):
    """Reject RAM mounts even when reached through a filesystem alias."""
    path = Path(path).resolve(strict=True)
    matches = []
    for line in Path('/proc/self/mountinfo').read_text().splitlines():
        left, right = line.split(' - ', 1)
        mount = left.split()[4]
        for escaped, literal in [('\\040', ' '), ('\\011', '\t'), ('\\012', '\n'), ('\\134', '\\')]:
            mount = mount.replace(escaped, literal)
        if path == Path(mount) or Path(mount) in path.parents:
            matches.append((len(Path(mount).parts), right.split()[0]))
    if not matches or max(matches)[1] in {'tmpfs', 'ramfs', 'devtmpfs'}:
        raise ValueError('Durable state requires a disk-backed filesystem')


def atomic_json(path: Path, value, *, replace: bool = True):
    path = Path(path)
    mkdir(path.parent)
    raw = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()
    fd, name = tempfile.mkstemp(prefix='.' + path.name + '.', suffix='.partial', dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, 'wb') as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        if replace:
            if path.is_symlink():
                raise ValueError('Refusing to replace a symlink')
            os.replace(temporary, path)
        else:
            os.link(temporary, path)
        sync_directory(path.parent)
        if path.read_bytes() != raw:
            raise IOError('Published JSON failed readback')
    finally:
        temporary.unlink(missing_ok=True)


def copy_verified(source: Path, target: Path, *, expected_sha256: str, expected_bytes: int):
    """Publish without overwriting an existing artifact; verify stored bytes."""
    source, target = Path(source), Path(target)
    if source.is_symlink() or not source.is_file():
        raise ValueError('Source must be a regular file')
    mkdir(target.parent)
    require_disk(target.parent)
    if target.exists():
        if target.is_symlink() or target.stat().st_size != expected_bytes or sha256(target) != expected_sha256:
            raise ValueError('Existing durable artifact differs')
        return
    fd, name = tempfile.mkstemp(prefix='.' + target.name + '.', suffix='.partial', dir=target.parent)
    temporary = Path(name)
    try:
        digest, count = hashlib.sha256(), 0
        with source.open('rb') as inp, os.fdopen(fd, 'wb') as out:
            while data := inp.read(8 << 20):
                out.write(data)
                digest.update(data)
                count += len(data)
            out.flush()
            os.fchmod(out.fileno(), 0o444)
            os.fsync(out.fileno())
        if count != expected_bytes or digest.hexdigest() != expected_sha256:
            raise ValueError('Source artifact differs from expected identity')
        if sha256(temporary) != expected_sha256:
            raise IOError('Durable artifact failed readback')
        os.link(temporary, target)
        sync_directory(target.parent)
    finally:
        temporary.unlink(missing_ok=True)
