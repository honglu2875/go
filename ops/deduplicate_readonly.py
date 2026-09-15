#!/usr/bin/env python3
"""Share identical immutable artifact bytes while retaining every logical path."""
import argparse
import os
from pathlib import Path
import stat
import sys
import time
import uuid

sys.dont_write_bytecode = True
SOURCE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE / 'packages/gozero/src'))
from gozero.checkpoints import sha256
from gozero.model_artifacts import artifact
from gozero.snapshots import canonical_json, read_json, verify


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workspace-root', type=Path, required=True)
    p.add_argument('--plan', type=Path, required=True)
    p.add_argument('--expected-plan-sha256', required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args(); verify(SOURCE); root = args.workspace_root.resolve()
    if sha256(args.plan) != args.expected_plan_sha256:
        raise ValueError('Storage plan changed')
    plan = read_json(args.plan)
    if plan['kind'] != 'readonly_artifact_deduplication' or args.output.exists():
        raise ValueError('Wrong plan kind or existing receipt')
    for name, expected in plan['evidence_sha256'].items():
        if sha256(artifact(root, name)) != expected:
            raise ValueError('Artifact evidence changed: ' + name)
    groups = []; seen = set()
    for group in plan['groups']:
        paths = []
        for name in group['paths']:
            path = artifact(root, name)
            allowed = (name.startswith('.gozero/native/') and path.suffix == '.so') or (
                name.startswith('runs/pod-') and '/artifacts/checkpoints/turn-' in name and path.name == 'arrays.npz')
            info = path.stat()
            if (not allowed or path in seen or not stat.S_ISREG(info.st_mode) or info.st_mode & 0o222
                    or str(path.relative_to(root)) != name or sha256(path) != group['sha256']):
                raise ValueError('Only distinct, verified, read-only native/checkpoint paths are allowed')
            seen.add(path); paths.append((path, info))
        if len(paths) < 2 or len({(s.st_dev, stat.S_IMODE(s.st_mode), s.st_uid, s.st_gid) for _, s in paths}) != 1:
            raise ValueError('Group must share a filesystem, mode and ownership')
        groups.append((group['sha256'], paths))
    report = {'schema_version': 1, 'kind': 'readonly_artifact_deduplication', 'operator_snapshot': SOURCE.name,
              'plan_sha256': args.expected_plan_sha256, 'started_unix': time.time(), 'status': 'running', 'groups': [],
              'content_and_logical_paths_preserved': True,
              'metadata_note': 'Hard links share inode timestamps. Original per-path times are retained here for optional materialization. This does not add external durability.'}
    try:
        for digest, paths in groups:
            source = paths[0][0]; row = {'sha256': digest, 'source': str(source.relative_to(root)), 'files': []}
            report['groups'].append(row)
            for path, old in paths:
                now = path.stat()
                if (now.st_dev, now.st_ino, now.st_size, now.st_mtime_ns, now.st_mode) != (
                        old.st_dev, old.st_ino, old.st_size, old.st_mtime_ns, old.st_mode):
                    raise ValueError('Artifact changed after validation')
                changed = now.st_ino != source.stat().st_ino
                if changed:
                    temporary = path.with_name('.' + path.name + '.dedup-' + uuid.uuid4().hex)
                    try:
                        os.link(source, temporary, follow_symlinks=False)
                        os.replace(temporary, path)
                        fd = os.open(path.parent, os.O_DIRECTORY)
                        try: os.fsync(fd)
                        finally: os.close(fd)
                    finally:
                        temporary.unlink(missing_ok=True)
                if sha256(path) != digest or path.stat().st_mode & 0o222:
                    raise ValueError('Published artifact bytes or read-only mode differ')
                row['files'].append({'path': str(path.relative_to(root)), 'changed_inode': changed,
                    'bytes': old.st_size, 'original_mode': stat.S_IMODE(old.st_mode),
                    'original_atime_ns': old.st_atime_ns, 'original_mtime_ns': old.st_mtime_ns,
                    'released_allocated_bytes_if_last_link': old.st_blocks * 512 if changed and old.st_nlink == 1 else 0})
        report['status'] = 'passed'
    except BaseException as error:
        report['error'] = repr(error); report['status'] = 'failed'; raise
    finally:
        verify(SOURCE); report['finished_unix'] = time.time()
        report['released_allocated_bytes'] = sum(f['released_allocated_bytes_if_last_link'] for g in report['groups'] for f in g['files'])
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('xb') as f:
            f.write(canonical_json(report)); f.flush(); os.fsync(f.fileno())
        args.output.chmod(0o444)
        print({'status': report['status'], 'released_allocated_bytes': report['released_allocated_bytes'], 'sha256': sha256(args.output)})


if __name__ == '__main__':
    main()
