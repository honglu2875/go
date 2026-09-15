#!/usr/bin/env python3
"""Pack, verify, evict or restore completed JSON/SGF records without changing bytes."""
import argparse
import json
from pathlib import Path
import shutil
import sys
import time

sys.dont_write_bytecode = True
SOURCE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE / 'packages/gozero/src'))
from gozero import game_archives
from gozero.checkpoints import sha256
from gozero.snapshots import canonical_json, read_json, verify


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['pack', 'verify', 'evict', 'restore'])
    parser.add_argument('--workspace-root', type=Path, required=True)
    parser.add_argument('--artifacts', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    verify(SOURCE)
    root = args.workspace_root.resolve(strict=True)
    artifacts = args.artifacts.resolve(strict=True)
    if not artifacts.is_relative_to(root / 'runs'):
        raise ValueError('Only workspace training-run artifacts may be archived')
    result = read_json(artifacts / 'result.json')
    verify(root / '.gozero/snapshots' / result['snapshot_id'])
    store = root / '.gozero/game-archives'
    report = {'schema_version': 1, 'kind': 'game_archive_operation', 'status': 'failed',
              'action': args.action, 'operator_snapshot': SOURCE.name,
              'artifacts': str(artifacts.relative_to(root)),
              'training_result_sha256': sha256(artifacts / 'result.json'),
              'started_unix': time.time(), 'free_bytes_before': shutil.disk_usage(root).free,
              'scope': 'Local reversible archival only. No claim of external durability or host-loss recovery.'}
    try:
        if args.output.exists():
            raise FileExistsError(args.output)
        if args.action == 'pack':
            report['archive_id'] = game_archives.pack(artifacts, store)
        else:
            report.update(getattr(game_archives, args.action)(artifacts, store))
        verify(SOURCE)
        report['status'] = 'passed'
    except Exception as error:
        report['error'] = repr(error)
        raise
    finally:
        report.update(finished_unix=time.time(), free_bytes_after=shutil.disk_usage(root).free)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('xb') as stream:
            stream.write(canonical_json(report))
        args.output.chmod(0o444)
        print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
