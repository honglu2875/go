"""Relocate two closed historical checkpoint bundles to two verified disk peers.

Only the two large owner payloads named below are retired locally. All source
metadata remains, and both complete four-rank bundles are preserved on each
peer before any local unlink. This is not a general pruning policy.
"""
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
STUDY = Path(__file__).parent
sys.path.insert(0, str(ROOT / 'packages/gozero/src'))
from gozero.disk_mirror import inventory, publish
from gozero.durable_files import atomic_json, sha256
from gozero.pod import load_hosts

ATTEMPTS = ('pod-20260913T203547Z-920ab564', 'pod-20260913T184112Z-f2c9b9e4')
TURN = 'turn-000001024'


def main():
    plan_path = STUDY / 'archive-20260927-plan-001.json'
    copies_path = STUDY / 'archive-20260927-copies-001.json'
    result_path = STUDY / 'archive-20260927-result-001.json'
    if result_path.exists():
        raise ValueError('This relocation already has a closure record')
    names = []
    retire = []
    for name in ATTEMPTS:
        attempt = ROOT / 'runs' / name
        if json.loads((attempt / 'result.json').read_text())['status'] != 'passed':
            raise ValueError('Only these closed successful historical attempts may move')
        names += [str((attempt / n).relative_to(ROOT)) for n in ('launch.json', 'result.json')]
        for rank in range(4):
            rank_root = attempt / f'rank-{rank}'
            checkpoint = rank_root / 'artifacts/checkpoints' / TURN
            manifest = json.loads((checkpoint / 'manifest.json').read_text())
            for n, expected in manifest['files'].items():
                p = checkpoint / n
                if p.is_symlink() or p.stat().st_size != expected['bytes'] or sha256(p) != expected['sha256']:
                    raise ValueError('Historical checkpoint differs: ' + str(p))
                names.append(str(p.relative_to(ROOT)))
            names.append(str((checkpoint / 'manifest.json').relative_to(ROOT)))
            for p in checkpoint.parent.glob(TURN + '.*'):
                if p.is_file():
                    names.append(str(p.relative_to(ROOT)))
            for relative in ('result.json', 'artifacts/result.json', 'artifacts/resolved_config.json'):
                p = rank_root / relative
                if p.is_file():
                    names.append(str(p.relative_to(ROOT)))
        p = attempt / f'rank-0/artifacts/checkpoints/{TURN}/arrays.npz'
        stat = p.stat()
        if stat.st_nlink != 1 or stat.st_size < 2_000_000_000:
            raise ValueError('Unexpected historical owner payload identity')
        retire.append(dict(path=str(p.relative_to(ROOT)), device=stat.st_dev,
                           inode=stat.st_ino, bytes=stat.st_size, sha256=sha256(p)))
    files = inventory(ROOT, sorted(set(names)))
    plan = dict(status='prepared', kind='historical_checkpoint_disk_relocation',
                operator_sha256=sha256(Path(__file__)), attempts=list(ATTEMPTS),
                source_root=str(ROOT), files=files, retire=retire, peers=[1, 3],
                target=str(ROOT / '.gozero/retained-checkpoint-archives/20260927-001'),
                policy='Preserve complete four-rank bundles on two disk peers; retire only named large local arrays after both copies verify.')
    if plan_path.exists():
        if json.loads(plan_path.read_text()) != plan:
            raise ValueError('Existing relocation plan differs')
    else:
        atomic_json(plan_path, plan, replace=False)
    hosts = {h.rank: h for h in load_hosts(ROOT / 'ops/hosts.json')}
    copies = []
    for rank in plan['peers']:
        print(json.dumps(dict(kind='copy_started', peer_rank=rank)), flush=True)
        copy = publish(ROOT, Path(plan['target']), files=files,
                       peer=hosts[rank].ssh, python='python3', floor_bytes=8 << 30)
        copies.append(dict(peer_rank=rank, **copy))
        print(json.dumps(dict(kind='copy_verified', peer_rank=rank, bytes=copy['bytes'])), flush=True)
    receipts = dict(status='passed', plan_sha256=sha256(plan_path), copies=copies)
    if copies_path.exists():
        if json.loads(copies_path.read_text()) != receipts:
            raise ValueError('Existing mirror receipt differs')
    else:
        atomic_json(copies_path, receipts, replace=False)
    # Check every source file again before retirement; metadata must stay intact.
    if inventory(ROOT, files) != files:
        raise ValueError('Source bundle changed during copying')
    before = os.statvfs(ROOT)
    for row in retire:
        p = ROOT / row['path']
        st = p.stat()
        if p.is_symlink() or (st.st_dev, st.st_ino, st.st_size, st.st_nlink) != (row['device'], row['inode'], row['bytes'], 1):
            raise ValueError('Payload identity changed before retirement')
    for row in retire:
        p = ROOT / row['path']
        p.unlink()
        fd = os.open(p.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    remaining = {k: v for k, v in files.items() if k not in {r['path'] for r in retire}}
    if inventory(ROOT, remaining) != remaining:
        raise ValueError('Retained metadata changed')
    after = os.statvfs(ROOT)
    result = dict(status='passed', kind=plan['kind'], finished=time.time(),
                  plan_sha256=sha256(plan_path), copies_sha256=sha256(copies_path),
                  retired=retire, retired_bytes=sum(r['bytes'] for r in retire),
                  disk_free_before=before.f_bavail * before.f_frsize,
                  disk_free_after=after.f_bavail * after.f_frsize,
                  local_metadata_verified=True, complete_disk_copies=2)
    atomic_json(result_path, result, replace=False)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
