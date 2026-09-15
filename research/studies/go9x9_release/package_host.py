"""Replay-audit every published game, then package exact NPZ bytes into tar shards."""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import tarfile
import time

CONTRACT = 'f593931d66505372580a8ae9e9624cbb4940cf5eb9ded6c1378b9fe220ba87bb'
SNAPSHOT = Path('/dev/shm/gozero/environments/0780619799010a206823')
sys.path.insert(0, str(SNAPSHOT / 'site-packages'))
import numpy as np
from flygo.data.corpus import audit_game, atomic_json, identity


def audit(path):
    path = Path(path)
    meta = audit_game(path)  # Frozen producer: complete Rust history, masks, labels, outcomes, splits.
    assert meta['contract_id'] == CONTRACT
    assert meta['game_id'] == path.stem
    assert identity([CONTRACT, meta['host_index'], meta['worker_index'], meta['sequence']]) == path.stem
    with np.load(path, allow_pickle=False) as a:
        n = meta['rows']
        shapes = {'stones': (n, 9, 9), 'legal': (n, 82), 'raw_policy': (n, 82),
                  'raw_value': (n,), 'search_policy': (n, 82), 'search_value': (n,),
                  'search_policy_valid': (n,), 'root_edge_visits': (n, 82),
                  'raw_score': (n,), 'raw_score_valid': (n,), 'teacher_visits': (n,), 'actions': (n,)}
        for key, shape in shapes.items():
            assert a[key].shape == shape, (path.name, key)
        assert np.isfinite(a['search_value']).all() and np.all(np.abs(a['search_value']) <= 1.00001)
        assert np.isfinite(a['raw_score']).all()
        assert np.all(a['root_edge_visits'] >= 0)
        expert = 1 + np.arange(n) % 2 == meta['expert_color']
        assert not np.any(a['search_policy_valid'][~expert])
        assert meta['terminal'] or n == 324
    return dict(meta, bytes=path.stat().st_size, source=str(path))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--host', type=int, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--workers', type=int, default=16)
    args = p.parse_args()
    os.sched_setaffinity(0, set(range(args.workers)))
    run = Path('/dev/shm/gozero/runs/expert-v1')
    sup = json.loads((run / 'supervisor.json').read_text())
    assert sup['state'] == 'stopped' and (run / 'stop').exists()
    config = json.loads((run / 'config.json').read_text())
    assert identity(config['contract']) == CONTRACT and config['host_index'] == args.host
    root = Path('/dev/shm/gozero/corpora') / CONTRACT / f'host-{args.host}'
    paths = sorted(root.glob('worker-*/*.npz'))
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    assert not (output / 'result.json').exists(), 'A completed release is immutable'
    for name in ('data', 'index', 'provenance'):
        (output / name).mkdir(exist_ok=True)
    started = time.time()
    atomic_json(output / 'status.json', dict(state='auditing', games=len(paths), started=started))
    stats = Counter()
    opponents = {}
    shards = []
    archive = None
    shard_bytes = 0
    shard_games = 0
    shard_positions = 0
    shard_path = None

    def close_shard():
        nonlocal archive
        if archive is None:
            return
        archive.close()
        with shard_path.open('rb') as stream:
            sha = hashlib.file_digest(stream, 'sha256').hexdigest()
        shards.append(dict(path=str(shard_path.relative_to(output)), sha256=sha,
                           bytes=shard_path.stat().st_size, games=shard_games, positions=shard_positions))
        archive = None

    index_path = output / 'index' / f'host-{args.host}.jsonl.gz'
    # Fixed gzip timestamp and tar metadata make packaging reproducible.
    with index_path.open('wb') as raw_index, gzip.GzipFile(fileobj=raw_index, mode='wb', mtime=0, filename='') as idx:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for k, meta in enumerate(pool.map(audit, paths, chunksize=32)):
                assert meta['teacher_sha256'] == config['contract']['teacher']['sha256']
                assert meta['opponent_sha256'] == config['contract']['opponents'][meta['opponent_index']]['sha256']
                payload = Path(meta.pop('source')).read_bytes()
                sha = hashlib.sha256(payload).hexdigest()
                if archive is None or shard_bytes + len(payload) > 256 * 1024**2:
                    close_shard()
                    shard_path = output / 'data' / f'host-{args.host}-{len(shards):05d}.tar'
                    assert not shard_path.exists()
                    archive = tarfile.open(shard_path, 'w', format=tarfile.USTAR_FORMAT)
                    shard_bytes = shard_games = shard_positions = 0
                member = f"{meta['split']}/{meta['game_id']}.npz"
                info = tarfile.TarInfo(member)
                info.size = len(payload)
                info.mode = 0o444
                offset = archive.offset + 512
                archive.addfile(info, io.BytesIO(payload))
                shard_bytes += len(payload)
                shard_games += 1
                shard_positions += meta['rows']
                row = dict(meta, sha256=sha, shard=str(shard_path.relative_to(output)), member=member, offset=offset)
                idx.write((json.dumps(row, sort_keys=True, separators=(',', ':')) + '\n').encode())
                stats['games'] += 1
                stats['positions'] += meta['rows']
                stats['source_bytes'] += len(payload)
                stats['terminal_games' if meta['terminal'] else 'capped_games'] += 1
                stats['terminal_positions' if meta['terminal'] else 'capped_positions'] += meta['rows']
                stats[meta['split'] + '_games'] += 1
                stats[meta['split'] + '_positions'] += meta['rows']
                op = opponents.setdefault(str(meta['opponent_index']), Counter())
                op['games'] += 1
                op['positions'] += meta['rows']
                op['terminal_games' if meta['terminal'] else 'capped_games'] += 1
                if (k + 1) % 1000 == 0:
                    atomic_json(output / 'status.json', dict(state='auditing', done=k+1, total=len(paths), elapsed=time.time()-started))
    close_shard()
    assert sorted(root.glob('worker-*/*.npz')) == paths
    assert stats['games'] == len(paths)
    provenance = output / 'provenance'
    atomic_json(provenance / f'host-{args.host}-config.json', config)
    atomic_json(provenance / f'host-{args.host}-stopped.json', sup)
    result = dict(host=args.host, contract_id=CONTRACT, stats=dict(stats), opponents=opponents,
                  shards=shards, elapsed_seconds=time.time()-started, completed=time.time(),
                  index=dict(path=str(index_path.relative_to(output)), bytes=index_path.stat().st_size,
                             sha256=hashlib.sha256(index_path.read_bytes()).hexdigest()),
                  audit='All published games replayed by the frozen producer audit, plus schema/value/identity checks',
                  status='passed')
    atomic_json(output / 'result.json', result)
    atomic_json(output / 'status.json', dict(state='complete', games=len(paths), stats=dict(stats), elapsed=time.time()-started))
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
