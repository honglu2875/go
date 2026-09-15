"""Assemble a hash-checked, explicit upload directory entirely in shared memory."""
from __future__ import annotations

from collections import Counter
import gzip
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import time

WORK = Path('/workspace/go/research/studies/go9x9_release')
ROOT = Path('/dev/shm/go9x9-release-v1')
OUT = ROOT / 'publish'
SSH = ['ssh', '-F', '/dev/null', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10']


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False)+'\n')


def main():
    fs = os.statvfs('/dev/shm')
    assert fs.f_bavail * fs.f_frsize > 72 * 1024**3
    OUT.mkdir(exist_ok=True)
    results = []
    for host in range(4):
        target = f'go-user@worker-{host}.example.invalid'
        source = ROOT / f'host-{host}'
        result = json.loads(subprocess.check_output(SSH + [target, 'cat '+str(source / 'result.json')], text=True))
        assert result['status'] == 'passed' and result['host'] == host
        results.append(result)
        for name in ('data', 'index', 'provenance'):
            dest = OUT / name
            dest.mkdir(exist_ok=True)
            if host == 0:
                for path in (source / name).iterdir():
                    if not (dest / path.name).exists():
                        os.link(path, dest / path.name)
            else:
                subprocess.run(['rsync', '-a', '--ignore-existing', '-e', shlex.join(SSH),
                                f'{target}:{source / name}/', str(dest)+'/'], check=True)
        for row in [*result['shards'], result['index']]:
            path = OUT / row['path']
            assert path.stat().st_size == row['bytes'] and sha(path) == row['sha256']
        save(OUT / 'provenance' / f'host-{host}-audit.json', result)
        save(WORK / f'host-{host}-audit.json', result)
        print(json.dumps(dict(host=host, copied_and_hashed=True)), flush=True)
    contract = json.loads((OUT / 'provenance/host-0-config.json').read_text())['contract']
    for host in range(4):
        assert json.loads((OUT / f'provenance/host-{host}-config.json').read_text())['contract'] == contract
    save(OUT / 'contract.json', contract)
    stats = Counter()
    opponents = {}
    for result in results:
        stats.update(result['stats'])
        for key, value in result['opponents'].items():
            opponents.setdefault(key, Counter()).update(value)
    families = {}
    ids = set()
    index_stats = Counter()
    for path in sorted((OUT / 'index').glob('*.jsonl.gz')):
        with gzip.open(path, 'rt') as stream:
            for line in stream:
                row = json.loads(line)
                assert row['game_id'] not in ids
                ids.add(row['game_id'])
                previous = families.setdefault(row['opening_family'], row['split'])
                assert previous == row['split'], 'Opening family leaks across splits'
                index_stats['games'] += 1
                index_stats['positions'] += row['rows']
    assert index_stats['games'] == stats['games'] and index_stats['positions'] == stats['positions']
    snapshot = Path('/dev/shm/gozero/environments/0780619799010a206823')
    producer_sources = ['go.py', 'gtp.py', 'runtime.py', 'storage.py', 'data/corpus.py',
                        'data/generate.py', 'data/katago.py', 'data/label.py', 'data/service.py']
    for name in producer_sources:
        dest = OUT / 'provenance/producer' / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(snapshot / 'site-packages/flygo' / name, dest)
    shutil.copyfile(snapshot / 'snapshot.json', OUT / 'provenance/producer-snapshot.json')
    for name in ('stop-requests.json', 'stopped-hosts.json', 'package_host.py'):
        shutil.copyfile(WORK / name, OUT / 'provenance' / name)
    shards = [row for result in results for row in result['shards']]
    indices = [result['index'] for result in results]
    identity_payload = dict(contract_id=results[0]['contract_id'], shards=shards, indices=indices)
    release_id = hashlib.sha256(json.dumps(identity_payload, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    manifest = dict(schema_version=1, release='final-9x9-v1', release_id=release_id,
        contract_id=results[0]['contract_id'], generated_by='KataGo mixed-strength play; fixed strong 9x9 teacher',
        finalized_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), stats=dict(stats),
        opponents={k:dict(v, name=contract['opponents'][int(k)]['name']) for k,v in opponents.items()},
        opening_families=len(families), opening_families_by_split=dict(Counter(families.values())),
        duplicate_game_ids=0, cross_split_opening_families=0, shards=shards, indices=indices,
        audit='Every published game replayed by the original frozen native rules engine; all passed.',
        split_rule='D4-canonical first-eight-action family SHA256; int(hash[:8],16)%100: <5 test, <10 validation, else train',
        stopped=True, generation_will_not_resume=True,
        incomplete_inflight=dict(published=False,
            stale_generating_workers=sum(w['state']=='generating' for h in json.loads((WORK/'stopped-hosts.json').read_text()) for w in h['workers']),
            note='Status records can remain generating after bounded supervisor shutdown; the exact number of unfinished games is not recoverable from these counters.'),
        exclusions='No published NPZ excluded; no model weights, credentials, caches, or unrelated project files included.')
    save(OUT / 'manifest.json', manifest)
    save(WORK / 'manifest.json', manifest)
    rows='\n'.join(f"| {split} | {stats[split+'_games']:,} | {stats[split+'_positions']:,} |" for split in ('train','validation','test'))
    op_rows='\n'.join(f"| {k} | {contract['opponents'][int(k)]['name']} | {v['games']:,} | {v['positions']:,} |" for k,v in sorted(opponents.items()))
    readme=(WORK / 'README.template.md').read_text().replace('{{SPLITS}}', rows).replace('{{OPPONENTS}}', op_rows).replace('{{RELEASE_ID}}',release_id).replace('{{FAMILIES}}',f'{len(families):,}')
    (OUT / 'README.md').write_text(readme)
    shutil.copyfile(WORK / 'read_games.py', OUT / 'read_games.py')
    files=[]
    for path in sorted(OUT.rglob('*')):
        if path.is_file() and path.name != 'SHA256SUMS':
            assert path.name not in ('.env', '.env.hf')
            files.append((sha(path), str(path.relative_to(OUT))))
    (OUT / 'SHA256SUMS').write_text(''.join(f'{h}  {p}\n' for h,p in files))
    save(WORK / 'upload-allowlist.json', dict(files=[p for _,p in files]+['SHA256SUMS'], bytes=sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file()), release_id=release_id))
    print(json.dumps(dict(state='ready', directory=str(OUT), stats=dict(stats), release_id=release_id)),flush=True)


if __name__ == '__main__':
    main()
