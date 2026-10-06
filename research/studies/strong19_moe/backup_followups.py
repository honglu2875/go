"""Preserve prepared follow-ups and their frozen sources on two disk peers."""
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[3]
STUDY=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'packages/gozero/src'))
from gozero.disk_mirror import inventory,publish
from gozero.durable_files import atomic_json,sha256
from gozero.pod import load_hosts
from gozero.snapshots import verify

names=('strong19_moe','strong19_moe_remat','strong19_moe_permute','strong19_moe_temporal')
paths={ROOT/'packages/gozero/src/gozero/moe.py',ROOT/'tests/test_moe.py'}
for name in names:
    for category in ('recipes','studies'):
        folder=ROOT/'research'/category/name
        paths.update(p for p in folder.iterdir() if p.is_file()
                     and p.suffix in ('.json','.py','.md','.txt','.npz')
                     and not p.name.startswith('followup-backup'))
snapshots=('0678b23f78ceda2e9e34e2ad3cb9bfb45634ab317958422868fef1bc65adaf77',
           'aa32c6b9d7cd7d10d2d7a16980f43af554be6525c9a5e70c1d97e9ba256612b5')
for identity in snapshots:
    folder=ROOT/'.gozero/snapshots'/identity
    verify(folder)
    paths.update(p for p in folder.rglob('*') if p.is_file())
files=inventory(ROOT,sorted(str(p.relative_to(ROOT)) for p in paths))
atomic_json(STUDY/'followup-backup-plan-001.json',dict(files=files,snapshots=snapshots,operator_sha256=sha256(Path(__file__))),replace=False)
hosts={h.rank:h for h in load_hosts(ROOT/'ops/hosts.json')}
copies=[]
for rank in (1,3):
    copies.append(dict(peer_rank=rank,**publish(ROOT,ROOT/'.gozero/retained-source-archives/20261004-moe-followups-001',files=files,peer=hosts[rank].ssh,python='python3',floor_bytes=8<<30)))
result=dict(status='passed',files=files,copies=copies,snapshots=snapshots,
            operator_sha256=sha256(Path(__file__)),scope='Private disk backup of frozen follow-ups, recipe/library source, qualifications and study notes. Active learning payloads remain under their checkpoint replication protocol.')
atomic_json(STUDY/'followup-backups-001.json',result,replace=False)
print(json.dumps(dict(status='passed',files=len(files),bytes=sum(x['bytes'] for x in files.values()),verified_disk_peers=2)))
