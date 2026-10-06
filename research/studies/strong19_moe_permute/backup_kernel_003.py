"""Back up the amended immutable kernel screen to two private disk peers."""
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[3];STUDY=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'packages/gozero/src'))
from gozero.disk_mirror import inventory,publish
from gozero.durable_files import atomic_json,sha256
from gozero.pod import load_hosts
from gozero.snapshots import verify
plan=json.loads((STUDY/'kernel-registration-003.json').read_text())
snapshot=ROOT/'.gozero/snapshots'/plan['snapshot'];verify(snapshot)
paths={p for p in snapshot.rglob('*') if p.is_file()}
paths.update(p for p in STUDY.iterdir() if p.is_file() and p.suffix in ('.py','.json','.md') and not p.name.startswith('kernel-backup'))
paths.update(ROOT/p for p in ('research/recipes/strong19_moe/README.md','ops/STATUS.md','research/studies/strong19_moe/OPTIMIZATION_NOTES.md'))
files=inventory(ROOT,sorted(str(p.relative_to(ROOT)) for p in paths))
atomic_json(STUDY/'kernel-backup-plan-003.json',dict(files=files,snapshot=snapshot.name,operator_sha256=sha256(Path(__file__))),replace=False)
hosts={h.rank:h for h in load_hosts(ROOT/'ops/hosts.json')};copies=[]
for rank in (1,3):
    copies.append(dict(peer_rank=rank,**publish(ROOT,ROOT/'.gozero/retained-source-archives/20261004-moe-kernel-003',files=files,peer=hosts[rank].ssh,python='python3',floor_bytes=8<<30)))
atomic_json(STUDY/'kernel-backups-003.json',dict(status='passed',files=files,copies=copies,snapshot=snapshot.name,operator_sha256=sha256(Path(__file__))),replace=False)
print(json.dumps(dict(status='passed',files=len(files),bytes=sum(x['bytes'] for x in files.values()),verified_disk_peers=2)))
