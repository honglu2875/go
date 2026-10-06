"""Preserve the small registered research closure on two disk peers."""
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[3];STUDY=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'packages/gozero/src'))
from gozero.disk_mirror import inventory,publish
from gozero.durable_files import atomic_json,sha256
from gozero.pod import load_hosts
paths=[p for p in STUDY.iterdir() if p.is_file() and p.suffix in ('.json','.py','.md','.txt') and not p.name.startswith('registration-backup')]
files=inventory(ROOT,[str(p.relative_to(ROOT)) for p in paths])
hosts={h.rank:h for h in load_hosts(ROOT/'ops/hosts.json')};copies=[]
for rank in (1,3):
    copies.append(dict(peer_rank=rank,**publish(ROOT,ROOT/'.gozero/retained-source-archives/20261004-moe-001',files=files,peer=hosts[rank].ssh,python='python3',floor_bytes=8<<30)))
result=dict(status='passed',operator_sha256=sha256(Path(__file__)),files=files,copies=copies,scope='Study registration, operators, configs, qualifications and storage receipts. Frozen library/recipe snapshot is already staged on all four hosts.')
atomic_json(STUDY/'registration-backups-001.json',result,replace=False)
print(json.dumps(dict(status='passed',files=len(files),bytes=sum(x['bytes'] for x in files.values()),complete_disk_copies=2)))
