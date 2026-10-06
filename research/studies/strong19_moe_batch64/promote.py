"""Add two complete disk copies of a selected RAM trial; preserve every source."""
import json
import os
from pathlib import Path
import shutil
import time
from execute_run import ROOT,STUDY,read,sha,publish,require
from gozero.disk_mirror import inventory,publish as mirror
from gozero.pod import load_hosts


def promote(record,label):
    result=read(ROOT/'runs'/record['attempt']/'rank-0/artifacts/result.json');cp=result['latest_checkpoint']
    if 'disk' in cp:
        return dict(status='passed',already_on_disk=True,attempt=record['attempt'],existing_disk_receipt=cp['disk'])
    require(sha(ROOT/record['audit'])==record['audit_sha256'] and read(ROOT/record['audit'])['status']=='passed','Selected trial not audited')
    source=Path(cp['path']);require(source.is_relative_to('/dev/shm/gozero-staged-checkpoints'),'Wrong new-state location')
    require(sha(source/'manifest.json')==cp['manifest_sha256'],'Primary identity changed')
    # Add small metadata alongside the original checkpoint. Its array remains
    # in place; publication streams it directly without duplicating the RAM payload.
    staging=source.parent;metadata=staging/'promotion-metadata';metadata.mkdir(exist_ok=False)
    paths=[source/n for n in ('manifest.json','state.json','actors.json','arrays.npz')]+[source.with_suffix('.group.json')]
    restore=[]
    for rank in range(4):
        r=read(ROOT/'runs'/record['attempt']/f'rank-{rank}/artifacts/result.json');path=Path(r['latest_checkpoint']['path'])
        require(r['latest_checkpoint']['group_sha256']==cp['group_sha256'],'Checkpoint groups differ')
        restore.append(dict(host_rank=rank,original_path=str(path),group_sha256=cp['group_sha256']))
        if rank:
            destination=metadata/f'host-{rank}';destination.mkdir()
            for n in ('manifest.json','state.json','actors.json','arrays.npz'):
                shutil.copyfile(path/n,destination/n);(destination/n).chmod(0o444);paths.append(destination/n)
            shutil.copyfile(path.with_suffix('.group.json'),destination/'group.json');paths.append(destination/'group.json')
    publish(metadata/'restore.json',dict(status='prepared',snapshot=result['snapshot_id'],attempt=record['attempt'],
        paths=restore,owner_bundle_prefix=source.name,restore='Restore complete rank files and group JSON to the recorded locations. Do not bypass source or logical-rank topology validation.'))
    paths.append(metadata/'restore.json')
    files=inventory(staging,[str(p.relative_to(staging)) for p in paths]);hosts={h.rank:h for h in load_hosts(ROOT/'ops/hosts.json')}
    copies=[]
    for rank in (1,3):
        copies.append(dict(peer_rank=rank,**mirror(staging,ROOT/'.gozero/retained-selected'/record['attempt'],
            files=files,peer=hosts[rank].ssh,python=str(ROOT/'.gozero/environments/a587255ebe0f78b1ec98bbb12ab1cb3315ec668424a08820c80da7353de40192/bin/python'),floor_bytes=2<<30)))
    answer=dict(status='passed',attempt=record['attempt'],created=time.time(),files=files,copies=copies,
                durability='Two checksum-verified fsynced disk bundles including all rank state. Original RAM and disk files are preserved.')
    publish(STUDY/(label+'-disk-promotion.json'),answer);return answer
