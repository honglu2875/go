"""Disk checkpoint admission, complete multi-rank peer copies, and retention."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import tempfile

from . import checkpoints
from .disk_mirror import inventory, publish
from .durable_files import atomic_json, mkdir, require_disk, sha256, sync_directory
from .pod import load_hosts, SSH_OPTIONS

NAMES = ('manifest.json','state.json','actors.json','arrays.npz')


def admit(path, arrays, *, floor_bytes):
    mkdir(Path(path).parent)
    require_disk(Path(path).parent)
    # Level-1 ZIP is normally smaller; bound the uncompressed payload plus ZIP
    # headers/metadata and retain the reserve even for incompressible arrays.
    pending = int(sum(a.nbytes for a in arrays.values())*1.01)+(16<<20)
    v=os.statvfs(Path(path).parent)
    if v.f_bavail*v.f_frsize < pending+floor_bytes:
        raise OSError('Insufficient durable checkpoint space')


def backup(path, records, *, source, config, python):
    path,source=Path(path),Path(source)
    group=path.with_suffix('.group.json')
    group_data=json.loads(group.read_text())
    identity=hashlib.sha256(str(path.parent.parent).encode()).hexdigest()[:24]
    target=Path(config['root'])/source.name/identity/path.name
    hosts=load_hosts(source/'ops/hosts.json') if len(records)>1 or config['peer'] is not None else []
    with tempfile.TemporaryDirectory(prefix='.checkpoint-peer-',dir=path.parent) as name:
        staging=Path(name)
        for record in records:
            h=record['host'];local=staging/f'host-{h}';local.mkdir()
            remote=Path(record['path'])
            if h==0:
                if remote!=path:raise ValueError('Owner checkpoint path differs')
                for member in NAMES:os.link(path/member,local/member)
                shutil.copyfile(group,local/'group.json')
            else:
                subprocess.run(['rsync','-a','--rsync-path=taskset -c 0,1 rsync',
                    '-e',shlex.join(['ssh',*SSH_OPTIONS]),hosts[h].ssh+':'+str(remote)+'/',str(local)+'/'],
                    check=True,timeout=120)
                subprocess.run(['rsync','-a','-e',shlex.join(['ssh',*SSH_OPTIONS]),
                    hosts[h].ssh+':'+str(remote.with_suffix('.group.json')),str(local/'group.json')],
                    check=True,timeout=60)
            if sha256(local/'manifest.json')!=record['manifest_sha256'] or sha256(local/'group.json')!=sha256(group):
                raise ValueError('Checkpoint rank manifest or group differs')
            m=json.loads((local/'manifest.json').read_text())
            if set(m['files'])!=set(NAMES)-{'manifest.json'}:raise ValueError('Incomplete checkpoint')
            for member,r in m['files'].items():
                if (local/member).stat().st_size!=r['bytes'] or sha256(local/member)!=r['sha256']:
                    raise ValueError('Checkpoint rank bytes differ')
            state=json.loads((local/'state.json').read_text())
            if state['host_rank']!=h or state['turn']!=group_data['turn']:
                raise ValueError('Checkpoint rank progress differs')
        atomic_json(staging/'restore.json',dict(group=group_data,source_paths=records,
            restore='Restore all original host directories and group files. Host/JAX rank changes require an audited logical-rank migration; never bypass trainer topology checks.'),replace=False)
        files=inventory(staging,[str(p.relative_to(staging)) for p in staging.rglob('*') if p.is_file()])
        result=publish(staging,target,files=files,
            peer=hosts[config['peer']].ssh if config['peer'] is not None else None,
            python=python,floor_bytes=config['peer_floor_bytes'])
    return dict(result,group_sha256=sha256(group),turn=group_data['turn'],all_rank_states=True,
                durability='Disk primary plus verified fsynced disk bundle; RAM is not a recovery dependency.')


def retire(path, *, keep):
    """Retire only previous local full states with sealed peer-copy receipts."""
    path=Path(path)
    candidates=sorted(p for p in path.parent.glob('turn-*') if p.is_dir() and not p.is_symlink())
    for old in candidates[:-keep]:
        receipt=old.with_suffix('.disk.json')
        if not receipt.is_file() or json.loads(receipt.read_text()).get('status')!='passed':
            raise ValueError('Cannot retire an unmirrored checkpoint')
        # Preserve small manifests, source/RNG metadata, and explicit retirement
        # records; only the reproducible parameter/moment payload is removed.
        payload=old/'arrays.npz'
        if not payload.exists():continue
        atomic_json(old.with_suffix('.retired.json'),dict(array_sha256=sha256(payload),
            bytes=payload.stat().st_size,mirror_receipt_sha256=sha256(receipt),
            replaced_by=path.name),replace=False)
        payload.unlink();sync_directory(old)
