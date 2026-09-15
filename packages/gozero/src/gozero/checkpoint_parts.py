"""Lossless durable checkpoint storage across fragmented pod disks.

This is a byte archive, not a model format. Reassembly restores the exact
ordinary four checkpoint files, including optimizer state. Allocations are
explicit and consumed only for a newly selected checkpoint.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import tempfile

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gozero import checkpoints
from gozero.checkpoint_archive import ROOT, FILES, SSH, publish, safe_path, verify_files
from gozero.snapshots import read_json


def part_path(spec, root=ROOT):
    digest=spec['sha256']
    if not re.fullmatch('[0-9a-f]{64}',digest) or type(spec['bytes']) is not int or spec['bytes']<=0:
        raise ValueError('Invalid part identity')
    base=Path(root)/'.gozero/checkpoint-parts'
    return safe_path(base/digest,base)


def verify_part(spec, root=ROOT):
    path=part_path(spec,root)
    if (not path.is_file() or path.is_symlink() or path.stat().st_mode & 0o222
            or path.stat().st_size!=spec['bytes'] or checkpoints.sha256(path)!=spec['sha256']):
        raise ValueError('Checkpoint part differs')
    return {'status':'verified','path':str(path),'bytes':spec['bytes'],'sha256':spec['sha256']}


def receive(spec, stream, root=ROOT):
    destination=part_path(spec,root)
    if destination.exists():
        # Avoid silently ignoring caller input: repeats are explicit verifies.
        raise FileExistsError(destination)
    base=Path(root)/'.gozero/checkpoint-part-reservations'
    reservation=safe_path(Path(root)/spec['reservation'],base)
    if (not reservation.is_file() or reservation.stat().st_nlink!=1
            or reservation.stat().st_size<spec['bytes']):raise ValueError('Missing or undersized part reservation')
    destination.parent.mkdir(parents=True,exist_ok=True)
    staging=destination.with_name('.'+destination.name+'.incoming')
    if staging.exists():raise FileExistsError(staging)
    os.rename(reservation,staging)
    remaining=spec['bytes']; digest=hashlib.sha256()
    with staging.open('r+b') as f:
        while remaining:
            block=stream.read(min(8*2**20,remaining))
            if not block:raise ValueError('Truncated part stream')
            f.write(block);digest.update(block);remaining-=len(block)
        if stream.read(1) or digest.hexdigest()!=spec['sha256']:raise ValueError('Part stream checksum differs')
        f.truncate(spec['bytes']);f.flush();os.fsync(f.fileno());os.fchmod(f.fileno(),0o444)
    staging.rename(destination);checkpoints._sync_directory(destination.parent)
    return verify_part(spec,root)


def command(host, action, spec, python):
    if type(host) is not int or host not in (0,1,2,3):raise ValueError('Unauthorized checkpoint host')
    argv=[str(python),'-B',str(Path(__file__).resolve()),action,'--spec',json.dumps(spec,separators=(',',':'))]
    return argv if host==0 else [*SSH,f'go-user@worker-{host}.example.invalid',shlex.join(argv)]


def remote_verify(host,spec,python):
    done=subprocess.run(command(host,'verify',spec,python),capture_output=True,timeout=180,check=True)
    return json.loads(done.stdout)


def send_part(source,offset,spec,host,python):
    process=subprocess.Popen(command(host,'receive',spec,python),stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    try:
        with Path(source).open('rb') as f:
            f.seek(offset);remaining=spec['bytes']
            while remaining:
                block=f.read(min(8*2**20,remaining))
                if not block:raise ValueError('Truncated source checkpoint')
                process.stdin.write(block);remaining-=len(block)
        process.stdin.close();process.stdin=None
        out,err=process.communicate(timeout=300)
        if process.returncode:raise RuntimeError('Checkpoint part transfer failed: '+err.decode()[-3000:])
        return json.loads(out)
    except BaseException:
        process.kill();process.communicate();raise


def digest_range(path,offset,size):
    result=hashlib.sha256();remaining=size
    with Path(path).open('rb') as f:
        f.seek(offset)
        while remaining:
            block=f.read(min(8*2**20,remaining))
            if not block:raise ValueError('Truncated checkpoint')
            result.update(block);remaining-=len(block)
    return result.hexdigest()


def promote(receipt, allocations, *, python, root=ROOT):
    """Allocate no new disk: consume the explicitly supplied reserved slots."""
    if receipt['kind']!='temporary_ram_checkpoint' or receipt['status']!='temporary':raise ValueError('Expected an explicit temporary checkpoint')
    cache=Path(receipt['cache_path']);verify_files(cache,receipt)
    if sum(x['bytes'] for x in allocations)<receipt['files']['arrays.npz']['bytes']:
        raise ValueError('Reservations cannot hold the selected checkpoint')
    destination=Path(root)/'.gozero/checkpoint-compositions'/receipt['manifest_sha256']
    if destination.exists():raise FileExistsError(destination)
    parts=[];offset=0;total=receipt['files']['arrays.npz']['bytes']
    for allocation in allocations:
        if offset==total:break
        size=min(allocation['bytes'],total-offset)
        spec={'bytes':size,'sha256':digest_range(cache/'arrays.npz',offset,size),'reservation':allocation['reservation']}
        verified=send_part(cache/'arrays.npz',offset,spec,allocation['host'],python)
        parts.append({'host':allocation['host'],'offset':offset,**verified});offset+=size
    if offset!=total:raise ValueError('Incomplete checkpoint composition')
    result={'kind':'partitioned_checkpoint_archive','status':'committed',
        'manifest_sha256':receipt['manifest_sha256'],'files':receipt['files'],'parts':parts,
        'temporary_receipt_sha256':receipt.get('receipt_sha256'),
        'source_snapshot':receipt['source_snapshot'],
        'restore_contract':'Concatenate verified array parts by offset, combine the three stored metadata files, and verify/read the standard four-file checkpoint.'}
    destination.parent.mkdir(parents=True,exist_ok=True)
    staging=Path(tempfile.mkdtemp(prefix='.composition-',dir=destination.parent))
    for name in sorted(FILES-{'arrays.npz'}):
        with (staging/name).open('xb') as f:
            f.write((cache/name).read_bytes());f.flush();os.fsync(f.fileno());os.fchmod(f.fileno(),0o444)
    publish(staging/'parts.json',result)
    checkpoints._sync_directory(staging);staging.rename(destination);checkpoints._sync_directory(destination.parent)
    return {'descriptor':str(destination/'parts.json'),'descriptor_sha256':checkpoints.sha256(destination/'parts.json'),**result}


def validate_composition(record):
    if record['kind']!='partitioned_checkpoint_archive' or record['status']!='committed':raise ValueError('Uncommitted checkpoint composition')
    if set(record['files'])!=FILES or record['manifest_sha256']!=record['files']['manifest.json']['sha256']:
        raise ValueError('Invalid checkpoint composition files')
    offset=0
    for part in record['parts']:
        if (part['offset']!=offset or type(part['host']) is not int or part['host'] not in (0,1,2,3)
                or part['path']!=str(part_path(part))):raise ValueError('Invalid checkpoint part placement')
        offset+=part['bytes']
    if offset!=record['files']['arrays.npz']['bytes']:raise ValueError('Checkpoint part coverage differs')


def restore(descriptor,destination,*,python):
    descriptor=Path(descriptor);record=read_json(descriptor);validate_composition(record)
    destination=Path(destination)
    if destination.exists() or destination.is_symlink():raise FileExistsError(destination)
    destination.parent.mkdir(parents=True,exist_ok=True)
    container=Path(tempfile.mkdtemp(prefix='.parts-restore-',dir=destination.parent));staging=container/'checkpoint';staging.mkdir()
    with (staging/'arrays.npz').open('xb') as f:
        for part in record['parts']:
            done=subprocess.Popen(command(part['host'],'read',part,python),stdout=subprocess.PIPE,stderr=subprocess.PIPE)
            digest=hashlib.sha256();remaining=part['bytes']
            try:
                while remaining:
                    block=done.stdout.read(min(8*2**20,remaining))
                    if not block:raise ValueError('Truncated archived checkpoint part')
                    f.write(block);digest.update(block);remaining-=len(block)
                extra,err=done.communicate(timeout=180)
                if extra or done.returncode or digest.hexdigest()!=part['sha256']:
                    raise ValueError('Archived checkpoint part differs: '+err.decode()[-1000:])
            except BaseException:
                done.kill();done.communicate();raise
        f.flush();os.fsync(f.fileno());os.fchmod(f.fileno(),0o444)
    for name in sorted(FILES-{'arrays.npz'}):
        with (staging/name).open('xb') as f:
            f.write((descriptor.parent/name).read_bytes());f.flush();os.fsync(f.fileno());os.fchmod(f.fileno(),0o444)
    verify_files(staging,record)
    checkpoints.read(staging,expected_manifest_sha256=record['manifest_sha256'])
    staging.rename(destination);container.rmdir();checkpoints._sync_directory(destination.parent)
    return record['manifest_sha256']


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['receive','verify','read','promote','restore'])
    p.add_argument('--spec');p.add_argument('--receipt',type=Path);p.add_argument('--allocations',type=Path)
    p.add_argument('--descriptor',type=Path);p.add_argument('--destination',type=Path);p.add_argument('--output',type=Path)
    a=p.parse_args()
    if a.action=='promote':
        result=promote(read_json(a.receipt),read_json(a.allocations)['allocations'],python=sys.executable)
        if a.output:publish(a.output,result)
    elif a.action=='restore':result={'restored_manifest_sha256':restore(a.descriptor,a.destination,python=sys.executable)}
    else:
        spec=json.loads(a.spec)
        if a.action=='read':
            verify_part(spec)
            with part_path(spec).open('rb') as f:
                while block:=f.read(8*2**20):sys.stdout.buffer.write(block)
            return
        result=receive(spec,sys.stdin.buffer) if a.action=='receive' else verify_part(spec)
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
