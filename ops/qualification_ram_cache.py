"""Reclaim or restore qualification arrays with two verified peer RAM copies.

Only completed runs of at most four updates are eligible. All local metadata
stays in place. Learning-run arrays and other projects are outside this scope.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import stat
import subprocess
import time

ROOT=Path('/workspace/go')
RAM=Path('/dev/shm/gozero-staged-checkpoints')
PEER_RAM=Path('/dev/shm/gozero-staged-replicas')
SSH=['ssh','-F','/dev/null','-o','BatchMode=yes','-o','ConnectTimeout=10']
PYTHON='/home/go-user/.local/share/uv/python/cpython-3.12.13-linux-x86_64-gnu/bin/python3.12'


def sha(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def read(path):return json.loads(path.read_text())


def regular(path):
    if any(p.is_symlink() for p in (path,*path.parents)):
        raise ValueError('Symlink is outside this operation')
    s=path.stat()
    if not stat.S_ISREG(s.st_mode) or s.st_nlink!=1 or s.st_mode&0o222:
        raise ValueError('Expected a single-link read-only regular file')
    return s


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--attempt',required=True)
    p.add_argument('--peer-receipts',type=Path,nargs=2,required=True)
    p.add_argument('--operation',choices=('inspect','evict','restore'),default='inspect')
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if not re.fullmatch(r'pod-[0-9]{8}T[0-9]{6}Z-[a-f0-9]{8}',a.attempt):raise ValueError('Invalid attempt')
    attempt=ROOT/'runs'/a.attempt;closed=read(attempt/'result.json')
    report=read(attempt/'rank-0/artifacts/result.json')
    config=read(ROOT/'.gozero/snapshots'/closed['snapshot_id']/'resolved_config.json')
    if closed['status']!='passed' or report['status']!='passed' or not 1<=config['steps']<=4 or closed.get('resume_attempt'):
        raise ValueError('Only closed uninterrupted short qualifications may be evicted')
    for launch in (ROOT/'runs').glob('pod-*/launch.json'):
        if not (launch.parent/'result.json').exists() and read(launch).get('resume_attempt')==a.attempt:
            raise ValueError('A live continuation refers to this qualification')
    checkpoint=report['latest_checkpoint'];source=Path(checkpoint['path'])
    if source!=RAM/a.attempt/'rank-0/artifacts/checkpoints'/f"turn-{config['steps']:09d}":
        raise ValueError('Unexpected RAM checkpoint path')
    expected=checkpoint['temporary']['files'];group=source.with_suffix('.group.json')
    if set(expected)!={'actors.json','arrays.npz','manifest.json','state.json'}:
        raise ValueError('Unexpected payload set')
    if sha(source/'manifest.json')!=checkpoint['manifest_sha256'] or sha(group)!=checkpoint['group_sha256']:
        raise ValueError('Local checkpoint metadata changed')
    peers={}
    for path in a.peer_receipts:
        receipt=read(path)
        if receipt['status']!='passed' or receipt['attempt']!=a.attempt or receipt['snapshot']!=closed['snapshot_id']:
            raise ValueError('Wrong replica receipt')
        for copy in receipt['copies']:
            host=copy['host']
            if host==0:continue
            target=PEER_RAM/a.attempt/source.name
            if (host not in (1,2,3) or copy['path']!=str(target) or copy['files']!=expected
                    or copy['manifest_sha256']!=checkpoint['manifest_sha256'] or copy['group_sha256']!=checkpoint['group_sha256']):
                raise ValueError('Replica identity differs')
            peers[host]=target
    if len(peers)!=2:raise ValueError('Two different peer hosts are required')
    verified=[]
    for host,target in sorted(peers.items()):
        remote_code='''from pathlib import Path
import hashlib,json,stat,subprocess,time
p=Path(TARGET);expected=EXPECTED
paths=[p/name for name in expected]+[p.with_suffix('.group.json')]
def sha(path):
 with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
for path in paths:
 assert not any(x.is_symlink() for x in (path,*path.parents))
 s=path.stat();assert stat.S_ISREG(s.st_mode) and s.st_nlink==1 and not s.st_mode&0o222
for name,item in expected.items():assert (p/name).stat().st_size==item['bytes'] and sha(p/name)==item['sha256']
assert sha(paths[-1])==GROUP
if SEAL:subprocess.run(['sudo','-n','chown','0:0','--',*map(str,paths)],check=True)
assert all(path.stat().st_uid==0 for path in paths) if SEAL else True
print(json.dumps(dict(host=HOST,path=str(p),verified_unix=time.time(),readonly=True,root_owned=all(path.stat().st_uid==0 for path in paths))))
'''.replace('TARGET',repr(str(target))).replace('EXPECTED',repr(expected)).replace('GROUP',repr(checkpoint['group_sha256'])).replace('SEAL',repr(a.operation!='inspect')).replace('HOST',repr(host))
        command='taskset -c 0,1 '+shlex.quote(PYTHON)+' -c '+shlex.quote(remote_code)
        verified.append(json.loads(subprocess.check_output(SSH+[f'go-user@worker-{host}.example.invalid',command],text=True,timeout=120)))
    result=dict(kind='qualification_ram_array_cache',operation=a.operation,attempt=a.attempt,
        checkpoint_manifest_sha256=checkpoint['manifest_sha256'],checkpoint_group_sha256=checkpoint['group_sha256'],
        local_array_path=str(source/'arrays.npz'),array=expected['arrays.npz'],peers=verified,
        replica_receipt_sha256={str(p):sha(p) for p in a.peer_receipts},operator_sha256=sha(Path(__file__)),
        durability='Two verified peer RAM copies; ownership shown per peer; still volatile across host restarts.',
        restore='Run this operator with the same attempt and peer receipts, --operation restore, and a new output path.')
    proof=a.output.with_suffix('.verified.json')
    with proof.open('x') as f:json.dump(result,f,indent=2);f.flush();os.fsync(f.fileno())
    arrays=source/'arrays.npz'
    if a.operation=='evict':
        regular(arrays)
        if arrays.stat().st_size!=expected['arrays.npz']['bytes'] or sha(arrays)!=expected['arrays.npz']['sha256']:
            raise ValueError('Local arrays changed')
        arrays.unlink()
        result['reclaimed_local_bytes']=expected['arrays.npz']['bytes']
    elif a.operation=='restore':
        if arrays.exists():raise FileExistsError('Local arrays already present')
        fs=os.statvfs(RAM)
        if fs.f_bavail*fs.f_frsize<expected['arrays.npz']['bytes']+64*(1<<30):
            raise ValueError('Restore would violate the RAM floor')
        temp=source/'arrays.restoring.npz'
        if temp.exists():raise FileExistsError(temp)
        host=min(peers);remote=f'go-user@worker-{host}.example.invalid:'+str(peers[host]/'arrays.npz')
        subprocess.run(['rsync','-t','--chmod=F444','--rsync-path=taskset -c 0,1 rsync','-e',shlex.join(SSH),remote,str(temp)],check=True,timeout=180)
        regular(temp)
        if temp.stat().st_size!=expected['arrays.npz']['bytes'] or sha(temp)!=expected['arrays.npz']['sha256']:
            raise ValueError('Restored array hash differs')
        os.link(temp,arrays);temp.unlink()
        subprocess.run(['sudo','-n','chown','0:0','--',str(arrays)],check=True)
        result['restored_bytes']=expected['arrays.npz']['bytes']
    result.update(status='passed',local_arrays_present=arrays.exists(),completed_unix=time.time())
    with a.output.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    a.output.chmod(0o444);proof.chmod(0o444)
    print(json.dumps({k:result[k] for k in ('status','operation','attempt','local_arrays_present')},sort_keys=True))


if __name__=='__main__':main()
