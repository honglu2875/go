"""Move only a completed release mirror after exact peer read-back verification.

The original game NPZ files, packed training view and checkpoints are outside
this operator's allowed roots. SSH destination identity is verified by hash.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import tempfile
import time

ROOT=Path(__file__).resolve().parents[3]
RAM_ROOT=Path('/dev/shm/go9x9-release-v1')
SOURCE=RAM_ROOT/'publish'
STUDY=Path(__file__).resolve().parent
MANIFEST_SHA='68b05a2dca33859a9d0b3b9ee89d4ed242c45d427d0e3233c8093c91f3f16096'
SSH=['ssh','-F','/dev/null','-o','BatchMode=yes','-o','ConnectTimeout=10']

REMOTE=r'''
import hashlib,json,os,socket,sys
from pathlib import Path
c=json.loads(sys.argv[1]);mode=c['mode'];root=Path('/dev/shm/go9x9-release-v1')
final=root/'publish';stage=root/c['stage'];fs=os.statvfs('/dev/shm')
mem={l.split(':')[0]:int(l.split()[1])*1024 for l in Path('/proc/meminfo').read_text().splitlines()}
identity=hashlib.sha256(socket.gethostname().encode()).hexdigest()
if c.get('node_sha256') and c['node_sha256']!=identity:raise ValueError('Destination node changed')
if mode in ('preflight','reserve'):
 if final.exists() or final.is_symlink() or stage.exists():raise ValueError('Destination is already occupied')
 if fs.f_bavail*fs.f_frsize<64*(1<<30)+c['bytes']+(1<<28) or mem['MemAvailable']<96*(1<<30):raise ValueError('Destination RAM floor')
 if mode=='reserve':root.mkdir(exist_ok=True);stage.mkdir(exist_ok=False)
 result=dict(node_sha256=identity,shm_free=fs.f_bavail*fs.f_frsize,memory_available=mem['MemAvailable'])
else:
 path=stage if mode=='seal' else final
 if path.is_symlink() or not path.is_dir():raise ValueError('Invalid mirror directory')
 found={str(p.relative_to(path)) for p in path.rglob('*') if p.is_file()}
 if found!=set(c['files']):raise ValueError('Mirror file coverage differs')
 for name,expected in c['files'].items():
  p=path/name
  if p.is_symlink() or p.stat().st_size!=expected['bytes']:raise ValueError('Mirror file differs')
  h=hashlib.sha256()
  with p.open('rb') as f:
   for block in iter(lambda:f.read(1<<20),b''):h.update(block)
  if h.hexdigest()!=expected['sha256']:raise ValueError('Mirror checksum differs: '+name)
  if mode=='seal':p.chmod(0o444)
 if mode=='seal':
  if final.exists():raise ValueError('Final destination appeared')
  stage.rename(final)
 result=dict(node_sha256=identity,files=len(found),bytes=sum(x['bytes'] for x in c['files'].values()),path=str(final),status='passed')
print(json.dumps(result))
'''


def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def publish(path,value):
    with path.open('x') as f:json.dump(value,f,indent=2);f.write('\n')
    path.chmod(0o444)


def remote(target,mode,plan):
    payload=dict(mode=mode,stage=plan['stage'],bytes=plan['bytes'],files=plan.get('files',{}),
                 node_sha256=plan.get('destination_node_sha256'))
    command=shlex.join(['taskset','-c','0,1','python3','-c',REMOTE,json.dumps(payload,separators=(',',':'))])
    return json.loads(subprocess.check_output(SSH+[target,command],text=True,timeout=120))


def path_identity(path):
    if path.is_symlink() or not path.is_file():raise ValueError('Expected a regular archive member')
    for parent in path.parents:
        if parent==RAM_ROOT:break
        if parent.is_symlink():raise ValueError('Archive member has a symlink parent')
    s=path.stat()
    return dict(device=s.st_dev,inode=s.st_ino,bytes=s.st_size,mtime_ns=s.st_mtime_ns,blocks=s.st_blocks*512,links=s.st_nlink)


def prepare(a):
    if SOURCE.resolve()!=SOURCE or sha(SOURCE/'manifest.json')!=MANIFEST_SHA:raise ValueError('Release mirror identity differs')
    allow=json.loads((STUDY/'upload-allowlist.json').read_text())
    expected={}
    for line in (SOURCE/'SHA256SUMS').read_text().splitlines():
        digest,name=line.split('  ',1)
        if name in expected:raise ValueError('Duplicate checksum member')
        expected[name]=digest
    if set(allow['files'])!=set(expected)|{'SHA256SUMS'}:raise ValueError('Release file coverage differs')
    expected['SHA256SUMS']=sha(SOURCE/'SHA256SUMS')
    files={};removals={};inodes={}
    for name,digest in sorted(expected.items()):
        relative=Path(name)
        if relative.is_absolute() or '..' in relative.parts:raise ValueError('Invalid archive path')
        path=SOURCE/relative;identity=path_identity(path)
        if sha(path)!=digest:raise ValueError('Release checksum differs: '+name)
        files[name]=dict(bytes=identity['bytes'],sha256=digest)
        removals[str(path)]=dict(**identity,sha256=digest)
        inodes[identity['device'],identity['inode']]=digest
    # Include only additional hard links to these exact already covered archive
    # members, within the completed host-0 packaging directory.
    for path in (RAM_ROOT/'host-0').rglob('*'):
        if not path.is_file() or path.is_symlink():continue
        identity=path_identity(path);key=identity['device'],identity['inode']
        if key in inodes:removals[str(path)]=dict(**identity,sha256=inodes[key])
    unique={}
    for record in removals.values():
        key=record['device'],record['inode'];unique.setdefault(key,dict(paths=0,**record));unique[key]['paths']+=1
    reclaim=sum(r['blocks'] for r in unique.values() if r['paths']==r['links'])
    plan=dict(kind='completed_release_mirror_relocation',created=time.time(),source=str(SOURCE),destination_host=3,
        destination=str(SOURCE),stage='.incoming-release-6a26ffa743f3',files=files,bytes=sum(x['bytes'] for x in files.values()),
        removals=removals,expected_reclaimed_blocks=reclaim,operator_sha256=sha(Path(__file__)),manifest_sha256=MANIFEST_SHA,
        scope='Exact RAM mirror relocation only. Original game NPZs, packed training arrays and every learning checkpoint are preserved.')
    preflight=remote(a.target,'preflight',plan);plan['destination_node_sha256']=preflight['node_sha256'];plan['destination_preflight']=preflight
    publish(a.plan,plan)
    print(json.dumps(dict(status='prepared',plan_sha256=sha(a.plan),files=len(files),bytes=plan['bytes'],removal_paths=len(removals),expected_reclaimed_blocks=reclaim)),flush=True)


def apply(a):
    if sha(a.plan)!=a.plan_sha256:raise ValueError('Relocation plan changed')
    plan=json.loads(a.plan.read_text())
    if plan['operator_sha256']!=sha(Path(__file__)) or plan['source']!=str(SOURCE):raise ValueError('Operator/source changed')
    if a.receipt.exists():raise FileExistsError(a.receipt)
    for name,expected in plan['removals'].items():
        p=Path(name)
        if not (p.is_relative_to(SOURCE) or p.is_relative_to(RAM_ROOT/'host-0')):raise ValueError('Removal escapes completed archive roots')
        if path_identity(p)!={k:v for k,v in expected.items() if k!='sha256'} or sha(p)!=expected['sha256']:
            raise ValueError('Original archive member changed')
    remote(a.target,'reserve',plan)
    with tempfile.TemporaryDirectory(prefix='gozero-release-relocation-') as tmp:
        listing=Path(tmp)/'files.txt';listing.write_text(''.join(name+'\n' for name in plan['files']))
        argv=['rsync','-rlt','--files-from='+str(listing),'--rsync-path=taskset -c 0,1 rsync','-e',shlex.join(SSH),
              str(SOURCE)+'/',a.target+':'+str(RAM_ROOT/plan['stage'])+'/']
        subprocess.run(argv,check=True,timeout=600)
    copied=remote(a.target,'seal',plan)
    publish(a.receipt.with_name(a.receipt.stem+'-copied.json'),dict(plan_sha256=a.plan_sha256,created=time.time(),destination=copied))
    verified=remote(a.target,'verify',plan)
    # Check all source paths before unlinking any, then retain a per-path journal.
    for name,expected in plan['removals'].items():
        p=Path(name)
        if path_identity(p)!={k:v for k,v in expected.items() if k!='sha256'} or sha(p)!=expected['sha256']:
            raise ValueError('Source changed before retirement')
    before=os.statvfs('/dev/shm');journal=a.receipt.with_name(a.receipt.stem+'-retired.jsonl')
    with journal.open('x') as log:
        for name,expected in plan['removals'].items():
            p=Path(name);s=p.stat()
            if (s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns)!=(expected['device'],expected['inode'],expected['bytes'],expected['mtime_ns']):
                raise ValueError('Archive identity changed during retirement')
            p.unlink();log.write(json.dumps(dict(path=name,sha256=expected['sha256']))+'\n');log.flush()
    after=os.statvfs('/dev/shm')
    result=dict(kind=plan['kind'],status='passed',created=time.time(),plan_sha256=a.plan_sha256,destination=verified,
        retired_paths=len(plan['removals']),expected_reclaimed_blocks=plan['expected_reclaimed_blocks'],
        observed_shm_free_change=(after.f_bavail-before.f_bavail)*after.f_frsize,
        original_distributed_games_preserved=True,packed_training_arrays_preserved=True,learning_checkpoints_preserved=True)
    publish(a.receipt,result)
    publish(STUDY/'local-mirror-001.json',dict(host=plan['destination_host'],path=plan['destination'],manifest_sha256=MANIFEST_SHA,
        relocation_receipt=str(a.receipt.relative_to(ROOT)),relocation_receipt_sha256=sha(a.receipt),files=len(plan['files']),bytes=plan['bytes']))
    print(json.dumps(dict(status='passed',receipt_sha256=sha(a.receipt),expected_reclaimed_blocks=plan['expected_reclaimed_blocks'])),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','apply']);p.add_argument('--target',required=True)
    p.add_argument('--plan',type=Path,required=True);p.add_argument('--plan-sha256');p.add_argument('--receipt',type=Path);a=p.parse_args()
    if a.action=='prepare':prepare(a)
    else:
        if a.plan_sha256 is None or a.receipt is None:raise ValueError('Exact plan and new receipt required')
        apply(a)


if __name__=='__main__':main()
