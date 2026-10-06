"""Checksummed, fsynced publication of an immutable file bundle on a disk peer.

Transfers can be retried after interruption. A receipt is published only after
all files have been verified and fsynced; incomplete directories are not backups.
SSH destinations come from the private host configuration, never model metadata.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import tempfile

from .durable_files import sha256
from .pod import SSH_OPTIONS
from .snapshots import canonical_json


# Keep the remote verifier self-contained so it does not trust a mutable import.
VERIFY = r'''import hashlib,json,os,sys
from pathlib import Path
p=json.load(sys.stdin);root=Path(p['target']);expected=p['files']
def sync(d):
 fd=os.open(d,os.O_RDONLY|os.O_DIRECTORY)
 try:os.fsync(fd)
 finally:os.close(fd)
root.mkdir(parents=True,exist_ok=True)
resolved=root.resolve(strict=True)
if root.is_symlink():raise ValueError('Mirror root is a symlink')
mounts=[]
for line in Path('/proc/self/mountinfo').read_text().splitlines():
 left,right=line.split(' - ',1);m=Path(left.split()[4])
 if resolved==m or m in resolved.parents:mounts.append((len(m.parts),right.split()[0]))
if not mounts or max(mounts)[1] in ('tmpfs','ramfs','devtmpfs'):raise ValueError('Mirror is not on disk')
v=os.statvfs(root)
if v.f_bavail*v.f_frsize<p['floor_bytes']:raise ValueError('Mirror disk reserve exhausted')
if p['mode']=='prepare':
 if v.f_bavail*v.f_frsize<p['floor_bytes']+sum(r['bytes'] for r in expected.values()):
  raise ValueError('Insufficient space for mirror publication')
 print(json.dumps({'status':'prepared'}));sys.exit(0)
for name,r in expected.items():
 f=root/name
 if f.is_symlink() or not f.is_file() or f.stat().st_size!=r['bytes']:raise ValueError('Mirror file differs: '+name)
 with f.open('rb') as stream:
  h=hashlib.sha256()
  while b:=stream.read(8<<20):h.update(b)
  if h.hexdigest()!=r['sha256']:raise ValueError('Mirror checksum differs: '+name)
  os.fsync(stream.fileno())
for directory in sorted({root,*[q for name in expected for q in (root/name).parents if q==root or root in q.parents]},key=lambda q:len(q.parts),reverse=True):sync(directory)
receipt=root/'mirror.json';raw=(json.dumps({'schema_version':1,'status':'passed','files':expected,'bundle_sha256':p['identity']},sort_keys=True,separators=(',',':'))+'\n').encode()
if receipt.exists():
 if receipt.read_bytes()!=raw:raise ValueError('Existing mirror receipt differs')
else:
 temporary=root/'.mirror.partial'
 with temporary.open('wb') as stream:stream.write(raw);stream.flush();os.fsync(stream.fileno())
 os.replace(temporary,receipt)
sync(root)
for directory in root.parents:
 sync(directory)
 if directory==Path('/'):break
print(json.dumps({'status':'passed','bundle_sha256':p['identity'],'receipt_sha256':hashlib.sha256(raw).hexdigest(),'bytes':sum(r['bytes'] for r in expected.values())}))
'''


def inventory(root: Path, paths):
    root = Path(root)
    result = {}
    for name in paths:
        relative = Path(name)
        if relative.is_absolute() or '..' in relative.parts or name == 'mirror.json':
            raise ValueError('Invalid mirror member')
        p = root/relative
        if p.is_symlink() or not p.is_file():
            raise ValueError('Mirror source must be a regular file')
        result[str(relative)] = dict(bytes=p.stat().st_size, sha256=sha256(p))
    if not result:
        raise ValueError('Empty mirror')
    return result


def publish(source: Path, target: Path, *, files: dict, peer: str | None,
            python: str, floor_bytes: int = 8 << 30):
    source, target = Path(source), Path(target)
    if not source.is_absolute() or not target.is_absolute():
        raise ValueError('Mirror paths must be absolute')
    if inventory(source, files) != files:
        raise ValueError('Mirror source no longer matches its planned bytes')
    identity = hashlib.sha256(canonical_json(files)).hexdigest()
    command = ['taskset','-c','0,1',python,'-B','-c',VERIFY]
    argv = ['ssh',*SSH_OPTIONS,peer,shlex.join(command)] if peer else command
    payload = dict(target=str(target),files=files,identity=identity,floor_bytes=floor_bytes)
    def verify(mode):
        result = subprocess.run(argv,input=json.dumps(dict(payload,mode=mode)),text=True,
                                capture_output=True,timeout=300,check=True)
        return json.loads(result.stdout)
    verify('prepare')
    with tempfile.NamedTemporaryFile(mode='w',suffix='.mirror-files') as listing:
        listing.write(''.join(name+'\n' for name in sorted(files)));listing.flush()
        args = ['rsync','-a','--ignore-existing','--files-from='+listing.name]
        if peer:
            args += ['--rsync-path=taskset -c 0,1 rsync','-e',shlex.join(['ssh',*SSH_OPTIONS])]
        destination = (peer+':' if peer else '')+str(target)+'/'
        subprocess.run(args+[str(source)+'/',destination],check=True,timeout=600)
    result = verify('seal')
    if result['status'] != 'passed' or result['bundle_sha256'] != identity:
        raise ValueError('Peer did not verify the requested bundle')
    return dict(result,peer=peer,target=str(target))
