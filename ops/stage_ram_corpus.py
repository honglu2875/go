"""Copy and verify one immutable packed corpus in RAM on all four hosts."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'packages/gozero/src'))
from gozero.corpus_format import KIND

SSH=['ssh','-F','/dev/null','-o','BatchMode=yes','-o','ConnectTimeout=10']
PYTHON='/home/go-user/.local/share/uv/python/cpython-3.12.13-linux-x86_64-gnu/bin/python3.12'


def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


REMOTE=r'''
import hashlib,json,os,stat,subprocess,sys,time
from pathlib import Path
request=json.load(sys.stdin);base=Path(request['dataset']);files=request['files']
assert base.is_absolute() and base.is_relative_to('/dev/shm/gozero-datasets') and '..' not in base.parts
for parent in (base,*base.parents):
 assert not parent.is_symlink(), 'Symlink in RAM corpus path'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  while True:
   b=f.read(8*1024**2)
   if not b:break
   h.update(b)
 return h.hexdigest()
pending=0
paths=[]
for name,item in files.items():
 relative=Path(name)
 assert not relative.is_absolute() and '..' not in relative.parts
 p=base/relative
 for parent in (p,*p.parents):assert not parent.is_symlink()
 if p.exists():
  s=p.stat()
  assert stat.S_ISREG(s.st_mode) and s.st_size==item['bytes'] and s.st_nlink==1
  assert s.st_uid in (0,os.getuid()) and not s.st_mode&0o222
  assert sha(p)==item['sha256'], 'Conflicting existing corpus file'
 else:pending+=item['bytes']
 paths.append(p)
v=os.statvfs('/dev/shm')
memory=int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')))*1024
assert v.f_bavail*v.f_frsize>64*(1<<30)+pending+(1<<30)
assert memory>96*(1<<30)
if request['phase']=='prepare':
 base.mkdir(parents=True,exist_ok=True)
 print(json.dumps(dict(missing_bytes=pending,shm_free=v.f_bavail*v.f_frsize,memory_available=memory)))
else:
 assert pending==0
 # Protect only this new, verified corpus from the pod's logout cleanup.
 # Data remains volatile across reboot; source games are backed up on the Hub.
 subprocess.run(['sudo','-n','--','chown','--no-dereference','0:0','--',*map(str,paths)],check=True,timeout=30)
 for name,item in files.items():
  p=base/name
  assert p.stat().st_uid==0 and p.stat().st_gid==0 and sha(p)==item['sha256']
 print(json.dumps(dict(status='passed',files=len(files),bytes=sum(x['bytes'] for x in files.values()),
  manifest_sha256=files['manifest.json']['sha256'],verified_unix=time.time(),
  retention='Root-owned read-only files resist logout cleanup; RAM remains volatile.')))
'''


def main():
    p=argparse.ArgumentParser();p.add_argument('--dataset',type=Path,required=True)
    p.add_argument('--expected-sha256',required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();base=a.dataset.absolute()
    if not base.is_relative_to('/dev/shm/gozero-datasets') or sha(base/'manifest.json')!=a.expected_sha256:
        raise ValueError('Unexpected corpus root or manifest')
    manifest=json.loads((base/'manifest.json').read_text())
    if manifest['kind']!=KIND or not manifest['complete_train_validation'] or manifest['test_arrays_included']:
        raise ValueError('Only complete qualified train/validation corpora are staged')
    files={'manifest.json':dict(bytes=(base/'manifest.json').stat().st_size,sha256=a.expected_sha256)}
    for shard in manifest['shards']:
        for record in shard['files'].values():
            name=record['path']
            if name in files:raise ValueError('Duplicate corpus member')
            files[name]=dict(bytes=record['bytes'],sha256=record['sha256'])
    if sum(x['bytes'] for x in files.values())>16*(1<<30):
        raise ValueError('Corpus exceeds its registered 16-GiB staging budget')
    a.output.mkdir(parents=True,exist_ok=False);started=time.time()
    request=dict(dataset=str(base),files=files)
    def remote(host,phase,log):
        target=f'go-user@worker-{host}.example.invalid'
        command='taskset -c 0,1 '+shlex.join([PYTHON,'-B','-c',REMOTE])
        done=subprocess.run(SSH+[target,command],input=json.dumps({**request,'phase':phase}),text=True,
            stdout=subprocess.PIPE,stderr=log,check=True,timeout=300)
        return json.loads(done.stdout)
    def stage(host):
        target=f'go-user@worker-{host}.example.invalid'
        with (a.output/f'host-{host}.log').open('x') as log:
            before=remote(host,'prepare',log)
            if before['missing_bytes']:
                names=b'\0'.join(name.encode() for name in sorted(files))+b'\0'
                subprocess.run(['rsync','-a','--ignore-existing','--from0','--files-from=-',
                    '--rsync-path=taskset -c 0,1 rsync','-e',shlex.join(SSH),str(base)+'/',target+':'+str(base)+'/'],
                    input=names,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=600)
            return dict(host=host,preflight=before,**remote(host,'verify',log))
    receipt=dict(kind='ram_corpus_staging',dataset=str(base),manifest_sha256=a.expected_sha256,
        files=files,started_unix=started,operator_sha256=sha(Path(__file__)))
    try:
        with ThreadPoolExecutor(4) as pool:receipt['hosts']=list(pool.map(stage,range(4)))
        receipt['status']='passed'
    except BaseException as error:
        receipt.update(status='failed',error=repr(error));raise
    finally:
        receipt['finished_unix']=time.time()
        (a.output/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
        (a.output/'receipt.json').chmod(0o444)
    print(json.dumps(dict(status='passed',dataset=str(base),hosts=len(receipt['hosts']),bytes_per_host=sum(x['bytes'] for x in files.values()))),flush=True)


if __name__=='__main__':main()
