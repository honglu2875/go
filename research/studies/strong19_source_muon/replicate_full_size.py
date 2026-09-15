"""Verify and replicate a closed experiment's temporary checkpoint on one peer."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import time


def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--workspace-root',type=Path,required=True)
    p.add_argument('--attempt',required=True)
    p.add_argument('--peer',type=int,default=2,choices=(1,2,3))
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();root=a.workspace_root.resolve()
    if not re.fullmatch(r'pod-[0-9]{8}T[0-9]{6}Z-[a-f0-9]{8}',a.attempt):raise ValueError('Invalid attempt')
    attempt=root/'runs'/a.attempt
    closed=json.loads((attempt/'result.json').read_text())
    report=json.loads((attempt/'rank-0/artifacts/result.json').read_text())
    assert closed['status']==report['status']=='passed'
    ck=report['latest_checkpoint'];source=Path(ck['path'])
    assert source.is_relative_to('/dev/shm/gozero-staged-checkpoints') and source.name.startswith('turn-')
    manifest=json.loads((source/'manifest.json').read_text())
    assert sha(source/'manifest.json')==ck['manifest_sha256']
    expected={name:dict(bytes=(source/name).stat().st_size,sha256=sha(source/name)) for name in ('manifest.json',*manifest['files'])}
    for name,item in manifest['files'].items():assert expected[name]['sha256']==item['sha256']
    group=source.with_suffix('.group.json');assert sha(group)==ck['group_sha256']
    target=Path('/dev/shm/gozero-staged-replicas')/a.attempt/source.name
    ssh=['ssh','-F','/dev/null','-o','BatchMode=yes','-o','ConnectTimeout=10']
    peer=f'go-user@worker-{a.peer}'
    total=sum(v['bytes'] for v in expected.values())+group.stat().st_size
    prepare=f'''from pathlib import Path
import os
p=Path({str(target)!r});s=os.statvfs('/dev/shm')
assert s.f_bavail*s.f_frsize>{total}+64*(1<<30)
p.mkdir(parents=True,exist_ok=False)
'''
    subprocess.run(ssh+[peer,'taskset -c 0,1 python3 -c '+shlex.quote(prepare)],check=True)
    subprocess.run(['rsync','-a','--rsync-path=taskset -c 0,1 rsync','-e',shlex.join(ssh),str(source)+'/',peer+':'+str(target)+'/'],check=True)
    subprocess.run(['rsync','-a','--rsync-path=taskset -c 0,1 rsync','-e',shlex.join(ssh),str(group),peer+':'+str(target.with_suffix('.group.json'))],check=True)
    code=f'''from pathlib import Path
import hashlib,json,time
p=Path({str(target)!r}); expected={expected!r}
def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  while True:
   b=f.read(8*1024**2)
   if not b:break
   h.update(b)
 return h.hexdigest()
for name,item in expected.items():assert (p/name).stat().st_size==item['bytes'] and sha(p/name)==item['sha256']
assert sha(p.with_suffix('.group.json'))=={ck['group_sha256']!r}
print(json.dumps(dict(host={a.peer},path=str(p),files=expected,manifest_sha256={ck['manifest_sha256']!r},group_sha256={ck['group_sha256']!r},verified=time.time())))
'''
    remote=json.loads(subprocess.check_output(ssh+[peer,'taskset -c 0,1 python3 -c '+shlex.quote(code)],text=True))
    local=dict(host=0,path=str(source),files=expected,manifest_sha256=ck['manifest_sha256'],group_sha256=ck['group_sha256'],verified=time.time())
    result=dict(status='passed',attempt=a.attempt,snapshot=closed['snapshot_id'],copies=[local,remote],
        durability='volatile RAM; selected final models require persistent promotion',
        restore='Copy the peer payload and adjacent group JSON back to the recorded source path on host0, then run the ordinary checkpoint reader and full hash audit.',completed=time.time())
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    a.output.chmod(0o444)
    print(json.dumps(dict(status='passed',output=str(a.output),bytes=total,peer=a.peer)),flush=True)


if __name__=='__main__':main()
