"""Release redundant source-CNN transport caches, preserving two sealed replicas."""
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import time

ROOT=Path(__file__).resolve().parents[3]
STUDY=Path(__file__).resolve().parent
SSH=['ssh','-F','/dev/null','-o','BatchMode=yes','-o','ConnectTimeout=10']
PYTHON='/home/go-user/.local/share/uv/python/cpython-3.12.13-linux-x86_64-gnu/bin/python3.12'
ATTEMPT='pod-20260915T202618Z-f34ea433'


def remote(host,code):
    return json.loads(subprocess.check_output(SSH+[f'go-user@worker-{host}.example.invalid',
        'taskset -c 0,1 '+shlex.quote(PYTHON)+' -c '+shlex.quote(code)],text=True,timeout=90))


def main():
    assert not any(not (p.parent/'result.json').exists() for p in (ROOT/'runs').glob('pod-*/launch.json'))
    eviction=ROOT/'ops/cold_checkpoint_storage/source-qualification-001/prefix-eviction.json'
    record=json.loads(eviction.read_text());assert record['status']=='passed' and record['attempt']==ATTEMPT
    report=json.loads((ROOT/'runs'/ATTEMPT/'rank-0/artifacts/result.json').read_text())
    cp=report['latest_checkpoint'];assert report['turn']==2
    expected=cp['temporary']['files'];retained=[]
    for peer in record['peers']:
        assert peer['host'] in (1,2) and peer['path']==f'/dev/shm/gozero-staged-replicas/{ATTEMPT}/turn-000000002'
        code=f'''from pathlib import Path
import hashlib,json,stat
p=Path({peer['path']!r});expected={expected!r}
for name,item in expected.items():
 s=(p/name).stat();assert s.st_uid==0 and not s.st_mode&0o222 and s.st_size==item['bytes']
 with (p/name).open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==item['sha256']
with p.with_suffix('.group.json').open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()=={cp['group_sha256']!r}
print(json.dumps(dict(status='passed',path=str(p),root_owned=True)))
'''
        retained.append(dict(host=peer['host'],**remote(peer['host'],code)))
    assert len({r['host'] for r in retained})==2
    source=Path(cp['path'])/'arrays.npz';item=expected['arrays.npz']
    assert str(source)==f'/dev/shm/gozero-staged-checkpoints/{ATTEMPT}/rank-0/artifacts/checkpoints/turn-000000002/arrays.npz'
    code=f'''from pathlib import Path
import hashlib,json,stat
p=Path({str(source)!r});e={item!r}
assert not any(x.is_symlink() for x in (p,*p.parents))
s=p.stat();assert stat.S_ISREG(s.st_mode) and not s.st_mode&0o222 and s.st_size==e['bytes']
with p.open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==e['sha256']
p.unlink()
print(json.dumps(dict(status='passed',reclaimed_bytes=e['bytes'],path=str(p),metadata_preserved=True)))
'''
    intent=dict(created=time.time(),attempt=ATTEMPT,retained=retained,remove_only=str(source),hosts=[1,2,3],
        operator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    with (STUDY/'source-resume-release-intent-001.json').open('x') as f:json.dump(intent,f,indent=2);f.write('\n')
    rows=[]
    for host in (1,2,3):
        row=dict(host=host,**remote(host,code));rows.append(row)
        with (STUDY/'source-resume-release-events-001.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
    result=dict(status='passed',completed=time.time(),retained=retained,released=rows)
    with (STUDY/'source-resume-release-result-001.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result))


if __name__=='__main__':main()
