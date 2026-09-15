"""Deploy the qualified immutable 19x19 producer to the four supplied hosts."""
from __future__ import annotations

import json
from pathlib import Path
import shlex
import subprocess
import time

SOURCE=Path(__file__).resolve().parent
SSH=['ssh','-F','/dev/null','-o','BatchMode=yes','-o','ConnectTimeout=10']
PYTHON='/home/go-user/.local/share/uv/python/cpython-3.12.13-linux-x86_64-gnu/bin/python3.12'


def main():
    d=json.loads((SOURCE/'deployment.json').read_text())
    root=Path(d['root']);environment=Path(d['environment'])
    qualification=json.loads((root/'qualification/result.json').read_text())
    assert qualification['status']=='passed' and qualification['board_size']==19
    assert qualification['producer_snapshot']==d['snapshot'] and qualification['full_worker_regression']['state']=='stopped'
    contract=json.loads(Path(d['contract']).read_text())
    receipts=[]
    for host in range(4):
        target=f'go-user@worker-{host}.example.invalid'
        if host:
            subprocess.run(SSH+[target,'mkdir -p '+str(root/'environments')],check=True)
            subprocess.run(['rsync','-a','--ignore-existing','-e',shlex.join(SSH),str(environment),f'{target}:{root}/environments/'],check=True)
        config=dict(storage_root=str(root),run_id=d['run_id'],host_index=host,concurrent_games=8,
            research_cpus=[0,1],drain_seconds=3600,contract=contract,
            storage_limits=dict(files_cap=24*(1<<30),free_files_floor=64*(1<<30),available_memory_floor=96*(1<<30)),
            workers=[dict(cpus=list(range(start,start+8)),opponent_index=i) for i,start in enumerate((8,16,24,32,40,48,56,64))],
            producer_snapshot=d['snapshot'],dataset_location='RAM only; completed 9x9 remains stopped in its separate root')
        code='''from pathlib import Path
import os,json,hashlib,subprocess,time,fcntl
root=Path(ROOT); env=Path(ENV); c=CONFIG
def sha(p):
 with p.open('rb') as f:
  h=hashlib.sha256()
  while True:
   b=f.read(8*1024**2)
   if not b:break
   h.update(b)
  return h.hexdigest()
old=Path('/dev/shm/gozero/runs/expert-v1')
assert (old/'stop').exists() and json.loads((old/'supervisor.json').read_text())['state']=='stopped'
memory={l.split(':')[0]:int(l.split()[1])*1024 for l in Path('/proc/meminfo').read_text().splitlines()}
s=os.statvfs(root)
assert s.f_bavail*s.f_frsize>68*(1<<30) and memory['MemAvailable']>200*(1<<30)
snapshot=json.loads((env/'snapshot.json').read_text())
assert snapshot['snapshot']==c['producer_snapshot']
for name,h in snapshot['source_sha256'].items():assert sha(env/'site-packages/flygo'/name)==h
native=next((env/'site-packages/flygo').glob('_native*.so'));assert sha(native)==snapshot['native_sha256']
records=[c['contract']['teacher'],*c['contract']['opponents']]
for record in records:
 suffix='txt.gz' if record.get('url','').endswith('.txt.gz') else 'bin.gz'
 src=Path('/dev/shm/gozero/artifacts')/record['sha256']/('model.'+suffix)
 assert sha(src)==record['sha256']
 dst=root/'artifacts'/record['sha256']/src.name;dst.parent.mkdir(parents=True,exist_ok=True)
 if not dst.exists():os.link(src,dst)
 assert sha(dst)==record['sha256']
h=c['contract']['engine_sha256'];src=Path('/dev/shm/gozero/artifacts')/h/'katago'
assert sha(src)==h
dst=root/'artifacts'/h/'katago';dst.parent.mkdir(parents=True,exist_ok=True)
if not dst.exists():os.link(src,dst)
run=root/'runs'/c['run_id'];run.mkdir(parents=True,exist_ok=True)
assert not (run/'supervisor.json').exists(), 'Never duplicate or silently restart a run'
assert not (run/'stop').exists()
with (run/'deployment.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 path=run/'config.json';path.write_text(json.dumps(c,indent=2)+'\\n')
 (run/'contract.json').write_text(json.dumps(c['contract'],indent=2)+'\\n')
 with (run/'supervisor.log').open('ab') as log:
  p=subprocess.Popen([__PYTHON_EXE__,'-B',str(env/'entry.py'),'serve',str(path)],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1'))
 rec=dict(host=c['host_index'],pid=p.pid,root=str(root),config=str(path),environment=str(env),launched=time.time())
 (run/'launch.json').write_text(json.dumps(rec,indent=2)+'\\n')
 print(json.dumps(rec))
'''.replace('ROOT',repr(str(root))).replace('ENV',repr(str(environment))).replace('CONFIG',repr(config)).replace('__PYTHON_EXE__',repr(PYTHON))
        compile(code, '<remote-deployment>', 'exec')
        receipt=json.loads(subprocess.check_output(SSH+[target,'python3 -c '+shlex.quote(code)],text=True))
        receipts.append(receipt)
        (SOURCE/'cluster.json').write_text(json.dumps(receipts,indent=2)+'\n')
        print(json.dumps(receipt),flush=True)
    (SOURCE/'qualification-result.json').write_text(json.dumps(qualification,indent=2)+'\n')


if __name__=='__main__':main()
