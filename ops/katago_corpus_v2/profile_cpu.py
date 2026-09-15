"""Read per-engine CPU use inside the owned corpus workers only."""
from concurrent.futures import ThreadPoolExecutor
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import time


CODE='''from pathlib import Path
import hashlib,json,os,time
run=Path('/dev/shm/go-corpus19/runs')/RUN
config=json.loads((run/'config.json').read_text())
def read(pid):
 try:
  fields=Path('/proc',str(pid),'stat').read_text().rsplit(')',1)[1].split()
  return dict(identity=fields[19],state=fields[0],cpu_ticks=int(fields[11])+int(fields[12]),rss_bytes=int(fields[21])*os.sysconf('SC_PAGE_SIZE'))
 except FileNotFoundError:return None
records=[]
for index,worker in enumerate(config['workers']):
 directory=run/f'worker-{index:02d}'
 p=directory/'engine-processes.json'
 if not p.exists():continue
 for role,record in zip(('teacher','opponent'),json.loads(p.read_text())):
  pid=record['pid'];before=read(pid)
  cfg=directory/role/'analysis.cfg'
  if before and before['identity']==record['identity'] and before['state']!='Z':
   records.append(dict(worker=index,role=role,pid=pid,cpus=worker['cpus'],before=before,config_sha256=hashlib.sha256(cfg.read_bytes()).hexdigest()))
start=time.monotonic();time.sleep(5);elapsed=time.monotonic()-start
for row in records:
 after=read(row['pid']);before=row.pop('before')
 row['stable_process']=after is not None and after['identity']==before['identity'] and after['state']!='Z'
 if row['stable_process']:
  row['cpu_cores_used']=(after['cpu_ticks']-before['cpu_ticks'])/os.sysconf('SC_CLK_TCK')/elapsed
  row['rss_bytes']=after['rss_bytes']
print(json.dumps(dict(run=RUN,seconds=elapsed,rows=records,completed=time.time())))
'''


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',choices=('expert19-v1','expert19-v2'),required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    def inspect(host):
        cmd='taskset -c 0,1 python3 -c '+shlex.quote(CODE.replace('RUN',repr(a.run)))
        out=subprocess.check_output(['ssh','-F','/dev/null','-o','BatchMode=yes','-o','ConnectTimeout=10',
            f'go-user@worker-{host}.example.invalid',cmd],text=True,timeout=30)
        return dict(host=host,**json.loads(out))
    with ThreadPoolExecutor(4) as pool:hosts=list(pool.map(inspect,range(4)))
    result=dict(kind='corpus_engine_cpu_profile',run=a.run,hosts=hosts,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),completed=time.time(),
        scope='Five-second CPU accounting samples; no throughput or post-change performance inference.')
    with a.output.open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps([dict(host=h['host'],cpu_cores=sum(r.get('cpu_cores_used',0) for r in h['rows']),
        stable_engines=sum(r['stable_process'] for r in h['rows'])) for h in hosts]))


if __name__=='__main__':main()
