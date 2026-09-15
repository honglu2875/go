"""Read-only cluster monitor, including the predecessor drain and CPU handover."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import shlex
import subprocess
import time

SOURCE = Path(__file__).resolve().parent
SSH = ['ssh', '-F', '/dev/null', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10']
CODE = '''import json,os,time
from pathlib import Path
root=Path('/dev/shm/go-corpus19')
def read(p):return json.loads(p.read_text()) if p.exists() else None
def proc(pid):
 try:
  a=Path('/proc',str(pid),'stat').read_text().rpartition(')')[2].split()
  return dict(pid=pid,state=a[0],identity=a[19])
 except FileNotFoundError:return None
def active(record):
 p=proc(record['pid'])
 return p is not None and p['state']!='Z' and p['identity']==record['identity']
new=root/'runs/expert19-v2';config=read(new/'config.json')
assert config
out=dict(host=config['host_index'],updated=time.time(),runs={},overlaps=[])
for name in ('expert19-v1','expert19-v2'):
 run=root/'runs'/name;s=read(run/'supervisor.json');workers=[]
 for i in range(8):
  w=read(run/f'worker-{i:02d}/status.json')
  if w is None:continue
  p=proc(w['pid']);live=p is not None and p['state']!='Z'
  workers.append(dict(index=i,live=live,process=p,state=w['state'],updated=w['updated'],
    games=w['completed_games'],positions=w['positions'],in_flight_games=w.get('in_flight_games'),
    in_flight_positions=w.get('in_flight_positions'),teacher_threads=w.get('teacher_threads'),
    opponent_threads=w.get('opponent_threads')))
  if name=='expert19-v2' and live:
   old=config['handover']['workers'][str(i)]['processes']
   if any(active(r) for r in old):out['overlaps'].append(i)
 out['runs'][name]=dict(supervisor=s,supervisor_process=proc(s['pid']) if s else None,
   stop_requested=(run/'stop').exists(),workers=workers,
   completed_games=sum(w['games'] for w in workers),positions=sum(w['positions'] for w in workers),
   live_workers=sum(w['live'] for w in workers))
stat=os.statvfs(root);out['shm_free']=stat.f_bavail*stat.f_frsize
out['memory_available']=next(int(l.split()[1])*1024 for l in Path('/proc/meminfo').read_text().splitlines() if l.startswith('MemAvailable:'))
out['drain_guard_released']=read(new/'drain-guard-released.json')
print(json.dumps(out))
'''


def main():
    def one(host):
        command = 'taskset -c 0,1 python3 -c '+shlex.quote(CODE)
        return json.loads(subprocess.check_output(SSH+[f'go-user@worker-{host}.example.invalid',command],text=True,timeout=60))
    with ThreadPoolExecutor(4) as pool: rows=list(pool.map(one,range(4)))
    with (SOURCE/'progress-checks.jsonl').open('a') as f:
        for row in rows: f.write(json.dumps(row)+'\n')
    for row in rows:
        old,new = (row['runs'][name] for name in ('expert19-v1','expert19-v2'))
        print(json.dumps(dict(host=row['host'],updated=row['updated'],
            old_live_workers=old['live_workers'],new_live_workers=new['live_workers'],
            completed_games=old['completed_games']+new['completed_games'],
            completed_positions=old['positions']+new['positions'],
            new_in_flight_positions=sum(w['in_flight_positions'] or 0 for w in new['workers']),
            new_failures=new['supervisor'].get('failures',{}),
            waiting_previous=new['supervisor'].get('waiting_previous'),
            overlaps=row['overlaps'],shm_free=row['shm_free'],memory_available=row['memory_available'])),flush=True)
    if any(row['overlaps'] for row in rows): raise RuntimeError('Recorded old/new generation processes overlap')


if __name__ == '__main__': main()
