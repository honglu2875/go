"""Read producer status on all supplied hosts without changing running jobs."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import shlex
import subprocess


CODE = """from pathlib import Path
import json,os,time
run=Path('/dev/shm/go-corpus19/runs/expert19-v1')
s=json.loads((run/'supervisor.json').read_text())
rows=[json.loads(p.read_text()) for p in sorted(run.glob('worker-*/status.json'))]
memory={x.split(':')[0]:int(x.split()[1])*1024 for x in Path('/proc/meminfo').read_text().splitlines()}
v=os.statvfs('/dev/shm')
print(json.dumps(dict(supervisor_state=s['state'],supervisor_updated=s['updated'],failures=s['failures'],
 live_workers=sum(Path('/proc',str(pid)).exists() for pid in s['workers'].values()),
 completed_games=sum(x['completed_games'] for x in rows),positions=sum(x['positions'] for x in rows),
 strata=[dict(index=i,games=x['completed_games'],positions=x['positions'],state=x['state']) for i,x in enumerate(rows)],
 shm_free=v.f_bavail*v.f_frsize,memory_available=memory['MemAvailable'],updated=time.time())))
"""


def inspect(host):
    output=subprocess.check_output(['ssh','-F','/dev/null','-o','BatchMode=yes','-o','ConnectTimeout=10',
        f'go-user@worker-{host}.example.invalid','python3 -c '+shlex.quote(CODE)],text=True,timeout=30)
    return dict(host=host,**json.loads(output))


if __name__=='__main__':
    with ThreadPoolExecutor(4) as pool:
        rows=list(pool.map(inspect,range(4)))
    with (Path(__file__).parent/'progress-checks.jsonl').open('a') as stream:
        for row in rows:
            stream.write(json.dumps(row)+'\n')
            print(json.dumps(row))
