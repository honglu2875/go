"""Read-only pod occupancy and storage observation after experiment closure."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import shlex
import subprocess
import time

from execute_run import ROOT, STUDY
from gozero.durable_files import atomic_json
from gozero.pod import load_hosts, SSH_OPTIONS

CODE = '''import json,os
from pathlib import Path
holders=[];unreadable=0
for process in Path('/proc').iterdir():
 if not process.name.isdigit():continue
 try: descriptors=list((process/'fd').iterdir())
 except (PermissionError,FileNotFoundError,ProcessLookupError):unreadable+=1;continue
 matched=[]
 for descriptor in descriptors:
  try: target=os.readlink(descriptor)
  except OSError:continue
  if target.startswith(('/dev/accel','/dev/vfio')):matched.append(target)
 if matched:holders.append(dict(pid=int(process.name),devices=sorted(set(matched))))
d=os.statvfs('/workspace/go');s=os.statvfs('/dev/shm')
print(json.dumps(dict(visible_device_holders=holders,unreadable_process_fd_tables=unreadable,
 disk_free_bytes=d.f_bavail*d.f_frsize,shm_free_bytes=s.f_bavail*s.f_frsize)))
'''


def main():
    def one(host):
        argv = ['ssh', *SSH_OPTIONS, host.ssh, shlex.join(['taskset', '-c', '0,1', 'python3', '-c', CODE])]
        result = subprocess.run(argv, capture_output=True, text=True, timeout=45)
        if result.returncode:
            raise RuntimeError('Inspection failed on rank ' + str(host.rank) + ': ' + result.stderr)
        return dict(host_rank=host.rank, **json.loads(result.stdout))
    with ThreadPoolExecutor(4) as pool: hosts = list(pool.map(one, load_hosts(ROOT / 'ops/hosts.json')))
    opened = [p.parent.name for p in (ROOT / 'runs').glob('pod-*/launch.json') if not (p.parent / 'result.json').exists()]
    status = 'passed' if not opened and not any(h['visible_device_holders'] for h in hosts) else 'occupied'
    result = dict(status=status, created=time.time(), hosts=hosts, open_attempts=opened,
        inspection_code_sha256=hashlib.sha256(CODE.encode()).hexdigest(),
        scope='No JAX initialization. Visible device descriptors checked; unreadable foreign descriptor tables counted explicitly.')
    atomic_json(STUDY / 'pod-release-001.json', result, replace=False)
    print(json.dumps(dict(status=status, open_attempts=opened, hosts=[dict(rank=h['host_rank'],
        visible_device_holders=len(h['visible_device_holders']), disk_free_gib=round(h['disk_free_bytes']/2**30, 2),
        shm_free_gib=round(h['shm_free_bytes']/2**30, 2)) for h in hosts])))


if __name__ == '__main__': main()
