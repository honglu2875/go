"""Existing read-only resource checks and token-aware cancellation."""
from concurrent.futures import ThreadPoolExecutor
import json
import shlex
import subprocess
import time
from execute_run import ROOT, publish


def resource_observation(snapshot):
    from gozero.pod import load_hosts,SSH_OPTIONS
    hosts=load_hosts(snapshot/'ops/hosts.json')
    code="import os,json;from pathlib import Path;s=os.statvfs('/dev/shm');d=os.statvfs('.');m={l.split(':')[0]:int(l.split()[1])*1024 for l in Path('/proc/meminfo').read_text().splitlines()};print(json.dumps(dict(shm_free=s.f_bavail*s.f_frsize,disk_free=d.f_bavail*d.f_frsize,memory_available=m['MemAvailable'])))"
    def one(host):
        command='cd '+shlex.quote(str(ROOT))+' && taskset -c 0,1 python3 -c '+shlex.quote(code)
        value=json.loads(subprocess.check_output(['ssh',*SSH_OPTIONS,host.ssh,command],text=True,timeout=30))
        return dict(rank=host.rank,**value)
    with ThreadPoolExecutor(4) as pool:return list(pool.map(one,hosts))


def request_cancellation(snapshot,attempt,folder,reason):
    """Ask the existing token-aware remote supervisors to close this attempt."""
    from gozero.pod import load_hosts,SSH_OPTIONS
    hosts=load_hosts(snapshot/'ops/hosts.json')
    def one(host):
        argv=['python3',str(snapshot/'ops/cancel_host.py'),'--snapshot',str(snapshot),
              '--attempt',str(ROOT/'runs'/attempt)]
        result=subprocess.run(['ssh',*SSH_OPTIONS,host.ssh,shlex.join(argv)],capture_output=True,text=True,timeout=45)
        return dict(rank=host.rank,returncode=result.returncode,stdout=result.stdout,stderr=result.stderr)
    with ThreadPoolExecutor(4) as pool:rows=list(pool.map(one,hosts))
    publish(folder/(attempt+'-cancellation.json'),dict(reason=reason,created=time.time(),ranks=rows))
