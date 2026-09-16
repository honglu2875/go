"""Run, audit and retain one predeclared objective-only transformer prefix."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import subprocess
import time

ROOT=Path(__file__).resolve().parents[3]
STUDY=Path(__file__).resolve().parent
SSH=['ssh','-F','/dev/null','-o','BatchMode=yes','-o','ConnectTimeout=10']


def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--plan-sha256',required=True);args=parser.parse_args()
    if sha(args.plan)!=args.plan_sha256:raise ValueError('Plan changed')
    p=json.loads(args.plan.read_text());folder=STUDY/p['output_directory'];folder.mkdir(exist_ok=False)
    for name,digest in p['operators'].items():
        if sha(ROOT/name)!=digest:raise ValueError('Operator changed: '+name)
    if sha(ROOT/p['control_audit'])!=p['control_audit_sha256']:raise ValueError('Control audit changed')
    snapshot=ROOT/'.gozero/snapshots'/p['snapshot'];started=time.time();attempt=None
    def command(argv,label,timeout):
        env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
        env.pop('JAX_PLATFORMS',None)
        log=folder/(label+'.log')
        with log.open('xb') as f:subprocess.run(list(map(str,argv)),cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,timeout=timeout,check=True)
        return log
    try:
        code="""import os,json
from pathlib import Path
f=os.statvfs('/dev/shm');d=os.statvfs('/workspace/go')
m={l.split(':')[0]:int(l.split()[1])*1024 for l in Path('/proc/meminfo').read_text().splitlines()}
active=[]
for p in Path('/proc').glob('[0-9]*/cmdline'):
 try:
  if any(x.endswith(b'/train.py') for x in p.read_bytes().split(b'\\0')):active.append(int(p.parent.name))
 except (FileNotFoundError,PermissionError,ProcessLookupError):pass
print(json.dumps(dict(shm_free=f.f_bavail*f.f_frsize,disk_free=d.f_bavail*d.f_frsize,memory_available=m['MemAvailable'],train_pids=active)))
"""
        def preflight(host):
            r=json.loads(subprocess.check_output(SSH+[f'go-user@worker-{host}.example.invalid',
                'taskset -c 0,1 python3 -c '+shlex.quote(code)],timeout=30,text=True))
            reserve=p['checkpoint_reserve_bytes'] if host in (0,p['replica_peer']) else 0
            if (r['train_pids'] or r['shm_free']<64*(1<<30)+reserve+512*(1<<20)
                    or r['disk_free']<2*(1<<30) or r['memory_available']<96*(1<<30)):
                raise ValueError('Prefix resource floor: '+str(dict(host=host,**r)))
            return dict(host=host,**r)
        with ThreadPoolExecutor(4) as pool:resources=list(pool.map(preflight,range(4)))
        (folder/'resources.json').write_text(json.dumps(resources,indent=2)+'\n')
        log=command([ROOT/'.venv/bin/python','-B',snapshot/'ops/pod_run.py','--snapshot',snapshot,
            '--workspace-root',ROOT,'--timeout',p['timeout_seconds'],'--prepare-timeout','600',
            '--controller-cpus','6','--controller-cpu-list','2,3,4,5,6,7',
            '--stop-after-turn',p['stop_turn']],'controller',p['timeout_seconds']+900)
        headers=[json.loads(x) for x in log.read_text().splitlines() if x.startswith('{')]
        headers=[x for x in headers if x.get('kind')=='pod_attempt']
        if len(headers)!=1:raise ValueError('Expected one attempt')
        attempt=ROOT/'runs'/headers[0]['attempt_id']
        closed=json.loads((attempt/'result.json').read_text())
        if closed['status']!='passed' or closed['snapshot_id']!=p['snapshot'] or closed['stop_after_turn']!=p['stop_turn']:
            raise ValueError('Prefix did not close')
        for rank in range(4):
            r=json.loads((attempt/f'rank-{rank}/result.json').read_text())
            if r['status']!='passed' or not r['source_integrity']:raise ValueError('Rank closure failed')
        command([p['python'],'-B',STUDY/'audit_prefix.py','--snapshot',snapshot,'--artifacts',
            *[attempt/f'rank-{i}/artifacts' for i in range(4)],'--purpose','learning','--stop-turn',p['stop_turn'],
            '--output',folder/'audit.json'],'audit',600)
        audit=json.loads((folder/'audit.json').read_text())
        control=json.loads((ROOT/p['control_audit']).read_text())
        if (audit['initial_parameters_sha256']!=control['initial_parameters_sha256']
                or audit['positions']!=p['expected_positions'] or audit['parameters']!=control['parameters']):
            raise ValueError('Prefix initialization, model size or draws differ')
        for key in ('validation_history','training_probe_history'):
            before=control[key][0];after=audit[key][0]
            if before['episode_ids_sha256']!=after['episode_ids_sha256']:
                raise ValueError('Initial evaluation population differs')
            for metric,value in before['metrics'].items():
                if not math.isclose(value,after['metrics'][metric],rel_tol=p['initial_metric_rtol'],abs_tol=p['initial_metric_atol']):
                    raise ValueError('Initial evaluation differs: '+metric)
        command([ROOT/'.venv/bin/python','-B',ROOT/'research/studies/strong19_source_muon/replicate_full_size.py',
            '--workspace-root',ROOT,'--attempt',attempt.name,'--peer',p['replica_peer'],'--output',folder/'replica.json'],'replica',600)
        result=dict(status='passed',attempt=attempt.name,audit_sha256=sha(folder/'audit.json'),replica_sha256=sha(folder/'replica.json'),
            initial_parameters_and_validation_matched=True,positions=audit['positions'])
    except BaseException as e:
        result=dict(status='failed',attempt=attempt.name if attempt else None,error=repr(e))
        raise
    finally:
        result.update(kind='objective_only_joint_prefix',started=started,finished=time.time(),plan_sha256=args.plan_sha256)
        (folder/'result.json').write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(result),flush=True)


if __name__=='__main__':main()
