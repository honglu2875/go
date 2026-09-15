"""Launch one frozen LR-only stage after the completed-study evidence gates."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import time

import lr_source

ROOT=Path(__file__).resolve().parents[3]
STUDY=Path(__file__).resolve().parent


def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def read(path):return json.loads(path.read_text())


def pinned(record):
    path=ROOT/record['path']
    if sha(path)!=record['sha256']:raise ValueError('Pinned evidence changed: '+record['path'])
    return read(path)


def inspect(plan,stage):
    if plan['kind']!='strong9_learning_rate_intervention' or set(plan['stages'])!={'seed1','seed2'}:
        raise ValueError('Expected an explicitly paired LR intervention')
    if sha(Path(__file__))!=plan['launch_operator_sha256'] or sha(Path(lr_source.__file__))!=plan['source_helper_sha256']:
        raise ValueError('LR launch/source operator changed')
    if sha(STUDY/'lr_compare.py')!=plan['comparison_operator_sha256']:raise ValueError('Comparison operator changed')
    if not re.fullmatch('[a-z0-9-]{1,48}',plan['trial']):raise ValueError('Invalid trial name')
    conclusion=pinned(plan['preceding_conclusion'])
    if (conclusion['kind']!='strong9_two_seed_conclusion' or conclusion['status']!='passed'
            or conclusion['decision']['outcome']!='criterion_not_met' or conclusion['sustained_overfit']
            or conclusion['test_targets_decoded']):
        raise ValueError('Completed-study evidence does not support this optimization probe')
    selection=pinned(plan['review'])
    if (selection['status']!='completed' or selection['next_intervention']!='learning_rate_only'
            or selection['preceding_conclusion_sha256']!=plan['preceding_conclusion']['sha256']
            or selection['peak_learning_rate']!=plan['peak_learning_rate']):
        raise ValueError('Completed evidence review required')
    rates=[];gates=[]
    for name,seed in (('seed1',91312427),('seed2',91312428)):
        item=plan['stages'][name]
        parent=ROOT/'.gozero/snapshots'/item['parent_snapshot']
        candidate=ROOT/'.gozero/snapshots'/item['snapshot']
        lr_source.identical_source(parent,candidate)
        c=read(candidate/'resolved_config.json')
        if c['seed']!=seed or c['platform']!='tpu' or c['expected_processes']!=4 or c['expected_devices']!=16:
            raise ValueError('Unregistered seed/topology')
        audit=pinned(item['parent_audit'])
        if audit['status']!='passed' or audit['steps']!=4096 or audit['training_snapshot']!=parent.name:
            raise ValueError('Parent learning audit differs')
        rates.append(c['learner']['learning_rate'])
    if len(set(rates))!=1 or rates[0]!=plan['peak_learning_rate']:
        raise ValueError('Paired seeds must test the same rate')
    q=pinned(plan['qualification'])
    if q['status']!='passed':raise ValueError('Full-shape transformer qualification missing')
    item=plan['stages'][stage];snapshot=ROOT/'.gozero/snapshots'/item['snapshot']
    manifest=lr_source.verify(snapshot);c=read(snapshot/'resolved_config.json')
    if q['model']!=c['model']:raise ValueError('Qualified tensor shapes differ')
    for name,digest in q['numerical_sources'].items():
        if sha(snapshot/manifest['recipe']/name)!=digest:raise ValueError('Qualified numerical source changed')
    if stage=='seed2':
        result_path=STUDY/plan['trial']/'seed1-contrast-001.json'
        first=read(result_path)
        if (first['kind']!='strong9_learning_rate_paired_contrast' or first['status']!='passed'
                or first['plan_sha256']!=plan['_sha256'] or not first['screen_passed']
                or first['candidate']['snapshot']!=plan['stages']['seed1']['snapshot']):
            raise ValueError('First seed did not qualify for confirmation')
        gates.append(dict(path=str(result_path.relative_to(ROOT)),sha256=sha(result_path)))
    return snapshot,gates


def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',type=Path,required=True)
    p.add_argument('--plan-sha256',required=True);p.add_argument('--stage',choices=('seed1','seed2'),required=True)
    p.add_argument('--inspect',action='store_true');a=p.parse_args()
    if sha(a.plan)!=a.plan_sha256:raise ValueError('LR registration changed')
    plan=read(a.plan);plan['_sha256']=a.plan_sha256
    snapshot,gates=inspect(plan,a.stage)
    if a.inspect:
        print(json.dumps(dict(status='passed',snapshot=snapshot.name,stage=a.stage,mutations=False)));return
    # The pod controller independently enforces exclusive accelerator ownership.
    # This study lock additionally prevents duplicate launches of its own stage.
    with (STUDY/'.lr-launch.lock').open('a+b') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for launch in (ROOT/'runs').glob('pod-*/launch.json'):
            if not (launch.parent/'result.json').exists():raise ValueError('Another pod attempt remains open: '+launch.parent.name)
        folder=STUDY/plan['trial'];folder.mkdir(parents=True,exist_ok=True)
        receipt=folder/(a.stage+'-process-001.json');log_path=folder/(a.stage+'-controller-001.log')
        if receipt.exists() or log_path.exists():raise ValueError('Stage already started; inspect its receipt/log')
        code="import os,json;from pathlib import Path;s=os.statvfs('/dev/shm');m={l.split(':')[0]:int(l.split()[1])*1024 for l in Path('/proc/meminfo').read_text().splitlines()};print(json.dumps(dict(shm_free=s.f_bavail*s.f_frsize,memory_available=m['MemAvailable'])))"
        def check(host):
            out=json.loads(subprocess.check_output(['ssh','-F','/dev/null','-o','BatchMode=yes','-o','ConnectTimeout=10',
                f'go-user@worker-{host}.example.invalid','taskset -c 0,1 python3 -c '+shlex.quote(code)],text=True,timeout=30))
            if out['shm_free']<=68*(1<<30) or out['memory_available']<=96*(1<<30):
                raise ValueError('Insufficient RAM headroom on host index '+str(host))
            # Reserve a complete final checkpoint plus two GiB of producer
            # growth before the established 64 GiB RAM floor on owner and peer.
            if host in (0,2) and out['shm_free']<=64*(1<<30)+3_000_000_000+2*(1<<30):
                raise ValueError('Checkpoint and producer-growth reserve unavailable')
            return dict(host_index=host,**out)
        with ThreadPoolExecutor(4) as pool:preflight=list(pool.map(check,range(4)))
        argv=[str(ROOT/'.venv/bin/python'),'-B',str(snapshot/'ops/pod_run.py'),'--snapshot',str(snapshot),
            '--workspace-root',str(ROOT),'--timeout','10800','--prepare-timeout','600',
            '--controller-cpus','6','--controller-cpu-list','2,3,4,5,6,7']
        with log_path.open('xb') as log:
            process=subprocess.Popen(argv,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
                start_new_session=True,env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1'))
        stat=Path('/proc',str(process.pid),'stat').read_text().rpartition(')')[2].split()
        result=dict(kind='strong9_lr_stage_process',stage=a.stage,trial=plan['trial'],pid=process.pid,
            process_identity=stat[19],started=time.time(),snapshot=snapshot.name,argv=argv,
            log=str(log_path),plan_sha256=a.plan_sha256,preflight=preflight,gates=gates,
            operator_sha256=sha(Path(__file__)))
        with receipt.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
        receipt.chmod(0o444)
        print(json.dumps(dict(status='started',stage=a.stage,pid=process.pid,snapshot=snapshot.name,receipt=str(receipt.relative_to(ROOT)))))


if __name__=='__main__':main()
