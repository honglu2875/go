"""Launch exactly one preregistered larger-data stage after its evidence gates."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

ROOT=Path('/workspace/go');STUDY=Path(__file__).resolve().parent
STAGES=('cnn_qualification','cnn_seed1','transformer_qualification','transformer_seed1','cnn_seed2','transformer_seed2')


def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def read(path):return json.loads(path.read_text())


def main():
    p=argparse.ArgumentParser();p.add_argument('--registration',type=Path,required=True)
    p.add_argument('--registration-sha256',required=True);p.add_argument('--stage',choices=STAGES,required=True)
    a=p.parse_args()
    if sha(a.registration)!=a.registration_sha256:raise ValueError('Registration changed')
    reg=read(a.registration);protocol_path=ROOT/reg['protocol']['path']
    if sha(protocol_path)!=reg['protocol']['sha256']:raise ValueError('Protocol changed')
    protocol=read(protocol_path)
    decision_path=ROOT/reg['preceding_decision']['path']
    if sha(decision_path)!=reg['preceding_decision']['sha256'] or read(decision_path)['status']!='completed':
        raise ValueError('Small-data architecture selection has not closed')
    for launch in (ROOT/'runs').glob('pod-*/launch.json'):
        if not (launch.parent/'result.json').exists():raise ValueError('Another pod attempt remains open: '+launch.parent.name)
    arm,stage=a.stage.split('_',1);snapshot=ROOT/'.gozero/snapshots'/reg['snapshots'][arm][stage]
    sys.path.insert(0,str(snapshot/'packages/gozero/src'))
    from gozero.snapshots import verify
    manifest=verify(snapshot);recipe=snapshot/manifest['recipe'];config=read(snapshot/'resolved_config.json')
    for name,digest in protocol['numerical_source_sha256'].items():
        if sha(recipe/name)!=digest:raise ValueError('Registered numerical source changed: '+name)
    for name,digest in reg['shared_library_sha256'].items():
        if sha(snapshot/name)!=digest:raise ValueError('Qualified shared library changed: '+name)
    if config['model']!=reg['models'][arm] or config['dataset']['manifest_sha256']!=protocol['data']['manifest_sha256']:
        raise ValueError('Model or data differs from registration')
    def audit(name):
        path=STUDY/(name+'-audit-001.json');result=read(path)
        previous_arm,previous_stage=name.split('-',1)
        if (result['status']!='passed' or result['steps']!=4096
                or result['training_snapshot']!=reg['snapshots'][previous_arm][previous_stage]):
            raise ValueError('Prerequisite audit failed or wrong run: '+name)
        return dict(path=str(path),sha256=sha(path))
    gates=[]
    if stage!='qualification':
        gate_path=STUDY/(arm+'-qualification-audit-001.json');gate=read(gate_path)
        if gate['status']!='passed' or gate['snapshot']!=reg['snapshots'][arm]['qualification'] or gate['model']!=config['model']:
            raise ValueError('Missing or wrong full-shape qualification')
        for name,digest in gate['numerical_sources'].items():
            if sha(recipe/name)!=digest:raise ValueError('Qualification/learning numerical source differs')
        gates.append(dict(path=str(gate_path),sha256=sha(gate_path)))
    if a.stage in ('transformer_qualification','transformer_seed1'):gates.append(audit('cnn-seed1'))
    if a.stage=='cnn_seed2':
        gates.extend((audit('cnn-seed1'),audit('transformer-seed1')))
        contrast_path=STUDY/'seed1-contrast-001.json'
        contrast=read(contrast_path)
        if contrast['status']!='passed' or contrast['registration_sha256']!=a.registration_sha256:
            raise ValueError('First paired seed was not contrasted')
        gates.append(dict(path=str(contrast_path),sha256=sha(contrast_path)))
    if a.stage=='transformer_seed2':gates.append(audit('cnn-seed2'))
    if stage=='qualification':
        budget_path=ROOT/reg['cpu_budgets'][arm]['path']
        if sha(budget_path)!=reg['cpu_budgets'][arm]['sha256'] or read(budget_path)['status']!='passed':
            raise ValueError('Missing CPU budget qualification')
        if config['steps']!=4 or config['learner']['warmup_steps']!=2:
            raise ValueError('Unexpected qualification horizon')
        gates.append(dict(path=str(budget_path),sha256=sha(budget_path)))
    elif config['steps']!=4096 or config['seed']!=protocol['training']['seeds'][int(stage[-1])-1]:
        raise ValueError('Unregistered learning horizon/seed')
    code="import os,json;from pathlib import Path;s=os.statvfs('/dev/shm');m={l.split(':')[0]:int(l.split()[1])*1024 for l in Path('/proc/meminfo').read_text().splitlines()};print(json.dumps(dict(shm_free=s.f_bavail*s.f_frsize,memory_available=m['MemAvailable'])))"
    def check(host):
        out=json.loads(subprocess.check_output(['ssh','-F','/dev/null','-o','BatchMode=yes','-o','ConnectTimeout=10',
            f'go-user@worker-{host}.example.invalid','taskset -c 0,1 python3 -c '+shlex.quote(code)],text=True,timeout=30))
        if out['shm_free']<=68*(1<<30) or out['memory_available']<=96*(1<<30):
            raise ValueError('Insufficient RAM headroom: '+str((host,out)))
        return dict(host=host,**out)
    with ThreadPoolExecutor(4) as pool:preflight=list(pool.map(check,range(4)))
    name=a.stage.replace('_','-');process_path=STUDY/(name+'-process-001.json')
    if process_path.exists():raise ValueError('Stage already launched; inspect its receipt')
    log_path=STUDY/(name+'-controller-001.log')
    timeout='2400' if stage=='qualification' else '10800'
    argv=[str(ROOT/'.venv/bin/python'),'-B',str(snapshot/'ops/pod_run.py'),'--snapshot',str(snapshot),
          '--workspace-root',str(ROOT),'--timeout',timeout,'--prepare-timeout','600',
          '--controller-cpus','6','--controller-cpu-list','0-5']
    with log_path.open('xb') as log:
        process=subprocess.Popen(argv,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
            start_new_session=True,env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1'))
    result=dict(stage=a.stage,pid=process.pid,started=time.time(),snapshot=snapshot.name,argv=argv,
                log=str(log_path),registration_sha256=a.registration_sha256,gates=gates,preflight=preflight,
                operator_sha256=sha(Path(__file__)))
    with process_path.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    process_path.chmod(0o444)
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
