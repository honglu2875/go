"""Execute a registered encoder wave serially; stop on failure or deadline."""
import argparse
from datetime import datetime,timezone
import fcntl
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
SOURCE=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(SOURCE/'packages/gozero/src'))
from gozero.snapshots import verify,read_json,canonical_json
from gozero.checkpoints import sha256
from gozero.checkpoint_archive import publish,verify_files


def main():
    p=argparse.ArgumentParser();p.add_argument('--workspace-root',type=Path,required=True)
    p.add_argument('--registration',type=Path,required=True);p.add_argument('--registration-sha256',required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--check-only',action='store_true')
    a=p.parse_args();root=a.workspace_root.resolve();verify(SOURCE)
    if root!=Path('/workspace/go') or sha256(a.registration)!=a.registration_sha256:raise ValueError('Wrong workspace/registration')
    reg=read_json(a.registration)
    if reg['operator_snapshot']!=SOURCE.name:raise ValueError('Wrong frozen operator')
    for name,digest in reg['prerequisite_files'].items():
        if sha256(root/name)!=digest:raise ValueError('Prerequisite changed: '+name)
    for case in reg['candidates']:
        source=root/'.gozero/snapshots'/case['snapshot'];verify(source)
        if sha256(source/'resolved_config.json')!=case['config_sha256']:raise ValueError('Candidate configuration changed')
        q=read_json(root/case['qualification_report'])
        if q['status']!='passed' or q['initial_parameter_elements_sha256']!=case['initial_parameter_elements_sha256']:
            raise ValueError('Missing or mismatched TPU qualification')
        qualified=root/'.gozero/snapshots'/q['snapshot'];verify(qualified)
        if read_json(qualified/'resolved_config.json')['model']!=case['model']:
            raise ValueError('Qualified model differs from registered experiment')
        for name,digest in q['numerical_sources'].items():
            if sha256(source/'research/recipes/visual_token_encoder'/name)!=digest:
                raise ValueError('Numerical implementation changed after qualification: '+name)
    if a.check_only:
        print(json.dumps({'status':'checked','candidates':[x['label'] for x in reg['candidates']]}));return
    a.output.mkdir(parents=True,exist_ok=False)
    env={**os.environ,'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','PYTHONDONTWRITEBYTECODE':'1'}
    cpu={**env,'JAX_PLATFORMS':'cpu','MPLCONFIGDIR':str(root/'.gozero/cache/matplotlib')}
    result={'kind':'encoder_intervention_execution','status':'running','registration_sha256':a.registration_sha256,
        'operator_snapshot':SOURCE.name,'started_at':datetime.now(timezone.utc).isoformat(),'audits':[]}
    audits=[(x['label'],str(root/x['audit_path']),x['audit_sha256']) for x in reg['references']]
    def event(kind,**fields):
        row={'kind':kind,'time':datetime.now(timezone.utc).isoformat(),**fields}
        with (a.output/'events.jsonl').open('ab') as f:f.write(canonical_json(row));f.flush();os.fsync(f.fileno())
        print(json.dumps(row),flush=True)
    try:
        with (root/'runs/.registered-lr-sweep.lock').open('a+b') as lock:
            fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
            for index,case in enumerate(reg['candidates'],1):
                if datetime.now(timezone.utc)>=datetime.fromisoformat(reg['stop_launching_after_utc']):
                    result['status']='deadline_reached';event('deadline_reached',next_label=case['label']);break
                launches=list((root/'runs').glob('pod-*/launch.json'))
                if any(not x.with_name('result.json').exists() for x in launches):raise ValueError('Another pod attempt is open')
                if any(read_json(x)['snapshot_id']==case['snapshot'] for x in launches):raise ValueError('Candidate already attempted; no automatic retry')
                fs=os.statvfs(root);ram=os.statvfs('/dev/shm')
                if fs.f_bavail*fs.f_frsize<150_000_000 or ram.f_bavail*ram.f_frsize<15_000_000_000:
                    raise ValueError('Insufficient metadata or temporary checkpoint space')
                allocations=read_json(root/reg['best_checkpoint_reservations'])['allocations']
                for allocation in allocations:
                    code="from pathlib import Path; p=Path("+repr(str(root/allocation['reservation']))+"); assert not p.is_symlink() and p.is_file() and p.stat().st_size=="+str(allocation['bytes'])
                    argv=['python3','-c',code]
                    if allocation['host']:
                        argv=['ssh','-F','/dev/null','-o','BatchMode=yes',f"go-user@worker-{allocation['host']}",shlex.join(argv)]
                    subprocess.run(argv,check=True,timeout=30)
                source=root/'.gozero/snapshots'/case['snapshot'];label=f"{index}-{case['label']}"
                event('arm_launch',label=case['label'],snapshot=case['snapshot'])
                with (a.output/(label+'-pod.log')).open('xb') as f:
                    done=subprocess.run([sys.executable,'-B',str(source/'ops/pod_run.py'),'--snapshot',str(source),'--workspace-root',str(root),
                        '--timeout',str(reg['maximum_seconds_per_attempt']),'--prepare-timeout','180','--controller-cpus','32'],cwd=root,env=env,stdout=f,stderr=subprocess.STDOUT)
                matches=[x for x in (root/'runs').glob('pod-*/launch.json') if read_json(x)['snapshot_id']==case['snapshot']]
                if len(matches)!=1:raise ValueError('Cannot identify unique attempt')
                attempt=matches[0].parent;closed=read_json(attempt/'result.json')
                event('arm_closed',label=case['label'],attempt=attempt.name,status=closed['status'])
                if done.returncode or closed['status']!='passed':raise ValueError('Experiment failed; no numerical retry')
                audit=a.output/(label+'-audit.json')
                with (a.output/(label+'-audit.log')).open('xb') as f:
                    subprocess.run(['taskset','-c','64-95',sys.executable,'-B',str(Path(__file__).with_name('audit_learning.py')),
                        '--workspace-root',str(root),'--attempt',str(attempt),'--output',str(audit)],cwd=root,env=cpu,stdout=f,stderr=subprocess.STDOUT,check=True)
                owner=read_json(attempt/'rank-0/artifacts/result.json');record=owner['latest_checkpoint']['temporary']
                if sha256(Path(record['receipt']))!=record['receipt_sha256']:raise ValueError('Temporary receipt changed')
                verify_files(Path(record['cache_path']),record)
                audits.append((case['label'],str(audit),sha256(audit)))
                publication=a.output/(label+'-analysis')
                argv=[str(root/'.gozero/analysis-environments/plotting/bin/python'),'-B',str(Path(__file__).with_name('analyze_encoder.py')),
                    '--workspace-root',str(root),'--registration',str(a.registration),'--registration-sha256',a.registration_sha256,'--output',str(publication)]
                for arm in audits:argv.extend(['--audit',*arm])
                with (a.output/(label+'-analysis.log')).open('xb') as f:subprocess.run(argv,cwd=root,env=cpu,stdout=f,stderr=subprocess.STDOUT,check=True)
                analysis=read_json(publication/'comparison.json')
                result['audits'].append({'label':case['label'],'attempt':attempt.name,'path':str(audit),'sha256':sha256(audit),
                    'publication':str(publication),'publication_manifest_sha256':sha256(publication/'manifest.json')})
                event('arm_audited',label=case['label'],best_completed=analysis['best_completed'],pending=analysis['pending'],publication=str(publication))
            else:result['status']='passed'
    except BaseException as error:
        result.update(status='failed',error=repr(error));event('queue_stopped',error=repr(error));raise
    finally:
        if not any(not x.with_name('result.json').exists() for x in (root/'runs').glob('pod-*/launch.json')):
            ledger=root/'research/studies/runtime_qualification/reservation_ledger.json'
            with (a.output/'ledger.log').open('xb') as f:
                done=subprocess.run([sys.executable,'-B',str(SOURCE/'ops/update_reservation_ledger.py'),'--workspace-root',str(root),
                    '--expected-previous-sha256',sha256(ledger)],cwd=root,env=env,stdout=f,stderr=subprocess.STDOUT)
            result['ledger_update_returncode']=done.returncode
        result['finished_at']=datetime.now(timezone.utc).isoformat();publish(a.output/'result.json',result)
        print(json.dumps(result),flush=True)


if __name__=='__main__':main()
