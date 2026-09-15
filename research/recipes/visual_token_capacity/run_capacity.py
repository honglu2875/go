"""Run the registered width's LR cells serially, auditing before advancing."""
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
from gozero.checkpoint_archive import publish,remote


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
    qualification=read_json(root/reg['qualification_report'])
    if qualification['status']!='passed':raise ValueError('Full TPU qualification has not passed')
    for case in reg['candidates']:
        source=root/'.gozero/snapshots'/case['snapshot'];verify(source)
        if sha256(source/'resolved_config.json')!=case['config_sha256']:raise ValueError('Candidate configuration changed')
    if a.check_only:
        print(json.dumps({'status':'checked','rates':[x['peak_learning_rate'] for x in reg['candidates']]}));return
    a.output.mkdir(parents=True,exist_ok=False)
    env={**os.environ,'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','PYTHONDONTWRITEBYTECODE':'1'}
    cpu={**env,'JAX_PLATFORMS':'cpu'}
    result={'kind':'encoder_capacity_width_execution','status':'running','registration_sha256':a.registration_sha256,
        'operator_snapshot':SOURCE.name,'started_at':datetime.now(timezone.utc).isoformat(),'audits':[]}
    audits=[(str(x['encoder_channels']),str(x['peak_learning_rate']),str(root/x['audit_path']),x['audit_sha256']) for x in reg['references']]
    def event(kind,**fields):
        row={'kind':kind,'time':datetime.now(timezone.utc).isoformat(),**fields}
        with (a.output/'events.jsonl').open('ab') as f:f.write(canonical_json(row));f.flush();os.fsync(f.fileno())
        print(json.dumps(row),flush=True)
    try:
        with (root/'runs/.registered-lr-sweep.lock').open('a+b') as lock:
            fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
            for index,case in enumerate(reg['candidates'],1):
                launches=list((root/'runs').glob('pod-*/launch.json'))
                if any(not x.with_name('result.json').exists() for x in launches):raise ValueError('Another pod attempt is open')
                if any(read_json(x)['snapshot_id']==case['snapshot'] for x in launches):raise ValueError('Candidate already attempted; no automatic retry')
                fs=os.statvfs(root)
                if fs.f_bavail*fs.f_frsize<150000000:raise ValueError('Insufficient owner metadata space')
                archive=case['checkpoint_archive']
                code="from pathlib import Path; p=Path("+repr(str(root/archive['reservation']))+"); assert not p.is_symlink() and p.is_file() and p.stat().st_size==3000000000"
                subprocess.run(['ssh','-F','/dev/null','-o','BatchMode=yes',f"go-user@worker-{archive['host']}",shlex.join(['python3','-c',code])],check=True,timeout=30)
                source=root/'.gozero/snapshots'/case['snapshot'];label=f"{index}-{case['peak_learning_rate']:.0e}"
                event('arm_launch',rate=case['peak_learning_rate'],snapshot=case['snapshot'])
                with (a.output/(label+'-pod.log')).open('xb') as f:
                    done=subprocess.run([sys.executable,'-B',str(source/'ops/pod_run.py'),'--snapshot',str(source),'--workspace-root',str(root),
                        '--timeout',str(reg['maximum_seconds_per_attempt']),'--prepare-timeout','180','--controller-cpus','32'],cwd=root,env=env,stdout=f,stderr=subprocess.STDOUT)
                matches=[x for x in (root/'runs').glob('pod-*/launch.json') if read_json(x)['snapshot_id']==case['snapshot']]
                if len(matches)!=1:raise ValueError('Cannot identify unique attempt')
                attempt=matches[0].parent;closed=read_json(attempt/'result.json')
                event('arm_closed',rate=case['peak_learning_rate'],attempt=attempt.name,status=closed['status'])
                if done.returncode or closed['status']!='passed':raise ValueError('Arm failed; queue stopped without numerical retries')
                # On this pod, user-owned /dev/shm files may disappear when the
                # last SSH session closes. Recover the committed peer copy and
                # protect only checkpoint files, within a live owner session.
                recovery=a.output/(label+'-cache-recovery.json')
                argv=[sys.executable,'-B',str(SOURCE/'packages/gozero/src/gozero/ram_checkpoints.py'),
                    '--attempt',str(attempt),'--output',str(recovery)]
                with (a.output/(label+'-cache-recovery.log')).open('xb') as f:
                    subprocess.run(['ssh','-F','/dev/null','-o','BatchMode=yes','go-user@worker-0.example.invalid',shlex.join(argv)],
                        cwd=root,env=env,stdout=f,stderr=subprocess.STDOUT,check=True,timeout=300)
                audit=a.output/(label+'-audit.json')
                with (a.output/(label+'-audit.log')).open('xb') as f:
                    subprocess.run(['taskset','-c','64-95',sys.executable,'-B',str(Path(__file__).with_name('audit_learning.py')),
                        '--workspace-root',str(root),'--attempt',str(attempt),'--output',str(audit)],cwd=root,env=cpu,stdout=f,stderr=subprocess.STDOUT,check=True)
                owner=read_json(attempt/'rank-0/artifacts/result.json');record=owner['latest_checkpoint']['archive']
                if sha256(Path(record['receipt']))!=record['receipt_sha256']:raise ValueError('Archive receipt changed')
                verified=remote(record['host'],'commit',{'manifest_sha256':record['manifest_sha256'],'token':'0'*32,'files':record['files']},
                    python=root/'.gozero/environments'/closed['runtime_key']/'bin/python')
                publication=a.output/(label+'-analysis')
                archive_audit=a.output/(label+'-archive-audit.json');publish(archive_audit,{'status':'passed','receipt_sha256':record['receipt_sha256'],'remote_verification':verified})
                audits.append((str(case['encoder_channels']),str(case['peak_learning_rate']),str(audit),sha256(audit)))
                argv=[str(root/'.gozero/analysis-environments/plotting/bin/python'),'-B',str(Path(__file__).with_name('analyze_capacity.py')),
                    '--workspace-root',str(root),'--registration',str(a.registration),'--registration-sha256',a.registration_sha256,'--output',str(publication)]
                for arm in audits:argv.extend(['--audit',*arm])
                with (a.output/(label+'-analysis.log')).open('xb') as f:subprocess.run(argv,cwd=root,env=cpu,stdout=f,stderr=subprocess.STDOUT,check=True)
                analysis=read_json(publication/'comparison.json')
                result['audits'].append({'rate':case['peak_learning_rate'],'attempt':attempt.name,'path':str(audit),'sha256':sha256(audit),
                    'archive_audit_sha256':sha256(archive_audit),'publication':str(publication),'publication_manifest_sha256':sha256(publication/'manifest.json')})
                event('arm_audited',rate=case['peak_learning_rate'],best_completed=analysis['best_completed'],pending=analysis['pending'],publication=str(publication))
            result['status']='passed'
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
