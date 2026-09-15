"""Run one registered SGD attempt, audit it, publish results and update costs."""
import argparse
from datetime import datetime, timezone
import fcntl
import os
from pathlib import Path
import subprocess
import sys
SOURCE = Path(__file__).resolve().parents[3]
sys.path.insert(0,str(SOURCE/'packages/gozero/src'))
from gozero.snapshots import canonical_json, read_json, verify
from gozero.checkpoints import sha256
from train_sgd import publish


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--workspace-root',type=Path,required=True)
    p.add_argument('--registration',type=Path,required=True)
    p.add_argument('--registration-sha256',required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--check-only',action='store_true')
    a=p.parse_args();verify(SOURCE);root=a.workspace_root.resolve()
    if root!=Path('/workspace/go') or sha256(a.registration)!=a.registration_sha256:raise ValueError('Wrong workspace or registration')
    reg=read_json(a.registration)
    if reg['training_snapshot']!=SOURCE.name:raise ValueError('Wrong training snapshot')
    for name,digest in reg['prerequisite_files'].items():
        if sha256(root/name)!=digest:raise ValueError('Prerequisite changed: '+name)
    if read_json(root/reg['preflight_path'])['status']!='passed':raise ValueError('Failed preflight')
    if a.check_only:
        print(canonical_json({'status':'checked','snapshot':SOURCE.name}).decode(),flush=True);return
    a.output.mkdir(parents=True,exist_ok=False)
    env={**os.environ,'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','PYTHONDONTWRITEBYTECODE':'1'}
    cpu_env={**env,'JAX_PLATFORMS':'cpu'}
    report={'kind':'registered_historical_sgd_execution','status':'running','snapshot':SOURCE.name,
            'registration_sha256':a.registration_sha256,'started_at':datetime.now(timezone.utc).isoformat()}
    try:
        with (root/'runs/.registered-lr-sweep.lock').open('a+b') as lock:
            fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
            launches=list((root/'runs').glob('pod-*/launch.json'))
            if any(not x.with_name('result.json').exists() for x in launches):raise ValueError('Another pod attempt is open')
            if any(read_json(x)['snapshot_id']==SOURCE.name for x in launches):raise ValueError('SGD already has an attempt; no automatic retry')
            disk=os.statvfs(root)
            if disk.f_bavail*disk.f_frsize<reg['minimum_writer_free_bytes']:raise ValueError('Insufficient room for reserved checkpoint and logs')
            command=[sys.executable,'-B',str(SOURCE/'ops/pod_run.py'),'--snapshot',str(SOURCE),'--workspace-root',str(root),
                     '--timeout',str(reg['maximum_seconds_per_attempt']),'--prepare-timeout','180','--controller-cpus','32']
            with (a.output/'pod.log').open('xb') as log:
                result=subprocess.run(command,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT)
            matches=[x for x in (root/'runs').glob('pod-*/launch.json') if read_json(x)['snapshot_id']==SOURCE.name]
            if len(matches)!=1:raise ValueError('Cannot identify the unique SGD attempt')
            attempt=matches[0].parent;closed=read_json(attempt/'result.json')
            report.update(attempt=attempt.name,closed_result_sha256=sha256(attempt/'result.json'))
            if result.returncode or closed['status']!='passed':raise ValueError('SGD stopped; preserve failure for diagnosis without numerical retries')
            audit=a.output/'audit.json';reference=reg['comparison_references'][0]
            with (a.output/'audit.log').open('xb') as log:
                subprocess.run(['taskset','-c','64-95',sys.executable,'-B',str(Path(__file__).with_name('audit_sgd.py')),
                    '--workspace-root',str(root),'--reference-audit',str(root/reference['audit_path']),
                    '--reference-audit-sha256',reference['audit_sha256'],'--attempt',str(attempt),
                    '--preflight',str(root/reg['preflight_path']),'--preflight-sha256',reg['preflight_sha256'],
                    '--output',str(audit)],cwd=root,env=cpu_env,stdout=log,stderr=subprocess.STDOUT,check=True)
            report['audit_sha256']=sha256(audit)
            with (a.output/'publication.log').open('xb') as log:
                subprocess.run([str(root/'.gozero/analysis-environments/plotting/bin/python'),'-B',str(Path(__file__).with_name('publish_results.py')),
                    '--workspace-root',str(root),'--registration',str(a.registration),'--registration-sha256',a.registration_sha256,
                    '--sgd-audit',str(audit),'--output',str(root/reg['publication_directory'])],cwd=root,env=cpu_env,
                    stdout=log,stderr=subprocess.STDOUT,check=True)
            report['publication_manifest_sha256']=sha256(root/reg['publication_directory']/'manifest.json')
            report['status']='passed'
    except BaseException as error:
        report.update(status='failed',error=repr(error));raise
    finally:
        if not any(not x.with_name('result.json').exists() for x in (root/'runs').glob('pod-*/launch.json')):
            ledger=root/'research/studies/runtime_qualification/reservation_ledger.json'
            with (a.output/'ledger.log').open('xb') as log:
                result=subprocess.run([sys.executable,'-B',str(SOURCE/'ops/update_reservation_ledger.py'),
                    '--workspace-root',str(root),'--expected-previous-sha256',sha256(ledger)],cwd=root,env=env,
                    stdout=log,stderr=subprocess.STDOUT)
            report['ledger_update_returncode']=result.returncode
            if result.returncode==0:report['ledger_sha256']=sha256(ledger)
        report['finished_at']=datetime.now(timezone.utc).isoformat();publish(a.output/'result.json',report)
        print(canonical_json(report).decode(),flush=True)

if __name__=='__main__':main()
