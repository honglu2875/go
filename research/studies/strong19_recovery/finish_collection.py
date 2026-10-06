"""Wait for the fixed quota, freeze the replacement corpus, and back it up.

Never selects a partial/adaptive learning population. The matched pair starts
only after all predeclared recovery gates, source checks and disk backups pass.
"""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[3]
STUDY=Path(__file__).parent
REPORT=ROOT/'ops/recovery_20260925'
sys.path.insert(0,str(ROOT/'packages/gozero/src'))
from gozero.disk_mirror import inventory,publish
from gozero.durable_files import atomic_json,sha256
from gozero.pod import load_hosts


def main():
    with (STUDY/'finish.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        registration=json.loads((STUDY/'collection-followup-001.json').read_text())
        for name,digest in registration['operators'].items():
            if sha256(ROOT/name)!=digest:raise ValueError('Followup operator changed: '+name)
        for name,digest in registration['prerequisites'].items():
            if sha256(ROOT/name)!=digest or json.loads((ROOT/name).read_text())['status']!='passed':
                raise ValueError('Followup prerequisite differs: '+name)
        if sha256(REPORT/'collection-plan-001.json')!=registration['collection_plan_sha256']:
            raise ValueError('Collection contract changed')
        atomic_json(STUDY/'collection-followup-status.json',dict(status='waiting_for_fixed_collection',updated=time.time()))
        deadline=registration['created']+7*86400
        while not (REPORT/'inventory-001.json').exists():
            if time.time()>deadline:raise TimeoutError('Collection did not finish within seven days')
            time.sleep(60)
        status=json.loads((REPORT/'collection-status.json').read_text())
        if status['status']!='complete' or status['total_terminal_games']!=3200:
            raise ValueError('Collection did not reach the registered balanced quota')
        inventory_path=REPORT/'inventory-001.json'
        python=registration['python'];plan=STUDY/'cohort-plan-001.json';result=STUDY/'cohort-result-001.json'
        def call(argv,label):
            with (STUDY/(label+'.log')).open('xb') as log:
                subprocess.run([python,'-B',str(STUDY/'prepare_cohort.py'),*argv],stdout=log,stderr=subprocess.STDOUT,
                    check=True,timeout=7200,env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1'))
        atomic_json(STUDY/'collection-followup-status.json',dict(status='packing',updated=time.time()))
        call(['select','--inventory',str(inventory_path),'--inventory-sha256',sha256(inventory_path),'--plan',str(plan)],'select-001')
        call(['prepare','--plan',str(plan),'--plan-sha256',sha256(plan),'--reserved-bytes',str(65<<30),'--output',str(result)],'pack-001')
        packed=json.loads(result.read_text());assert packed['status']=='passed'
        source=Path(packed['target']);files=inventory(source,[str(p.relative_to(source)) for p in source.rglob('*') if p.is_file()])
        peers=load_hosts(ROOT/'ops/hosts.json');copies=[]
        for h in (2,3):
            copies.append(publish(source,ROOT/'.gozero/datasets'/source.name,files=files,peer=peers[h].ssh,python=python))
        atomic_json(STUDY/'cohort-disk-backups-001.json',dict(status='passed',manifest_sha256=packed['manifest_sha256'],copies=copies),replace=False)
        atomic_json(STUDY/'collection-followup-status.json',dict(status='dataset_ready',updated=time.time(),
            manifest_sha256=packed['manifest_sha256'],games=packed['games'],positions=packed['positions'],
            next_action='Register and execute the fixed matched comparison after rechecking all recovery gates.'),replace=True)
        # Recheck the future operators after waiting, before making any learning
        # decision; edits during collection invalidate this registration.
        for name,digest in registration['operators'].items():
            if sha256(ROOT/name)!=digest:raise ValueError('Followup operator changed while waiting: '+name)
        with (STUDY/'prepare-pair-001.log').open('xb') as log:
            subprocess.run([python,'-B',str(STUDY/'prepare_pair.py')],stdout=log,stderr=subprocess.STDOUT,
                timeout=900,check=True,env=dict(os.environ,JAX_PLATFORMS='cpu',PYTHONDONTWRITEBYTECODE='1'))
        paired=STUDY/'registration-001.json'
        atomic_json(STUDY/'collection-followup-status.json',dict(status='running_registered_pair',updated=time.time(),registration_sha256=sha256(paired)))
        with (STUDY/'pair-controller-001.log').open('xb') as log:
            subprocess.run([python,'-B',str(STUDY/'execute_pair.py'),'--registration',str(paired),
                '--registration-sha256',sha256(paired),'--run'],stdout=log,stderr=subprocess.STDOUT,timeout=54000,check=True)
        atomic_json(STUDY/'collection-followup-status.json',dict(status='complete',updated=time.time(),registration_sha256=sha256(paired)))


if __name__=='__main__':
    try:main()
    except BaseException as error:
        atomic_json(STUDY/'collection-followup-error.json',dict(status='failed',error=repr(error),updated=time.time()))
        raise
