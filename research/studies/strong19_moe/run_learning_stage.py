"""One immutable screen followed by its full audit and dense-prefix comparison."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[3];STUDY=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'packages/gozero/src'))
from gozero.snapshots import canonical_json

def main():
    p=argparse.ArgumentParser();p.add_argument('--label',required=True);p.add_argument('--plan-sha256',required=True);a=p.parse_args()
    plan=STUDY/(a.label+'-plan.json')
    if hashlib.sha256(plan.read_bytes()).hexdigest()!=a.plan_sha256:raise ValueError('Plan differs')
    log=STUDY/(a.label+'-controller.log');started=time.time();result=dict(status='running',started=started,plan_sha256=a.plan_sha256)
    env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
    try:
        with log.open('xb') as stream:
            subprocess.run([sys.executable,'-B',str(STUDY/'execute_run.py'),'--plan',str(plan),'--plan-sha256',a.plan_sha256,'--run'],cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,check=True,timeout=14500)
            subprocess.run([sys.executable,'-B',str(STUDY/'compare_learning.py'),'--label',a.label],cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,check=True,timeout=300)
        result['status']='passed'
    except BaseException as error:
        result.update(status='failed',error=repr(error));raise
    finally:
        result['finished']=time.time()
        with (STUDY/(a.label+'-sequence.json')).open('xb') as stream:stream.write(canonical_json(result));stream.flush();os.fsync(stream.fileno())

if __name__=='__main__':main()
