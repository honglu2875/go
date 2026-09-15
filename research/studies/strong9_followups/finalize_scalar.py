"""Audit, replicate and contrast one completed registered scalar intervention."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from scalar_compare import read,sha
import scalar_source

ROOT=Path(__file__).resolve().parents[3]
STUDY=Path(__file__).resolve().parent


def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',type=Path,required=True)
    p.add_argument('--plan-sha256',required=True);p.add_argument('--stage',choices=('seed1','seed2'),required=True)
    p.add_argument('--inspect',action='store_true');a=p.parse_args()
    if sha(a.plan)!=a.plan_sha256:raise ValueError('Scalar plan changed')
    plan=read(a.plan)
    if plan['kind']!='strong9_scalar_intervention' or sha(Path(__file__))!=plan['finalize_operator_sha256']:
        raise ValueError('Wrong finalization plan/operator')
    if sha(STUDY/'scalar_compare.py')!=plan['comparison_operator_sha256'] or sha(Path(scalar_source.__file__))!=plan['source_helper_sha256']:
        raise ValueError('Comparison source changed')
    folder=STUDY/plan['trial'];receipt=read(folder/(a.stage+'-process-001.json'))
    item=plan['stages'][a.stage];source=ROOT/'.gozero/snapshots'/item['snapshot']
    manifest=scalar_source.verify(source)
    if receipt['snapshot']!=source.name or receipt['plan_sha256']!=a.plan_sha256 or receipt['stage']!=a.stage:
        raise ValueError('Wrong stage launch receipt')
    log=folder/(a.stage+'-controller-001.log')
    if receipt['log']!=str(log):raise ValueError('Unexpected controller log')
    attempt=None
    for line in log.read_text().splitlines()[:10]:
        record=json.loads(line)
        if record.get('kind')=='pod_attempt':
            attempt=Path(record['attempt'])
            if attempt.parent!=ROOT/'runs' or record['snapshot_id']!=source.name or attempt.name!=record['attempt_id']:
                raise ValueError('Wrong launched attempt')
            break
    if attempt is None:raise ValueError('Launch has not published its attempt')
    launch=read(attempt/'launch.json')
    if launch['snapshot_id']!=source.name or not -30<=launch['start_unix_time']-receipt['started']<=180:
        raise ValueError('Attempt does not belong to this stage')
    closed=attempt/'result.json'
    if a.inspect:
        print(json.dumps(dict(stage=a.stage,attempt=attempt.name,status=read(closed)['status'] if closed.exists() else 'running',mutations=False)));return
    if not closed.exists() or read(closed)['status']!='passed':raise ValueError('Attempt is not successfully closed')
    env=dict(os.environ,JAX_PLATFORMS='cpu',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
    def invoke(argv,name):
        path=folder/(a.stage+'-'+name+'.log')
        if path.exists():raise FileExistsError('Prior finalization command exists; inspect it before recovery')
        with path.open('xb') as stream:
            subprocess.run([sys.executable,'-B',*map(str,argv)],cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,check=True,timeout=900)
    replica=folder/(a.stage+'-ram-replicas-001.json')
    if not replica.exists():
        invoke([source/'ops/replicate_ram_checkpoint.py','--workspace-root',ROOT,'--attempt',attempt.name,
                '--peer','2','--output',replica],'replica-001')
    audit=folder/(a.stage+'-audit-001.json')
    if not audit.exists():
        invoke([source/manifest['recipe']/'audit_learning.py','--workspace-root',ROOT,'--attempt',attempt,'--output',audit],'audit-001')
    for path in (replica,audit):
        value=read(path)
        if (value['status']!='passed' or value['attempt']!=attempt.name
                or value.get('training_snapshot',value.get('snapshot'))!=source.name):
            raise ValueError('Wrong completed-stage evidence')
    contrast=folder/(a.stage+'-contrast-001.json')
    if not contrast.exists():
        invoke([STUDY/'scalar_compare.py','--plan',a.plan,'--plan-sha256',a.plan_sha256,'--stage',a.stage,
                '--candidate-audit',audit,'--candidate-audit-sha256',sha(audit),'--output',contrast],'contrast-001')
    result=read(contrast)
    if result['status']!='passed' or result['plan_sha256']!=a.plan_sha256 or result['candidate']['snapshot']!=source.name:
        raise ValueError('Wrong Scalar contrast')
    print(json.dumps(dict(status='passed',stage=a.stage,attempt=attempt.name,screen_passed=result['screen_passed'],
        audit_sha256=sha(audit),replica_sha256=sha(replica),contrast_sha256=sha(contrast))))


if __name__=='__main__':main()
