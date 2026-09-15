"""Run original-source checks and independent fresh-process state continuation."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'packages/gozero/src'))
from gozero import checkpoints
from gozero.snapshots import read_json,verify


def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',type=Path,required=True)
    p.add_argument('--plan-sha256',required=True);p.add_argument('--python',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    sha=checkpoints.sha256
    if a.output.exists():raise FileExistsError(a.output)
    if sha(a.plan)!=a.plan_sha256:raise ValueError('Plan changed')
    plan=read_json(a.plan)
    if sha(Path(__file__))!=plan['operator_sha256']:raise ValueError('Operator changed')
    source=ROOT/'.gozero/snapshots'/plan['snapshot_id'];manifest=verify(source)
    folder=ROOT/plan['run_directory'];folder.mkdir(exist_ok=False)
    started=time.monotonic();report=dict(kind='lookahead_cpu_continuation_qualification',status='running',
        created=time.time(),snapshot=source.name,plan_sha256=a.plan_sha256,stages={})
    try:
        for mode,extra in (('full',[]),('prefix',['--stop-after-events','12']),('resumed',['--resume',str(folder/'prefix')])):
            argv=[str(a.python),'-B',str(source/manifest['recipe']/'train.py'),'--workspace-root',str(ROOT),
                '--config',str(source/'resolved_config.json'),'--output',str(folder/mode),*extra]
            env=dict(os.environ,JAX_PLATFORMS='cpu',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
            print(json.dumps(dict(stage=mode,status='starting')),flush=True)
            with (folder/(mode+'.log')).open('xb') as log:
                process=subprocess.run(argv,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=180)
            path=folder/mode/'result.json'
            report['stages'][mode]=dict(returncode=process.returncode,result_sha256=sha(path) if path.exists() else None,
                log_sha256=sha(folder/(mode+'.log')))
            if process.returncode or read_json(path)['status']!='passed':raise ValueError('Stage failed: '+mode)
        import numpy as np
        full=read_json(folder/'full/result.json');prefix=read_json(folder/'prefix/result.json');resumed=read_json(folder/'resumed/result.json')
        if not full['all_events_complete'] or prefix['all_events_complete'] or not resumed['all_events_complete']:
            raise ValueError('Wrong continuation completion boundary')
        matched=[]
        for left,middle,right in zip(full['cases'],prefix['cases'],resumed['cases']):
            if middle['cursor']!=12 or right['initial_cursor']!=12 or left['cursor']!=right['cursor']:
                raise ValueError('Wrong saved/continued event')
            sa,aa,_=checkpoints.read(Path(left['checkpoint']),expected_manifest_sha256=left['checkpoint_manifest_sha256'])
            sb,bb,_=checkpoints.read(Path(right['checkpoint']),expected_manifest_sha256=right['checkpoint_manifest_sha256'])
            if sa!=sb or set(aa)!=set(bb):raise ValueError('Complete state differs')
            for key in aa:np.testing.assert_array_equal(aa[key],bb[key],err_msg=key)
            if left['checkpoint_manifest_sha256']!=right['checkpoint_manifest_sha256']:
                raise ValueError('Continuation manifest differs')
            boundary,_,_=checkpoints.read(Path(middle['checkpoint']),expected_manifest_sha256=middle['checkpoint_manifest_sha256'])
            matched.append(dict(case=left['case'],config=left['config'],checkpoint_arrays=len(aa),
                boundary_counter=boundary['lookahead']['counter'],full_events=left['cursor'],
                checkpoint_manifest_sha256=left['checkpoint_manifest_sha256'],
                reference_comparisons=left['comparisons'],maximum_absolute_error=left['maximum_absolute_error']))
        if matched[-1]['boundary_counter']!=5:raise ValueError('Required unsynchronized checkpoint was not exercised')
        report.update(status='passed',cases=matched,all_fast_slow_arrays_exact=True,all_clocks_exact=True,
            scope='Four source-reference configurations, donated JAX CPU execution and exact fresh-process continuation across a mid-cycle checkpoint. Synthetic fast-optimizer increments; no neural learner or TPU integration.')
    except BaseException as error:
        report.update(status='failed',error=repr(error));raise
    finally:
        report['seconds']=time.monotonic()-started
        with a.output.open('x') as f:json.dump(report,f,indent=2);f.write('\n')
        a.output.chmod(0o444)
        print(json.dumps(dict(status=report['status'],seconds=report['seconds'],sha256=sha(a.output))),flush=True)


if __name__=='__main__':main()
