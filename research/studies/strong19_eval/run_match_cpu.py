"""Run the two frozen trained-model KataGo CPU qualifications sequentially."""
import argparse
import hashlib
import json
import os
import re
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[3]
STUDY=Path(__file__).resolve().parent


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',type=Path,required=True);p.add_argument('--sha256',required=True)
    a=p.parse_args()
    if sha(a.plan)!=a.sha256:raise ValueError('Plan changed')
    plan=json.loads(a.plan.read_text())
    if plan['operator_sha256']!=sha(Path(__file__)):raise ValueError('Controller changed')
    case=plan['case']
    if not isinstance(case,str) or re.fullmatch(r'[0-9]{3}',case) is None:raise ValueError('Invalid qualification case')
    out=STUDY/('match-cpu-'+case);out.mkdir(exist_ok=False)
    result_path=STUDY/('match-cpu-result-'+case+'.json')
    if result_path.exists():raise FileExistsError('Qualification result exists')
    started=time.monotonic();report=dict(kind='trained_joint_katago_cpu_pair',status='running',created=time.time(),plan_sha256=a.sha256,models={})
    try:
        for arm in ('cnn','transformer'):
            source=ROOT/'.gozero/snapshots'/plan['snapshots'][arm]
            command=[str(ROOT/plan['python']),'-B',str(source/'research/recipes/strong19_eval/train.py'),
                     '--config',str(source/'resolved_config.json'),'--workspace-root',str(ROOT),'--output',str(out/arm)]
            env=dict(os.environ,JAX_PLATFORMS='cpu',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',
                     XLA_FLAGS='--xla_force_host_platform_device_count=4 --xla_cpu_multi_thread_eigen=false')
            with (out/(arm+'.log')).open('x') as log:
                done=subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,env=env,timeout=2100)
            path=out/arm/'result.json'
            if done.returncode or not path.exists():raise RuntimeError(arm+' qualification failed; inspect its retained log/result')
            record=json.loads(path.read_text())
            if record['status']!='passed' or record['snapshot']!=source.name:
                raise ValueError('Unsuccessful or foreign qualification')
            report['models'][arm]=dict(path=str(path.relative_to(ROOT)),sha256=sha(path),
                                      summary=record['summary'],checked_positions=record['checked_positions'],comparisons=len(record['comparisons']),
                                      seconds=record['seconds'],maximum_absolute_errors=record['maximum_absolute_errors'])
            print(json.dumps(dict(architecture=arm,status='passed',seconds=record['seconds'])),flush=True)
        report['status']='passed'
    except BaseException as error:
        report.update(status='failed',error=repr(error));raise
    finally:
        report['seconds']=time.monotonic()-started
        report['scope']='Small trained19 models against real KataGo with exact board/score and selected full JAX prediction checks. No architecture strength claim or TPU timing.'
        with result_path.open('x') as stream:json.dump(report,stream,indent=2);stream.write('\n')
        result_path.chmod(0o444)
        print(json.dumps(dict(status=report['status'],seconds=report['seconds'],sha256=sha(result_path))),flush=True)


if __name__=='__main__':main()
