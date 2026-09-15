"""Bounded CPU analysis of the registered main-only CNN after successful closure."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
SOURCE=Path(__file__).resolve().parents[3];RECIPE=Path(__file__).resolve().parent
sys.path.insert(0,str(SOURCE/'packages/gozero/src'))
from gozero.snapshots import verify,read_json,canonical_json
from gozero.checkpoints import sha256


def main():
    p=argparse.ArgumentParser();p.add_argument('--workspace-root',type=Path,required=True)
    p.add_argument('--attempt',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();verify(SOURCE);root=a.workspace_root.resolve();a.output.mkdir(parents=True,exist_ok=False)
    candidate='df4516c5ffc6b0c2595c241b52dcc472ae0b641152b3c065a5bc3c81f47aee1e'
    reference=root/'runs/cnn-lr-followup-f3de116f/sweep/2-1e-03-audit.json'
    expected='38d3b55a36dad73d60307a1a97ff2cbe4b5e7ca407dd3cf6cd00dfaf23861bb7'
    if sha256(reference)!=expected or read_json(a.attempt/'launch.json')['snapshot_id']!=candidate:raise ValueError('Registered input differs')
    env={**os.environ,'JAX_PLATFORMS':'cpu','OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1',
         'MPLCONFIGDIR':'/tmp/gozero-spatial-followups-mpl','PYTHONDONTWRITEBYTECODE':'1'}
    python=root/'.venv/bin/python';plotting=root/'.gozero/analysis-environments/plotting/bin/python'
    commands=[]
    def run(name,exe,script,args):
        cmd=[str(exe),'-B',str(RECIPE/script),*map(str,args)];commands.append(cmd)
        print(json.dumps({'phase':name,'status':'starting'}),flush=True)
        with (a.output/(name+'.log')).open('xb') as f:subprocess.run(cmd,cwd=root,env=env,stdout=f,stderr=subprocess.STDOUT,check=True,timeout=600)
        print((a.output/(name+'.log')).read_text(),end='',flush=True)
    started=time.time()
    try:
        deadline=time.monotonic()+7300
        while not (a.attempt/'result.json').exists():
            if time.monotonic()>deadline:raise TimeoutError('Learning attempt did not close')
            time.sleep(10)
        closed=read_json(a.attempt/'result.json')
        if closed['status']!='passed' or closed['snapshot_id']!=candidate:raise ValueError('Learning did not pass')
        audit=a.output/'learning-audit.json'
        run('audit',python,'audit_learning.py',['--workspace-root',root,'--attempt',a.attempt,'--output',audit])
        comparison=a.output/'comparison'
        run('compare',plotting,'report_comparison.py',['--workspace-root',root,
            '--audit','CNN with helper',reference,expected,'--audit','CNN main only',audit,sha256(audit),'--output',comparison])
        contrast=a.output/'contrast.json'
        run('contrast',python,'compare_main.py',['--workspace-root',root,'--comparison',comparison/'comparison.json',
            '--comparison-sha256',sha256(comparison/'comparison.json'),'--output',contrast])
        result={'status':'passed','audit':str(audit),'audit_sha256':sha256(audit),'contrast':str(contrast),
                'contrast_sha256':sha256(contrast),'comparison':str(comparison/'comparison.json')}
    except BaseException as error:
        result={'status':'failed','error':repr(error)}
        raise
    finally:
        result.update(operator_snapshot=SOURCE.name,started_unix=started,ended_unix=time.time(),commands=commands,automatic_retry=False)
        with (a.output/'result.json').open('xb') as f:f.write(canonical_json(result))
        for path in a.output.iterdir():
            if path.is_file():path.chmod(0o444)
        print(json.dumps(result),flush=True)


if __name__=='__main__':main()
