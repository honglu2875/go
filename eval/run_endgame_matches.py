#!/usr/bin/env python3
"""Execute registered fixed-model value-rescaling KataGo panels once."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

sys.dont_write_bytecode=True
SOURCE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(SOURCE/'packages/gozero/src'))
from gozero.checkpoints import sha256
from gozero.causal_artifacts import validate
from gozero.evaluation_inputs import verify as verify_inputs
from gozero.snapshots import canonical_json,read_json,verify


def require(value,message):
    if not value:raise ValueError(message)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--workspace-root',type=Path,required=True)
    p.add_argument('--protocol',type=Path,required=True);p.add_argument('--expected-protocol-sha256',required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();verify(SOURCE)
    require(sha256(a.protocol)==a.expected_protocol_sha256,'Registration changed');protocol=read_json(a.protocol);root=a.workspace_root.resolve()
    require(protocol['kind']=='endgame_rescaling_katago_ablation'and protocol['snapshot']==SOURCE.name
        and protocol['maximum_attempts']==1 and a.output.resolve()==root/protocol['output'],'Study identity differs')
    verify_inputs(SOURCE,protocol['evaluation_input_closure'])
    for name,expected in protocol['evaluation_code_sha256'].items():require(sha256(SOURCE/name)==expected,'Evaluation code changed')
    validate(root,read_json(SOURCE/protocol['candidate']))
    a.output=a.output.resolve();a.output.mkdir(parents=True,exist_ok=False)
    result={'schema_version':1,'kind':'endgame_katago_ablation_attempt','status':'running','snapshot':SOURCE.name,
        'protocol_sha256':a.expected_protocol_sha256,'started_unix':time.time(),'panels':[]}
    try:
        for panel in protocol['panels']:
            require(time.time()-result['started_unix']+panel['maximum_seconds']<protocol['maximum_seconds'],'Remaining budget cannot cover panel')
            path=SOURCE/panel['path'];require(sha256(path)==panel['sha256'],'Panel specification changed')
            command=[sys.executable,'-B',str(SOURCE/'eval/panel.py'),'--spec',str(path),'--artifacts-root',str(root),'--output',str(a.output/panel['id'])]
            print(json.dumps({'kind':'endgame_panel_start','panel':panel['id']}),flush=True)
            with (a.output/(panel['id']+'.log')).open('x')as log:
                child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT)
                try:code=child.wait(timeout=panel['maximum_seconds'])
                except BaseException:
                    child.terminate()
                    try:child.wait(timeout=15)
                    except subprocess.TimeoutExpired:child.kill();child.wait()
                    raise
            path=a.output/panel['id']/'result.json';data=read_json(path);summaries=list(data['summaries'].values())
            require(code in (0,1)and sum(s['scheduled_games']for s in summaries)==panel['games']
                and not any(s['failed_games']for s in summaries)and 'error'not in data
                and not any(m['timed_out']or not m['result_sha256']for m in data['matches']),'Panel process/integrity failure; no retry')
            result['panels'].append({'id':panel['id'],'returncode':code,'result_sha256':sha256(path),'summaries':data['summaries']})
            (a.output/'progress.json').write_bytes(canonical_json(result));print(json.dumps(result['panels'][-1]),flush=True)
        result['status']='passed'
    except BaseException as e:result.update(status='failed',error=repr(e));raise
    finally:
        result['finished_unix']=time.time();(a.output/'result.json').write_bytes(canonical_json(result))
        verify(SOURCE);require(sha256(a.protocol)==a.expected_protocol_sha256,'Protocol changed')
        print(json.dumps(result))


if __name__=='__main__':main()
