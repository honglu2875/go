#!/usr/bin/env python3
"""Run one registered board-input arm's real KataGo panels without retries."""
import argparse
import fcntl
import json
from pathlib import Path
import subprocess
import sys
import time

sys.dont_write_bytecode=True
SOURCE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(SOURCE/'packages/gozero/src'))
from gozero.causal_artifacts import validate
from gozero.checkpoints import sha256
from gozero.evaluation_inputs import verify as verify_inputs
from gozero.snapshots import canonical_json,read_json,verify


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workspace-root',type=Path,required=True);p.add_argument('--protocol',type=Path,required=True)
    p.add_argument('--expected-protocol-sha256',required=True);p.add_argument('--arm',choices=['empty','exact'],required=True)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args();verify(SOURCE)
    if sha256(args.protocol)!=args.expected_protocol_sha256:raise ValueError('Board study registration differs')
    protocol=read_json(args.protocol);root=args.workspace_root.resolve()
    if protocol['kind']!='board_state_distillation_pilot':raise ValueError('Wrong study kind')
    for name,expected in protocol['evaluation_code_sha256'].items():
        if sha256(SOURCE/name)!=expected:raise ValueError('Registered evaluator changed: '+name)
    verify_inputs(SOURCE,protocol['evaluation_input_closure'])
    arm=protocol['arms'][args.arm];descriptor_path=SOURCE/arm['descriptor'];descriptor=read_json(descriptor_path)
    trained=validate(root,descriptor)
    if descriptor['kind']!='board_causal_history_policy' or descriptor['training_snapshot']!=arm['snapshot_id'] or trained['config']['model']['board_mode']!=args.arm:
        raise ValueError('Evaluation candidate differs from registered training arm')
    args.output=args.output.resolve();args.output.mkdir(parents=True,exist_ok=False)
    report={'schema_version':1,'kind':'board_state_katago_suite','status':'running','snapshot_id':SOURCE.name,
            'protocol_sha256':args.expected_protocol_sha256,'arm':args.arm,'candidate_sha256':sha256(descriptor_path),
            'model_export_sha256':descriptor['model_export_sha256'],'started_unix':time.time(),'panels':[]}
    try:
        with (args.output/'.operator.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            for item in arm['evaluation_panels']:
                c=read_json(SOURCE/item['path']);children=[read_json(SOURCE/m['spec']) for m in c['matches']]
                bound=max(2*len(child['openings'])*child['game_timeout_seconds'] for child in children)+180
                if time.time()-report['started_unix']+bound>protocol['maximum_eval_seconds_per_arm']:
                    raise TimeoutError('Remaining study budget cannot cover the bounded panel')
                if sha256(SOURCE/item['path'])!=item['sha256']:raise ValueError('Panel specification differs')
                output=args.output/item['id'];command=[sys.executable,'-B',str(SOURCE/'eval/panel.py'),'--spec',str(SOURCE/item['path']),'--artifacts-root',str(root),'--output',str(output)]
                print(json.dumps({'kind':'board_state_panel_start','arm':args.arm,'id':item['id']}),flush=True)
                with (args.output/(item['id']+'.log')).open('x') as log:
                    child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT)
                    try:code=child.wait(timeout=bound)
                    except BaseException:
                        child.terminate()
                        try:child.wait(timeout=60)
                        except subprocess.TimeoutExpired:child.kill();child.wait()
                        raise
                path=output/'result.json';result=read_json(path);summaries=list(result['summaries'].values())
                entry={'id':item['id'],'spec_sha256':item['sha256'],'result_sha256':sha256(path),'returncode':code,
                       'elapsed_seconds':result['finished_unix']-result['started_unix'],'summaries':result['summaries']}
                report['panels'].append(entry);(args.output/'progress.json').write_bytes(canonical_json(report))
                print(json.dumps({'kind':'board_state_panel_finished',**entry}),flush=True)
                if code not in (0,1) or sum(s['scheduled_games'] for s in summaries)!=item['games'] or any(s['failed_games'] for s in summaries) or 'error' in result or any(m['timed_out'] or not m['result_sha256'] for m in result['matches']):
                    raise RuntimeError('Evaluation process/integrity failure; records retained, no retry')
                verify(SOURCE)
        if sha256(args.protocol)!=args.expected_protocol_sha256:raise ValueError('Registration changed during evaluation')
        report['status']='passed'
    except BaseException as e:report.update(status='failed',error=repr(e));raise
    finally:
        report['finished_unix']=time.time();(args.output/'result.json').write_bytes(canonical_json(report))


if __name__=='__main__':main()
