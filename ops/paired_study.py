#!/usr/bin/env python3
"""Run a registered sequence of bounded, interleaved, frozen pod experiments."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
sys.dont_write_bytecode=True
SOURCE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(SOURCE/'packages/gozero/src'))
from gozero.snapshots import canonical_json,read_json,verify
from gozero.checkpoints import sha256


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workspace-root',type=Path,required=True);p.add_argument('--spec',type=Path,required=True)
    p.add_argument('--spec-sha256',required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();verify(SOURCE)
    if sha256(args.spec)!=args.spec_sha256:raise ValueError('Study specification hash mismatch')
    spec=read_json(args.spec);root=args.workspace_root.resolve();output=args.output.resolve();output.mkdir(parents=True,exist_ok=False)
    variants={key:spec[key+'_snapshot'] for key in ('baseline','candidate')}
    runner=root/'ops/pod_run.py'
    runner_sha=sha256(runner)
    for snapshot in variants.values():
        manifest=verify(root/'.gozero/snapshots'/snapshot)
        if manifest['files']['ops/pod_run.py']['sha256']!=runner_sha:raise ValueError('Controller differs from the frozen experiment source')
    if not 1<=spec['measured_pairs']<=3 or not 1<=spec['timeout_seconds_per_attempt']<=180:raise ValueError('Study exceeds the bounded systems-screen contract')
    result={'schema_version':1,'controller_snapshot':SOURCE.name,'controller_runner_sha256':runner_sha,'spec_sha256':args.spec_sha256,
            'status':'running','started_unix':time.time(),'executions':[]}
    try:
        for pair in range(spec['measured_pairs']):
            for variant,snapshot in variants.items():
                if sha256(runner)!=runner_sha or sha256(args.spec)!=args.spec_sha256:raise ValueError('Study inputs changed during execution')
                argv=[sys.executable,'-B',str(runner),'--snapshot',str(root/'.gozero/snapshots'/snapshot),
                      '--native-receipt',str(root/'.gozero/native'/snapshot/'receipt.json'),'--timeout',str(spec['timeout_seconds_per_attempt']),
                      '--prepare-timeout','180','--controller-cpus','32']
                logfile=output/f'{pair}-{variant}.log'
                print(json.dumps({'kind':'study_start','pair':pair,'variant':variant}),flush=True)
                with logfile.open('w') as log:
                    completed=subprocess.run(argv,cwd=root,stdout=log,stderr=subprocess.STDOUT,timeout=spec['timeout_seconds_per_attempt']+480)
                events=[]
                for line in logfile.read_text().splitlines():
                    try:events.append(json.loads(line))
                    except ValueError:pass
                attempts=[x for x in events if x.get('kind')=='pod_attempt']
                if len(attempts)!=1:raise ValueError('Cannot identify pod execution attempt')
                attempt=root/'runs'/attempts[0]['attempt_id'];record=read_json(attempt/'result.json')
                ranks=[read_json(attempt/f'rank-{host}/artifacts/result.json') for host in range(4)]
                entry={'pair':pair,'variant':variant,'snapshot_id':snapshot,'attempt':attempt.name,'returncode':completed.returncode,
                       'pod_result_sha256':sha256(attempt/'result.json'),'slowest_training_seconds':max(r['elapsed_segment_seconds'] for r in ranks),
                       'slowest_inference_seconds':max(r['counters']['inference_seconds'] for r in ranks),
                       'slowest_native_seconds':max(r['counters']['native_seconds'] for r in ranks),
                       'model_sha256':ranks[0]['model_export_sha256'],'global_active_neural_evaluations':sum(r['counters']['active_neural_evaluations'] for r in ranks),
                       'global_neural_slots':sum(r['counters']['neural_slots'] for r in ranks),'reserved_attempt_chip_hours':record['reserved_chip_hours']}
                result['executions'].append(entry)
                (output/'progress.json').write_bytes(canonical_json(result))
                if completed.returncode or record['status']!='passed' or any(r['status']!='passed' for r in ranks):raise RuntimeError('A registered experiment failed; retained in ledger')
                if len({r['model_export_sha256'] for r in ranks})!=1:raise RuntimeError('Host model exports differ')
                print(json.dumps({'kind':'study_finished',**entry}),flush=True)
        result['status']='passed'
        verify(SOURCE)
    except BaseException as error:
        result.update(status='failed',error=repr(error));raise
    finally:
        result['finished_unix']=time.time();(output/'result.json').write_bytes(canonical_json(result))
        print(json.dumps({'kind':'study_result','status':result['status'],'completed_executions':len(result['executions'])}),flush=True)


if __name__=='__main__':main()
