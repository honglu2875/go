#!/usr/bin/env python3
"""Verify equal-work paired runs and summarize the registered systems measurements."""
import argparse
import json
from pathlib import Path
import statistics
import sys
sys.dont_write_bytecode=True
SOURCE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(SOURCE/'packages/gozero/src'))
from gozero import checkpoints
from gozero.snapshots import canonical_json,read_json,verify


def compare(root,a,b):
    import numpy as np
    reports=[]
    for host in range(4):
        relative=Path(f'rank-{host}/artifacts');left=root/'runs'/a/relative;right=root/'runs'/b/relative
        ra=read_json(left/'result.json');rb=read_json(right/'result.json')
        if ra['status']!='passed' or rb['status']!='passed' or ra['config_sha256']!=rb['config_sha256']:
            raise ValueError('Run status or scientific configuration differs')
        sa,aa,ga=checkpoints.read(left/'checkpoints/turn-000000512',expected_manifest_sha256=ra['latest_checkpoint']['manifest_sha256'])
        sb,ab,gb=checkpoints.read(right/'checkpoints/turn-000000512',expected_manifest_sha256=rb['latest_checkpoint']['manifest_sha256'])
        if aa.keys()!=ab.keys():raise ValueError('Checkpoint array trees differ')
        for k in aa:np.testing.assert_array_equal(aa[k],ab[k],err_msg=f'host {host} {k}')
        ga=json.loads(ga);gb=json.loads(gb)
        if ga['actors']!=gb['actors'] or ga['config']!=gb['config']:raise ValueError('Actor configuration or saved states differ')
        for k in ('turn','learner_ready','replay_count','replay_cursor','numpy_rng','last_metrics','model_schema'):
            if sa[k]!=sb[k]:raise ValueError('Learner state differs: '+k)
        for k in ('real_moves','completed_games','truncated_games','eligible_rows','updates','active_neural_evaluations'):
            if sa['counters'][k]!=sb['counters'][k]:raise ValueError('Work counters differ: '+k)
        files={p.name for p in (left/'games').iterdir()}
        if files!={p.name for p in (right/'games').iterdir()}:raise ValueError('Game sets differ')
        for name in files:
            if (left/'games'/name).read_bytes()!=(right/'games'/name).read_bytes():raise ValueError('Game record differs: '+name)
        reports.append({'host':host,'arrays_exact':len(aa),'game_artifacts_exact':len(files),'all_actor_and_learner_states_exact':True})
    return reports


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--workspace-root',type=Path,required=True)
    p.add_argument('--study-result',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();verify(SOURCE);study=read_json(args.study_result)
    if study['status']!='passed' or len(study['executions'])!=6:raise ValueError('Registered experiment sequence incomplete')
    comparisons=[]
    for pair in range(3):
        a=next(x for x in study['executions'] if x['pair']==pair and x['variant']=='baseline')
        b=next(x for x in study['executions'] if x['pair']==pair and x['variant']=='candidate')
        for e in (a,b):
            if Path(e['attempt']).name!=e['attempt'] or checkpoints.sha256(args.workspace_root/'runs'/e['attempt']/'result.json')!=e['pod_result_sha256']:
                raise ValueError('Attempt identity differs from the controller ledger')
        hosts=compare(args.workspace_root,a['attempt'],b['attempt'])
        comparisons.append({'pair':pair,'baseline':a,'candidate':b,'equivalence':hosts,
                            'training_speedup':a['slowest_training_seconds']/b['slowest_training_seconds'],
                            'inference_speedup':a['slowest_inference_seconds']/b['slowest_inference_seconds']})
        print(json.dumps({'kind':'compared','pair':pair,'all_saved_arrays_and_games_exact':True}),flush=True)
    speeds=[c['training_speedup'] for c in comparisons]
    result={'schema_version':1,'status':'passed','analysis_snapshot':SOURCE.name,'study_result_sha256':checkpoints.sha256(args.study_result),
            'spec_sha256':study['spec_sha256'],'comparisons':comparisons,'median_paired_training_speedup':statistics.median(speeds),
            'paired_training_speedup_range':[min(speeds),max(speeds)],'median_paired_inference_speedup':statistics.median(c['inference_speedup'] for c in comparisons),
            'measured_attempt_chip_hours':sum(e['reserved_attempt_chip_hours'] for e in study['executions']),
            'claims_sample_efficiency':False,'claims_mfu':False,'limitations':'Three sequential pairs on one fixed 9x9 workload and seed. Hardware/general workload uncertainty is not estimated. Replica and game equality are exact; timing is empirical. Attempt chip-hours exclude reservation time outside these attempts.'}
    verify(SOURCE)
    with args.output.open('xb') as stream:stream.write(canonical_json(result))
    print(json.dumps({k:v for k,v in result.items() if k!='comparisons'},sort_keys=True))


if __name__=='__main__':main()
