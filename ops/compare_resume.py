#!/usr/bin/env python3
"""Verify exact fresh-process continuation, excluding only elapsed-time counters."""
import argparse
import json
from pathlib import Path
import sys
sys.dont_write_bytecode=True
SOURCE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(SOURCE/'packages/gozero/src'))
from gozero import checkpoints
from gozero.snapshots import canonical_json,read_json,verify


def compare(left,right,resume_turn,final_turn,origin=None):
    import numpy as np
    a=read_json(left/'result.json');b=read_json(right/'result.json')
    if a['status']!='passed' or b['status']!='passed': raise ValueError('Training attempts must pass')
    if b['resumed_from']['turn']!=resume_turn: raise ValueError('Resume boundary differs')
    name=f'turn-{final_turn:09d}'
    sa,aa,ga=checkpoints.read(left/'checkpoints'/name,expected_manifest_sha256=a['latest_checkpoint']['manifest_sha256'])
    sb,ab,gb=checkpoints.read(right/'checkpoints'/name,expected_manifest_sha256=b['latest_checkpoint']['manifest_sha256'])
    if set(aa)!=set(ab): raise ValueError('Saved array trees differ')
    for k in aa: np.testing.assert_array_equal(aa[k],ab[k],err_msg=k)
    if json.loads(ga)!=json.loads(gb): raise ValueError('Complete native actor state differs')
    for state in (sa,sb):
        state['counters']={k:v for k,v in state['counters'].items() if not k.endswith('_seconds')}
    if sa!=sb: raise ValueError('Non-timing scientific state differs')
    base=(left if origin is None else origin)/'checkpoints'/f'turn-{resume_turn:09d}'
    if origin is not None:
        expected_state,expected_arrays,expected_actors=checkpoints.read(left/'checkpoints'/base.name)
        origin_state,origin_arrays,origin_actors=checkpoints.read(base)
        if set(expected_arrays)!=set(origin_arrays):raise ValueError('Resume origin array tree differs')
        for key in expected_arrays:np.testing.assert_array_equal(expected_arrays[key],origin_arrays[key],err_msg='origin '+key)
        if json.loads(expected_actors)!=json.loads(origin_actors):raise ValueError('Resume origin actors differ')
        for state in (expected_state,origin_state):
            state['counters']={k:v for k,v in state['counters'].items() if not k.endswith('_seconds')}
        if expected_state!=origin_state:raise ValueError('Resume origin scientific state differs')
    group=base.parent/(base.name+'.group.json')
    if checkpoints.sha256(group)!=b['resumed_from']['group_sha256']: raise ValueError('Resume group identity differs')
    start=read_json(base/'state.json')
    expected=sum(sa['counters'][k]-start['counters'][k] for k in ('completed_games','truncated_games'))
    records=list((right/'games').glob('*.json'))
    if len(records)!=expected: raise ValueError('Subsequent game count differs')
    for record in records:
        for suffix in ('.json','.sgf'):
            p=record.with_suffix(suffix)
            if p.read_bytes()!=(left/'games'/p.name).read_bytes(): raise ValueError('Subsequent game differs: '+p.name)
    if a['model_export_sha256']!=b['model_export_sha256']: raise ValueError('Exported model differs')
    return {'status':'passed','snapshot_id':sa['snapshot_id'],'jax_rank':sa['jax_rank'],'resume_turn':resume_turn,
            'final_turn':final_turn,'compared_arrays':len(aa),'all_arrays_exact':True,'all_actor_states_exact':True,
            'scientific_state_exact':True,'subsequent_game_records_exact':True,'compared_subsequent_games':expected}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--continuous',type=Path,required=True);p.add_argument('--resumed',type=Path,required=True)
    p.add_argument('--resume-turn',type=int,required=True);p.add_argument('--final-turn',type=int,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--hosts',type=int,default=1)
    p.add_argument('--resume-origin',type=Path,help='Actual checkpoint attempt, if different from the uninterrupted control')
    args=p.parse_args();verify(SOURCE)
    if args.hosts not in (1,4): p.error('Supported qualification layouts are one local or four pod hosts')
    report={'schema_version':1,'analysis_snapshot':SOURCE.name,'status':'running',
            'qualification_scope':'Fresh-process continuation on unchanged topology. Optional separate checkpoint origin is compared at the resume boundary; failure injection requires a separate receipt. Only timing counters excluded.','hosts':[]}
    try:
        for host in range(args.hosts):
            rel=Path('.') if args.hosts==1 else Path(f'rank-{host}/artifacts')
            origin=None if args.resume_origin is None else args.resume_origin/rel
            report['hosts'].append(compare(args.continuous/rel,args.resumed/rel,args.resume_turn,args.final_turn,origin))
        report['status']='passed'
    except BaseException as error:
        report.update(status='failed',error=repr(error));raise
    finally:
        with args.output.open('xb') as f: f.write(canonical_json(report))
        print(json.dumps(report),flush=True)


if __name__=='__main__':main()
