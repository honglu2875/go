#!/usr/bin/env python3
"""Recheck frozen trained trace artifacts, event prefixes and useful work."""
import argparse
from collections import Counter
from pathlib import Path
import sys

import numpy as np

sys.dont_write_bytecode=True
SOURCE=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(SOURCE/'packages/gozero/src'))
from gozero.causal_artifacts import validate
from gozero.checkpoints import sha256
from gozero.model_artifacts import artifact
from gozero.snapshots import canonical_json,read_json,verify

REGISTRATION='14427f90a0e2c8d46dd184d5deeeea3252c856086ca84f7ccd8ab9705686511b'


def require(condition,message):
    if not condition:raise ValueError(message)


def audit(root,protocol,condition):
    source=artifact(root,'.gozero/snapshots/'+condition['snapshot']);verify(source)
    config=read_json(source/'resolved_config.json');output=artifact(root,condition['output'])
    result=read_json(output/'result.json')
    require(result['status']=='passed' and result['snapshot_id']==source.name and result['config_sha256']==condition['config_sha256']==sha256(source/'resolved_config.json'),'Run or configuration differs')
    require(protocol['registered_unix']<result['started_unix']<=result['finished_unix'] and result['finished_unix']-result['started_unix']<protocol['maximum_seconds_per_condition'],'Run order or budget differs')
    native=artifact(root,'.gozero/native/'+source.name+'/receipt.json');receipt=read_json(native)
    require(sha256(native)==condition['native_receipt_sha256'] and receipt==result['native'] and sha256(native.parent/receipt['filename'])==condition['native_binary_sha256']==receipt['binary_sha256'],'Native identity differs')
    descriptor=read_json(source/config['candidate']);trained=validate(root,descriptor)
    require(result['candidate']==descriptor and result['candidate_sha256']==sha256(source/config['candidate']),'Trained candidate differs')
    require(trained['model_code_sha256']==sha256(source/'research/recipes/learned_trace/model.py'),'Execution model differs from training')
    with np.load(output/'loaded_parameters.npz',allow_pickle=False) as p, np.load(output/'untrained_behavior_parameters.npz',allow_pickle=False) as control:
        require(set(p.files)==set(control.files)==set(trained['arrays']),'Parameter names differ')
        for i,entry in enumerate(trained['model_schema']):
            name=f'p_{i:04d}';expected=trained['arrays'][name]
            require(p[name].dtype==expected.dtype and p[name].shape==expected.shape and p[name].tobytes()==expected.tobytes(),'Loaded parameter elements differ')
            require(control[name].dtype==expected.dtype and control[name].shape==expected.shape and np.isfinite(control[name]).all(),'Invalid control parameters')
            if not entry['path'].startswith("['behavior"):
                require(control[name].tobytes()==p[name].tobytes(),'Control changed the own policy or trunk')
    require(sha256(output/'loaded_parameters.npz')==result['parameter_sha256'] and sha256(output/'untrained_behavior_parameters.npz')==result['untrained_behavior_parameter_sha256'],'Parameter file hash differs')
    for name,compiled in result['compilation'].items():
        if 'hlo_sha256' in compiled:require(sha256(output/(name+'.hlo.txt'))==compiled['hlo_sha256'],'Compiled HLO differs')
    streams={};rows={}
    require(set(result['modes'])=={'sequential','behavior_independent','behavior_untrained','known_policy_independent','known_policy_joint'},'Execution modes differ')
    for name,counts in result['modes'].items():
        path=output/(name+'-events.json');data=read_json(path);events=data['events'];packets=data['packets']
        require(sha256(path)==counts['event_sha256'] and len(events)==config['games'],'Event artifact differs')
        require(len(packets)>0 and (name=='sequential' or len(packets)==config['packets']),'Registered packet count differs')
        reconstructed=[[] for _ in events];measured=Counter();depths=Counter()
        for packet,record in enumerate(packets):
            require(record['packet']==packet and len(record['tickets'])==len(record['resolutions'])==config['games'],'Packet coverage differs')
            for g,(ticket,resolution) in enumerate(zip(record['tickets'],record['resolutions'])):
                actions=resolution['actions'];require(bool(actions),'Live game failed to advance')
                measured['valid_prefix_tokens']+=ticket['move_number']+1
                measured['real_moves']+=len(actions);measured['legality_corrections']+=resolution['legality_corrections']
                measured['stop_'+resolution['stop']]+=1;depths['resolved_depth_'+str(len(actions))]+=1
                for depth,action in enumerate(actions):
                    last=depth==len(actions)-1
                    reconstructed[g].append({'episode':ticket['episode'],'ply':ticket['move_number']+depth,'action':action,
                        'ending':resolution['stop'] if last and resolution['stop'] in ('terminal','move_limit') else None,
                        'terminal_white_score':resolution['terminal_white_score'] if last else None})
        require(reconstructed==events,'Event stream differs from native resolutions')
        require(all(counts[k]==v for k,v in (measured+depths).items()),'Native work counter differs')
        horizon=1 if name=='sequential' else config['horizon'];samples=1 if name in ('sequential','known_policy_joint') else config['samples']
        require(counts['dispatches']==len(packets) and counts['prefill_token_slots']==len(packets)*config['games']*config['model']['max_tokens'],'Dispatch or prefill work differs')
        require(counts['decoder_append_token_slots']==len(packets)*config['games']*2*samples*(horizon-1),'Append work differs')
        actions=config['actors']['size']**2+1
        expected_bytes=len(packets)*config['games']*horizon*4*(2*samples*(1+actions)+actions)
        require(counts['output_bytes']==expected_bytes,'Returned output bytes differ')
        moves_per_dispatch=measured['real_moves']/(config['games']*len(packets))
        require(counts['moves_per_game_per_dispatch']==moves_per_dispatch,'Resolved-depth mean differs')
        streams[name]=events
        rows[name]={'real_moves':measured['real_moves'],'dispatches':len(packets),'moves_per_game_per_dispatch':moves_per_dispatch,
            'resolved_depths':dict(depths),'terminal_episodes':measured['stop_terminal'],'capped_episodes':measured['stop_move_limit'],
            'own_legality_corrections':measured['legality_corrections'],'first_forecast_distinct_fraction':counts['first_forecast_distinct_actions']/counts['first_forecast_samples'],
            'prefill_token_slots':counts['prefill_token_slots'],'valid_prefix_tokens':counts['valid_prefix_tokens'],
            'prefill_padding_fraction':1-counts['valid_prefix_tokens']/counts['prefill_token_slots'],
            'decoder_append_token_slots':counts['decoder_append_token_slots'],'output_bytes':counts['output_bytes'],
            'elapsed_seconds_diagnostic':counts['elapsed_seconds'],'event_sha256':sha256(path)}
    for name,events in streams.items():
        require(all(a==b[:len(a)] for a,b in zip(events,streams['sequential'])),'Sequential event prefix differs')
        rows[name]['sequential_prefix_exact']=True
    return {'name':condition['name'],'snapshot':source.name,'result_sha256':sha256(output/'result.json'),'modes':rows,
            'trained_model_export_sha256':descriptor['model_export_sha256'],'model_parameters':result['parameter_count'],
            'trained_parameter_elements_exact':True,'baseline_reference_events':sum(map(len,streams['sequential']))}


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--artifacts-root',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();verify(SOURCE);root=args.artifacts_root.resolve()
    protocol_path=artifact(root,'research/studies/learned_trace/corrected_spec.json')
    require(sha256(protocol_path)==REGISTRATION,'Study registration differs');protocol=read_json(protocol_path)
    attempts_path=artifact(root,'research/studies/learned_trace/corrected_attempts.json');attempts=read_json(attempts_path)
    require(attempts['protocol_sha256']==sha256(protocol_path) and len(attempts['attempts'])==len(protocol['conditions']),'Attempt coverage differs')
    for attempt,condition in zip(attempts['attempts'],protocol['conditions']):
        require(attempt['name']==condition['name'] and attempt['status']=='passed' and attempt['exit_code']==0,'Attempt failed or order changed')
        require(sha256(artifact(root,condition['output']+'.log'))==attempt['log_sha256'],'Attempt log differs')
    rows=[audit(root,protocol,c) for c in protocol['conditions']]
    result={'schema_version':1,'kind':'trained_causal_trace_audit','status':'passed','operator_snapshot':SOURCE.name,
        'protocol_sha256':sha256(protocol_path),'attempts_sha256':sha256(attempts_path),'conditions':rows,
        'primary_exact_execution_criterion_met':True,'claims_training_improvement':False,'claims_throughput_improvement':False,'claims_mfu':False,
        'limitations':protocol['limitations']+['Per-mode game episodes overlap; counts across conditions cannot be added as independent games.','No MCTS decisions were executed in these policy traces. Greedy initial-state runs can be identical across game IDs.','Forecast distinctness was instrumented in the frozen runner; only accepted events and work counters were independently reconstructed here.']}
    verify(SOURCE);require(sha256(protocol_path)==REGISTRATION,'Study registration changed during audit');args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('xb') as stream:stream.write(canonical_json(result))
    args.output.chmod(0o444);print(sha256(args.output))


if __name__=='__main__':main()
