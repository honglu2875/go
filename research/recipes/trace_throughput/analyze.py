#!/usr/bin/env python3
"""Independently audit equal-work trace artifacts and repeated pod timing."""
import argparse
from collections import Counter
import gzip
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

sys.dont_write_bytecode=True
SOURCE=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(SOURCE/'packages/gozero/src'))
from gozero.causal_artifacts import validate as validate_candidate
from gozero.checkpoints import sha256
from gozero.model_artifacts import artifact
from gozero.snapshots import canonical_json,read_json,verify


def require(condition,message):
    if not condition:raise ValueError(message)


def audit_segment(directory,row,c,model_id):
    path=directory/row['artifact'];require(sha256(path)==row['artifact_sha256'],'Compressed event artifact differs')
    data=json.loads(gzip.decompress(path.read_bytes()));events=data['events'];packets=data['packets'];final=data['final_states']
    require(len(events)==len(final)==c['games'],'Game coverage differs')
    mode=row['mode'];horizon=1 if mode=='sequential' else c['horizon'];samples=c['samples']if mode=='behavior'else 1
    target=row['target_moves_per_game'];remaining=[target]*c['games'];reconstructed=[[]for _ in events];counts=Counter(inactive_game_slots=0)
    for number,packet in enumerate(packets):
        require(packet['packet']==number and len(packet['tickets'])==len(packet['resolutions'])==len(packet['allowances'])==c['games'],'Packet coverage differs')
        counts['request_json_bytes']+=len(json.dumps(packet['tickets'],separators=(',',':')).encode())
        counts['response_json_bytes']+=len(json.dumps(packet['resolutions'],separators=(',',':')).encode())
        for game,(ticket,resolution,allowance)in enumerate(zip(packet['tickets'],packet['resolutions'],packet['allowances'])):
            require(allowance==min(remaining[game],horizon),'Per-game allowance differs from remaining accepted work')
            actions=resolution['actions'];stop=resolution['stop'];counts['stop_'+stop]+=1;counts['resolved_depth_'+str(len(actions))]+=1
            counts['legality_corrections']+=resolution['legality_corrections']
            require(stop in ('horizon','work_limit','state_mismatch','missing_continuation','terminal','move_limit'),'Unknown stop reason')
            require(len(actions)<=allowance and len(resolution['selected_samples'])==len(actions)
                    and all(type(s)is int and 0<=s<samples for s in resolution['selected_samples'])
                    and 0<=resolution['legality_corrections']<=len(actions),'Native move/sample dimensions differ')
            if ticket is None:
                require(allowance==0 and not actions and stop=='work_limit' and resolution['terminal_white_score']is None,'Inactive game advanced or acquired an outcome')
                counts['inactive_game_slots']+=1;continue
            require(allowance>0 and actions and ticket['models']==[model_id,model_id] and ticket['contexts']==[0,0],'Active ticket/model/work differs')
            counts['active_prefix_tokens']+=ticket['move_number']+1
            require((stop=='terminal')==(resolution['terminal_white_score']is not None),'Outcome assigned to nonterminal prefix')
            if stop=='horizon':require(len(actions)==horizon,'Horizon stopped early')
            if stop=='work_limit':require(len(actions)==allowance<horizon,'Work-limit stop differs')
            if stop in ('state_mismatch','missing_continuation'):require(len(actions)<allowance,'Mismatch consumed its full allowance')
            for depth,action in enumerate(actions):
                require(type(action)is int and 0<=action<=c['actors']['size']**2,'Invalid actual action')
                last=depth==len(actions)-1
                event={'episode':ticket['episode'],'ply':ticket['move_number']+depth,'action':action,
                    'ending':stop if last and stop in ('terminal','move_limit')else None,
                    'terminal_white_score':resolution['terminal_white_score']if last else None}
                prior=reconstructed[game][-1]if reconstructed[game]else None
                if prior is None:require(event['episode']==0 and event['ply']==0,'Game did not start from empty episode0')
                elif prior['ending']in ('terminal','move_limit'):
                    require(event['episode']==prior['episode']+1 and event['ply']==0,'Terminal/cap reset differs')
                else:require(event['episode']==prior['episode'] and event['ply']==prior['ply']+1,'Action prefix is discontinuous')
                reconstructed[game].append(event)
            remaining[game]-=len(actions)
    require(not any(remaining) and reconstructed==events and all(len(e)==target for e in events),'Registered accepted work or event reconstruction differs')
    for stream,state in zip(events,final):
        last=stream[-1];suffix=[e['action']for e in stream if e['episode']==last['episode']]
        require(state['episode']==last['episode'] and state['moves']==suffix and state['to_play']==1+len(suffix)%2
                and state['terminal']==(last['ending']=='terminal') and len(state['stones'])==c['actors']['size']**2
                and all(s in (0,1,2)for s in state['stones']),'Final episode/tape/terminal state differs, or an inactive game was recycled')
    dispatches=len(packets);games=c['games'];actions=c['actors']['size']**2+1
    counts.update(dispatches=dispatches,real_moves=games*target,prefill_token_slots=dispatches*games*c['model']['max_tokens'],
        input_array_bytes=dispatches*games*(4*c['model']['max_tokens']+8+actions+actions-1),
        output_array_bytes=dispatches*games*horizon*(4*(2*samples*(1+actions)+actions)+2*samples*(actions-1)),
        board_encode_rows=dispatches*games*2*samples*horizon,
        decoder_append_token_slots=dispatches*games*2*samples*(horizon-1))
    actual=row['counters'];timers=('native_start_and_frame_decode_seconds','inference_and_transfer_seconds','native_resolve_and_response_decode_seconds')
    require(set(actual)-set(timers)==set(counts) and all(actual[k]==v for k,v in counts.items()),'Logical work or transfer accounting differs')
    require(math.isfinite(row['elapsed_seconds']) and row['elapsed_seconds']>0 and row['process_cpu_seconds']>=0
            and all(math.isfinite(actual[k]) and actual[k]>=0 for k in timers)
            and sum(actual[k]for k in timers)<=row['elapsed_seconds']+.001,'Invalid segment timing')
    require(row['average_active_cpu_cores']==row['process_cpu_seconds']/row['elapsed_seconds']
            and row['moves_per_second']==games*target/row['elapsed_seconds']
            and row['moves_per_game_per_dispatch']==target/dispatches,'Derived throughput differs')
    return events,final


def audit_host(root,directory,protocol,host):
    result=read_json(directory/'result.json');source=artifact(root,'.gozero/snapshots/'+protocol['snapshot']);verify(source)
    c=read_json(source/'resolved_config.json')
    require(result['kind']=='equal_prefix_trace_throughput' and result['status']=='passed' and 'error'not in result
            and result['snapshot_id']==source.name and result['host_rank']==host
            and result['config_sha256']==protocol['config_sha256']==sha256(source/'resolved_config.json')
            and result['world_size']==protocol['expected_hosts'],'Host result or configuration differs')
    require(protocol['registered_unix']<result['started_unix']<=result['finished_unix'],'Run preceded registration')
    receipt=artifact(root,'.gozero/native/'+source.name+'/receipt.json');native=read_json(receipt)
    require(native==result['native'] and sha256(receipt)==protocol['native_receipt_sha256']
            and sha256(receipt.parent/native['filename'])==native['binary_sha256'],'Loaded native artifact differs')
    descriptor=read_json(source/c['candidate']);trained=validate_candidate(root,descriptor)
    require(result['candidate']==descriptor and result['candidate_sha256']==sha256(source/c['candidate'])
            and trained['model_code_sha256']==sha256(source/'research/recipes/trace_throughput/model.py'),'Fixed execution model differs')
    elements=hashlib.sha256(b''.join(trained['arrays'][f'p_{i:04d}'].tobytes()for i in range(len(trained['model_schema'])))).hexdigest()
    require(elements==result['parameter_elements_sha256'] and result['parameter_count']==sum(a.size for a in trained['arrays'].values()),'Loaded parameter elements differ')
    for name,entry in result['compilation'].items():
        require(sha256(directory/(name+'.hlo.txt.gz'))==entry['hlo_sha256'] and sha256(directory/(name+'.cost.json'))==entry['cost_sha256'],'Compiled program or cost artifact differs')
    require([r['mode']for r in result['qualification']]==['sequential','behavior','joint'],'Qualification modes differ')
    reference=None;reference_final=None
    for row in result['qualification']:
        require(row['target_moves_per_game']==c['qualification_moves_per_game'],'Qualification work differs')
        events,final=audit_segment(directory,row,c,int(elements[:16],16))
        if reference is None:reference=events;reference_final=final
        require(events==reference and final==reference_final and row['sequential_events_exact'] and row['final_native_states_exact'] and not row['first_differences'],'Qualification numerical execution differs')
    require([(r['repetition'],r['mode'])for r in result['timings']]==[(i,m)for i,order in enumerate(c['orders'])for m in order],'Timing order/coverage differs')
    first_reference=None
    for repetition in range(len(c['orders'])):
        pairs={}
        for row in [r for r in result['timings']if r['repetition']==repetition]:
            require(row['target_moves_per_game']==c['timed_moves_per_game'],'Timing accepted work differs')
            pairs[row['mode']]=audit_segment(directory,row,c,int(elements[:16],16))
            require(row['sequential_events_exact'] and row['final_native_states_exact'] and not row['first_differences'],'Timed qualification flags differ')
        require(all(value==pairs['sequential']for value in pairs.values()),'Equal-work events or final state differ')
        if repetition==0:first_reference=pairs['sequential'][0]
    if result['profile']['status']in ('captured','failed_to_stop'):
        row=result['profile']['run'];events,_=audit_segment(directory,row,c,int(elements[:16],16))
        require(events==[e[:c['profile_moves_per_game']]for e in first_reference],'Profiled execution changed the reference prefix')
        for name,expected in result['profile']['files'].items():require(sha256(directory/name)==expected,'Profile file changed')
    return result,c


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--workspace-root',type=Path,required=True)
    p.add_argument('--protocol',type=Path,required=True);p.add_argument('--expected-protocol-sha256',required=True)
    p.add_argument('--attempt',required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();verify(SOURCE);root=a.workspace_root.resolve()
    require(sha256(a.protocol)==a.expected_protocol_sha256,'Registration differs');protocol=read_json(a.protocol)
    attempt=artifact(root,a.attempt);world=protocol.get('expected_hosts',1);protocol={**protocol,'expected_hosts':world}
    rows=[]
    for host in range(world):
        directory=attempt/f'rank-{host}/artifacts'if world>1 else attempt
        row,c=audit_host(root,directory,protocol,host);rows.append(row)
    require({r['jax_rank']for r in rows}==set(range(world)) and len({r['parameter_elements_sha256']for r in rows})==1,'Rank coverage or replicated parameters differ')
    pod=read_json(attempt/'result.json')if world>1 else None
    if pod is not None:
        require(pod['status']=='passed' and pod['snapshot_id']==protocol['snapshot'] and pod['elapsed_seconds']<protocol['maximum_seconds']
                and pod['start_unix_time']>protocol['registered_unix'],'Pod status, timing or registration differs')
        launches=[p for p in (root/'runs').glob('pod-*/launch.json')if read_json(p)['snapshot_id']==protocol['snapshot']]
        require(len(launches)==protocol['maximum_attempts'] and launches[0].parent==attempt,'Pod attempt budget differs')
    else:require(rows[0]['finished_unix']-rows[0]['started_unix']<protocol['maximum_seconds'] and a.attempt==protocol['output'],'CPU attempt or budget differs')
    repetitions=[]
    for repetition in range(len(c['orders'])):
        summary={}
        for mode in ('sequential','behavior','joint'):
            per_host=[next(x for x in r['timings']if x['repetition']==repetition and x['mode']==mode)for r in rows]
            elapsed=max(x['elapsed_seconds']for x in per_host);moves=sum(x['counters']['real_moves']for x in per_host)
            summary[mode]={'global_moves':moves,'slowest_host_seconds':elapsed,'global_moves_per_second':moves/elapsed,
                'host_average_active_cpu_cores':[x['average_active_cpu_cores']for x in per_host],
                'host_dispatches':[x['counters']['dispatches']for x in per_host]}
        for mode in ('behavior','joint'):summary[mode]['speed_ratio']=summary['sequential']['slowest_host_seconds']/summary[mode]['slowest_host_seconds']
        repetitions.append(summary)
    ratios={mode:[r[mode]['speed_ratio']for r in repetitions]for mode in ('behavior','joint')}
    gate=protocol.get('primary_speed_criterion');met=None
    if gate is not None:
        values=ratios[gate['mode']];met=statistics.median(values)>=gate['minimum_median_ratio'] and min(values)>=gate['minimum_each_repetition_ratio']
    report={'schema_version':1,'kind':'equal_prefix_trace_audit','status':'passed','analysis_snapshot':SOURCE.name,
        'protocol_sha256':a.expected_protocol_sha256,'attempt':a.attempt,
        'rank_results_sha256':[sha256((attempt/f'rank-{h}/artifacts'if world>1 else attempt)/'result.json')for h in range(world)],
        'all_equal_work_event_streams_and_final_states_exact':True,'repetitions':repetitions,'ratios':ratios,
        'median_ratios':{k:statistics.median(v)for k,v in ratios.items()},'registered_speed_criterion_met':met,
        'recorded_attempt_chip_hours':pod['reserved_chip_hours']if pod else 0.,
        'profile_status_by_host':[r['profile']['status']for r in rows],
        'claims_go_strength':False,'claims_mfu':False,'claims_rl_sample_efficiency':False,
        'limitations':protocol.get('limitations',[])}
    verify(SOURCE);require(sha256(a.protocol)==a.expected_protocol_sha256,'Registration changed during audit')
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('xb')as f:f.write(canonical_json(report))
    a.output.chmod(0o444);print(sha256(a.output))


if __name__=='__main__':main()
