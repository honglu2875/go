#!/usr/bin/env python3
"""Audit a completed numerical-qualification failure without promoting timings."""
import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path

from analyze import SOURCE,artifact,audit_segment,canonical_json,read_json,require,sha256,validate_candidate,verify


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workspace-root',type=Path,required=True)
    p.add_argument('--protocol',type=Path,required=True)
    p.add_argument('--expected-protocol-sha256',required=True)
    p.add_argument('--attempt',required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();verify(SOURCE);root=a.workspace_root.resolve()
    require(sha256(a.protocol)==a.expected_protocol_sha256,'Registration differs')
    protocol=read_json(a.protocol);attempt=artifact(root,a.attempt)
    source=artifact(root,'.gozero/snapshots/'+protocol['snapshot']);verify(source)
    c=read_json(source/'resolved_config.json');pod=read_json(attempt/'result.json')
    require(pod['status']=='failed' and pod['snapshot_id']==source.name
            and protocol['registered_unix']<pod['start_unix_time']<pod['end_unix_time']
            and pod['elapsed_seconds']<protocol['maximum_seconds'],'Attempt identity, failure or budget differs')
    launches=[f for f in (root/'runs').glob('pod-*/launch.json')if read_json(f)['snapshot_id']==source.name]
    require(len(launches)==protocol['maximum_attempts']==1 and launches[0].parent==attempt,'Attempt count differs')
    descriptor=read_json(source/c['candidate']);trained=validate_candidate(root,descriptor)
    require(trained['model_code_sha256']==sha256(source/'research/recipes/trace_throughput/model.py'),'Model code differs')
    elements=hashlib.sha256(b''.join(trained['arrays'][f'p_{i:04d}'].tobytes()for i in range(len(trained['model_schema'])))).hexdigest()
    receipt=artifact(root,'.gozero/native/'+source.name+'/receipt.json');native=read_json(receipt)
    require(sha256(receipt)==protocol['native_receipt_sha256'] and sha256(receipt.parent/native['filename'])==native['binary_sha256'],'Native artifact differs')
    hosts=[];ranks=set();totals=Counter()
    for host in range(protocol['expected_hosts']):
        directory=attempt/f'rank-{host}/artifacts';result=read_json(directory/'result.json')
        require(result['status']=='failed' and result['error']=="ValueError('TPU/CPU numerical execution changed the equal-work reference')"
                and result['snapshot_id']==source.name and result['host_rank']==host
                and result['world_size']==protocol['expected_hosts'] and result['native']==native
                and result['candidate']==descriptor and result['candidate_sha256']==sha256(source/c['candidate'])
                and result['config_sha256']==protocol['config_sha256']==sha256(source/'resolved_config.json')
                and result['parameter_elements_sha256']==elements
                and result['parameter_count']==sum(v.size for v in trained['arrays'].values())
                and protocol['registered_unix']<result['started_unix']<=result['finished_unix']
                and not result.get('timings') and 'profile'not in result,'Host failure/lineage or stop-before-timing differs')
        ranks.add(result['jax_rank'])
        require(list(result['compilation'])==sorted(('sequential','behavior','joint')),'Compilation coverage differs')
        for name,entry in result['compilation'].items():
            require(sha256(directory/(name+'.hlo.txt.gz'))==entry['hlo_sha256']
                    and sha256(directory/(name+'.cost.json'))==entry['cost_sha256'],'Compiler evidence differs')
        require([r['mode']for r in result['qualification']]==['sequential','behavior','joint'],'Mode coverage differs')
        reference=None;reference_final=None;summary=[]
        for row in result['qualification']:
            require(row['target_moves_per_game']==c['qualification_moves_per_game'],'Qualification work differs')
            events,final=audit_segment(directory,row,c,int(elements[:16],16))
            if reference is None:reference=events;reference_final=final
            differences=[];packet_depths=Counter()
            data=json.loads(gzip.decompress((directory/row['artifact']).read_bytes()))
            for game,(ref,got)in enumerate(zip(reference,events)):
                if ref==got:continue
                first=next(i for i,(x,y)in enumerate(zip(ref,got))if x!=y)
                differences.append({'game':game,'first_event':first,'reference':ref[first],'candidate':got[first]})
                offset=0
                for packet in data['packets']:
                    actions=packet['resolutions'][game]['actions']
                    if first<offset+len(actions):packet_depths[first-offset]+=1;break
                    offset+=len(actions)
            require(row['first_differences']==differences[:8] and row['sequential_events_exact']==(not differences)
                    and row['final_native_states_exact']==(final==reference_final),'Recorded comparison differs')
            totals[row['mode']+'_differing_game_streams']+=len(differences)
            totals['qualification_moves']+=len(events)*c['qualification_moves_per_game']
            summary.append({'mode':row['mode'],'differing_game_streams':len(differences),
                'first_difference_packet_depths':dict(packet_depths),'first_differences':differences,
                'equal_accepted_move_count':True,'final_states_exact':final==reference_final,'artifact_sha256':row['artifact_sha256']})
        require(any(r['differing_game_streams']for r in summary),'Failure has no divergent event streams')
        hosts.append({'host':host,'jax_rank':result['jax_rank'],'result_sha256':sha256(directory/'result.json'),'modes':summary})
    require(ranks==set(range(protocol['expected_hosts'])),'Rank coverage differs')
    report={'schema_version':1,'kind':'failed_trace_qualification_audit','status':'passed','qualification_status':'failed',
        'analysis_snapshot':SOURCE.name,'protocol_sha256':a.expected_protocol_sha256,'attempt':a.attempt,
        'pod_result_sha256':sha256(attempt/'result.json'),'parameter_elements_sha256':elements,'hosts':hosts,'totals':dict(totals),
        'recorded_attempt_chip_hours':pod['reserved_chip_hours'],'timing_study_started':False,
        'claims_throughput_gain':False,'claims_strength_gain':False,'claims_numerical_cause_proven':False,
        'interpretation':'All observed first differences occur after incremental steps. This suggests numerical-path differences; the artifacts do not retain logits and cannot establish their magnitude or rule out another cause.'}
    verify(SOURCE);verify(source);require(sha256(a.protocol)==a.expected_protocol_sha256,'Registration changed')
    with a.output.open('xb')as f:f.write(canonical_json(report))
    a.output.chmod(0o444);print(sha256(a.output))


if __name__=='__main__':main()
