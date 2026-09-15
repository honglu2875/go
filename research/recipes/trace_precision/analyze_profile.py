#!/usr/bin/env python3
"""Summarize audited packet costs and measured XLA-module timeline occupancy."""
import argparse
from collections import Counter
import gzip
import json
import math
from pathlib import Path
import sys

sys.dont_write_bytecode=True
SOURCE=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(SOURCE/'packages/gozero/src'))
from gozero.checkpoints import sha256
from gozero.model_artifacts import artifact
from gozero.snapshots import canonical_json,read_json,verify


def require(value,message):
    if not value:raise ValueError(message)


def union_duration(intervals):
    total=0.;right=-math.inf
    for start,end in sorted(intervals):
        require(math.isfinite(start) and math.isfinite(end) and end>=start,'Invalid trace interval')
        total+=max(0.,end-max(start,right));right=max(right,end)
    return total


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--workspace-root',type=Path,required=True)
    p.add_argument('--timing-audit',type=Path,required=True);p.add_argument('--expected-timing-audit-sha256',required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();verify(SOURCE)
    root=a.workspace_root.resolve();require(sha256(a.timing_audit)==a.expected_timing_audit_sha256,'Timing audit differs')
    audit=read_json(a.timing_audit)
    require(audit['kind']=='equal_prefix_trace_audit' and audit['status']=='passed'
            and audit['all_equal_work_event_streams_and_final_states_exact'],'Raw timing audit did not pass')
    verify(artifact(root,'.gozero/snapshots/'+audit['analysis_snapshot']))
    attempt=artifact(root,audit['attempt']);hosts=[]
    for host,expected in enumerate(audit['rank_results_sha256']):
        path=attempt/f'rank-{host}/artifacts/result.json';require(sha256(path)==expected,'Audited host result changed')
        hosts.append(read_json(path))
    require(len(hosts)==4 and all(h['status']=='passed'for h in hosts),'Host coverage differs')
    profile=hosts[0]['profile'];require(profile['status']=='captured','No completed host-0 profile')
    directory=attempt/'rank-0/artifacts'
    for name,expected in profile['files'].items():require(sha256(directory/name)==expected,'Profile artifact differs')
    files=[name for name in profile['files']if name.endswith('.trace.json.gz')]
    require(len(files)==1,'Ambiguous Chrome trace artifact')
    data=json.loads(gzip.decompress((directory/files[0]).read_bytes()));events=data['traceEvents']
    markers=[e for e in events if e.get('name')=='gozero_packet_loop'and e.get('ph')=='X']
    require(len(markers)==1,'Packet-loop marker coverage differs')
    marker=markers[0];lo=marker['ts'];hi=lo+marker['dur'];duration=marker['dur']
    require(duration>0 and abs(duration*1e-6-profile['run']['elapsed_seconds'])<.005
            and marker['args']['mode']==profile['run']['mode']=='behavior'
            and int(marker['args']['target_moves_per_game'])==profile['run']['target_moves_per_game'],'Trace marker and measured loop differ')
    processes={e['pid']:e['args']['name']for e in events if e.get('ph')=='M'and e.get('name')=='process_name'}
    threads={(e['pid'],e.get('tid')):e['args']['name']for e in events if e.get('ph')=='M'and e.get('name')=='thread_name'}
    devices={pid:name for pid,name in processes.items()if name.startswith('/device:TPU:')}
    require(len(devices)==8,'Expected eight TPU trace tracks for four local dual-core chips')
    tracks=[]
    for pid,name in sorted(devices.items(),key=lambda item:int(item[1].rsplit(':',1)[1])):
        by_thread={};counts=Counter()
        for e in events:
            if e.get('pid')!=pid or e.get('ph')!='X' or e.get('dur',0)<=0:continue
            start=max(lo,e['ts']);end=min(hi,e['ts']+e['dur'])
            if start>=end:continue
            thread=threads.get((pid,e.get('tid')),'unlabeled');counts[thread]+=1
            by_thread.setdefault(thread,[]).append((start,end))
        require(counts['XLA Modules']==profile['run']['counters']['dispatches'],'Device module count differs from actual dispatches')
        occupied={name:union_duration(intervals)for name,intervals in by_thread.items()}
        tracks.append({'trace_device':name,'counts_by_track':dict(counts),'union_microseconds_by_track':occupied,
            'module_window_fraction':occupied['XLA Modules']/duration,
            'outside_module_window_fraction':1-occupied['XLA Modules']/duration})
    counters=[e for e in events if e.get('pid')in devices and e.get('ph')=='C'and lo<=e.get('ts',-1)<hi]
    modes={}
    for mode in ('sequential','behavior','joint'):
        rows=[r for h in hosts for r in h['timings']if r['mode']==mode]
        total=Counter()
        for row in rows:total.update(row['counters'])
        seconds=sum(r['elapsed_seconds']for r in rows)
        modes[mode]={'summed_host_loop_seconds':seconds,'summed_counters':dict(total),
            'host_process_cpu_cores_range':[min(r['average_active_cpu_cores']for r in rows),max(r['average_active_cpu_cores']for r in rows)],
            'inference_and_transfer_fraction':total['inference_and_transfer_seconds']/seconds,
            'native_and_frame_decode_fraction':(total['native_start_and_frame_decode_seconds']+total['native_resolve_and_response_decode_seconds'])/seconds,
            'active_history_fraction_of_prefill_slots':total['active_prefix_tokens']/total['prefill_token_slots'],
            'moves_per_game_per_dispatch':total['real_moves']/(64*total['dispatches'])}
    require(len({m['summed_counters']['real_moves']for m in modes.values()})==1,'Summed accepted work differs')
    baseline=modes['sequential']['summed_counters']
    for mode,summary in modes.items():
        values=summary['summed_counters'];summary['ratios_to_sequential']={k:values[k]/baseline[k]for k in
            ('dispatches','board_encode_rows','prefill_token_slots','input_array_bytes','output_array_bytes')}
    report={'schema_version':1,'kind':'trace_packet_cost_and_occupancy_audit','status':'passed','analysis_snapshot':SOURCE.name,
        'timing_audit_sha256':a.expected_timing_audit_sha256,'attempt':audit['attempt'],'timing_source':hosts[0]['snapshot_id'],
        'profile_files':profile['files'],'profile_event_count':len(events),'profile_loop_microseconds':duration,
        'profiled_dispatches_per_device':profile['run']['counters']['dispatches'],'profiled_real_moves':profile['run']['counters']['real_moves'],
        'profiled_trace_device_tracks':len(devices),'jax_local_device_count':len(hosts[0]['local_devices']),
        'device_tracks':tracks,'device_counter_event_count':len(counters),'unprofiled_timing_costs':modes,
        'claims_mfu':False,'claims_mxu_utilization':False,'claims_compute_bound':False,
        'limitations':['Module timeline occupancy includes device computation, internal copies, barriers and waits; it is not MXU utilization or MFU.',
            'Only the behavior mode on host 0 was profiled. Profile overhead and its shorter prefix prevent a cross-mode utilization comparison.',
            'Outside-module time includes host processing, dispatch, transfers and waits; this trace summary does not apportion it.',
            'Logical tensor byte counts are not physical hardware-link traffic; process CPU time includes Python, Rust and runtime threads.',
            'Eight TPU trace tracks represent the two cores of four local chips, not eight allocated chips.',
            'Active history slots describe repeated fixed-shape prefill work, not a measured FLOP utilization.']}
    verify(SOURCE);require(sha256(a.timing_audit)==a.expected_timing_audit_sha256,'Timing audit changed')
    with a.output.open('xb')as f:f.write(canonical_json(report))
    a.output.chmod(0o444);print(sha256(a.output))


if __name__=='__main__':main()
