#!/usr/bin/env python3
"""Verify closed ABBA cache profiles and retain every host's measurements."""
import argparse
from pathlib import Path
import statistics
import sys
SOURCE=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(SOURCE/'packages/gozero/src'))
from gozero.snapshots import canonical_json,read_json,verify
from gozero.checkpoints import sha256

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--workspace-root',type=Path,required=True)
    p.add_argument('--attempt',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();verify(SOURCE);root=a.workspace_root.resolve();attempt=a.attempt.resolve()
    closed=read_json(attempt/'result.json');assert closed['status']=='passed'
    snap=root/'.gozero/snapshots'/closed['snapshot_id'];verify(snap);c=read_json(snap/'resolved_config.json')
    assert c['kind']=='cached_decode_qualification' and c['cases']==['reference','carried','carried','reference']
    hosts=[];files={};initial=[];ranks=[]
    for host in range(4):
        folder=attempt/f'rank-{host}/artifacts';path=folder/'result.json';r=read_json(path)
        files[str(path.relative_to(root))]=sha256(path)
        assert r['status']=='passed' and r['snapshot_id']==snap.name and r['host_rank']==host
        assert r['dataset_manifest_sha256']==c['dataset']['manifest_sha256']
        initial.append(r['initial_parameter_elements_sha256']);ranks.append(r['jax_rank'])
        assert [x['case'] for x in r['cases']]==c['cases']
        case_rows=[]
        for i,x in enumerate(r['cases']):
            base=folder/f"case-{i}-{x['case']}";hlo=base/'decode.hlo';result=base/'result.json'
            assert x=={'case':x['case'],'index':i,**read_json(result)}
            assert sha256(hlo)==x['hlo_sha256'];files[str(hlo.relative_to(root))]=sha256(hlo)
            files[str(result.relative_to(root))]=sha256(result)
            assert x['cache_donated'] and x['cache_outputs_observed']
            assert (x['board_size'],x['batch_size'],x['past_moves'],x['allocated_cache_positions'],x['attention_key_extent'])==(9,128,128,512,4901)
            assert len(x['host_dispatch_latency_seconds'])==10 and min(x['host_dispatch_latency_seconds'])>0
            assert x['max_logit_error']<=.04 and x['max_policy_tv']<=.005
            assert not x['jaxpr']['unaccounted_primitives']
            assert x['jaxpr']['counts']['multiply_add_flops']==x['analytical']['multiply_add_flops_per_batch']
            assert x['analytical']==r['cases'][0]['analytical']
            assert x['jaxpr']['floating_operations_unit_cost']==r['cases'][0]['jaxpr']['floating_operations_unit_cost']
            case_rows.append({k:x[k] for k in ('case','index','host_dispatch_latency_seconds','memory_bytes','hlo_sha256','max_logit_error','max_policy_tv')})
        medians={name:statistics.median([v for x in r['cases'] if x['case']==name for v in x['host_dispatch_latency_seconds']]) for name in ('reference','carried')}
        hosts.append({'host':host,'cases':case_rows,'median_seconds':medians,'speedup':medians['reference']/medians['carried']})
    assert len(set(initial))==1 and sorted(ranks)==list(range(4))
    report={'kind':'complete_cached_decode_abba_audit','status':'passed','operator_snapshot':SOURCE.name,
        'attempt':attempt.name,'closed_result_sha256':sha256(attempt/'result.json'),'training_snapshot':snap.name,
        'input_files':files,'initial_parameter_elements_sha256':initial[0],'hosts':hosts,
        'reserved_chip_hours':closed['reserved_chip_hours'],
        'scope':'Warm complete neural decode; encoder and observed donated KV output included. CPU feature generation, transfers and pointer reset excluded. No training or Go strength measurement.'}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('xb') as f:f.write(canonical_json(report))
    a.output.chmod(0o444)
    print(canonical_json({'status':'passed','sha256':sha256(a.output),'host_speedups':[h['speedup'] for h in hosts]}).decode().strip())

if __name__=='__main__':main()
