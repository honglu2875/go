#!/usr/bin/env python3
"""Compare closed LR arms after checking every non-LR control and actual draw."""
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
    p.add_argument('--registration',type=Path,required=True);p.add_argument('--registration-sha256',required=True)
    p.add_argument('--audit',nargs=3,action='append',required=True,metavar=('RATE','PATH','SHA256'))
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();verify(SOURCE);root=a.workspace_root.resolve()
    if sha256(a.registration)!=a.registration_sha256:raise ValueError('Registration changed')
    registration=read_json(a.registration);expected={x['peak_learning_rate']:x['snapshot'] for x in registration['candidates_in_execution_order']}
    reference=registration['reference'];expected[reference['peak_learning_rate']]=reference['training_snapshot']
    rows=[];controls=[];draws=[];initial=[];seen=set()
    for rate,path,digest in a.audit:
        rate=float(rate);path=Path(path)
        if rate in seen or rate not in expected or sha256(path)!=digest:raise ValueError('Unknown/repeated rate or changed audit')
        seen.add(rate);audit=read_json(path)
        if audit['status']!='passed' or audit['steps']!=1024 or audit['training_snapshot']!=expected[rate]:raise ValueError('Arm differs from registration')
        for name,wanted in audit['input_files'].items():
            if sha256(root/name)!=wanted:raise ValueError('Audited bytes changed')
        attempt=root/'runs'/audit['attempt'];closed=read_json(attempt/'result.json')
        if closed['status']!='passed' or sha256(attempt/'result.json')!=audit['closed_result_sha256']:raise ValueError('Run changed')
        snapshot=root/'.gozero/snapshots'/audit['training_snapshot'];verify(snapshot);c=read_json(snapshot/'resolved_config.json')
        if c['learner']['learning_rate']!=rate:raise ValueError('Peak learning rate differs')
        control={**c,'learner':{k:v for k,v in c['learner'].items() if k not in ('learning_rate','end_learning_rate')}}
        controls.append(control)
        import json
        by_rank={}
        for host in range(4):
            folder=attempt/f'rank-{host}/artifacts';r=read_json(folder/'result.json')
            log=[json.loads(line) for line in (folder/'metrics.jsonl').read_text().splitlines()]
            initial.append(r['initial_parameter_elements_sha256'])
            by_rank[r['jax_rank']]=[{k:x[k] for k in ('turn','bucket','local_entries_sha256','local_symmetries','expert_positions')} for x in log]
            if host==0:owner=r;owner_log=log
        draws.append(by_rank)
        times={0:0.,**{x['turn']:x['cumulative_learning_seconds'] for x in owner_log}}
        curve=[{'turn':x['turn'],'learning_seconds':times[x['turn']],'validation_ids_sha256':x['episode_ids_sha256'],**x['metrics']} for x in audit['validation_curve']]
        rows.append({'peak_learning_rate':rate,'end_learning_rate':c['learner']['end_learning_rate'],'training_snapshot':snapshot.name,
            'attempt':attempt.name,'audit_path':str(path),'audit_sha256':digest,'curve':curve,'endpoint':curve[-1],
            'timing':audit['timing'],'clipped_updates':audit['clipped_updates'],'position_exposures':audit['expert_positions'],
            'reserved_chip_hours':closed['reserved_chip_hours'],
            'decode_median_seconds':statistics.median(owner['decode_profile']['host_dispatch_latency_seconds']),
            'checkpoint':owner['latest_checkpoint']})
    if len(set(initial))!=1 or any(x!=controls[0] for x in controls) or any(x!=draws[0] for x in draws):
        raise ValueError('Non-LR controls, initial weights or actual data draws differ')
    for row in rows:
        if [x['turn'] for x in row['curve']]!=[x['turn'] for x in rows[0]['curve']]:raise ValueError('Evaluation cadence changed')
        if any(x['validation_ids_sha256']!=rows[0]['curve'][0]['validation_ids_sha256'] for x in row['curve']):raise ValueError('Validation population changed')
    rows.sort(key=lambda x:x['peak_learning_rate']);best=min(rows,key=lambda x:(x['endpoint']['expert_kl'],x['peak_learning_rate']))
    result={'kind':'controlled_learning_rate_sweep_analysis','status':'passed','operator_snapshot':SOURCE.name,
        'registration_sha256':a.registration_sha256,'all_non_lr_settings_and_draws_identical':True,
        'initial_parameter_elements_sha256':initial[0],'completed_arms':rows,'pending_rates':sorted(set(expected)-seen),
        'complete':seen==set(expected),'lowest_completed_endpoint_rate':best['peak_learning_rate'],
        'selection_scope':'Lowest completed final validation KL; do not call this the sweep winner while registered rates remain pending. One seed and fixed data; test closed. No global optimum, convergence, Go strength or RL efficiency claim.'}
    with a.output.open('xb') as f:f.write(canonical_json(result))
    a.output.chmod(0o444);print(canonical_json({'status':'passed','sha256':sha256(a.output),'pending_rates':result['pending_rates'],'best_completed_rate':best['peak_learning_rate']}).decode().strip())

if __name__=='__main__':main()
