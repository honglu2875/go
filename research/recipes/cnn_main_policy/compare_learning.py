#!/usr/bin/env python3
"""Compare closed runs only after checking common settings and every data draw."""
import argparse
import json
from pathlib import Path
import statistics
import sys
SOURCE=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(SOURCE/'packages/gozero/src'))
from gozero.snapshots import canonical_json,read_json,verify
from gozero.checkpoints import sha256


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--workspace-root',type=Path,required=True)
    p.add_argument('--audit',nargs=3,action='append',required=True,metavar=('LABEL','PATH','SHA256'))
    p.add_argument('--output',type=Path,required=True);args=p.parse_args();verify(SOURCE);root=args.workspace_root.resolve()
    if len(args.audit)!=2 or args.audit[0][0]==args.audit[1][0]:raise ValueError('Two uniquely named arms required')
    arms=[];settings=[];draws=[]
    for label,path,digest in args.audit:
        path=Path(path)
        if sha256(path)!=digest:raise ValueError('Audit changed')
        audit=read_json(path)
        if audit['status']!='passed':raise ValueError('Run is not qualified')
        for name,wanted in audit['input_files'].items():
            if sha256(root/name)!=wanted:raise ValueError('Audited input changed')
        attempt=root/'runs'/audit['attempt'];closed=read_json(attempt/'result.json')
        if sha256(attempt/'result.json')!=audit['closed_result_sha256'] or closed['status']!='passed':raise ValueError('Attempt changed')
        snapshot=root/'.gozero/snapshots'/audit['training_snapshot'];verify(snapshot);c=read_json(snapshot/'resolved_config.json')
        settings.append({key:c[key] for key in ['dataset','seed','steps','learner','evaluation','eval_every','expected_devices','expected_processes']})
        reports=[read_json(attempt/f'rank-{host}/artifacts/result.json') for host in range(c['expected_processes'])]
        rank_draws={}
        for host,report in enumerate(reports):
            log=[json.loads(x) for x in (attempt/f'rank-{host}/artifacts/metrics.jsonl').read_text().splitlines()]
            rank_draws[report['jax_rank']]=[{k:x[k] for k in ['turn','bucket','local_entries_sha256','local_symmetries','expert_positions']} for x in log]
        draws.append(rank_draws);r=reports[0]
        times={0:0.,**{x['turn']:x['cumulative_learning_seconds'] for x in [json.loads(y) for y in (attempt/'rank-0/artifacts/metrics.jsonl').read_text().splitlines()]}}
        curve=[{'turn':x['turn'],'learning_seconds':times[x['turn']],'validation_ids_sha256':x['episode_ids_sha256'],**x['metrics']} for x in audit['validation_curve']]
        decode=r.get('decode_profile')
        arms.append({'label':label,'audit_path':str(path),'audit_sha256':digest,'training_snapshot':snapshot.name,
            'parameter_count':r['parameter_count'],'curve':curve,'timing':audit['timing'],'clipped_updates':audit['clipped_updates'],
            'position_exposures':audit['expert_positions'],'reserved_chip_hours':closed['reserved_chip_hours'],
            'decode_median_seconds':statistics.median(decode['host_dispatch_latency_seconds']) if decode else None,
            'decode_scope':decode['scope'] if decode else None,'checkpoint':r['latest_checkpoint']})
    if settings[0]!=settings[1] or draws[0]!=draws[1]:raise ValueError('Controlled settings or draws differ')
    left,right=arms;paired=[]
    if len(left['curve'])!=len(right['curve']):raise ValueError('Evaluation cadence differs')
    for x,y in zip(left['curve'],right['curve']):
        if x['turn']!=y['turn'] or x['validation_ids_sha256']!=y['validation_ids_sha256'] or x['expert_count']!=y['expert_count']:
            raise ValueError('Validation population differs')
        paired.append({'turn':x['turn'],'right_minus_left_kl':y['expert_kl']-x['expert_kl'],
            'right_minus_left_top1':y['expert_top1']-x['expert_top1'],
            'left_learning_seconds':x['learning_seconds'],'right_learning_seconds':y['learning_seconds'],
            'phase_kl_differences':{k:y[k]-x[k] for k in x if k.startswith('phase_') and k.endswith('_kl')}})
    result={'kind':'paired_sequential_learning_comparison','status':'passed','operator_snapshot':SOURCE.name,
        'common_settings':settings[0],'all_episode_and_augmentation_draws_identical':True,'arms':arms,'paired_curve':paired,
        'scope':'One seed, shared fixed data and optimizer schedule; test closed. Descriptive learnability differences, not convergence, causal attention isolation, Go strength or RL efficiency.'}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('xb') as f:f.write(canonical_json(result))
    args.output.chmod(0o444)
    print(json.dumps({'status':'passed','endpoint':paired[-1],'sha256':sha256(args.output)}),flush=True)


if __name__=='__main__':main()
