"""Compare an audited sparse endpoint with the same-draw dense prefix."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[3];STUDY=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'packages/gozero/src'))
from gozero.snapshots import canonical_json
p=argparse.ArgumentParser();p.add_argument('--label',required=True);a=p.parse_args()
plan_path=STUDY/(a.label+'-plan.json');plan=json.loads(plan_path.read_text())
stage=STUDY/plan['output_directory'];closed=json.loads((stage/'result.json').read_text());assert closed['status']=='passed'
moe=json.loads((stage/'audit.json').read_text());dense=json.loads((ROOT/'research/studies/strong19_recovery/flat-stage-003/audit.json').read_text());assert moe['status']==dense['status']=='passed'
assert moe['dataset_manifest_sha256']==dense['dataset_manifest_sha256'] and moe['steps']==plan['steps'] and moe['positions']==plan['expected_positions']
read_rows=lambda attempt,name:[json.loads(s) for s in (ROOT/'runs'/attempt/'rank-0/artifacts'/name).read_text().splitlines()]
mr=read_rows(closed['attempt'],'metrics.jsonl');dr=read_rows(plan['reference_attempt'],'metrics.jsonl')
for i,row in enumerate(mr):
    old=dr[i]
    for name in ('turn','bucket','positions','learning_rate'):assert row[name]==old[name],(name,i)
# Rank-local draw hashes are audited independently against the same logical-rank
# RNG schedule; host-to-logical-rank mappings are not assumed to remain fixed.
md={r['turn']:r for r in moe['validation_history']};dd={r['turn']:r for r in dense['validation_history']}
mt={r['turn']:r for r in moe['training_probe_history']};dt={r['turn']:r for r in dense['training_probe_history']}
turns=sorted(md);curves=[]
for turn in turns:
    assert md[turn]['episode_ids_sha256']==dd[turn]['episode_ids_sha256']
    assert mt[turn]['episode_ids_sha256']==dt[turn]['episode_ids_sha256']
    def metrics(rows):return {k:rows[turn]['metrics'][k] for k in ('expert_kl','family_kl','expert_top1','value_mse','value_family_mse')}
    curves.append(dict(turn=turn,moe_validation=metrics(md),dense_validation=metrics(dd),moe_training_probe=metrics(mt),dense_training_probe=metrics(dt)))
endpoint=curves[-1]
seconds=dict(moe=mr[-1]['cumulative_learning_seconds'],dense=dr[len(mr)-1]['cumulative_learning_seconds'])
wall_turns=[t for t in sorted(dd) if t>0 and dr[t-1]['cumulative_learning_seconds']<=seconds['moe']]
matched_time_turn=max(wall_turns) if wall_turns else 0
router_names=[k for k in mr[0] if k.startswith('moe_')]
router=dict(last={k:mr[-1][k] for k in router_names},maximum_dead_expert_fraction=max(r['moe_dead_expert_fraction'] for r in mr),
            maximum_load_cv2=max(r['moe_load_cv2'] for r in mr),all_tokens_retained=all(r['moe_dropped_tokens']==0 for r in mr))
result=dict(status='passed',kind='same_draw_moe_dense_prefix_comparison',steps=plan['steps'],positions=moe['positions'],
    parameters=dict(dense=dense['parameters'],moe=moe['parameters']),active_flop_ratio=plan['nominal_active_flop_ratio'],
    learning_seconds=seconds,learning_time_ratio=seconds['moe']/seconds['dense'],curves=curves,router=router,
    endpoint_policy_kl_relative_change=endpoint['moe_validation']['expert_kl']/endpoint['dense_validation']['expert_kl']-1,
    endpoint_value_mse_relative_change=endpoint['moe_validation']['value_mse']/endpoint['dense_validation']['value_mse']-1,
    historical_learning_time_reference=dict(turn=matched_time_turn,metrics=dd[matched_time_turn]['metrics'],
        note='Earlier dense validation endpoint within the sparse run recorded learning-time budget. Historical timing, not a simultaneous equal-wall-time experiment.'),
    claims_strength=False,limitations=['One paired seed; warmup-length short screen, not a convergence or generality result.',
        'Equal logical active matrix FLOPs excludes physical padding, routing, sorting, optimizer and communication work.',
        'No KataGo match evaluation or test-target access in this architecture screen.'],
    inputs_sha256={str(q.relative_to(ROOT)):hashlib.sha256(q.read_bytes()).hexdigest() for q in [plan_path,stage/'audit.json',stage/'result.json',ROOT/'research/studies/strong19_recovery/flat-stage-003/audit.json']})
output=STUDY/(a.label+'-comparison.json')
with output.open('xb') as f:f.write(canonical_json(result))
lines=['# Sparse/dense learning screen','',f"Completed {plan['steps']} updates and {moe['positions']:,} matched position exposures per model.",'',
       '| Model | Parameters | Validation policy KL | Validation value MSE | Learning hours |',
       '|---|---:|---:|---:|---:|']
for name in ('dense','moe'):
    m=endpoint[name+'_validation'];lines.append(f"| {name} | {result['parameters'][name]:,} | {m['expert_kl']:.6f} | {m['value_mse']:.6f} | {seconds[name]/3600:.3f} |")
lines+=['',f"Policy KL change: {100*result['endpoint_policy_kl_relative_change']:+.2f}%. Value MSE change: {100*result['endpoint_value_mse_relative_change']:+.2f}% (lower is better).",
        f"Learning time ratio: {result['learning_time_ratio']:.3f}x. Logical active matrix FLOP ratio: {result['active_flop_ratio']:.6f}x.",'',
        'All updates, logical-rank data/augmentation replay, full model/AdamW state, and disk peer replicas passed their audits. Router load, train/validation curves and historical time comparison are in the accompanying JSON.','',*result['limitations']]
with (STUDY/(a.label+'-RESULTS.md')).open('x') as f:f.write('\n'.join(lines)+'\n')
print(json.dumps({k:result[k] for k in ('status','steps','positions','learning_time_ratio','endpoint_policy_kl_relative_change','endpoint_value_mse_relative_change')}))
