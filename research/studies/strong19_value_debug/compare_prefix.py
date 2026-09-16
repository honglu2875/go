"""Compare the registered CE prefix with the original pilot at the same turn."""
import argparse
import hashlib
import json
from pathlib import Path
import time

ROOT=Path(__file__).resolve().parents[3]
STUDY=Path(__file__).resolve().parent


def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    plan=json.loads((STUDY/'ce-prefix-plan-002.json').read_text())
    closure=json.loads((STUDY/'ce-prefix-001/result.json').read_text())
    if closure['status']!='passed' or closure['plan_sha256']!=sha(STUDY/'ce-prefix-plan-002.json'):
        raise ValueError('Prefix audit/retention did not pass')
    control=ROOT/plan['control_audit'];candidate=STUDY/'ce-prefix-001/audit.json'
    if sha(control)!=plan['control_audit_sha256'] or sha(candidate)!=closure['audit_sha256']:
        raise ValueError('Audit identity changed')
    old,new=json.loads(control.read_text()),json.loads(candidate.read_text());turn=plan['stop_turn']
    if old['initial_parameters_sha256']!=new['initial_parameters_sha256'] or new['positions']!=plan['expected_positions']:
        raise ValueError('Initialization or exposure mismatch')
    observations={}
    for key in ('validation_history','training_probe_history'):
        before=next(r for r in old[key] if r['turn']==turn);after=next(r for r in new[key] if r['turn']==turn)
        if before['episode_ids_sha256']!=after['episode_ids_sha256'] or before['raw_totals']['expert_count']!=after['raw_totals']['expert_count']:
            raise ValueError('Comparison population differs')
        metrics={}
        for m in ('expert_kl','family_kl','value_mse','value_family_mse','value_mean_prediction'):
            b=before['metrics'][m];n=after['metrics'][m]
            metrics[m]=dict(original_mse=b,cross_entropy=n,difference=n-b,ratio=n/b if b else None)
        observations[key]=dict(positions=after['raw_totals']['expert_count'],episode_ids_sha256=after['episode_ids_sha256'],
            metrics=metrics,probabilities={k:v for k,v in after['metrics'].items() if k.startswith('logit_')})
    paths=dict(original=ROOT/'runs/pod-20260916T010420Z-754c0b22/rank-0/artifacts/metrics.jsonl',
               cross_entropy=ROOT/'runs'/closure['attempt']/'rank-0/artifacts/metrics.jsonl')
    curves={name:[json.loads(line) for line in path.read_text().splitlines()][:turn] for name,path in paths.items()}
    for x,y in zip(curves['original'],curves['cross_entropy']):
        for k in ('turn','bucket','positions','learning_rate'):
            if x[k]!=y[k]:raise ValueError('Matched update clock differs: '+k)
    result=dict(kind='objective_only_value_prefix_comparison',status='passed',created=time.time(),turn=turn,
        positions=new['positions'],parameters=new['parameters'],observations=observations,
        input_sha256={'plan':sha(STUDY/'ce-prefix-plan-002.json'),'controller':sha(STUDY/'ce-prefix-001/result.json'),
            'control_audit':sha(control),'candidate_audit':sha(candidate),**{k:sha(v) for k,v in paths.items()}},
        initialization_identical=True,batches_and_schedule_independently_verified=True,
        curves={name:[{k:v for k,v in row.items() if k in ('turn','main_value_mse','gradient_norm_value','policy_loss',
            'main_value_probability_win','main_value_probability_loss','main_value_probability_neutral')} for row in data] for name,data in curves.items()},
        learning_seconds={name:data[-1]['cumulative_learning_seconds'] for name,data in curves.items()},
        operator_sha256=sha(Path(__file__)),
        scope='Nine-update single-seed objective screen with fixed complete validation; no final-horizon, architecture ranking, MFU or playing-strength claim.')
    with a.output.open('x') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(dict(status='passed',positions=result['positions'],validation=observations['validation_history'],
        learning_seconds=result['learning_seconds'])))


if __name__=='__main__':main()
