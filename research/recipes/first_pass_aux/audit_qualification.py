"""Gate a full learning run on full-shape TPU and durable checkpoint evidence."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys
import os
SOURCE=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(SOURCE/'packages/gozero/src'))
from gozero.snapshots import verify,read_json,canonical_json
from gozero.checkpoints import sha256

NUMERICAL_FILES=('encoder.py','causal.py','compute_budget.py','policy_model.py','policy_config.py',
    'policy_optimizer.py','profile_causal.py','train_policy.py','train.py','observation_attention.py',
    'draft_model.py','draft_mask.py','profile_draft.py','qualify_draft_runtime.py','training_arithmetic.py','reference_draft.py')


def main():
    p=argparse.ArgumentParser();p.add_argument('--workspace-root',type=Path,required=True)
    p.add_argument('--attempt',type=Path,required=True);p.add_argument('--cpu-budget',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--audit-output',type=Path,required=True)
    a=p.parse_args();root=a.workspace_root.resolve();closed=read_json(a.attempt/'result.json')
    source=root/'.gozero/snapshots'/closed['snapshot_id'];m=verify(source);recipe=source/m['recipe'];c=read_json(source/'resolved_config.json')
    cpu=read_json(a.cpu_budget)
    if (closed['status']!='passed' or c['steps']!=4 or c['learner']['games_per_host']!=32
            or c['expected_processes']!=4 or c['expected_devices']!=16 or c['model']['width']!=768
            or c['dataset']['warmup_buckets']!=[128,256,384] or cpu['config']['model']!=c['model']
            or c.get('checkpoint_temporary') or c.get('checkpoint_archive')):raise ValueError('Incomplete/wrong full-shape qualification')
    numerical={n:sha256(recipe/n) for n in NUMERICAL_FILES}
    if any(cpu['source_sha256'][n]!=v for n,v in numerical.items()):raise ValueError('CPU/TPU numerical source differs')
    with a.audit_output.with_suffix('.log').open('xb') as log:
        subprocess.run(['taskset','-c','112-119',sys.executable,'-B',str(recipe/'audit_learning.py'),
            '--workspace-root',str(root),'--attempt',str(a.attempt),'--output',str(a.audit_output)],
            env={**os.environ,'JAX_PLATFORMS':'cpu','OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1'},
            stdout=log,stderr=subprocess.STDOUT,check=True,timeout=600)
    reports=[read_json(a.attempt/f'rank-{h}/artifacts/result.json') for h in range(4)]
    for r in reports:
        if set(r['compiled_updates'])!={'128','256','384'}:raise ValueError('Some training buckets were not exercised')
        if r['parameter_count']!=cpu['cases']['128']['analytical']['trainable_parameters']:raise ValueError('TPU schema differs from CPU')
        for key in ('decode_profile','trained_decode_profile'):
            q=r[key]
            if (q['max_logit_error']>q['max_logit_tolerance'] or q['max_policy_tv']>q['max_policy_tv_tolerance']
                    or not q['cache_donated'] or not q['cache_outputs_observed']):raise ValueError('Cached/full policy comparison failed')
        if c['model'].get('first_pass_aux_weight',0):
            kernel=r['draft_kernel_qualification']
            if kernel['status']!='passed' or [x['bucket'] for x in kernel['cases']]!=[128,256,384]:raise ValueError('Incomplete draft kernel qualification')
            for row in kernel['cases']:
                if any(a>row['absolute_tolerance'] or v>row['relative_l2_tolerance'] for a,v in row['output_q_k_v_errors']):raise ValueError('Draft kernel errors exceed tolerance')
            for key in ('draft_decode_profile','trained_draft_decode_profile'):
                q=r[key]
                if (q['max_logit_error']>.04 or q['max_policy_tv']>.005 or q['joint_main_max_logit_error']>.04
                        or q['joint_main_max_policy_tv']>.005 or not q['full_pass_history_cache']
                        or not q['cache_donated'] or not q['cache_outputs_observed']):raise ValueError('Draft/full history semantics failed')
            val=r['draft_validation'];main=r['validation_history'][-1]
            if val['episode_ids_sha256']!=main['episode_ids_sha256'] or val['metrics']['expert_count']!=main['metrics']['expert_count']:raise ValueError('Draft validation population differs')
            reference=r['draft_reference_evaluation']
            if reference['audit_sha256']!=c['draft_reference']['audit_sha256']:raise ValueError('Wrong parent draft reference')
            if r['initial_parameter_elements_sha256']!=reference['parent_initial_parameter_elements_sha256']:raise ValueError('Paired initial weights differ')
            if reference['draft_validation']['episode_ids_sha256']!=val['episode_ids_sha256']:raise ValueError('Parent and trained draft validation populations differ')
        if str(r['latest_checkpoint']['path']).startswith('/dev/shm/'):raise ValueError('Checkpoint must be persistent')
    r=reports[0]
    out={'status':'passed','kind':'shared_encoder_full_tpu_qualification','snapshot':source.name,'attempt':a.attempt.name,
        'model':c['model'],'seed':c['seed'],'dataset':c['dataset'],'numerical_sources':numerical,
        'cpu_budget_sha256':sha256(a.cpu_budget),'audit_path':str(a.audit_output),'audit_sha256':sha256(a.audit_output),
        'initial_parameter_elements_sha256':r['initial_parameter_elements_sha256'],'parameter_count':r['parameter_count'],
        'initial_decode_median_ms':1000*statistics.median(r['decode_profile']['host_dispatch_latency_seconds']),
        'trained_decode_median_ms':1000*statistics.median(r['trained_decode_profile']['host_dispatch_latency_seconds']),
        'compiled_updates':r['compiled_updates'],'checkpoint':r['latest_checkpoint']}
    with a.output.open('xb') as f:f.write(canonical_json(out))
    print(json.dumps({'status':'passed','output':str(a.output),'initial_decode_ms':out['initial_decode_median_ms'],'trained_decode_ms':out['trained_decode_median_ms']}),flush=True)


if __name__=='__main__':main()
