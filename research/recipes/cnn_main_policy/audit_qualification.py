"""Gate the main-only CNN on full-batch TPU updates, identity and checkpoint audit."""
import argparse
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
SOURCE=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(SOURCE/'packages/gozero/src'))
from gozero.snapshots import verify,read_json,canonical_json
from gozero.checkpoints import sha256


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--workspace-root',type=Path,required=True)
    p.add_argument('--attempt',type=Path,required=True)
    p.add_argument('--cpu-budget',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--audit-output',type=Path,required=True)
    a=p.parse_args(); verify(SOURCE); root=a.workspace_root.resolve()
    closed=read_json(a.attempt/'result.json'); source=root/'.gozero/snapshots'/closed['snapshot_id']
    manifest=verify(source); recipe=source/manifest['recipe']; c=read_json(source/'resolved_config.json')
    cpu=read_json(a.cpu_budget); expected=cpu['config']
    if (closed['status']!='passed' or c['steps']!=4 or c['learner']['games_per_host']!=32
            or c['expected_processes']!=4 or c['expected_devices']!=16
            or c['model']!=expected['model'] or c['dataset']!=expected['dataset']
            or c['seed']!=expected['seed'] or c.get('checkpoint_temporary') or c.get('checkpoint_archive')
            or cpu['status']!='passed'):
        raise ValueError('Wrong full-shape qualification')
    for n,wanted in cpu['source_sha256'].items():
        if sha256(recipe/n)!=wanted: raise ValueError('CPU/TPU numerical source differs')
    with a.audit_output.with_suffix('.log').open('xb') as log:
        subprocess.run([sys.executable,'-B',str(recipe/'audit_learning.py'),
            '--workspace-root',str(root),'--attempt',str(a.attempt),'--output',str(a.audit_output)],
            env={**os.environ,'JAX_PLATFORMS':'cpu','OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1'},
            stdout=log,stderr=subprocess.STDOUT,check=True,timeout=600)
    historical=read_json(root/'runs/cnn-lr-followup-f3de116f/sweep/2-1e-03-audit.json')
    if historical['training_snapshot']!=cpu['baseline_snapshot']: raise ValueError('Reference changed')
    old=read_json(root/'runs'/historical['attempt']/'rank-0/artifacts/result.json')
    reports=[read_json(a.attempt/f'rank-{h}/artifacts/result.json') for h in range(4)]
    for r in reports:
        if set(r['compiled_updates'])!={'128','256','384'}: raise ValueError('Missing full training bucket')
        if r['parameter_count']!=cpu['stored_parameters']: raise ValueError('Parameter schema differs')
        if r['initial_parameter_elements_sha256']!=old['initial_parameter_elements_sha256']:
            raise ValueError('Initial CNN arrays differ from the historical control')
        q=r['decode_profile']
        if (q['analytical']!=cpu['decode']['analytical'] or q['jaxpr']!=cpu['decode']['traced']
                or q['batch_size']!=128 or q['board_size']!=9): raise ValueError('TPU inference arithmetic differs')
        if not Path(r['latest_checkpoint']['owner_checkpoint_path']).is_relative_to(root/'runs'):
            raise ValueError('Checkpoint is not persistent in the workspace')
    r=reports[0]
    result={'status':'passed','kind':'cnn_main_full_tpu_qualification','snapshot':source.name,
        'attempt':a.attempt.name,'cpu_budget_sha256':sha256(a.cpu_budget),
        'audit_path':str(a.audit_output),'audit_sha256':sha256(a.audit_output),
        'historical_audit_sha256':sha256(root/'runs/cnn-lr-followup-f3de116f/sweep/2-1e-03-audit.json'),
        'exact_initial_array_identity':True,'stored_parameters':cpu['stored_parameters'],
        'active_parameters':cpu['active_parameters'],'unused_helper_slots':cpu['unused_helper_slots'],
        'initial_decode_median_ms':1000*statistics.median(r['decode_profile']['host_dispatch_latency_seconds']),
        'checkpoint':r['latest_checkpoint'],'compiled_updates':r['compiled_updates']}
    with a.output.open('xb') as f:f.write(canonical_json(result))
    a.output.chmod(0o444)
    print(json.dumps({'status':'passed','output':str(a.output)}),flush=True)


if __name__=='__main__':main()
