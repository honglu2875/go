"""Audit a full-size four-update TPU probe and publish the learning gate."""
import argparse
import os
from pathlib import Path
import statistics
import subprocess
import sys
SOURCE=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(SOURCE/'packages/gozero/src'))
from gozero.snapshots import verify,read_json
from gozero.checkpoints import sha256
from gozero.checkpoint_archive import publish,verify_files
from prepare_variant import NUMERICAL_FILES


def main():
    p=argparse.ArgumentParser();p.add_argument('--workspace-root',type=Path,required=True)
    p.add_argument('--attempt',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--audit-output',type=Path,required=True);a=p.parse_args();root=a.workspace_root.resolve();verify(SOURCE)
    closed=read_json(a.attempt/'result.json');snapshot=root/'.gozero/snapshots'/closed['snapshot_id'];verify(snapshot)
    c=read_json(snapshot/'resolved_config.json')
    if (closed['status']!='passed' or c['steps']!=4 or c['learner']['games_per_host']!=32
            or c['expected_processes']!=4 or c['expected_devices']!=16 or c['checkpoint_temporary'] is not True
            or c['dataset']['warmup_buckets']!=[128,256,384] or c['model']['width']!=768):
        raise ValueError('Wrong or incomplete full-shape qualification')
    with a.audit_output.with_suffix('.log').open('xb') as f:
        subprocess.run(['taskset','-c','64-95',sys.executable,'-B',str(snapshot/'research/recipes/visual_token_encoder/audit_learning.py'),
            '--workspace-root',str(root),'--attempt',str(a.attempt),'--output',str(a.audit_output)],
            env={**os.environ,'JAX_PLATFORMS':'cpu','OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1'},
            stdout=f,stderr=subprocess.STDOUT,check=True,timeout=300)
    report=read_json(a.attempt/'rank-0/artifacts/result.json');profile=report['decode_profile'];temporary=report['latest_checkpoint']['temporary']
    verify_files(Path(temporary['cache_path']),temporary)
    if (profile['max_logit_error']>profile['max_logit_tolerance'] or profile['max_policy_tv']>profile['max_policy_tv_tolerance']
            or not profile['cache_donated'] or not profile['cache_outputs_observed']):raise ValueError('Cached decoder qualification failed')
    files=[Path(temporary['cache_path'])/n for n in ['arrays.npz','manifest.json','state.json','actors.json']]
    files.append(Path(temporary['cache_path']).with_suffix('.group.json'))
    if any(p.stat().st_uid!=0 or p.stat().st_gid!=0 for p in files):raise ValueError('Temporary checkpoint is vulnerable to logout cleanup')
    numerical={n:sha256(snapshot/'research/recipes/visual_token_encoder'/n) for n in NUMERICAL_FILES}
    result={'kind':'encoder_full_tpu_qualification','status':'passed','snapshot':snapshot.name,'attempt':a.attempt.name,
        'model':c['model'],'numerical_sources':numerical,'initial_parameter_elements_sha256':report['initial_parameter_elements_sha256'],
        'parameter_count':report['parameter_count'],'audit_path':str(a.audit_output.relative_to(root)),
        'audit_sha256':sha256(a.audit_output),'decode_median_ms':1000*statistics.median(profile['host_dispatch_latency_seconds']),
        'max_policy_tv':profile['max_policy_tv'],'max_logit_error':profile['max_logit_error'],
        'decode_profile':profile,'temporary_checkpoint':temporary,
        'scope':'Full candidate, full training shapes/buckets, four finite updates, complete checkpoint/data audit, protected RAM files, full cached decode arithmetic and numerical agreement. The short LR schedule is qualification only.'}
    publish(a.output,result);print(a.output,sha256(a.output),flush=True)


if __name__=='__main__':main()
