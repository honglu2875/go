"""Verify a minimal spatial-readout intervention on a fixed-data comparison."""
import argparse
import json
import re
from pathlib import Path
import sys

SOURCE=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(SOURCE/'packages/gozero/src'))
from gozero.snapshots import verify,read_json,canonical_json
from gozero.checkpoints import sha256


def without_provenance(text):
    return re.sub(r'^ *# spatial study provenance begin\n.*?^ *# spatial study provenance end\n','',text,flags=re.M|re.S)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--workspace-root',type=Path,required=True)
    p.add_argument('--comparison',type=Path,required=True)
    p.add_argument('--comparison-sha256',required=True)
    p.add_argument('--control',required=True);p.add_argument('--candidate',required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();verify(SOURCE);root=a.workspace_root.resolve()
    if sha256(a.comparison)!=a.comparison_sha256:raise ValueError('Comparison identity differs')
    r=read_json(a.comparison)
    if r['status']!='passed' or not r['all_game_and_augmentation_draws_equal']:raise ValueError('Comparison did not verify draws')
    for name,h in r['input_files'].items():
        if sha256(root/name)!=h:raise ValueError('Comparison evidence changed: '+name)
    rows={x['label']:x for x in r['arms']}
    if len(rows)!=len(r['arms']) or a.control==a.candidate:raise ValueError('Distinct labels required')
    left,right=rows[a.control],rows[a.candidate]
    lc,rc=dict(left['model']),dict(right['model'])
    if lc.pop('policy_spatial_bias',False) or rc.pop('policy_spatial_bias',False) is not True or lc!=rc:
        raise ValueError('Expected spatial correction as the only model-config intervention')
    for k in ('peak_lr','end_to_peak_lr','position_exposures'):
        if left[k]!=right[k]:raise ValueError('Non-readout setting differs: '+k)
    if right['parameters']-left['parameters']!=rc['encoder_width']+1:raise ValueError('Unexpected extra parameters')
    sources=[];reports=[];closure=[]
    for row in (left,right):
        source=root/'.gozero/snapshots'/row['snapshot'];manifest=verify(source);recipe=source/manifest['recipe']
        sources.append(recipe)
        closure.append({n:v['sha256'] for n,v in manifest['files'].items() if n.startswith('packages/') or n in ('uv.lock','pyproject.toml','Cargo.lock')})
        reports.append(read_json(root/'runs'/row['attempt']/'rank-0/artifacts/result.json'))
    if closure[0]!=closure[1]:raise ValueError('Shared library or dependencies changed')
    for n in ('policy_optimizer.py','policy_model.py','policy_config.py','train.py','observation_attention.py','profile_causal.py'):
        if (sources[0]/n).read_bytes()!=(sources[1]/n).read_bytes():raise ValueError('Unrelated numerical source changed: '+n)
    if without_provenance((sources[0]/'train_policy.py').read_text())!=without_provenance((sources[1]/'train_policy.py').read_text()):
        raise ValueError('Trainer change extends beyond marked provenance fields')
    if (reports[1]['initial_shared_parameter_elements_sha256']!=reports[0]['initial_parameter_elements_sha256']
            or reports[1]['initial_spatial_projection_zero'] is not True):raise ValueError('Initial common weights differ or correction is nonzero')
    x,y=left['curve'][0],right['curve'][0]
    if abs(x['expert_kl']-y['expert_kl'])>1e-6 or abs(x['expert_top1']-y['expert_top1'])>1e-6:
        raise ValueError('Initial validation policy differs')
    paired=[]
    if len(left['curve'])!=len(right['curve']):raise ValueError('Evaluation cadence differs')
    for x,y in zip(left['curve'],right['curve']):
        for k in ('turn','validation_ids_sha256','expert_count'):
            if x[k]!=y[k]:raise ValueError('Validation population differs')
        paired.append({'turn':x['turn'],'candidate_minus_control_kl':y['expert_kl']-x['expert_kl'],
            'candidate_minus_control_top1':y['expert_top1']-x['expert_top1'],
            'phase_kl_differences':{k:y[k]-x[k] for k in x if k.startswith('phase_') and k.endswith('_kl')}})
    improvement=1-right['curve'][-1]['expert_kl']/left['curve'][-1]['expert_kl']
    latency_ratio=right['decode_median_ms']/left['decode_median_ms']
    out={'status':'passed','kind':'minimal_spatial_readout_contrast','operator_snapshot':SOURCE.name,
        'comparison':str(a.comparison),'comparison_sha256':a.comparison_sha256,
        'control':a.control,'candidate':a.candidate,'common_initialization_sha256':reports[0]['initial_parameter_elements_sha256'],
        'additional_parameters':right['parameters']-left['parameters'],'paired_curve':paired,
        'relative_endpoint_kl_improvement':improvement,'warm_decode_latency_ratio':latency_ratio,
        'replication_screen_passed':improvement>=.01 and latency_ratio<=1.15,
        'scope':'Same data, optimizer, initial common weights, encoder allocation, tokens and depth. One added spatial correction, not a new policy target. One-seed screening does not establish playing strength.'}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('xb') as f:f.write(canonical_json(out))
    a.output.chmod(0o444)
    print(json.dumps({'status':'passed','relative_kl_improvement':improvement,'latency_ratio':latency_ratio,'replication_screen_passed':out['replication_screen_passed'],'sha256':sha256(a.output)}))


if __name__=='__main__':main()
