"""Audit one fixed-data within-board-attention intervention and its selection screen."""
import argparse,json,re
from pathlib import Path
import sys
SOURCE=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(SOURCE/'packages/gozero/src'))
from gozero.snapshots import verify,read_json,canonical_json
from gozero.checkpoints import sha256


def model(c):
    c=dict(c)
    if not c.get('policy_context_dim',0):c.pop('policy_context_dim',None)
    if not c.get('encoder_attention_blocks',0):
        for name in ('encoder_attention_blocks','encoder_attention_heads','encoder_attention_mlp_hidden','encoder_rope_theta'):c.pop(name,None)
    return c


def strip_provenance(t):return re.sub(r'^ *# spatial study provenance begin\n.*?^ *# spatial study provenance end\n','',t,flags=re.M|re.S)


def main():
    p=argparse.ArgumentParser();p.add_argument('--workspace-root',type=Path,required=True)
    p.add_argument('--comparison',type=Path,required=True);p.add_argument('--comparison-sha256',required=True)
    p.add_argument('--control',required=True);p.add_argument('--candidate',required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();verify(SOURCE);root=a.workspace_root.resolve()
    if sha256(a.comparison)!=a.comparison_sha256:raise ValueError('Comparison changed')
    r=read_json(a.comparison)
    if r['status']!='passed' or not r['all_game_and_augmentation_draws_equal']:raise ValueError('Draws not verified')
    for name,h in r['input_files'].items():
        if sha256(root/name)!=h:raise ValueError('Changed evidence: '+name)
    rows={x['label']:x for x in r['arms']};left,right=rows[a.control],rows[a.candidate]
    lc,rc=model(left['model']),model(right['model'])
    expected={'encoder_attention_blocks':4,'encoder_attention_heads':12,'encoder_attention_mlp_hidden':1024,'encoder_rope_theta':10000.}
    if any(rc.pop(k,None)!=v for k,v in expected.items()) or lc!=rc:raise ValueError('Unexpected model intervention')
    for name in ('peak_lr','end_to_peak_lr','position_exposures'):
        if left[name]!=right[name]:raise ValueError('Unpaired setting: '+name)
    if right['parameters']-left['parameters']!=-36864:raise ValueError('Unexpected parameter reallocation')
    sources=[];closures=[]
    for row in (left,right):
        source=root/'.gozero/snapshots'/row['snapshot'];m=verify(source);sources.append(source/m['recipe'])
        closures.append({n:v['sha256'] for n,v in m['files'].items() if n.startswith('packages/') or n in ('uv.lock','pyproject.toml','Cargo.lock')})
    if closures[0]!=closures[1]:raise ValueError('Dependencies changed')
    for name in ('policy_optimizer.py','policy_model.py','policy_config.py','train.py','observation_attention.py','profile_causal.py'):
        if (sources[0]/name).read_bytes()!=(sources[1]/name).read_bytes():raise ValueError('Unrelated numerical source changed: '+name)
    if strip_provenance((sources[0]/'train_policy.py').read_text())!=strip_provenance((sources[1]/'train_policy.py').read_text()):raise ValueError('Trainer changed outside provenance')
    improvement=1-right['curve'][-1]['expert_kl']/left['curve'][-1]['expert_kl'];latency=right['decode_median_ms']/left['decode_median_ms']
    out={'status':'passed','kind':'encoder_attention_contrast','operator_snapshot':SOURCE.name,
        'comparison':str(a.comparison),'comparison_sha256':a.comparison_sha256,'control':a.control,'candidate':a.candidate,
        'relative_endpoint_kl_improvement':improvement,'warm_decode_latency_ratio':latency,
        'replication_screen_passed':improvement>=.01 and latency<=1.15,
        'parameter_difference':right['parameters']-left['parameters'],
        'initial_validation_kl_difference':right['curve'][0]['expert_kl']-left['curve'][0]['expert_kl'],
        'scope':'Same samples, optimizer, width, temporal trunk and readout configuration; replace four encoder blocks. Parameter families differ; identical full initialization is not asserted. No Go-strength or MFU claim.'}
    with a.output.open('xb') as f:f.write(canonical_json(out))
    a.output.chmod(0o444);print(json.dumps(out),flush=True)


if __name__=='__main__':main()
