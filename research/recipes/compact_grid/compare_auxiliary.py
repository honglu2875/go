"""Audit paired full-policy learning and the parent/trained one-pass drafts."""
import argparse
import json
from pathlib import Path
import statistics
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
    if lc.pop('first_pass_aux_weight',0)!=0 or rc.pop('first_pass_aux_weight',0)!=.25 or lc!=rc:raise ValueError('Unexpected auxiliary model intervention')
    for name in ('peak_lr','end_to_peak_lr','position_exposures','parameters','initial_parameter_elements_sha256'):
        if left[name]!=right[name]:raise ValueError('Unpaired setting or initialization: '+name)
    sources=[];closures=[];reports=[]
    for row in (left,right):
        source=root/'.gozero/snapshots'/row['snapshot'];m=verify(source);sources.append(source/m['recipe'])
        closures.append({n:v['sha256'] for n,v in m['files'].items() if n.startswith('packages/') or n in ('uv.lock','pyproject.toml','Cargo.lock')})
        reports.append(read_json(root/'runs'/row['attempt']/'rank-0/artifacts/result.json'))
    if closures[0]!=closures[1]:raise ValueError('Dependency closure changed')
    for name in ('policy_optimizer.py','train.py','observation_attention.py','profile_causal.py'):
        if (sources[0]/name).read_bytes()!=(sources[1]/name).read_bytes():raise ValueError('Unrelated numerical source changed: '+name)
    if reports[0]['decode_profile']['jaxpr']!=reports[1]['decode_profile']['jaxpr']:raise ValueError('Full inference arithmetic changed')
    if abs(left['curve'][0]['expert_kl']-right['curve'][0]['expert_kl'])>1e-6:raise ValueError('Initial full policy differs')
    new=reports[1];reference=new['draft_reference_evaluation']
    if reference['snapshot']!=left['snapshot']:raise ValueError('Wrong parent draft checkpoint')
    parent=reference['draft_validation'];draft=new['draft_validation'];main=new['validation_history'][-1]
    for row in (parent,draft,main):
        if row['episode_ids_sha256']!=left['curve'][-1]['validation_ids_sha256'] or row['metrics']['expert_count']!=left['curve'][-1]['expert_count']:raise ValueError('Unpaired draft validation population')
    if abs(reference['main_validation']['metrics']['expert_kl']-left['curve'][-1]['expert_kl'])>1e-5:raise ValueError('Parent restoration changed validation')
    improvement=1-right['curve'][-1]['expert_kl']/left['curve'][-1]['expert_kl'];latency=right['decode_median_ms']/left['decode_median_ms']
    draft_ms=1000*statistics.median(new['trained_draft_decode_profile']['host_dispatch_latency_seconds'])
    parent_draft_ms=1000*statistics.median(reference['draft_decode_profile']['host_dispatch_latency_seconds'])
    draft_improvement=1-draft['metrics']['expert_kl']/parent['metrics']['expert_kl']
    overlap_increase=draft['metrics']['full_policy_overlap']-parent['metrics']['full_policy_overlap']
    draft_full_ratio=draft_ms/right['trained_decode_median_ms']
    main_screen=improvement>=.01 and latency<=1.15
    draft_screen=(improvement>=-.01 and latency<=1.15 and draft_improvement>=.05
                  and overlap_increase>=.02 and draft_full_ratio<=.65)
    out={'status':'passed','kind':'first_pass_auxiliary_contrast','operator_snapshot':SOURCE.name,
        'comparison':str(a.comparison),'comparison_sha256':a.comparison_sha256,'control':a.control,'candidate':a.candidate,
        'relative_endpoint_kl_improvement':improvement,'warm_decode_latency_ratio':latency,
        'replication_screen_passed':main_screen,'draft_utility_screen_passed':draft_screen,
        'replication_eligible':main_screen or draft_screen,
        'parent_draft_kl':parent['metrics']['expert_kl'],'trained_draft_kl':draft['metrics']['expert_kl'],
        'parent_draft_full_policy_overlap':parent['metrics']['full_policy_overlap'],
        'trained_draft_full_policy_overlap':draft['metrics']['full_policy_overlap'],
        'parent_draft_full_argmax_agreement':parent['metrics']['full_argmax_agreement'],
        'trained_draft_full_argmax_agreement':draft['metrics']['full_argmax_agreement'],
        'relative_draft_kl_improvement':draft_improvement,'draft_full_policy_overlap_increase':overlap_increase,
        'trained_draft_over_full_neural_latency':draft_full_ratio,
        'parent_draft_decode_ms':parent_draft_ms,'trained_draft_decode_ms':draft_ms,
        'trained_full_over_draft_neural_latency':right['trained_decode_median_ms']/draft_ms,
        'scope':'Same full architecture, initial arrays, optimizer and actual samples. Add shared first-pass supervision and masked training frames; full inference arithmetic preserved. Draft evaluation uses full-pass history in both arms. Added training work is separate; no speculative acceptance, Go strength or MFU claim.'}
    with a.output.open('xb') as f:f.write(canonical_json(out))
    a.output.chmod(0o444);print(json.dumps(out),flush=True)


if __name__=='__main__':main()
