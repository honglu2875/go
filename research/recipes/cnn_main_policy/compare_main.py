"""Verify the historical CNN versus its sole main-only loss intervention."""
import argparse
from pathlib import Path
import sys
SOURCE=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(SOURCE/'packages/gozero/src'))
from gozero.snapshots import verify,read_json,canonical_json
from gozero.checkpoints import sha256


def main():
    p=argparse.ArgumentParser();p.add_argument('--workspace-root',type=Path,required=True)
    p.add_argument('--comparison',type=Path,required=True);p.add_argument('--comparison-sha256',required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();verify(SOURCE);root=a.workspace_root.resolve()
    if sha256(a.comparison)!=a.comparison_sha256:raise ValueError('Comparison changed')
    r=read_json(a.comparison)
    if r['status']!='passed' or not r['all_game_and_augmentation_draws_equal']:raise ValueError('Unqualified comparison')
    for name,h in r['input_files'].items():
        if sha256(root/name)!=h:raise ValueError('Changed evidence')
    rows={x['label']:x for x in r['arms']};old=rows['CNN with helper'];new=rows['CNN main only']
    sources=[];closures=[];configs=[]
    for row in (old,new):
        snapshot=root/'.gozero/snapshots'/row['snapshot'];m=verify(snapshot);sources.append(snapshot/m['recipe'])
        configs.append(read_json(snapshot/'resolved_config.json'))
        closures.append({n:x['sha256'] for n,x in m['files'].items() if n.startswith('packages/') or n in ('uv.lock','Cargo.lock','pyproject.toml')})
    if configs[0]!=configs[1]:raise ValueError('Configuration changed')
    added=set(closures[1])-set(closures[0])
    storage_names=('checkpoint_archive','checkpoint_parts','checkpoint_stage','ram_checkpoints')
    expected_added={'packages/gozero/src/gozero/'+n+'.py' for n in storage_names}
    if added!=expected_added or any(closures[1].get(n)!=h for n,h in closures[0].items()):raise ValueError('A pre-existing dependency changed or an unexpected file appeared')
    # This historical snapshot predates four archival utilities. Every original
    # library module/lock is identical. Neither original library source nor the
    # exact executed recipe imports/references any of these added utilities.
    used_recipe=('katago.py','policy_optimizer.py','train_policy.py','policy_config.py','train.py','compute_budget.py','policy_model.py')
    original_root=sources[0].parents[2]
    old_modules=[original_root/n for n in closures[0] if n.startswith('packages/') and n.endswith('.py')]
    texts=[p.read_text() for p in old_modules]+[(sources[1]/n).read_text() for n in used_recipe]
    if any(name in text for name in storage_names for text in texts):raise ValueError('Added storage utility may be referenced by executed source')
    for name in ('katago.py','policy_optimizer.py','train_policy.py','policy_config.py','train.py','compute_budget.py'):
        if (sources[0]/name).read_bytes()!=(sources[1]/name).read_bytes():raise ValueError('Unexpected numerical change: '+name)
    if (sources[0]/'policy_model.py').read_text().split('\ndef losses(')[0]!=(sources[1]/'policy_model.py').read_text().split('\ndef losses(')[0]:raise ValueError('Changes outside objective')
    if old['initial_parameter_elements_sha256']!=new['initial_parameter_elements_sha256']:raise ValueError('Initial arrays differ')
    for key in ('expert_kl','expert_top1'):
        if abs(old['curve'][0][key]-new['curve'][0][key])>1e-6:raise ValueError('Initial policy differs')
    result={'status':'passed','kind':'cnn_main_only_contrast','operator_snapshot':SOURCE.name,
        'comparison':str(a.comparison),'comparison_sha256':a.comparison_sha256,
        'exact_initial_arrays_and_config':True,'stored_parameters':new['parameters'],'active_parameters':232134784,
        'unused_helper_slots':297088,'original_helper_kl':old['curve'][-1]['expert_kl'],
        'unchanged_original_dependency_closure':closures[0],
        'additional_unreferenced_storage_utilities':{n:closures[1][n] for n in sorted(added)},
        'main_only_kl':new['curve'][-1]['expert_kl'],
        'relative_endpoint_kl_improvement':1-new['curve'][-1]['expert_kl']/old['curve'][-1]['expert_kl'],
        'warm_decode_latency_ratio':new['decode_median_ms']/old['decode_median_ms'],
        'learning_seconds_ratio':new['timing']['learning_seconds']/old['timing']['learning_seconds'],
        'scope':'Same main architecture, every initial array, exact draws and training settings; remove helper loss and its batch-statistics path together. No separate isolation of these effects or Go-strength claim.'}
    with a.output.open('xb') as f:f.write(canonical_json(result))
    a.output.chmod(0o444);print(result)


if __name__=='__main__':main()
