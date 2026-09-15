"""Combine two audited context-readout pairs without selecting the better seed."""
import argparse
import json
from pathlib import Path
import statistics
import sys

SOURCE=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(SOURCE/'packages/gozero/src'))
from gozero.snapshots import canonical_json,read_json,verify
from gozero.checkpoints import sha256
from audit_qualification import NUMERICAL_FILES


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workspace-root',type=Path,required=True)
    p.add_argument('--pair',nargs=2,action='append',required=True,metavar=('CONTRAST','SHA256'))
    p.add_argument('--registration',type=Path,required=True);p.add_argument('--registration-sha256',required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();verify(SOURCE);root=a.workspace_root.resolve()
    if len(a.pair)!=2 or a.output.exists():raise ValueError('Two pairs and a new output required')
    if sha256(a.registration)!=a.registration_sha256:raise ValueError('Replication registration changed')
    registration=read_json(a.registration);inputs={str(a.registration):a.registration_sha256};pairs=[]
    for path,h in a.pair:
        path=Path(path)
        if sha256(path)!=h:raise ValueError('Contrast identity changed')
        contrast=read_json(path);inputs[str(path)]=h
        if contrast['status']!='passed' or contrast['kind']!='context_spatial_readout_contrast':raise ValueError('Invalid spatial audit')
        comparison=Path(contrast['comparison'])
        if sha256(comparison)!=contrast['comparison_sha256']:raise ValueError('Comparison identity changed')
        r=read_json(comparison);inputs[str(comparison)]=contrast['comparison_sha256']
        if r['status']!='passed' or not r['all_game_and_augmentation_draws_equal']:raise ValueError('Data draws not verified')
        for name,wanted in r['input_files'].items():
            if sha256(root/name)!=wanted:raise ValueError('Changed evidence: '+name)
            inputs[name]=wanted
        rows={x['label']:x for x in r['arms']};control=rows[contrast['control']];candidate=rows[contrast['candidate']]
        seed=r['common_non_lr_settings']['seed']
        pair={'seed':seed,'contrast':str(path),'contrast_sha256':h,'control':control,'candidate':candidate,
              'relative_kl_improvement':contrast['relative_endpoint_kl_improvement'],
              'warm_decode_latency_ratio':contrast['warm_decode_latency_ratio'],
              'replication_screen_passed':contrast['replication_screen_passed']}
        pairs.append(pair)
    if pairs[0]['seed']==pairs[1]['seed']:raise ValueError('Repeated seed is not a replication')
    if pairs[0]['contrast_sha256']!=registration['screen_contrast_sha256'] or pairs[1]['seed']!=registration['seed']:
        raise ValueError('Pairs do not match the prospective registration')
    for role,key in [('control','control_snapshot'),('candidate','candidate_snapshot')]:
        if pairs[1][role]['snapshot']!=registration[key]:raise ValueError('Replication snapshot differs from registration')
        configs=[];identities=[]
        for pair in pairs:
            row=pair[role];source=root/'.gozero/snapshots'/row['snapshot'];m=verify(source)
            c=read_json(source/'resolved_config.json')
            if c.pop('seed')!=pair['seed']:raise ValueError('Configuration seed differs')
            configs.append(c)
            hashes={n:v['sha256'] for n,v in m['files'].items() if n.startswith('packages/') or n in ('uv.lock','pyproject.toml','Cargo.lock')}
            hashes.update({'recipe/'+n:sha256(source/m['recipe']/n) for n in NUMERICAL_FILES})
            identities.append(hashes)
        if configs[0]!=configs[1] or identities[0]!=identities[1]:raise ValueError('Replication changed more than seed: '+role)
        if pairs[0][role]['initial_parameter_elements_sha256']==pairs[1][role]['initial_parameter_elements_sha256']:
            raise ValueError('Expected independent initialization across seeds')
    control_mean=statistics.mean(x['control']['curve'][-1]['expert_kl'] for x in pairs)
    candidate_mean=statistics.mean(x['candidate']['curve'][-1]['expert_kl'] for x in pairs)
    out={'status':'passed','kind':'two_seed_context_readout_replication','operator_snapshot':SOURCE.name,
        'registration':str(a.registration),'registration_sha256':a.registration_sha256,'pairs':pairs,'input_files':inputs,
        'mean_control_kl':control_mean,'mean_candidate_kl':candidate_mean,
        'relative_improvement_of_mean_kl':1-candidate_mean/control_mean,
        'direction_replicated':all(x['relative_kl_improvement']>0 for x in pairs),
        'each_seed_passed_one_percent_latency_screen':all(x['replication_screen_passed'] for x in pairs),
        'scope':'Two independent paired initializations on the same fixed dataset. Exact draws match within each seed, not across seeds. Report both effects; no precise statistical-confidence or Go-strength claim.'}
    a.output.mkdir(parents=True)
    (a.output/'replication.json').write_bytes(canonical_json(out))
    import matplotlib
    matplotlib.use('Agg');matplotlib.rcParams['svg.hashsalt']=SOURCE.name
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12,4.6),layout='constrained')
    for i,pair in enumerate(pairs):
        for role,color in [('control','C0'),('candidate','C1')]:
            curve=pair[role]['curve'][1:]
            axes[0].plot([x['turn'] for x in curve],[x['expert_kl'] for x in curve],
                         color=color,linestyle='-' if i==0 else '--',marker='o',markersize=3,label=f"{role}, seed {i+1}")
        x=pair['control']['curve'][1:];y=pair['candidate']['curve'][1:]
        axes[1].plot([r['turn'] for r in x],[b['expert_kl']-a['expert_kl'] for a,b in zip(x,y)],
                     marker='o',markersize=3,label=f"seed {i+1}")
    axes[0].set_ylabel('Validation KL');axes[1].set_ylabel('Context minus control KL')
    axes[1].axhline(0,color='black',linewidth=.8)
    for ax in axes:ax.set_xlabel('Optimizer updates');ax.grid(alpha=.2);ax.legend(fontsize=9)
    fig.suptitle('Context-conditioned policy readout: both paired seeds')
    fig.savefig(a.output/'learning.png',dpi=160)
    fig.savefig(a.output/'learning.svg',metadata={'Date':None})
    plt.close(fig)
    lines=['The context-readout replication retains both independent seeds. Configuration and numerical source change only by seed between corresponding runs; actual game and D4 draws match within each pair.',
           '', '| Seed | Position exposures per arm | Control KL | Spatial KL | Relative KL improvement | Decode latency ratio |',
           '|---|---:|---:|---:|---:|---:|']
    for pair in pairs:
        lines.append(f"| {pair['seed']} | {pair['control']['position_exposures']:,} | {pair['control']['curve'][-1]['expert_kl']:.6f} | {pair['candidate']['curve'][-1]['expert_kl']:.6f} | {pair['relative_kl_improvement']:.2%} | {pair['warm_decode_latency_ratio']:.3f} |")
    lines+=['',f"Mean KL: control {control_mean:.6f}; spatial {candidate_mean:.6f}. Relative improvement of the mean: {out['relative_improvement_of_mean_kl']:.2%}.",
            '',f"Direction replicated: {out['direction_replicated']}. Both seeds meet the exploratory 1% KL / 15% latency screen: {out['each_seed_passed_one_percent_latency_screen']}.",
            '',out['scope'],'','![Both paired learning curves](learning.png)']
    (a.output/'REPORT.md').write_text('\n'.join(lines)+'\n')
    files={p.name:sha256(p) for p in a.output.iterdir() if p.is_file()}
    (a.output/'manifest.json').write_bytes(canonical_json({'operator_snapshot':SOURCE.name,'files':files}))
    for p in a.output.iterdir():p.chmod(0o444)
    print(json.dumps({'status':'passed','direction_replicated':out['direction_replicated'],'relative_improvement_of_mean_kl':out['relative_improvement_of_mean_kl'],'output':str(a.output)}))


if __name__=='__main__':main()
