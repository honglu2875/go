"""Publish audited encoder interventions with verified data and training controls."""
import argparse
import json
from pathlib import Path
import statistics
import sys
SOURCE=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(SOURCE/'packages/gozero/src'))
from gozero.snapshots import read_json,verify
from gozero.checkpoints import sha256
from gozero.checkpoint_archive import publish


def main():
    p=argparse.ArgumentParser();p.add_argument('--workspace-root',type=Path,required=True)
    p.add_argument('--registration',type=Path,required=True);p.add_argument('--registration-sha256',required=True)
    p.add_argument('--audit',nargs=3,action='append',required=True,metavar=('LABEL','PATH','SHA'))
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();root=a.workspace_root.resolve();verify(SOURCE)
    if sha256(a.registration)!=a.registration_sha256:raise ValueError('Registration changed')
    reg=read_json(a.registration);expected={x['label']:x for x in reg['references']+reg['candidates']}
    if len(expected)!=len(reg['references'])+len(reg['candidates']):raise ValueError('Repeated experiment labels')
    arms=[];seen=set();controls=[];draws=[];validations=[];initial={}
    for label,name,digest in a.audit:
        if label in seen or label not in expected or sha256(Path(name))!=digest:raise ValueError('Unknown, repeated or changed audit')
        seen.add(label);case=expected[label];audit=read_json(Path(name))
        if audit['status']!='passed' or audit['steps']!=1024 or audit['training_snapshot']!=case['snapshot']:raise ValueError('Incomplete or wrong experiment')
        for f,wanted in audit['input_files'].items():
            if sha256(root/f)!=wanted:raise ValueError('Audited observation changed')
        source=root/'.gozero/snapshots'/case['snapshot'];verify(source);c=read_json(source/'resolved_config.json')
        if c['model']!=case['model'] or c['learner']['learning_rate']!=case['peak_learning_rate']:raise ValueError('Unregistered model or learning rate')
        common={k:v for k,v in c.items() if k not in ('checkpoint_archive','checkpoint_temporary')}
        common['learner']={k:v for k,v in c['learner'].items() if k not in ('learning_rate','end_learning_rate')}
        common['model']={k:v for k,v in c['model'].items() if k not in ('encoder','encoder_channels','encoder_depth','observation_attention','readout_pooling','policy_spatial_bias','encoder_rematerialize','mlp_hidden','layers')}
        controls.append(common)
        attempt=root/'runs'/audit['attempt']
        if sha256(attempt/'result.json')!=audit['closed_result_sha256']:raise ValueError('Attempt closure changed')
        by_rank={};fingerprints=[]
        for host in range(4):
            folder=attempt/f'rank-{host}/artifacts';report=read_json(folder/'result.json')
            log=[json.loads(line) for line in (folder/'metrics.jsonl').read_text().splitlines()]
            fingerprints.append(report['initial_parameter_elements_sha256'])
            by_rank[report['jax_rank']]=[{k:r[k] for k in ('turn','bucket','local_entries_sha256','local_symmetries','expert_positions')} for r in log]
            if host==0:owner=report;owner_log=log
        if len(set(fingerprints))!=1 or fingerprints[0]!=case['initial_parameter_elements_sha256']:
            raise ValueError('Initial parameters differ from registered qualification/control')
        # Attention masks do not change parameter initialization; same shapes
        # retain one seed fingerprint across these isolated interventions.
        architecture={**c['model']};architecture.pop('observation_attention',None);architecture.pop('readout_pooling',None);architecture.pop('encoder_rematerialize',None)
        if not architecture.get('policy_spatial_bias',False):architecture.pop('policy_spatial_bias',None)
        architecture['encoder_depth']=architecture.get('encoder_depth',2)
        if architecture['encoder']=='overlap_conv64':architecture.update(encoder='overlap_conv',encoder_channels=64)
        initial.setdefault(json.dumps(architecture,sort_keys=True),set()).update(fingerprints)
        draws.append(by_rank)
        counts={0:0};times={0:0.};count=0
        for row in owner_log:
            count+=int(row['expert_positions']);counts[row['turn']]=count;times[row['turn']]=row['cumulative_learning_seconds']
        curve=[{**row,'positions':counts[row['turn']],'learning_seconds':times[row['turn']]} for row in audit['validation_curve']]
        validations.append([(r['turn'],r['episode_ids_sha256'],r['metrics']['expert_count']) for r in curve])
        arms.append({'label':label,'model':c['model'],'peak_learning_rate':c['learner']['learning_rate'],
            'snapshot':source.name,'attempt':attempt.name,'audit_path':name,'audit_sha256':digest,'positions':count,
            'curve':curve,'timing':audit['timing'],'reserved_chip_hours':audit['reserved_chip_hours'],
            'parameter_count':owner['parameter_count'],'decode_median_seconds':statistics.median(owner['decode_profile']['host_dispatch_latency_seconds']),
            'clipped_updates':audit['clipped_updates'],'checkpoint':owner['latest_checkpoint']})
    if (any(x!=controls[0] for x in controls) or any(x!=draws[0] for x in draws) or any(x!=validations[0] for x in validations)
            or any(len(x)!=1 for x in initial.values()) or any(x['positions']!=11469333 for x in arms)):
        raise ValueError('Data, validation, initialization or unregistered controls differ')
    best=min(arms,key=lambda x:x['curve'][-1]['metrics']['expert_kl'])
    cnn_audit=root/reg['cnn_reference']['audit_path']
    if sha256(cnn_audit)!=reg['cnn_reference']['audit_sha256']:raise ValueError('CNN reference audit changed')
    cnn=read_json(cnn_audit)
    if cnn['status']!='passed':raise ValueError('Unaudited CNN reference')
    for name,digest in cnn['input_files'].items():
        if sha256(root/name)!=digest:raise ValueError('Audited CNN observations changed')
    cnn_draws={};cnn_times={0:0.};cnn_counts={0:0.};count=0
    for host in range(4):
        folder=root/'runs'/cnn['attempt']/f'rank-{host}/artifacts'
        report=read_json(folder/'result.json');log=[json.loads(line) for line in (folder/'metrics.jsonl').read_text().splitlines()]
        cnn_draws[report['jax_rank']]=[{k:r[k] for k in ('turn','bucket','local_entries_sha256','local_symmetries','expert_positions')} for r in log]
        if host==0:
            for row in log:
                count+=int(row['expert_positions']);cnn_counts[row['turn']]=count;cnn_times[row['turn']]=row['cumulative_learning_seconds']
    if cnn_draws!=draws[0]:raise ValueError('CNN and transformer actual draws differ')
    cnn_curve=[{**row,'positions':cnn_counts[row['turn']],'learning_seconds':cnn_times[row['turn']]} for row in cnn['validation_curve']]
    if [(row['turn'],row['episode_ids_sha256'],row['metrics']['expert_count']) for row in cnn_curve]!=validations[0]:
        raise ValueError('CNN and transformer validation populations differ')
    result={'kind':'encoder_intervention_comparison','status':'passed','complete':seen==set(expected),
        'operator_snapshot':SOURCE.name,'registration_sha256':a.registration_sha256,'arms':arms,
        'best_completed':{'label':best['label'],'kl':best['curve'][-1]['metrics']['expert_kl']},'pending':sorted(set(expected)-seen),
        'cnn_reference':{'audit_sha256':sha256(cnn_audit),'metrics':cnn['validation_curve'][-1]['metrics'],'timing':cnn['timing'],'curve':cnn_curve},
        'same_actual_data_draws':True,'same_full_validation_population':True,
        'scope':'Single seed, fixed weak-teacher targets, matched complete logical decoding budget at the registered reference context. Final validation KL compares learnability, not Go strength or RL efficiency. Choices between waves are adaptive.'}
    a.output.mkdir(parents=True,exist_ok=False);publish(a.output/'comparison.json',result)
    lines=['Encoder interventions use the same **11,469,333 training-position exposures**, exact game/D4 draws and full validation population. Decoder width remains 768. Registered decoder depth and FFN compensation maintain the full parameter/decoding budget when encoder capacity changes.\n',
        '| Experiment | Peak LR | Final validation KL | Top-move agreement | Learning minutes | Decode ms |',
        '|---|---:|---:|---:|---:|---:|']
    for x in arms:
        m=x['curve'][-1]['metrics']
        lines.append(f"| {x['label']} | {x['peak_learning_rate']:.0e} | {m['expert_kl']:.6f} | {100*m['expert_top1']:.2f}% | {x['timing']['learning_seconds']/60:.2f} | {1000*x['decode_median_seconds']:.2f} |")
    m=result['cnn_reference']['metrics'];lines.append(f"| Selected CNN AdamW | 1e-3 | {m['expert_kl']:.6f} | {100*m['expert_top1']:.2f}% | {cnn['timing']['learning_seconds']/60:.2f} | — |")
    lines+=['','Registered wave complete.' if result['complete'] else 'Pending: '+str(result['pending']),'',result['scope'],'','[Curves](learning.png) · [Audited comparison](comparison.json)']
    (a.output/'REPORT.md').write_text('\n'.join(lines)+'\n');(a.output/'REPORT.md').chmod(0o444)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(16,4),constrained_layout=True)
    for x in arms:
        y=[r['metrics']['expert_kl'] for r in x['curve']]
        axes[0].plot([r['positions']/1e6 for r in x['curve']],y,label=x['label'])
        axes[1].plot([r['learning_seconds']/60 for r in x['curve']],y,label=x['label'])
        axes[2].plot([r['positions']/1e6 for r in x['curve']],y,label=x['label'])
    for index in (0,2):axes[index].plot([r['positions']/1e6 for r in cnn_curve],[r['metrics']['expert_kl'] for r in cnn_curve],color='black',linestyle='--',label='CNN AdamW')
    axes[1].plot([r['learning_seconds']/60 for r in cnn_curve],[r['metrics']['expert_kl'] for r in cnn_curve],color='black',linestyle='--',label='CNN AdamW')
    for ax in axes:
        ax.axhline(m['expert_kl'],color='black',linestyle=':',label='CNN final KL');ax.set_ylabel('Validation policy KL');ax.grid(alpha=.25);ax.legend(fontsize=7)
    axes[0].set_xlabel('Training position exposures (millions)');axes[1].set_xlabel('Learning time (minutes)')
    axes[2].set_xlabel('Training position exposures (millions), late training');axes[2].set_xlim(6,11.6);axes[2].set_ylim(.35,1.0)
    for name in ('learning.png','learning.svg'):fig.savefig(a.output/name,dpi=160);(a.output/name).chmod(0o444)
    plt.close(fig)
    publish(a.output/'manifest.json',{'kind':result['kind'],'status':'passed','complete':result['complete'],
        'files':{n:sha256(a.output/n) for n in ['REPORT.md','comparison.json','learning.png','learning.svg']}})
    print(json.dumps({'status':'passed','best_completed':result['best_completed'],'complete':result['complete'],'output':str(a.output)}),flush=True)


if __name__=='__main__':main()
