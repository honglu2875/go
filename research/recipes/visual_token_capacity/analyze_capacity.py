"""Audit controls across encoder widths and publish each completed LR result."""
import argparse
import json
from pathlib import Path
import statistics
import sys
SOURCE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(SOURCE/'packages/gozero/src'))
from gozero.snapshots import read_json, verify
from gozero.checkpoints import sha256
from gozero.checkpoint_archive import publish


def main():
    p = argparse.ArgumentParser(); p.add_argument('--workspace-root', type=Path, required=True)
    p.add_argument('--registration', type=Path, required=True); p.add_argument('--registration-sha256', required=True)
    p.add_argument('--audit', nargs=4, action='append', required=True, metavar=('CHANNELS','RATE','PATH','SHA'))
    p.add_argument('--output', type=Path, required=True); a = p.parse_args(); root = a.workspace_root.resolve(); verify(SOURCE)
    if sha256(a.registration) != a.registration_sha256: raise ValueError('Registration changed')
    reg = read_json(a.registration)
    expected = {(x['encoder_channels'], x['peak_learning_rate']): x['snapshot'] for x in reg['references']+reg['candidates']}
    arms=[]; seen=set(); controls=[]; draws=[]; initial={}; validations=[]
    for channels, rate, name, wanted in a.audit:
        channels, rate = int(channels), float(rate); key = (channels,rate); path=Path(name)
        if key in seen or key not in expected or sha256(path) != wanted: raise ValueError('Unknown, repeated or changed audit')
        seen.add(key); audit=read_json(path)
        if audit['status'] != 'passed' or audit['steps'] != 1024 or audit['training_snapshot'] != expected[key]: raise ValueError('Incomplete/wrong arm')
        for f,digest in audit['input_files'].items():
            if sha256(root/f) != digest: raise ValueError('Audited observations changed')
        attempt=root/'runs'/audit['attempt']; source=root/'.gozero/snapshots'/expected[key]; verify(source)
        if sha256(attempt/'result.json') != audit['closed_result_sha256']: raise ValueError('Attempt closure changed')
        c=read_json(source/'resolved_config.json')
        common={k:v for k,v in c.items() if k!='checkpoint_archive'}
        common['learner']={k:v for k,v in c['learner'].items() if k not in ('learning_rate','end_learning_rate')}
        common['model']={k:v for k,v in c['model'].items() if k not in ('encoder','encoder_channels','mlp_hidden')}
        controls.append(common); by_rank={}; fingerprints=[]
        if c['model'].get('encoder_channels',64) != channels or c['learner']['learning_rate'] != rate: raise ValueError('Wrong encoder/rate')
        if c['model']['mlp_hidden'] != {64:2304,128:2296,256:2288}[channels]: raise ValueError('FFN compensation changed')
        for host in range(4):
            folder=attempt/f'rank-{host}/artifacts'; report=read_json(folder/'result.json')
            log=[json.loads(x) for x in (folder/'metrics.jsonl').read_text().splitlines()]
            fingerprints.append(report['initial_parameter_elements_sha256'])
            by_rank[report['jax_rank']]=[{k:r[k] for k in ('turn','bucket','local_entries_sha256','local_symmetries','expert_positions')} for r in log]
            if host==0: owner=report; owner_log=log
        if len(set(fingerprints))!=1: raise ValueError('Initial parameters differ across ranks')
        initial.setdefault(channels,set()).update(fingerprints)
        if channels == reg['encoder_channels'] and fingerprints[0] != reg['qualified_initial_parameter_elements_sha256']:
            raise ValueError('Initial parameters differ from full TPU qualification')
        draws.append(by_rank)
        counts={0:0}; times={0:0.}; count=0
        for r in owner_log:
            count+=int(r['expert_positions']); counts[r['turn']]=count; times[r['turn']]=r['cumulative_learning_seconds']
        curve=[{**r,'positions':counts[r['turn']],'learning_seconds':times[r['turn']]} for r in audit['validation_curve']]
        validations.append([(r['turn'],r['episode_ids_sha256'],r['metrics']['expert_count']) for r in curve])
        profile=owner['decode_profile']
        arms.append({'encoder_channels':channels,'peak_learning_rate':rate,'snapshot':source.name,'attempt':attempt.name,
            'audit_path':str(path),'audit_sha256':wanted,'positions':count,'curve':curve,'timing':audit['timing'],
            'reserved_chip_hours':audit['reserved_chip_hours'],'parameter_count':owner['parameter_count'],
            'decode_median_seconds':statistics.median(profile['host_dispatch_latency_seconds']),
            'clipped_updates':audit['clipped_updates'],'checkpoint':owner['latest_checkpoint']})
    if (any(x!=controls[0] for x in controls) or any(x!=draws[0] for x in draws)
            or any(len(x)!=1 for x in initial.values()) or any(x!=validations[0] for x in validations)
            or any(x['positions']!=11469333 for x in arms)):
        raise ValueError('Data, initialization, validation or unregistered controls differ')
    arms.sort(key=lambda x:(x['encoder_channels'],x['peak_learning_rate']))
    best={str(w):min([x for x in arms if x['encoder_channels']==w],key=lambda x:x['curve'][-1]['metrics']['expert_kl']) for w in sorted(initial)}
    result={'kind':'encoder_capacity_comparison','status':'passed','complete':seen==set(expected),'operator_snapshot':SOURCE.name,
        'registration_sha256':a.registration_sha256,'arms':arms,'pending':sorted(set(expected)-seen),
        'same_actual_data_draws':True,'same_initial_parameters_within_each_width':True,'same_full_validation_population':True,
        'best_completed':{w:{'rate':x['peak_learning_rate'],'kl':x['curve'][-1]['metrics']['expert_kl']} for w,x in best.items()},
        'scope':'One seed, fixed weak-teacher dataset, equal exposure and complete logical decoding-FLOP budget. Width and compensated FFN shapes differ. Results do not establish Go strength or RL efficiency.'}
    a.output.mkdir(parents=True,exist_ok=False);publish(a.output/'comparison.json',result)
    lines=['The encoder comparison uses the same **11,469,333 training-position exposures**, exact episode/D4 draws and full validation set. Transformer width is 768 with 34 causal layers; encoder channels and the registered FFN compensation vary.\n',
           '| Encoder channels | Peak LR | Final validation KL | Top-move agreement | Learning minutes | Decode ms |',
           '|---|---:|---:|---:|---:|---:|']
    for x in arms:
        m=x['curve'][-1]['metrics']
        lines.append(f"| {x['encoder_channels']} | {x['peak_learning_rate']:.0e} | {m['expert_kl']:.6f} | {100*m['expert_top1']:.2f}% | {x['timing']['learning_seconds']/60:.2f} | {1000*x['decode_median_seconds']:.2f} |")
    lines+=['', 'Grid complete.' if result['complete'] else 'Grid incomplete; pending cells: '+str(result['pending']), '', result['scope'], '', 'Selected CNN AdamW reference: validation KL 0.420361 and 66.66% top-move agreement. Historical CNN SGD: KL 0.424920 and 66.31%.', '', '[Curves](learning.png) · [Audited comparison](comparison.json)']
    (a.output/'REPORT.md').write_text('\n'.join(lines)+'\n');(a.output/'REPORT.md').chmod(0o444)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12,4),constrained_layout=True)
    for x in arms:
        label=f"C{x['encoder_channels']}, LR {x['peak_learning_rate']:.0e}"
        y=[r['metrics']['expert_kl'] for r in x['curve']]
        axes[0].plot([r['positions']/1e6 for r in x['curve']],y,label=label)
        axes[1].plot([r['learning_seconds']/60 for r in x['curve']],y,label=label)
    for ax in axes:
        ax.set_ylabel('Validation policy KL');ax.grid(alpha=.25);ax.legend(fontsize=8)
    axes[0].set_xlabel('Training position exposures (millions)');axes[1].set_xlabel('Learning time (minutes)')
    for name in ('learning.png','learning.svg'):fig.savefig(a.output/name,dpi=160);(a.output/name).chmod(0o444)
    plt.close(fig)
    publish(a.output/'manifest.json',{'kind':result['kind'],'status':'passed','complete':result['complete'],'files':{n:sha256(a.output/n) for n in ['REPORT.md','comparison.json','learning.png','learning.svg']}})
    print(json.dumps({'status':'passed','complete':result['complete'],'best_completed':result['best_completed'],'output':str(a.output)}),flush=True)


if __name__=='__main__':main()
