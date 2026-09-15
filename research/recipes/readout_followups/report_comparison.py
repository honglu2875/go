"""Compare fixed-data learning records, including independently tuned reference LRs."""
import argparse
import json
import hashlib
import math
from pathlib import Path
import statistics
import sys
SOURCE=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(SOURCE/'packages/gozero/src'))
from gozero.snapshots import read_json,canonical_json,verify
from gozero.checkpoints import sha256


def main():
    p=argparse.ArgumentParser();p.add_argument('--workspace-root',type=Path,required=True)
    p.add_argument('--audit',nargs=3,action='append',required=True,metavar=('LABEL','PATH','SHA256'))
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();verify(SOURCE);root=a.workspace_root.resolve()
    rows=[];shared_settings=None;shared_draws=None;shared_lr_ratio=None;inputs={}
    for label,path,digest in a.audit:
        path=Path(path)
        if sha256(path)!=digest:raise ValueError('Audit identity changed')
        inputs[str(path)]=digest;q=read_json(path)
        if q['status']!='passed':raise ValueError('Unsuccessful audit')
        for name,h in q['input_files'].items():
            if sha256(root/name)!=h:raise ValueError('Audited file changed: '+name)
            inputs[name]=h
        attempt=root/'runs'/q['attempt'];source=root/'.gozero/snapshots'/q['training_snapshot'];verify(source)
        if sha256(attempt/'result.json')!=q['closed_result_sha256']:raise ValueError('Attempt closure changed')
        c=read_json(source/'resolved_config.json')
        settings={k:c[k] for k in ('dataset','seed','steps','evaluation','eval_every','expected_devices','expected_processes')}
        settings['learner']={k:v for k,v in c['learner'].items() if k not in ('learning_rate','end_learning_rate')}
        lr_ratio=c['learner']['end_learning_rate']/c['learner']['learning_rate']
        records=[read_json(attempt/f'rank-{i}/artifacts/result.json') for i in range(4)];draws={}
        for i,r in enumerate(records):
            log=[json.loads(line) for line in (attempt/f'rank-{i}/artifacts/metrics.jsonl').read_text().splitlines()]
            draws[r['jax_rank']]=[{k:x[k] for k in ('turn','bucket','local_entries_sha256','local_symmetries','expert_positions')} for x in log]
            if i==0:times={0:0.,**{x['turn']:x['cumulative_learning_seconds'] for x in log}}
        if shared_settings is None:shared_settings,shared_draws,shared_lr_ratio=settings,draws,lr_ratio
        elif (settings!=shared_settings or draws!=shared_draws
              or not math.isclose(lr_ratio,shared_lr_ratio,rel_tol=1e-12,abs_tol=0.0)):
            raise ValueError('Non-LR training settings, LR decay ratio or actual draws differ')
        r=records[0];profile=r.get('decode_profile',{});trained=r.get('trained_decode_profile')
        memory=profile.get('memory_bytes',{})
        memory_total=(sum(memory[k] for k in ('argument_size_in_bytes','output_size_in_bytes','temp_size_in_bytes'))
                      -memory['alias_size_in_bytes']) if memory else None
        curves=[{'turn':x['turn'],'learning_seconds':times[x['turn']],
                 'validation_ids_sha256':x['episode_ids_sha256'],**x['metrics']} for x in q['validation_curve']]
        if rows and [(x['turn'],x['validation_ids_sha256'],x['expert_count']) for x in curves]!=[(x['turn'],x['validation_ids_sha256'],x['expert_count']) for x in rows[0]['curve']]:raise ValueError('Validation population changed')
        rows.append({'label':label,'snapshot':source.name,'attempt':attempt.name,'model':c['model'],
            'parameters':r['parameter_count'],'peak_lr':c['learner']['learning_rate'],'end_to_peak_lr':lr_ratio,
            'initial_parameter_elements_sha256':r['initial_parameter_elements_sha256'],
            'position_exposures':q['expert_positions'],'curve':curves,'timing':q['timing'],
            'decode_median_ms':1000*statistics.median(profile['host_dispatch_latency_seconds']) if profile else None,
            'trained_decode_median_ms':1000*statistics.median(trained['host_dispatch_latency_seconds']) if trained else None,
            'decode_memory_bytes':memory,'decode_compiler_accounted_bytes':memory_total,
            'checkpoint':r['latest_checkpoint'],'checkpoint_available_locally':Path(r['latest_checkpoint']['path'],'arrays.npz').is_file()})
    result={'status':'passed','operator_snapshot':SOURCE.name,'all_game_and_augmentation_draws_equal':True,
        'common_non_lr_settings':shared_settings,'common_end_to_peak_lr':shared_lr_ratio,
        'lr_ratio_relative_tolerance':1e-12,'arms':rows,'input_files':inputs,
        'scope':'Single-seed fixed weak-teacher learnability. Peak LR is reported per arm; other data/optimizer controls are verified. Historical checkpoint availability is separate from preserved metric evidence. No Go-strength or MFU claim.'}
    a.output.mkdir(parents=True,exist_ok=False);(a.output/'comparison.json').write_bytes(canonical_json(result))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12,4.7),layout='constrained')
    for row in rows:
        for ax,key,scale in [(axes[0],'turn',1),(axes[1],'learning_seconds',1/60)]:
            curve=row['curve'][1:]
            ax.plot([x[key]*scale for x in curve],[x['expert_kl'] for x in curve],marker='o',markersize=3,label=f"{row['label']} (LR {row['peak_lr']:.0e})")
    for ax in axes:ax.set_ylabel('Validation KL (lower is better)');ax.grid(alpha=.2)
    axes[0].set_xlabel('Optimizer updates');axes[1].set_xlabel('Measured learning minutes')
    axes[0].legend(fontsize=8);fig.suptitle('One board token: shared CNN + causal transformer')
    for suffix in ('png','svg'):fig.savefig(a.output/f'learning.{suffix}',dpi=160)
    lines=['Fixed-data comparison; all actual game and D4 draws verified. Peak learning rates are stated separately.\n',
           '| Model | Parameters | Peak LR | Final validation KL | Top-1 | Learning min | Warm decode ms |',
           '|---|---:|---:|---:|---:|---:|---:|']
    for row in rows:
        last=row['curve'][-1]
        lines.append(f"| {row['label']} | {row['parameters']:,} | {row['peak_lr']:.0e} | {last['expert_kl']:.6f} | {last['expert_top1']:.4f} | {row['timing']['learning_seconds']/60:.2f} | {row['decode_median_ms']:.2f} |")
    lines+=['','The latency column uses the initial-weight measurement shared by the historical reports. Updated-weight measurements for new models are included in comparison.json. Training time excludes compilation, evaluation and checkpoint writing. These results do not establish playing strength or reinforcement-learning improvement.',
            '', '| Cached model | Compiler-accounted decode GiB | Donated cache GiB |',
            '|---|---:|---:|']
    for row in rows:
        if row['decode_compiler_accounted_bytes'] is not None:
            lines.append(f"| {row['label']} | {row['decode_compiler_accounted_bytes']/2**30:.3f} | {row['decode_memory_bytes']['alias_size_in_bytes']/2**30:.3f} |")
    lines+=['', 'Memory uses the per-device compiler report: arguments + outputs + temporaries - aliases. This is not a measured peak resident-memory trace and excludes the environment/CPU pipeline.', '', '![Learning curves](learning.png)']
    (a.output/'REPORT.md').write_text('\n'.join(lines)+'\n')
    files={p.name:sha256(p) for p in a.output.iterdir() if p.is_file()}
    (a.output/'manifest.json').write_bytes(canonical_json({'operator_snapshot':SOURCE.name,'files':files}))
    print(json.dumps({'status':'passed','output':str(a.output),'final_kl':{r['label']:r['curve'][-1]['expert_kl'] for r in rows}}))


if __name__=='__main__':main()
