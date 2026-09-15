"""Contrast completed registered arms after their independent per-run audits."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time


def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def read(path):return json.loads(path.read_text())


def main():
    p=argparse.ArgumentParser();p.add_argument('--workspace-root',type=Path,required=True)
    p.add_argument('--registration',type=Path,required=True);p.add_argument('--registration-sha256',required=True)
    p.add_argument('--cnn-audit',type=Path,required=True);p.add_argument('--cnn-audit-sha256',required=True)
    p.add_argument('--transformer-audit',type=Path,required=True);p.add_argument('--transformer-audit-sha256',required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();root=a.workspace_root.resolve()
    for path,digest in ((a.registration,a.registration_sha256),(a.cnn_audit,a.cnn_audit_sha256),
                        (a.transformer_audit,a.transformer_audit_sha256)):
        if sha(path)!=digest:raise ValueError('Pinned comparison input changed: '+str(path))
    registration=read(a.registration);protocol_path=root/registration['protocol']['path']
    if sha(protocol_path)!=registration['protocol']['sha256']:raise ValueError('Protocol changed')
    protocol=read(protocol_path)
    audits=[read(a.cnn_audit),read(a.transformer_audit)]
    source=root/'.gozero/snapshots'/audits[0]['training_snapshot']
    sys.path.insert(0,str(source/'packages/gozero/src'))
    from gozero.snapshots import verify
    configs=[];reports=[];draws=[];arms=[]
    for label,audit,audit_path in zip(('cnn','transformer'),audits,(a.cnn_audit,a.transformer_audit)):
        if audit['status']!='passed':raise ValueError('Unqualified learning evidence')
        attempt=root/'runs'/audit['attempt'];source=root/'.gozero/snapshots'/audit['training_snapshot'];verify(source)
        config=read(source/'resolved_config.json');configs.append(config)
        if config['seed'] not in protocol['training']['seeds']:raise ValueError('Unregistered seed')
        stage='seed'+str(protocol['training']['seeds'].index(config['seed'])+1)
        if source.name!=registration['snapshots'][label][stage]:raise ValueError('Unregistered training snapshot')
        if sha(attempt/'result.json')!=audit['closed_result_sha256']:raise ValueError('Closed attempt changed')
        for name,digest in audit['input_files'].items():
            if sha(root/name)!=digest:raise ValueError('Audited input changed')
        rank_draws={}
        for host in range(4):
            report=read(attempt/f'rank-{host}/artifacts/result.json')
            rows=[json.loads(line) for line in (attempt/f'rank-{host}/artifacts/metrics.jsonl').read_text().splitlines()]
            rank_draws[report['jax_rank']]=[{key:r[key] for key in ('turn','bucket','local_entries_sha256','local_symmetries','expert_positions')} for r in rows]
        draws.append(rank_draws)
        report=read(attempt/'rank-0/artifacts/result.json');reports.append(report)
        population=audit['population_audit']
        arms.append(dict(label=label,attempt=attempt.name,snapshot=source.name,
            audit_path=str(audit_path),audit_sha256=sha(audit_path),parameter_count=report['parameter_count'],
            validation_curve=audit['validation_curve'],training_probe_curve=audit['training_probe_curve'],
            overfit_observations=audit['overfit_observations'],population_audit=population,
            expert_positions=audit['expert_positions'],timing=audit['timing'],reserved_chip_hours=audit['reserved_chip_hours'],
            checkpoint=report['latest_checkpoint'],
            initial_decode_median_ms=1000*statistics.median(report['decode_profile']['host_dispatch_latency_seconds']),
            trained_decode_median_ms=1000*statistics.median(report['trained_decode_profile']['host_dispatch_latency_seconds'])))
    for key in ('dataset','seed','steps','learner','eval_every','evaluation','expected_processes','expected_devices'):
        if configs[0][key]!=configs[1][key]:raise ValueError('Unpaired setting: '+key)
    if configs[0]['steps']!=protocol['training']['steps'] or draws[0]!=draws[1]:
        raise ValueError('Horizon or exact training draws differ')
    if configs[0]['model']['architecture']!='katago_nested_policy' or configs[1]['model']['architecture']!='causal_visual_policy':
        raise ValueError('Wrong architecture pair')
    if not configs[0]['model']['width']==configs[1]['model']['width']==768:
        raise ValueError('Width constraint differs')
    curves=[]
    for name in ('validation_curve','training_probe_curve'):
        left,right=(arm[name] for arm in arms)
        expected=[0,*range(256,4097,256)]
        if [r['turn'] for r in left]!=expected or [r['turn'] for r in right]!=expected:
            raise ValueError('Evaluation schedule differs')
        paired=[]
        for cnn,transformer in zip(left,right):
            if cnn['episode_ids_sha256']!=transformer['episode_ids_sha256']:
                raise ValueError('Evaluation population differs')
            for key in cnn['metrics']:
                if key.endswith(('_count','_target_entropy')):
                    if abs(cnn['metrics'][key]-transformer['metrics'][key])>1e-5*max(1,abs(cnn['metrics'][key])):
                        raise ValueError('Evaluation denominators/targets differ: '+key)
            paired.append(dict(turn=cnn['turn'],cnn=cnn['metrics'],transformer=transformer['metrics']))
        curves.append(paired)
    gains={};tail_gains={}
    for metric in ('expert_kl','family_kl'):
        endpoint=curves[0][-1]
        gains[metric]=1-endpoint['transformer'][metric]/endpoint['cnn'][metric]
        tail_gains[metric]=1-statistics.mean(r['transformer'][metric] for r in curves[0][-3:])/statistics.mean(r['cnn'][metric] for r in curves[0][-3:])
    result=dict(status='passed',kind='strong9_paired_seed_contrast',created=time.time(),seed=configs[0]['seed'],
        registration_sha256=sha(a.registration),operator_sha256=sha(Path(__file__)),arms=arms,
        relative_endpoint_kl_improvement=gains,relative_last_three_mean_kl_improvement=tail_gains,
        both_endpoint_metrics_improved=all(v>0 for v in gains.values()),
        paired_validation=curves[0],paired_training_probe=curves[1],
        all_training_draws_identical=True,all_evaluation_populations_identical=True,
        auxiliary_losses='CNN .2 main + .8 helper; transformer .75 final + .25 first pass',
        scope='One paired fixed-data seed; the registered conclusion requires both seeds. No strength or RL-speed claim.')
    with a.output.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    a.output.chmod(0o444)
    print(json.dumps(dict(status='passed',seed=result['seed'],relative_endpoint_kl_improvement=gains)),flush=True)


if __name__=='__main__':main()
