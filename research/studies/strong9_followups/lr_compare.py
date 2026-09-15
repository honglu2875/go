"""Contrast one LR-only intervention against its audited full-horizon parent."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
import time

import lr_source

ROOT=Path(__file__).resolve().parents[3]


def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def read(path):return json.loads(path.read_text())


def audited(root,path,digest):
    if sha(path)!=digest:raise ValueError('Audit identity changed')
    audit=read(path)
    if audit['status']!='passed' or audit['steps']!=4096:raise ValueError('Incomplete audited learning run')
    attempt=root/'runs'/audit['attempt']
    if sha(attempt/'result.json')!=audit['closed_result_sha256']:raise ValueError('Closed attempt changed')
    closed=read(attempt/'result.json')
    if closed['status']!='passed' or closed['snapshot_id']!=audit['training_snapshot']:raise ValueError('Wrong closed attempt')
    for name,digest in audit['input_files'].items():
        if sha(root/name)!=digest:raise ValueError('Audited input changed: '+name)
    snapshot=root/'.gozero/snapshots'/audit['training_snapshot']
    lr_source.verify(snapshot)
    reports=[];draws={}
    for host in range(4):
        folder=attempt/f'rank-{host}/artifacts'
        report=read(folder/'result.json');reports.append(report)
        if report['jax_rank'] in draws:raise ValueError('Duplicate JAX rank')
        rows=[json.loads(line) for line in (folder/'metrics.jsonl').read_text().splitlines()]
        draws[report['jax_rank']]=[{key:r[key] for key in
            ('turn','bucket','local_entries_sha256','local_symmetries','expert_positions')} for r in rows]
    if set(draws)!=set(range(4)):raise ValueError('Incomplete rank coverage')
    return dict(audit=audit,audit_path=str(path.relative_to(root)),audit_sha256=sha(path),
        snapshot=snapshot,config=read(snapshot/'resolved_config.json'),reports=reports,draws=draws)


def pair_curves(left,right):
    if [r['turn'] for r in left]!=list(range(0,4097,256)) or [r['turn'] for r in right]!=list(range(0,4097,256)):
        raise ValueError('Evaluation schedule differs')
    result=[]
    for a,b in zip(left,right):
        if a['episode_ids_sha256']!=b['episode_ids_sha256'] or a['split']!=b['split']:
            raise ValueError('Evaluation population differs')
        if set(a['metrics'])!=set(b['metrics']):raise ValueError('Metric schema differs')
        for key in a['metrics']:
            x,y=a['metrics'][key],b['metrics'][key]
            if not all(type(v) in (int,float) and math.isfinite(v) for v in (x,y)):
                raise ValueError('Nonfinite evaluation metric')
            if key.endswith(('_count','_target_entropy')) and abs(x-y)>1e-5*max(1,abs(x)):
                raise ValueError('Target/count population differs')
        result.append(dict(turn=a['turn'],parent=a['metrics'],candidate=b['metrics']))
    return result


def gains(curve):
    result={}
    for metric in ('expert_kl','family_kl'):
        a,b=(curve[-1][key][metric] for key in ('parent','candidate'))
        x,y=(statistics.mean(row[key][metric] for row in curve[-3:]) for key in ('parent','candidate'))
        if min(a,x)<=0 or min(b,y)<0:raise ValueError('Invalid KL ratio')
        result[metric]=dict(parent_endpoint=a,candidate_endpoint=b,relative_endpoint_gain=1-b/a,
            parent_last_three_mean=x,candidate_last_three_mean=y,relative_last_three_gain=1-y/x)
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',type=Path,required=True)
    p.add_argument('--plan-sha256',required=True);p.add_argument('--stage',choices=('seed1','seed2'),required=True)
    p.add_argument('--candidate-audit',type=Path,required=True);p.add_argument('--candidate-audit-sha256',required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    if sha(a.plan)!=a.plan_sha256:raise ValueError('Plan changed')
    plan=read(a.plan)
    if plan['kind']!='strong9_learning_rate_intervention' or plan['comparison_operator_sha256']!=sha(Path(__file__)):
        raise ValueError('Wrong comparison plan/operator')
    if sha(Path(lr_source.__file__))!=plan['source_helper_sha256']:raise ValueError('LR source validator changed')
    stage=plan['stages'][a.stage]
    parent=audited(ROOT,ROOT/stage['parent_audit']['path'],stage['parent_audit']['sha256'])
    candidate=audited(ROOT,a.candidate_audit.resolve(),a.candidate_audit_sha256)
    if parent['snapshot'].name!=stage['parent_snapshot'] or candidate['snapshot'].name!=stage['snapshot']:
        raise ValueError('Unregistered source pair')
    source_files=lr_source.identical_source(parent['snapshot'],candidate['snapshot'])
    if parent['draws']!=candidate['draws']:raise ValueError('Training game/symmetry draws differ')
    if parent['audit']['expert_positions']!=candidate['audit']['expert_positions']:raise ValueError('Position exposure differs')
    initialized=lambda arm:{r['initial_parameter_elements_sha256'] for r in arm['reports']}
    if len(initialized(parent))!=1 or initialized(parent)!=initialized(candidate):raise ValueError('Initial model parameters differ')
    validation=pair_curves(parent['audit']['validation_curve'],candidate['audit']['validation_curve'])
    probe=pair_curves(parent['audit']['training_probe_curve'],candidate['audit']['training_probe_curve'])
    result=gains(validation)
    threshold=plan['screen_min_relative_gain']
    if type(threshold) not in (int,float) or not 0<threshold<1:raise ValueError('Explicit screen threshold required')
    overfit=any(x['sustained'] for x in candidate['audit']['overfit_observations'])
    screen=all(r['relative_endpoint_gain']>=threshold and r['relative_last_three_gain']>=0 for r in result.values()) and not overfit
    def describe(arm):
        audit=arm['audit'];report=arm['reports'][0]
        return dict(audit_path=arm['audit_path'],audit_sha256=arm['audit_sha256'],snapshot=arm['snapshot'].name,
            attempt=audit['attempt'],peak_learning_rate=arm['config']['learner']['learning_rate'],
            parameter_count=report['parameter_count'],expert_positions=audit['expert_positions'],
            timing=audit['timing'],reserved_chip_hours=audit['reserved_chip_hours'],
            clipped_updates=audit['clipped_updates'],overfit_observations=audit['overfit_observations'],
            trained_decode_median_ms=1000*statistics.median(report['trained_decode_profile']['host_dispatch_latency_seconds']))
    report=dict(kind='strong9_learning_rate_paired_contrast',status='passed',created=time.time(),
        plan_sha256=a.plan_sha256,operator_sha256=sha(Path(__file__)),stage=a.stage,seed=parent['config']['seed'],
        parent=describe(parent),candidate=describe(candidate),identical_source_files=source_files,
        all_initial_parameters_identical=True,all_training_draws_identical=True,all_evaluation_populations_identical=True,
        endpoint_and_tail_gains=result,paired_validation=validation,paired_training_probe=probe,
        screen_min_relative_gain=threshold,screen_passed=screen,sustained_overfit=overfit,
        scope='One paired fixed-data supervised LR intervention. A promising first seed requires paired second-seed confirmation; this does not establish a globally optimal LR, architecture superiority, RL efficiency or playing strength.')
    with a.output.open('x') as f:json.dump(report,f,indent=2,allow_nan=False);f.write('\n')
    a.output.chmod(0o444)
    print(json.dumps(dict(status='passed',screen_passed=screen,gains=result,sha256=sha(a.output))))


if __name__=='__main__':main()
