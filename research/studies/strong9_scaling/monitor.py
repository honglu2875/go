"""Read complete live records and record conservative runtime estimates.

This operator observes only; it does not stop, modify, or schedule a run.
"""
import argparse
from collections import defaultdict
import importlib.util
import json
from pathlib import Path
import statistics
import time

ROOT=Path(__file__).resolve().parents[3]
STUDY=Path(__file__).resolve().parent


def rows(path):
    if not path.exists():return []
    text=path.read_text();lines=text.splitlines()
    if text and not text.endswith('\n'):lines=lines[:-1]
    return [json.loads(line) for line in lines]


def main():
    p=argparse.ArgumentParser();p.add_argument('--attempt',required=True)
    a=p.parse_args()
    attempt=ROOT/'runs'/a.attempt
    if attempt.parent!=ROOT/'runs' or not (attempt/'launch.json').exists():
        raise ValueError('Expected an existing local pod attempt')
    launch=json.loads((attempt/'launch.json').read_text())
    folder=attempt/'rank-0/artifacts'
    config=json.loads((folder/'resolved_config.json').read_text())
    if config['dataset']['manifest_sha256']!='37244b2e743b3b7f80e0b14eccddfc2942aa4d81120c31aab10c4b3b4f733fc2':
        raise ValueError('Wrong dataset')
    metrics=rows(folder/'metrics.jsonl');evaluations=rows(folder/'evaluations.jsonl')
    if not metrics:
        print(json.dumps(dict(attempt=a.attempt,status='initializing',evaluations=len(evaluations))));return
    val=[e for e in evaluations if e['kind']=='visual_heldout']
    probe=[e for e in evaluations if e['kind']=='visual_training_probe']
    paired=sorted(set(e['turn'] for e in val)&set(e['turn'] for e in probe))
    last=metrics[-1];now=time.time()
    snapshot=ROOT/'.gozero/snapshots'/launch['snapshot_id']
    manifest=json.loads((snapshot/'manifest.json').read_text())
    spec=importlib.util.spec_from_file_location('frozen_training_probe',snapshot/manifest['recipe']/'training_probe.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    observations=[module.overfit_observation(val,probe,metric=m) for m in ('expert_kl','family_kl')]
    per_bucket=defaultdict(list);prev_learning=prev_sampling=0.
    for r in metrics:
        dt=r['cumulative_learning_seconds']-prev_learning+r['cumulative_sampling_seconds']-prev_sampling
        per_bucket[r['bucket']].append(dt)
        prev_learning=r['cumulative_learning_seconds'];prev_sampling=r['cumulative_sampling_seconds']
    bucket_means={str(k):statistics.mean(v) for k,v in per_bucket.items()}
    cdata=config['dataset']
    average=sum(prob*bucket_means[str(bucket)] for bucket,prob in zip(cdata['buckets'],cdata['bucket_probabilities']))
    history=STUDY/(a.attempt+'-monitor.jsonl')
    previous=rows(history)
    sample=dict(attempt=a.attempt,observed_unix=now,turn=last['turn'],paired_evaluations=len(paired),
                cumulative_learning_seconds=prev_learning,cumulative_sampling_seconds=prev_sampling,
                average_update_seconds=average)
    cycle_overheads=[];cycle_points=[]
    for point in previous+[sample]:
        if not cycle_points or point['paired_evaluations']>cycle_points[-1]['paired_evaluations']:
            cycle_points.append(point)
    # Keep the first observation of each completed-evaluation count. Samples
    # taken partway through the next evaluation must not lose its earlier time.
    for before,after in zip(cycle_points,cycle_points[1:]):
        count=after['paired_evaluations']-before['paired_evaluations']
        if count>0 and before['paired_evaluations']>0:
            overhead=after['observed_unix']-before['observed_unix']-(after['cumulative_learning_seconds']-before['cumulative_learning_seconds'])-(after['cumulative_sampling_seconds']-before['cumulative_sampling_seconds'])
            if overhead>=0:cycle_overheads.append(overhead/count)
    elapsed=now-launch['start_unix_time']
    nonlearning=elapsed-prev_learning-prev_sampling
    # Until a complete evaluation interval is observed, startup is charged to
    # each cycle as a conservative upper estimate, not an ETA point estimate.
    evaluation_seconds=max(cycle_overheads) if cycle_overheads else nonlearning/max(len(paired),1)
    final_turn=config['steps'];remaining_evaluations=sum(t>max(paired,default=-1) for t in range(0,final_turn+1,config['eval_every']))
    remaining=average*(final_turn-last['turn'])+remaining_evaluations*evaluation_seconds+120.
    estimate=dict(method='observed evaluation-cycle overhead' if cycle_overheads else 'startup-inclusive upper estimate; not settled',
                  evaluation_cycle_samples=len(cycle_overheads),evaluation_seconds=evaluation_seconds,
                  remaining_seconds=remaining,finish_unix=now+remaining,
                  conservative_deadline_unix=launch['start_unix_time']+launch['timeout_seconds'],
                  deadline_margin_seconds=launch['start_unix_time']+launch['timeout_seconds']-now-remaining,
                  finalization_reserve_seconds=120)
    sample.update(runtime_estimate=estimate,bucket_update_seconds=bucket_means,
                  latest_evaluations=[dict(kind=x['kind'],turn=x['turn'],position_kl=x['metrics']['expert_kl'],family_kl=x['metrics']['family_kl']) for x in (val[-1:] + probe[-1:])],
                  overfit_observations=observations,status='closed' if (attempt/'result.json').exists() else 'running')
    with history.open('a') as f:f.write(json.dumps(sample)+'\n')
    print(json.dumps(sample),flush=True)


if __name__=='__main__':main()
