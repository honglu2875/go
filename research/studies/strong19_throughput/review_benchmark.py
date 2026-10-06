"""Audit every host and summarize a completed, registered runtime A/B screen."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time

ROOT=Path(__file__).resolve().parents[3]
STUDY=Path(__file__).resolve().parent


def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--attempt',required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    if '/' in args.attempt or not args.attempt.startswith('pod-'):raise ValueError('Invalid attempt')
    registration=read(STUDY/'registration-001.json')
    process_record=read(STUDY/'process-001.json')
    if sha(STUDY/'registration-001.json')!=process_record['registration_sha256']:
        raise ValueError('Registration changed after launch')
    snapshot=ROOT/'.gozero/snapshots'/registration['snapshot']
    sys.path.insert(0,str(snapshot/'packages/gozero/src'))
    from gozero.snapshots import verify
    verify(snapshot)
    qualification_path=STUDY/'cpu-equivalence-001.json'
    if sha(qualification_path)!=registration['cpu_qualification_sha256']:
        raise ValueError('CPU qualification changed')
    for name,expected in registration['qualified_model_sources'].items():
        if sha(snapshot/'research/recipes/strong19_throughput'/name)!=expected:
            raise ValueError('Numerical source differs from CPU qualification: '+name)
    config=read(snapshot/'resolved_config.json')
    if sha(snapshot/'resolved_config.json')!=registration['config_sha256']:raise ValueError('Config changed')
    folder=ROOT/'runs'/args.attempt;closed=read(folder/'result.json')
    if closed['status']!='passed' or closed['snapshot_id']!=snapshot.name:raise ValueError('Attempt did not pass')
    inputs={str(p.relative_to(ROOT)):sha(p) for p in
            (STUDY/'registration-001.json',folder/'result.json',snapshot/'resolved_config.json')}
    reports=[]
    expected_cases=[(v['name'],d['bucket']) for v in config['variants'] for d in config['draws']]
    for host in range(4):
        process_path=folder/f'rank-{host}/result.json';process=read(process_path)
        report_path=folder/f'rank-{host}/artifacts/result.json';report=read(report_path)
        config_path=folder/f'rank-{host}/artifacts/resolved_config.json'
        if sha(config_path)!=registration['config_sha256']:raise ValueError('Worker configuration changed')
        inputs[str(process_path.relative_to(ROOT))]=sha(process_path);inputs[str(report_path.relative_to(ROOT))]=sha(report_path)
        if (process['status']!='passed' or process['returncode']!=0 or not process['source_integrity']
                or process['timed_out'] or process['cancelled'] or report['status']!='passed'
                or report['snapshot_id']!=snapshot.name or report['host_rank']!=host
                or report['reference_config_sha256']!=config['reference_config_sha256']
                or report['parameter_count']!=232011540):raise ValueError('Rank scope/closure changed')
        if [(c['variant'],c['bucket']) for c in report['cases']]!=expected_cases:raise ValueError('Incomplete cases')
        for case in report['cases']:
            if case['status']!='passed' or [r['timed'] for r in case['updates']]!=[False,True,True]:
                raise ValueError('Incomplete timing case')
            if case['variant']!='baseline' and case['equivalence']['status']!='passed':
                raise ValueError('Numerical check failed')
            draw=next(d for d in config['draws'] if d['bucket']==case['bucket'])
            if any(r['metrics']['accepted']!=1 or r['metrics']['positions']!=draw['positions']
                   for r in case['updates']):raise ValueError('Draw/update differs')
        for actual,expected in zip(report['draws'],config['draws']):
            rank=report['jax_rank'];target=expected['ranks'][rank]
            if (actual['local_entries_sha256']!=target['local_entries_sha256']
                    or actual['local_symmetries']!=target['local_symmetries']
                    or actual['global_positions']!=expected['positions']):raise ValueError('Sampler differs')
        reports.append(report)
    if sorted(r['jax_rank'] for r in reports)!=list(range(4)):raise ValueError('Missing/duplicate JAX rank')
    cases=[]
    for name,bucket in expected_cases:
        rows=[next(c for c in r['cases'] if (c['variant'],c['bucket'])==(name,bucket)) for r in reports]
        latencies=[max(c['updates'][i]['seconds'] for c in rows) for i in (1,2)]
        latency=statistics.median(latencies);positions=next(d['positions'] for d in config['draws'] if d['bucket']==bucket)
        cases.append(dict(variant=name,bucket=bucket,global_step_seconds=latencies,median_step_seconds=latency,
            live_positions_per_second=positions/latency,max_compiled_peak_bytes=max(c['estimated_peak_bytes'] for c in rows),
            all_rank_state_exact=None if name=='baseline' else all(c['equivalence']['exactly_equal'] for c in rows),
            max_state_abs_error=0 if name=='baseline' else max(c['equivalence']['max_abs'] for c in rows),
            max_state_relative_l2=0 if name=='baseline' else max(c['equivalence']['relative_l2'] for c in rows)))
    comparisons={}
    for bucket in (512,768):
        a=next(c for c in cases if c['variant']=='baseline' and c['bucket']==bucket)
        b=next(c for c in cases if c['variant']=='skip_padding' and c['bucket']==bucket)
        comparisons[str(bucket)]=dict(latency_reduction=1-b['median_step_seconds']/a['median_step_seconds'],
            throughput_gain=a['median_step_seconds']/b['median_step_seconds']-1)
    original_counts={512:448,768:64}
    mean={name:sum(next(c['median_step_seconds'] for c in cases if c['variant']==name and c['bucket']==bucket)*count
                  for bucket,count in original_counts.items())/512 for name in ('baseline','skip_padding')}
    result=dict(kind='fixed_model_throughput_screen_review',status='passed',created=time.time(),attempt=args.attempt,
        snapshot=snapshot.name,input_sha256=inputs,operator_sha256=sha(Path(__file__)),cases=cases,comparisons=comparisons,
        observed_bucket_mix_estimate=dict(mean_step_seconds=mean,latency_reduction=1-mean['skip_padding']/mean['baseline'],
            throughput_gain=mean['baseline']/mean['skip_padding']-1),
        candidate='promising' if all(x['latency_reduction']>=.05 for x in comparisons.values()) else 'not_selected',
        scope='Two timed repeats per bucket, maximum host latency per synchronized update. All-rank complete state equivalence. '
              'Fixed real batches; weighted estimate uses the old run bucket counts, not a measured long training run. No MFU or new learning-quality claim.')
    with args.output.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    lines=['# Fixed-model transformer throughput screen','',result['scope'],'',
        '| Runtime | Bucket | Update seconds | Live positions/s | Compiled peak GB |',
        '| --- | ---: | ---: | ---: | ---: |']
    for c in cases:lines.append(f"| {c['variant']} | {c['bucket']} | {c['median_step_seconds']:.3f} | {c['live_positions_per_second']:.1f} | {c['max_compiled_peak_bytes']/1e9:.2f} |")
    lines+=['','| Bucket | Latency reduction | Throughput gain |','| --- | ---: | ---: |']
    for b,c in comparisons.items():lines.append(f"| {b} | {100*c['latency_reduction']:.2f}% | {100*c['throughput_gain']:.2f}% |")
    lines+=['',f"Candidate: **{result['candidate']}**. All numerical gates passed. Parameter count, model, objective, optimizer and sampled batches are unchanged.",
        '', 'The skipped work belongs only to fully padded encoder chunks. Partial chunks and causal histories retain their original order. The CPU qualification also matches full gradients and optimizer state exactly.',
        '', 'See [next architecture ideas](FUTURE_IDEAS.md) for separate, unscheduled model changes.']
    with args.output.with_suffix('.md').open('x') as f:f.write('\n'.join(lines)+'\n')
    print(json.dumps(dict(status='passed',candidate=result['candidate'],comparisons=comparisons,
        observed_bucket_mix_estimate=result['observed_bucket_mix_estimate'])),flush=True)


if __name__=='__main__':main()
