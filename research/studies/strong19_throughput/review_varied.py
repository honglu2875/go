"""Independent all-rank review of the prospective varied-batch runtime gates."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
STUDY = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--attempt', required=True)
    args = parser.parse_args()
    if '/' in args.attempt or not args.attempt.startswith('pod-'):
        raise ValueError('Invalid attempt')
    reg_path = STUDY / 'varied-registration-001.json'
    reg = read(reg_path)
    process = read(STUDY / 'varied-process-001.json')
    if process['registration_sha256'] != sha(reg_path):
        raise ValueError('Registration changed')
    bindings = {'cpu_qualification_sha256': 'diagnostic-cpu-001.json',
                'driver_check_sha256': 'varied-driver-check-001.json',
                'protocol_sha256': 'VARIED_PROTOCOL_001.md',
                'diagnostic_sha256': 'DIAGNOSTIC_001.json', 'screen001_sha256': 'SCREEN_001.json'}
    for key, name in bindings.items():
        if sha(STUDY / name) != reg[key]:
            raise ValueError('Registration input changed: ' + name)
    snapshot = ROOT / '.gozero/snapshots' / reg['snapshot']
    sys.path.insert(0, str(snapshot / 'packages/gozero/src'))
    from gozero.snapshots import verify
    verify(snapshot)
    config = read(snapshot / 'resolved_config.json')
    if sha(snapshot / 'resolved_config.json') != reg['config_sha256']:
        raise ValueError('Configuration changed')
    for name, expected in {**reg['qualified_sources'], **reg['driver_sources']}.items():
        if sha(snapshot / 'research/recipes/strong19_padding_varied' / name) != expected:
            raise ValueError('Qualified source changed: ' + name)
    folder = ROOT / 'runs' / args.attempt
    closed = read(folder / 'result.json')
    if closed['snapshot_id'] != snapshot.name:
        raise ValueError('Wrong attempt snapshot')
    inputs = {str(p.relative_to(ROOT)): sha(p) for p in (reg_path, folder / 'result.json', snapshot / 'resolved_config.json')}
    names = config['variants']
    draws = [d for d in config['replay_prefix'] if d['turn'] in config['draw_turns']]
    reports, exact_checks, drift_checks = [], [], []
    for host in range(4):
        path = folder / f'rank-{host}/artifacts/result.json'
        report = read(path)
        receipt_path = folder / f'rank-{host}/result.json'
        receipt = read(receipt_path)
        if (report['snapshot_id'] != snapshot.name or report['host_rank'] != host
                or report['parameter_count'] != 232011540 or receipt['timed_out']
                or not receipt['source_integrity']
                or report['reference_config_sha256'] != config['reference_config_sha256']
                or sha(path.with_name('resolved_config.json')) != reg['config_sha256']
                or [c['variant'] for c in report['cases']] != names):
            raise ValueError('Rank scope or closure differs')
        if len({c['initial_state_sha256'] for c in report['cases']}) != 1:
            raise ValueError('Initialization differs')
        if len(report['draws']) != len(draws):
            raise ValueError('Incomplete draw records')
        for actual, expected in zip(report['draws'], draws, strict=True):
            target = expected['ranks'][report['jax_rank']]
            if (actual['turn'] != expected['turn'] or actual['bucket'] != expected['bucket']
                    or actual['global_positions'] != expected['positions']
                    or actual['local_entries_sha256'] != target['local_entries_sha256']
                    or actual['local_symmetries'] != target['local_symmetries']):
                raise ValueError('Registered draw changed')
        for case in report['cases']:
            if set(case['compilations']) != {'512', '768'} or len(case['updates']) != len(draws):
                raise ValueError('Incomplete shape/update coverage')
            for bucket, comp in case['compilations'].items():
                sizes = comp['memory_bytes']
                peak = sizes['argument_size_in_bytes'] + sizes['output_size_in_bytes'] + sizes['temp_size_in_bytes'] - sizes['alias_size_in_bytes']
                upper = comp['all_rank_peak_upper_bytes']
                if (peak != comp['estimated_peak_bytes'] or len(upper) != 4
                        or max(upper) > config['compiled_memory_limit_bytes']
                        or peak > max(upper) or peak <= 2**31):
                    raise ValueError('Memory gate is inconsistent or overflowed')
                if case['variant'] == 'conditional_skip':
                    dense = report['cases'][1]['compilations'][bucket]
                    if comp['hlo_sha256'] != dense['hlo_sha256'] or not comp['reused_from_conditional_dense']:
                        raise ValueError('Conditional executable was not reused')
            for index, (row, draw) in enumerate(zip(case['updates'], draws, strict=True), 1):
                m = row['metrics']
                if (row['update'] != index or row['draw_turn'] != draw['turn'] or row['bucket'] != draw['bucket']
                        or m['accepted'] != 1 or m['positions'] != draw['positions']
                        or not all(math.isfinite(v) for v in m.values())
                        or not math.isclose(m['learning_rate'], index * .001 / 40, rel_tol=1e-6)):
                    raise ValueError('Update, optimizer or exposure differs')
                for reference, summary in row['comparisons'].items():
                    comparison_path = path.with_name(f"{case['variant']}-vs-{reference}-update-{index}.json")
                    full = read(comparison_path)
                    if len(full['leaves']) != 160 or any(full[k] != summary[k] for k in summary):
                        raise ValueError('Full-state comparison differs')
                    if reference == 'conditional_dense':
                        exact = full['exactly_equal'] and full['metrics_exactly_equal']
                        exact_checks.append(exact)
                        expected_gate = dict(status='passed' if exact else 'failed')
                    else:
                        reference_metrics = report['cases'][0]['updates'][index - 1]['metrics']
                        failures = []
                        for key, value in m.items():
                            if not math.isclose(value, reference_metrics[key], rel_tol=config['original_metric_rtol'], abs_tol=config['original_metric_atol']):
                                failures.append('metric/' + key)
                        for key, values in full['groups'].items():
                            part = key.split('/')[0]
                            if part in config['original_group_relative_l2_limits']:
                                if values['relative_l2'] > config['original_group_relative_l2_limits'][part]:
                                    failures.append('relative_l2/' + key)
                                if part == 'first' and values['cosine'] is not None and values['cosine'] < config['original_first_moment_min_cosine']:
                                    failures.append('cosine/' + key)
                            elif part == 'counter' and not values['exactly_equal']:
                                failures.append('counter')
                        drift_checks.append(not failures)
                        expected_gate = dict(status='passed' if not failures else 'failed', failures=failures,
                                             screen001_coordinate_gate_passed=full['groups']['all']['outside_screen001_tolerance'] == 0)
                    observed_gate = dict(full['gate'])
                    if 'failures' in expected_gate:
                        expected_gate['failures'] = sorted(expected_gate['failures'])
                        observed_gate['failures'] = sorted(observed_gate['failures'])
                    if expected_gate != observed_gate:
                        raise ValueError('Independently recalculated numerical gate differs')
                    inputs[str(comparison_path.relative_to(ROOT))] = sha(comparison_path)
        for p in (path, receipt_path):
            inputs[str(p.relative_to(ROOT))] = sha(p)
        reports.append(report)
    if {r['jax_rank'] for r in reports} != set(range(4)) or len({r['cases'][0]['initial_state_sha256'] for r in reports}) != 1:
        raise ValueError('Replicated rank/initial-state identity differs')
    if len(exact_checks) != 16 or len(drift_checks) != 32:
        raise ValueError('Incomplete numerical gate coverage')
    numerical = all(exact_checks) and all(drift_checks)
    if numerical and (closed['status'] != 'passed' or any(r['status'] != 'passed' for r in reports)):
        raise ValueError('Numerical pass has no successful closure')
    cases = []
    for case_index, name in enumerate(names):
        for bucket in (512, 768):
            indices = [i for i, d in enumerate(draws) if d['bucket'] == bucket]
            samples = [max(r['cases'][case_index]['updates'][i]['seconds'] for r in reports) for i in indices]
            peaks = [r['cases'][case_index]['compilations'][str(bucket)]['estimated_peak_bytes'] for r in reports]
            positions = [draws[i]['positions'] for i in indices]
            cases.append(dict(variant=name, bucket=bucket, global_step_seconds=samples,
                              median_seconds=statistics.median(samples),
                              live_positions_per_second=sum(positions) / sum(samples),
                              max_compiled_peak_bytes=max(peaks)))
    comparisons = {}
    for bucket in (512, 768):
        old = next(r for r in cases if r['variant'] == 'original_dense' and r['bucket'] == bucket)
        new = next(r for r in cases if r['variant'] == 'conditional_skip' and r['bucket'] == bucket)
        comparisons[str(bucket)] = dict(latency_reduction=1 - new['median_seconds'] / old['median_seconds'],
                                        throughput_gain=old['median_seconds'] / new['median_seconds'] - 1)
    speed = all(c['latency_reduction'] >= config['minimum_latency_reduction'] for c in comparisons.values())
    qualified = numerical and speed
    result = dict(kind='varied_padding_runtime_review', status='passed' if qualified else 'rejected', created=time.time(),
                  attempt=args.attempt, snapshot=snapshot.name, exact_padding_controls=all(exact_checks),
                  original_compiler_drift_gates=all(drift_checks), performance_gate=speed,
                  selected_for_research=qualified, cases=cases, comparisons=comparisons,
                  original_screen_status='rejected', input_sha256=inputs, operator_sha256=sha(Path(__file__)),
                  scope='Four varied real batches, same model/objective/optimizer. Exact shared-executable padding controls. '
                        'Bounded original-compiler drift; no bitwise historical-compiler, long-run quality, strength or MFU claim.')
    with (STUDY / 'VARIED_001.json').open('x') as f:
        json.dump(result, f, indent=2)
        f.write('\n')
    lines = ['# Varied-batch padding qualification', '',
             f"Outcome: **{'selected for research' if qualified else 'rejected'}**.", '', result['scope'], '',
             '| Runtime | Length | Median step | Live positions/s | Compiled peak |',
             '|---|---:|---:|---:|---:|']
    for row in cases:
        lines.append(f"| {row['variant']} | {row['bucket']} | {row['median_seconds']:.3f} s | {row['live_positions_per_second']:.1f} | {row['max_compiled_peak_bytes']/1e9:.2f} GB |")
    lines += ['', f"Exact padding controls: **{all(exact_checks)}**. Original-compiler group/metric gates: **{all(drift_checks)}**.", '',
              'The initial coordinate-wise screen remains rejected. Its diagnostic showed the discrepancy before any padding was skipped. '
              'This separate qualification preregistered fresh runtime batches and explicit group/metric bounds; '
              'see [the protocol](VARIED_PROTOCOL_001.md), [the failed screen](SCREEN_001.md) and [the diagnostic](DIAGNOSTIC_001.md).', '',
              'Two samples per length are a short systems test. Compilation and host state-comparison costs are excluded from step latency. '
              'No new long learning run was scheduled, and no historical results or checkpoint arrays were altered.']
    with (STUDY / 'VARIED_001.md').open('x') as f:
        f.write('\n'.join(lines) + '\n')
    print(json.dumps({k: result[k] for k in ('status', 'exact_padding_controls', 'original_compiler_drift_gates', 'selected_for_research', 'comparisons')}))


if __name__ == '__main__':
    main()
