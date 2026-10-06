"""Audit and summarize the separately registered conditional-control diagnostic."""
import argparse
import hashlib
import json
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
    reg_path = STUDY / 'diagnostic-registration-001.json'
    reg = read(reg_path)
    process = read(STUDY / 'diagnostic-process-001.json')
    if process['registration_sha256'] != sha(reg_path):
        raise ValueError('Registration changed')
    snapshot = ROOT / '.gozero/snapshots' / reg['snapshot']
    sys.path.insert(0, str(snapshot / 'packages/gozero/src'))
    from gozero.snapshots import verify
    verify(snapshot)
    config = read(snapshot / 'resolved_config.json')
    if sha(snapshot / 'resolved_config.json') != reg['config_sha256']:
        raise ValueError('Configuration changed')
    cpu_path = STUDY / 'diagnostic-cpu-001.json'
    if sha(cpu_path) != reg['cpu_qualification_sha256'] or read(cpu_path)['status'] != 'passed':
        raise ValueError('CPU qualification changed')
    for name, expected in reg['qualified_sources'].items():
        if sha(snapshot / 'research/recipes/strong19_throughput_diag' / name) != expected:
            raise ValueError('Qualified source changed: ' + name)
    folder = ROOT / 'runs' / args.attempt
    closed = read(folder / 'result.json')
    if closed['snapshot_id'] != snapshot.name:
        raise ValueError('Wrong snapshot')
    inputs = {str(p.relative_to(ROOT)): sha(p) for p in (reg_path, cpu_path, folder / 'result.json')}
    reports = []
    names = config['variants']
    for host in range(4):
        path = folder / f'rank-{host}/artifacts/result.json'
        report = read(path)
        receipt = read(folder / f'rank-{host}/result.json')
        if (report['snapshot_id'] != snapshot.name or report['host_rank'] != host
                or report['parameter_count'] != 232011540 or receipt['timed_out']
                or not receipt['source_integrity']
                or report['reference_config_sha256'] != config['reference_config_sha256']
                or sha(path.with_name('resolved_config.json')) != reg['config_sha256']):
            raise ValueError('Rank identity or scope changed')
        if [c['variant'] for c in report['cases']] != names:
            raise ValueError('Incomplete diagnostic')
        expected = config['draw']['ranks'][report['jax_rank']]
        if (report['draw']['local_entries_sha256'] != expected['local_entries_sha256']
                or report['draw']['local_symmetries'] != expected['local_symmetries']
                or report['draw']['global_positions'] != config['draw']['positions']):
            raise ValueError('Draw changed')
        if len({c['initial_state_sha256'] for c in report['cases']}) != 1:
            raise ValueError('Initialization differs')
        dense, skip = report['cases'][1:]
        if dense['hlo_sha256'] != skip['hlo_sha256'] or not skip['reused_executable']:
            raise ValueError('Conditional executable was not reused')
        for case in report['cases']:
            if len(case['updates']) != config['updates']:
                raise ValueError('Missing update')
            if (max(case['all_rank_peak_upper_bytes']) > config['compiled_memory_limit_bytes']
                    or case['estimated_peak_bytes'] > max(case['all_rank_peak_upper_bytes'])):
                raise ValueError('Memory gate differs')
            for row in case['updates']:
                if row['metrics']['accepted'] != 1 or row['metrics']['positions'] != config['draw']['positions']:
                    raise ValueError('Update rejected or exposure differs')
                for reference, summary in row['comparisons'].items():
                    full_path = path.with_name(f"{case['variant']}-vs-{reference}-update-{row['update']}.json")
                    full = read(full_path)
                    if any(full[k] != summary[k] for k in ('exactly_equal', 'metrics_exactly_equal', 'groups')):
                        raise ValueError('Comparison report differs')
                    if len(full['leaves']) != 160:
                        raise ValueError('Not all 53 parameter/moment leaves and counter were compared')
                    inputs[str(full_path.relative_to(ROOT))] = sha(full_path)
        inputs[str(path.relative_to(ROOT))] = sha(path)
        reports.append(report)
    if {r['jax_rank'] for r in reports} != set(range(4)):
        raise ValueError('Missing or repeated rank')
    if len({r['cases'][0]['initial_state_sha256'] for r in reports}) != 1:
        raise ValueError('Replicated initialization differs')
    cases = []
    for index, name in enumerate(names):
        rows = [r['cases'][index] for r in reports]
        times = [max(c['updates'][u]['seconds'] for c in rows) for u in range(config['updates'])]
        cases.append(dict(variant=name, global_step_seconds=times,
                          max_compiled_peak_bytes=max(c['estimated_peak_bytes'] for c in rows)))
    controls = [row['comparisons']['conditional_dense'] for r in reports for row in r['cases'][2]['updates']]
    exact = all(c['exactly_equal'] and c['metrics_exactly_equal'] for c in controls)
    if exact and (closed['status'] != 'passed' or any(r['status'] != 'passed' for r in reports)):
        raise ValueError('Passing controls have an incomplete process closure')
    conclusion = ('Padding skipping is exactly invariant within the shared conditional executable on both updates. '
                  'Compare original-vs-conditional groups to diagnose compiler lowering; screen 001 remains rejected.'
                  if exact else 'Skipping changes numerical results even within the same executable; the padding path remains unqualified.')
    output = dict(kind='conditional_padding_diagnostic_review', status='passed', created=time.time(),
                  attempt=args.attempt, attempt_status=closed['status'], padding_control_exact=exact,
                  optimization_qualified=False, conclusion=conclusion, cases=cases,
                  comparisons=reports[0]['cases'][2]['updates'],
                  input_sha256=inputs, operator_sha256=sha(Path(__file__)))
    with (STUDY / 'DIAGNOSTIC_001.json').open('x') as f:
        json.dump(output, f, indent=2)
        f.write('\n')
    lines = ['# Conditional padding diagnostic', '', conclusion, '',
             'All variants use the same fixed real 512-frame batch, model, optimizer and initial state. '
             'The two conditional variants reuse one compiled executable, with only the runtime encoder mask changed. '
             'Complete parameters, both AdamW moments and every metric are compared after each update on all ranks.', '',
             '| Path | First update | Second update | Compiled peak |', '|---|---:|---:|---:|']
    for case in cases:
        a, b = case['global_step_seconds']
        lines.append(f"| {case['variant']} | {a:.3f} s | {b:.3f} s | {case['max_compiled_peak_bytes']/1e9:.2f} GB |")
    lines += ['', 'This diagnoses numerical behavior; it does not qualify the optimization for production or change '
              'the failed first screen. The memory collective now uses int32 ceiling KiB to avoid int64 truncation.']
    with (STUDY / 'DIAGNOSTIC_001.md').open('x') as f:
        f.write('\n'.join(lines) + '\n')
    print(json.dumps(dict(status='passed', padding_control_exact=exact, conclusion=conclusion)))


if __name__ == '__main__':
    main()
