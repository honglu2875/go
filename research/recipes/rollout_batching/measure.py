#!/usr/bin/env python3
"""Execute a frozen, counterbalanced timing study after native trace qualification."""
import argparse
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'packages/gozero/src'))
from gozero.checkpoints import sha256
from gozero.snapshots import canonical_json, read_json, verify


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace-root', type=Path, required=True)
    parser.add_argument('--spec', type=Path, required=True)
    parser.add_argument('--spec-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); verify(ROOT)
    root = args.workspace_root.resolve(); spec = read_json(args.spec)
    if sha256(args.spec) != args.spec_sha256:
        raise ValueError('Study specification identity differs')
    variants = spec['variants']; order = spec['order']; limit = spec['timeout_seconds_per_attempt']
    if (not 2 <= len(variants) <= 3 or set(variants) - {'serial', 'async', 'scan'} or 'serial' not in variants
            or len(order) != 3 * len(variants) or any(order.count(v) != 3 for v in variants)
            or set(order) != set(variants) or not 1 <= limit <= 180):
        raise ValueError('Invalid bounded three-repetition timing protocol')
    runner = root / 'ops/pod_run.py'; runner_sha = sha256(runner)
    configs = []; native_sources = []
    for mode, entry in variants.items():
        source = root / '.gozero/snapshots' / entry['snapshot']
        manifest = verify(source)
        if manifest['files']['ops/pod_run.py']['sha256'] != runner_sha:
            raise ValueError('Live pod controller differs from frozen source')
        c = read_json(source / 'resolved_config.json')
        if c.get('dispatch_mode', 'serial') != mode or c['verify_trace'] or c.get('verify_outputs', False):
            raise ValueError('Wrong dispatch mode or verification work enabled during timing')
        c.pop('dispatch_mode', None); c.pop('verify_outputs', None); configs.append(c)
        native_sources.append({k: v['sha256'] for k, v in manifest['files'].items() if k.startswith('crates/')})
        if mode != 'serial':
            path = root / entry['qualification_path']
            if sha256(path) != entry['qualification_sha256']:
                raise ValueError('Qualification report identity differs')
            qualification = read_json(path)
            if qualification['status'] != 'passed' or len(qualification['hosts']) != 4:
                raise ValueError('Four-host exact-trace qualification required')
            qualified_source = root / '.gozero/snapshots' / qualification['hosts'][0]['snapshot_ids'][1]
            qualified_manifest = verify(qualified_source)
            qc = read_json(qualified_source / 'resolved_config.json')
            if qc.get('dispatch_mode') != mode:
                raise ValueError('Qualification used another dispatch mode')
            qc['verify_trace'] = False; qc.pop('dispatch_mode', None); qc.pop('verify_outputs', None)
            if qc != c:
                raise ValueError('Timing configuration differs from qualification')
            for filename in ('train.py', 'model.py'):
                if sha256(qualified_source / qualified_manifest['recipe'] / filename) != sha256(source / manifest['recipe'] / filename):
                    raise ValueError('Timing implementation differs from qualification')
    if any(c != configs[0] for c in configs) or any(n != native_sources[0] for n in native_sources):
        raise ValueError('Variants differ beyond dispatch mode')
    output = args.output.resolve(); output.mkdir(parents=True, exist_ok=False)
    report = {'schema_version': 1, 'analysis_snapshot': ROOT.name, 'spec_sha256': args.spec_sha256,
              'status': 'running', 'started_unix': time.time(), 'executions': [],
              'claims_mfu': False, 'claims_strength_improvement': False, 'claims_learning_speedup': False}
    reference_work = None
    try:
        for sequence, mode in enumerate(order):
            if sha256(runner) != runner_sha or sha256(args.spec) != args.spec_sha256:
                raise ValueError('Controller or protocol changed during execution')
            snapshot = variants[mode]['snapshot']
            command = [sys.executable, '-B', str(runner), '--snapshot', str(root / '.gozero/snapshots' / snapshot),
                       '--native-receipt', str(root / '.gozero/native' / snapshot / 'receipt.json'),
                       '--timeout', str(limit), '--prepare-timeout', '180', '--controller-cpus', '64']
            path = output / f'{sequence:02d}-{mode}.log'
            print(json.dumps({'kind': 'timing_start', 'sequence': sequence, 'mode': mode}), flush=True)
            with path.open('w') as log:
                completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, cwd=root, timeout=limit + 480)
            events = []
            for line in path.read_text().splitlines():
                try: events.append(json.loads(line))
                except ValueError: pass
            starts = [e for e in events if e.get('kind') == 'pod_attempt']
            if len(starts) != 1:
                raise ValueError('Cannot identify executed pod attempt')
            attempt = root / 'runs' / starts[0]['attempt_id']
            pod = read_json(attempt / 'result.json')
            entry = {'sequence': sequence, 'mode': mode, 'snapshot': snapshot, 'attempt': attempt.name,
                     'returncode': completed.returncode, 'pod_result_sha256': sha256(attempt / 'result.json'),
                     'status': pod['status'], 'reserved_attempt_chip_hours': pod['reserved_chip_hours']}
            report['executions'].append(entry)
            if completed.returncode or pod['status'] != 'passed':
                raise RuntimeError('Registered attempt failed; retained in result')
            ranks = [read_json(attempt / f'rank-{host}/artifacts/result.json') for host in range(4)]
            if any(r['status'] != 'passed' or r['snapshot_id'] != snapshot for r in ranks):
                raise ValueError('Incomplete or mismatched rank result')
            work = [{k: r['counters'][k] for k in ('real_moves', 'neural_slots', 'active_neural_evaluations', 'completed_games', 'truncated_games', 'eligible_rows')} for r in ranks]
            if reference_work is not None and reference_work != work:
                raise ValueError('Native work changed during timing repetitions')
            reference_work = work
            if any(r['counters']['verification_neural_slots'] or r['counters']['verification_seconds'] for r in ranks):
                raise ValueError('Verification work leaked into timing')
            entry.update(rank_result_sha256=[sha256(attempt / f'rank-{host}/artifacts/result.json') for host in range(4)],
                         slowest_segment_seconds=max(r['elapsed_segment_seconds'] for r in ranks),
                         slowest_inference_seconds=max(r['counters']['inference_seconds'] for r in ranks),
                         slowest_native_seconds=max(r['counters']['native_seconds'] for r in ranks),
                         average_cpu_cores_per_host=[r['process_cpu_segment_seconds'] / r['elapsed_segment_seconds'] for r in ranks],
                         work_per_host=work, observed_thread_affinity_counts=[r['observed_thread_affinity_counts'] for r in ranks])
            (output / 'progress.json').write_bytes(canonical_json(report))
            print(json.dumps({k: v for k, v in entry.items() if k not in ('work_per_host', 'observed_thread_affinity_counts')}), flush=True)
        base = [e for e in report['executions'] if e['mode'] == 'serial']
        report['comparisons'] = {}
        for mode in variants:
            if mode == 'serial': continue
            candidates = [e for e in report['executions'] if e['mode'] == mode]
            ratios = [a['slowest_segment_seconds'] / b['slowest_segment_seconds'] for a, b in zip(base, candidates)]
            inference = [a['slowest_inference_seconds'] / b['slowest_inference_seconds'] for a, b in zip(base, candidates)]
            report['comparisons'][mode] = {'segment_speedup_by_repetition': ratios, 'median_segment_speedup': statistics.median(ratios),
                'inference_speedup_by_repetition': inference, 'median_inference_speedup': statistics.median(inference),
                'meets_registered_systems_screen': statistics.median(ratios) >= spec['minimum_median_segment_speedup'] and all(x > 1 for x in ratios)}
        report['recorded_attempt_chip_hours'] = sum(e['reserved_attempt_chip_hours'] for e in report['executions'])
        report['status'] = 'passed'; verify(ROOT)
    except Exception as error:
        report.update(status='failed', error=repr(error)); raise
    finally:
        report['finished_unix'] = time.time()
        (output / 'result.json').write_bytes(canonical_json(report))
        print(json.dumps({k: v for k, v in report.items() if k != 'executions'}), flush=True)


if __name__ == '__main__':
    main()
