#!/usr/bin/env python3
"""Audit visual-model qualifications and descriptive supplied-path profiles."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

sys.dont_write_bytecode = True
SOURCE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(SOURCE / 'packages/gozero/src'))
from gozero.snapshots import canonical_json, read_json, verify
from gozero.checkpoints import sha256

INITIAL = ['pod-20260912T014956Z-27d5501c', 'pod-20260912T015403Z-751ad094',
           'pod-20260912T015749Z-ccf7d3b0', 'pod-20260912T020510Z-7afaadb0',
           'pod-20260912T021252Z-228bada3']


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workspace-root', type=Path, required=True)
    p.add_argument('--profile-result', type=Path)
    p.add_argument('--expected-profile-result-sha256')
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args(); verify(SOURCE); root = a.workspace_root.resolve()
    entries = [(name, None) for name in INITIAL]
    if a.profile_result is not None:
        if sha256(a.profile_result) != a.expected_profile_result_sha256:
            raise ValueError('Profile result identity differs')
        profiles = read_json(a.profile_result)
        entries += [(e['attempt'], e['id']) for e in profiles['attempts']]
    artifacts = {}
    def read(path):
        path = Path(path); artifacts[str(path.relative_to(root))] = sha256(path)
        return read_json(path)
    rows = []
    for attempt, case in entries:
        directory = root / 'runs' / attempt
        result = read(directory / 'result.json'); launch = read(directory / 'launch.json')
        snapshot = root / '.gozero/snapshots' / result['snapshot_id']; manifest = verify(snapshot)
        c = read(snapshot / 'resolved_config.json')
        if (result['attempt_id'] != attempt or result['snapshot_id'] != launch['snapshot_id']
                or manifest['recipe'] != 'research/recipes/visual_causal'
                or not math.isclose(result['reserved_chip_hours'], result['elapsed_seconds'] * 16 / 3600, abs_tol=1e-9)):
            raise ValueError('Attempt identity or cost differs')
        row = {'attempt': attempt, 'case': case, 'snapshot_id': result['snapshot_id'],
               'status': result['status'], 'attempt_chip_hours': result['reserved_chip_hours']}
        ranks = []
        for rank in range(4):
            wrapper = read(directory / f'rank-{rank}/result.json')
            report_path = directory / f'rank-{rank}/artifacts/result.json'
            if report_path.exists():
                report = read(report_path); ranks.append(report)
            if result['status'] == 'passed' and (wrapper['status'] != 'passed' or not wrapper['source_integrity'] or wrapper['timed_out']):
                raise ValueError('Passed attempt has an unqualified rank')
        if result['status'] == 'passed':
            if len(ranks) != 4 or any(r['status'] != 'passed' or r['snapshot_id'] != result['snapshot_id'] for r in ranks):
                raise ValueError('Model report coverage differs')
            schema = read(directory / 'rank-0/artifacts/parameter_schema.json')
            count = sum(item['elements'] for item in schema)
            if any(r['parameter_count'] != count or r['cache_max_abs_error'] > c['probe']['cache_tolerance'] for r in ranks):
                raise ValueError('Model count or cache qualification differs')
            if len({h for r in ranks for h in r['prediction_sha256_per_rank']}) != 1:
                raise ValueError('Cross-host model prediction identity differs')
            timing = {k: [max(r['supplied_path_timing_seconds'][k][i] for r in ranks)
                          for i in range(c['probe']['repetitions'])] for k in ('block', 'sequential')}
            if any(not math.isfinite(t) or t <= 0 for values in timing.values() for t in values):
                raise ValueError('Invalid timing sample')
            row.update(parameter_count=count, sequences_across_pod=c['probe']['sequences_per_host'] * 4,
                       sequences_per_device=c['probe']['sequences_per_host'] // 4,
                       positions=c['probe']['positions'], horizon=c['probe']['draft_plies'], board_size=c['probe']['board_size'],
                       critical_host_times_seconds=timing,
                       median_seconds={k: statistics.median(v) for k, v in timing.items()},
                       median_paired_sequential_over_block=statistics.median(s / b for s, b in zip(timing['sequential'], timing['block'])),
                       maximum_device_peak_bytes=max(m['peak_bytes_in_use'] for r in ranks for m in r['device_memory_stats']),
                       maximum_cache_abs_error=max(r['cache_max_abs_error'] for r in ranks),
                       maximum_prediction_abs_error=max(vv for r in ranks for v in r['prediction_errors'].values() for vv in v.values()),
                       gradient_qualified=c['probe']['gradient_check'])
            if c['probe']['gradient_check']:
                if any(not math.isfinite(r['gradient']['norm']) or r['gradient']['norm'] <= 0 for r in ranks):
                    raise ValueError('Gradient qualification differs')
                row['gradient_norms'] = [r['gradient']['norm'] for r in ranks]
        else:
            row['errors'] = [r.get('error') for r in ranks]
        rows.append(row)
    out = {'schema_version': 1, 'kind': 'visual_model_qualification_audit', 'operator_snapshot': SOURCE.name,
           'audit_status': 'passed', 'attempts': rows, 'artifacts': artifacts,
           'total_recorded_attempt_chip_hours': sum(r['attempt_chip_hours'] for r in rows),
           'claims_mfu': False, 'claims_go_strength': False, 'claims_rollout_speedup': False,
           'timing_scope': 'Synthetic device-resident supplied paths. Per-repetition maximum across hosts, then descriptive medians; hosts are not independent statistical replicates. Excludes drafting, CPU board work, acceptance, queueing and transfer.'}
    if a.output.exists():
        raise FileExistsError(a.output)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open('xb') as f:
        f.write(canonical_json(out))
    print(json.dumps({'status': out['audit_status'], 'output': str(a.output), 'sha256': sha256(a.output), 'attempts': len(rows)}))


if __name__ == '__main__':
    main()
