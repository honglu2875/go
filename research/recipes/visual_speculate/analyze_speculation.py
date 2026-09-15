"""Audit the fixed exact-dynamics early-exit packet screen, including failures."""
import argparse
from pathlib import Path
import statistics
import sys
sys.dont_write_bytecode = True
SOURCE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(SOURCE / 'packages/gozero/src'))
from gozero.checkpoints import sha256
from gozero.snapshots import canonical_json, read_json, verify


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--workspace-root', type=Path, required=True)
    p.add_argument('--registration', type=Path, required=True); p.add_argument('--registration-sha256', required=True)
    p.add_argument('--repair', type=Path, required=True); p.add_argument('--repair-sha256', required=True)
    p.add_argument('--attempt', required=True); p.add_argument('--output', type=Path, required=True); a = p.parse_args()
    verify(SOURCE); root = a.workspace_root.resolve()
    if sha256(a.registration) != a.registration_sha256 or sha256(a.repair) != a.repair_sha256 or a.output.exists():
        raise ValueError('Scientific registration changed or output exists')
    registration, repair = read_json(a.registration), read_json(a.repair)
    if repair['parent_registration_sha256'] != a.registration_sha256:
        raise ValueError('Repair is not attached to this study')
    snapshot = root / '.gozero/snapshots' / repair['replacement_snapshot']; verify(snapshot)
    if (sha256(snapshot / 'resolved_config.json') != repair['config_sha256']
            or repair['config_sha256'] != registration['sources'][1]['config_sha256']):
        raise ValueError('Repair changed the scientific configuration')
    evidence = {}
    def remember(path):
        evidence[str(path.relative_to(root))] = sha256(path); return read_json(path)
    directory = root / 'runs' / a.attempt; closed = remember(directory / 'result.json')
    if closed['status'] != 'passed' or closed['snapshot_id'] != snapshot.name:
        raise ValueError('Speculative screen did not pass execution qualification')
    reports = [remember(directory / f'rank-{h}/artifacts/result.json') for h in range(4)]
    if any(r['status'] != 'passed' or r['snapshot_id'] != snapshot.name or r['parameter_count'] != 233137152 for r in reports):
        raise ValueError('One rank differs from the qualified full-model contract')
    if len({r['candidate_sha256'] for r in reports}) != 1:
        raise ValueError('Hosts used different trained models')
    repetitions = len(reports[0]['one_full_step_seconds'])
    baseline = [max(r['one_full_step_seconds'][i] for r in reports) for i in range(repetitions)]
    reference = statistics.median(baseline); conditions = []
    for index, expected in enumerate(reports[0]['conditions']):
        records = [r['conditions'][index] for r in reports]
        if any(r['name'] != expected['name'] for r in records):
            raise ValueError('Hosts used different packet configurations')
        for h, record in enumerate(records):
            path = directory / f"rank-{h}/artifacts/{record['name']}.npz"
            if sha256(path) != record['packet_arrays_sha256']:
                raise ValueError('Saved speculative proposal packet changed')
            evidence[str(path.relative_to(root))] = sha256(path)
        times = [max(r['packet_seconds'][i] for r in records) for i in range(repetitions)]
        audited_times = [max(r['packet_seconds'][i] + r['coupling_and_native_audit_seconds'][i] for r in records) for i in range(repetitions)]
        accepted, resolved = [], []
        for i in range(repetitions):
            decisions = [d for r in records for d in r['resolutions'][i]]
            if any(len(d['actions']) != d['accepted_draft_moves'] + int(d['replacement'] is not None) for d in decisions):
                raise ValueError('Resolution is not a contiguous accepted prefix plus at most one replacement')
            accepted.append(statistics.mean(d['accepted_draft_moves'] for d in decisions))
            resolved.append(statistics.mean(len(d['actions']) for d in decisions))
        optimistic = [reference * n / t for n, t in zip(resolved, times)]
        with_audit = [reference * n / t for n, t in zip(resolved, audited_times)]
        conditions.append({'name': expected['name'], 'maximum_full_reference_policy_tv': max(r['maximum_target_vs_full_policy_tv'] for r in records),
            'exact_positions_checked': sum(r['exact_positions_checked'] for r in records),
            'critical_packet_seconds': times, 'critical_packet_and_audit_seconds': audited_times,
            'resolved_moves_per_root': resolved, 'accepted_draft_moves_per_root': accepted,
            'same_path_mean_overlap': statistics.mean(r['mean_same_path_overlap'] for r in records),
            'median_optimistic_amortized_speedup_excluding_all_cpu_audit': statistics.median(optimistic),
            'median_amortized_speedup_including_candidate_audit': statistics.median(with_audit),
            'screen_passed': statistics.median(with_audit) >= registration['screen']['minimum_amortized_independent_root_speedup']})
    attempts = [remember(root / 'runs' / name / 'result.json') for name in
                ['pod-20260912T044233Z-1ba5ec36', repair['retained_failed_attempt'], a.attempt]]
    result = {'schema_version': 1, 'kind': 'visual_exact_dynamics_speculation_audit', 'audit_status': 'passed',
        'operator_snapshot': SOURCE.name, 'registration_sha256': a.registration_sha256,
        'repair_sha256': a.repair_sha256, 'evidence': evidence, 'candidate_sha256': reports[0]['candidate_sha256'],
        'critical_one_full_step_seconds': baseline, 'conditions': conditions,
        'screen_passed': any(c['screen_passed'] for c in conditions),
        'attempt_chip_hours_including_failure': sum(r['reserved_chip_hours'] for r in attempts),
        'decision': 'Do not promote long-horizon speculation for this checkpoint; every configuration fails even the optimistic device-only amortization screen.',
        'limitations': 'Independent roots, one fixed proposal draw per root reused for timing with fresh acceptance draws. Candidate CPU audit replays complete histories; baseline timing excludes a corresponding Rust audit. The optimistic ratio excludes all candidate CPU audit and still fails. Root prefill and future cache repair are excluded. BF16 block probabilities have measured differences from serial/full evaluation. No completed-game throughput, MCTS equivalence, MFU or strength improvement claim.'}
    if any(c['median_optimistic_amortized_speedup_excluding_all_cpu_audit'] >= 1.2 for c in conditions):
        raise ValueError('Decision text requires reconsideration for an optimistic passing condition')
    a.output.write_bytes(canonical_json(result))
    print(canonical_json({k: result[k] for k in ['audit_status', 'screen_passed', 'conditions', 'attempt_chip_hours_including_failure']}).decode().strip())


if __name__ == '__main__':
    main()
