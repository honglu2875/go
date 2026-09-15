"""Audit registered local tensor-parallel latency and numerical qualification."""
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
    p.add_argument('--attempts', nargs=2, required=True); p.add_argument('--output', type=Path, required=True)
    a = p.parse_args(); verify(SOURCE); root = a.workspace_root.resolve()
    if a.output.exists() or sha256(a.registration) != a.registration_sha256: raise ValueError('Result exists or registration differs')
    registration = read_json(a.registration); evidence = {}; cost = 0.; stages = {}
    def remember(path):
        evidence[str(path.relative_to(root))] = sha256(path); return read_json(path)
    for declaration, attempt in zip(registration['stages'], a.attempts):
        snapshot = root / '.gozero/snapshots' / declaration['snapshot_id']; verify(snapshot)
        config = remember(snapshot / 'resolved_config.json')
        if sha256(snapshot / 'resolved_config.json') != declaration['config_sha256']: raise ValueError('Configuration changed')
        directory = root / 'runs' / attempt; closed = remember(directory / 'result.json')
        if closed['status'] != 'passed' or closed['snapshot_id'] != snapshot.name: raise ValueError('Attempt did not qualify')
        cost += closed['reserved_chip_hours']; hosts = []
        for host in range(4):
            report = remember(directory / f'rank-{host}/artifacts/result.json')
            if report['status'] != 'passed' or report['host_rank'] != host or report['snapshot_id'] != snapshot.name:
                raise ValueError('Host scientific identity differs')
            expected = {f'tp{t}-h{h}' for t in config['tensor_parallel_sizes'] for h in config['horizons']}
            if {q['name'] for q in report['qualification']} != expected or set(report['timings']) != expected:
                raise ValueError('Missing qualified or timed condition')
            for q in report['qualification']:
                if (max(q['maximum_legal_policy_tv'].values()) > config['maximum_policy_tv'] or
                        q['maximum_absolute_head_errors']['value'] > config['maximum_value_error']):
                    raise ValueError('Policy or value exceeded the numerical contract')
            hosts.append(report)
        critical = {key: [max(r['timings'][key][i] for r in hosts) for i in range(config['repetitions'])] for key in expected}
        medians = {key: statistics.median(values) for key, values in critical.items()}
        speedup = {key: medians['tp1-h' + key.split('-h')[1]] / value for key, value in medians.items()}
        stages[declaration['stage']] = {'attempt': attempt, 'critical_path_seconds': critical,
            'critical_median_seconds': medians, 'speedup_vs_same_horizon_tp1': speedup,
            'maximum_legal_policy_tv': max(max(q['maximum_legal_policy_tv'].values()) for r in hosts for q in r['qualification']),
            'physical_root_shards': [{k: v for k, v in r['compilation'].items() if k.endswith('-root')} for r in hosts]}
    final = stages['233m_latency_screen']; threshold = registration['primary_screen']['minimum_critical_median_speedup']
    candidates = {f'tp{t}': final['speedup_vs_same_horizon_tp1'][f'tp{t}-h1'] >= threshold for t in (2, 4)}
    result = {'schema_version': 1, 'kind': 'visual_tensor_parallel_latency_audit', 'audit_status': 'passed',
        'registration_sha256': a.registration_sha256, 'operator_snapshot': SOURCE.name, 'evidence': evidence,
        'stages': stages, 'candidate_speed_gates': candidates, 'screen_passed': any(candidates.values()),
        'attempt_chip_hours': cost, 'decision': 'Retain TP1; local tensor partitioning failed the fixed-batch latency screen.' if not any(candidates.values()) else 'Proceed to native search qualification.',
        'limitations': registration['scope']}
    a.output.write_bytes(canonical_json(result))
    print(canonical_json({'audit_status': 'passed', 'candidate_speed_gates': candidates,
        'full_model_speedups': final['speedup_vs_same_horizon_tp1']}).decode().strip())


if __name__ == '__main__': main()
