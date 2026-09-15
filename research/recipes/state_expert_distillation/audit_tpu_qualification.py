#!/usr/bin/env python3
"""Verify registered TPU attempts, continuation coverage and resource observations."""
import argparse
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
SOURCE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(SOURCE / 'packages/gozero/src'))
from gozero.checkpoints import sha256
from gozero.snapshots import canonical_json, read_json, verify


def require(value, message):
    if not value:
        raise ValueError(message)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workspace-root', type=Path, required=True)
    p.add_argument('--protocol', type=Path, required=True)
    p.add_argument('--expected-protocol-sha256', required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args(); root = a.workspace_root.resolve(); verify(SOURCE)
    require(sha256(a.protocol) == a.expected_protocol_sha256, 'Protocol changed')
    spec = read_json(a.protocol); raw = root / spec['output']; result = read_json(raw / 'result.json')
    verify(root / '.gozero/snapshots' / spec['runner_snapshot'])
    require(result['status'] == 'passed' and result['protocol_sha256'] == a.expected_protocol_sha256
            and result['runner_snapshot'] == spec['runner_snapshot']
            and [(r['arm'], r['mode']) for r in result['attempts']] == [('history', 'baseline'), ('history', 'resumed'), ('state', 'baseline'), ('state', 'resumed')],
            'Qualification identity or attempt coverage differs')
    for path, digest in spec['prerequisites_sha256'].items():
        require(sha256(root / path) == digest and read_json(root / path)['status'] == 'passed', 'Prerequisite changed')
    for row in result['attempts']:
        folder = root / 'runs' / row['attempt']; launch = read_json(folder / 'launch.json'); pod = read_json(folder / 'result.json')
        require(sha256(folder / 'result.json') == row['result_sha256'] and sha256(folder / 'launch.json') == row['launch_sha256']
                and row['log_sha256'] == sha256(raw / (row['arm'] + '-' + row['mode'] + '.log'))
                and pod['status'] == 'passed' and pod['snapshot_id'] == spec['arms'][row['arm']]['snapshot']
                and pod['start_unix_time'] > spec['registered_unix'] and pod['elapsed_seconds'] < spec['maximum_seconds_per_attempt']
                and pod['reserved_chips'] == 16 and len(pod['hosts']) == 4 and pod['controller_cpus'] == 32
                and row['returncode'] == 0, 'Attempt identity, resources or limits differ')
        require(launch['resume_attempt'] == (result['arms'][row['arm']]['baseline'] if row['mode'] == 'resumed' else None)
                and launch['resume_turn'] == (8 if row['mode'] == 'resumed' else None) and launch['stop_after_turn'] is None,
                'Continuation launch differs')
        for rank in range(4):
            host = read_json(folder / f'rank-{rank}/result.json')
            require(host['status'] == 'passed' and host['returncode'] == 0 and host['source_integrity']
                    and not host['timed_out'] and host['world_size'] == 4 and host['rank'] == rank, 'Host did not close cleanly')
    arms = {}
    for arm, entry in spec['arms'].items():
        source = root / '.gozero/snapshots' / entry['snapshot']; verify(source)
        require(sha256(source / 'resolved_config.json') == entry['config_sha256'], 'Configuration changed')
        for name, digest in spec['cpu_qualified_code_sha256'].items():
            require(sha256(source / name) == digest, 'CPU-qualified code changed')
        expected_attempts = {result['arms'][arm][k] for k in ('baseline', 'resumed')}
        actual_attempts = {f.parent.name for f in (root / 'runs').glob('pod-*/launch.json') if read_json(f)['snapshot_id'] == source.name}
        require(actual_attempts == expected_attempts, 'Unexpected extra qualification attempt')
        recovery_path = raw / (arm + '-recovery.json'); recovery = read_json(recovery_path)
        require(sha256(recovery_path) == result['arms'][arm]['recovery_sha256'] and recovery['status'] == 'passed'
                and sum(r['arrays_exact'] for r in recovery['ranks']) == entry['expected_recovery_arrays'], 'Recovery coverage differs')
        records = [read_json(root / 'runs' / result['arms'][arm]['baseline'] / f'rank-{rank}/artifacts/result.json') for rank in range(4)]
        first = records[0]
        require({r['jax_rank'] for r in records} == set(range(4)), 'Training rank coverage differs')
        for r in records:
            require(r['status'] == 'passed' and r['turn'] == 16 and r['training_complete'] and r['world_size'] == 4
                    and r['parameter_count'] == entry['expected_parameters'] and r['expert_architecture'] == arm
                    and r['behavior_training_enabled'] is False
                    and r['initial_parameter_elements_sha256'] == first['initial_parameter_elements_sha256']
                    and r['fixed_observer'] == first['fixed_observer']
                    and r['counters']['expert_token_exposures'] == first['counters']['expert_token_exposures'], 'Replicated architecture, sampler work or observer differs')
        arms[arm] = {**result['arms'][arm], 'parameter_count': first['parameter_count'],
                     'initial_parameter_elements_sha256': first['initial_parameter_elements_sha256'],
                     'global_expert_token_exposures': first['counters']['expert_token_exposures'],
                     'maximum_learner_seconds': max(r['counters']['learner_seconds'] for r in records),
                     'maximum_segment_seconds': max(r['elapsed_segment_seconds'] for r in records),
                     'maximum_compile_seconds': max(r['update_compile_seconds'] for r in records),
                     'maximum_process_rss_kib': max(r['peak_process_rss_kib'] for r in records),
                     'fixed_observer': first['fixed_observer']}
    require(arms['history']['global_expert_token_exposures'] == arms['state']['global_expert_token_exposures']
            and arms['history']['fixed_observer'] == arms['state']['fixed_observer'], 'Matched qualification inputs differ')
    report = {'schema_version': 1, 'kind': 'state_expert_registered_tpu_qualification_audit', 'status': 'passed',
              'analysis_snapshot': SOURCE.name, 'protocol_sha256': sha256(a.protocol), 'raw_result_sha256': sha256(raw / 'result.json'),
              'arms': arms, 'attempts': result['attempts'], 'recorded_attempt_chip_hours': result['recorded_attempt_chip_hours'],
              'claims_strength': False, 'claims_speedup': False, 'claims_mfu': False, 'claims_host_loss_recovery': False}
    with a.output.open('xb') as stream:
        stream.write(canonical_json(report))
    a.output.chmod(0o444)
    print(json.dumps({k: v for k, v in report.items() if k not in ('attempts', 'arms')}), flush=True)


if __name__ == '__main__':
    main()
