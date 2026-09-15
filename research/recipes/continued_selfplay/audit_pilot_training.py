#!/usr/bin/env python3
"""Independently audit complete online continuation state and phase work."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

sys.dont_write_bytecode = True
SOURCE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(SOURCE / 'packages/gozero/src'))
from gozero import checkpoints, checkpoint_forks as forks
from gozero.snapshots import canonical_json, read_json, verify


def require(value, message):
    if not value: raise ValueError(message)


def audit_arm(root, spec, execution, name):
    arm = spec['arms'][name]; source = root / '.gozero/snapshots' / arm['snapshot']; verify(source)
    config = read_json(source / 'resolved_config.json'); origin, descriptor, _, native = forks.contract(root, source, config)
    require(origin['parent_group_sha256'] == spec['parent_group_sha256'] and origin['parent_turn'] == spec['parent_turn']
            and config['selfplay_turns'] == spec['final_turn'] and checkpoints.sha256(source / 'resolved_config.json') == arm['config_sha256']
            and config['learner']['learning_rate'] == arm['learning_rate'], 'Arm configuration differs')
    for path, expected in spec['training_code_sha256'].items():
        require(checkpoints.sha256(source / path) == expected, 'Qualified training code differs: ' + path)
    row = execution['arms'][name]; directory = root / 'runs' / row['attempt']; pod = read_json(directory / 'result.json')
    launches = [p for p in (root / 'runs').glob('pod-*/launch.json') if read_json(p)['snapshot_id'] == source.name]
    require(launches == [directory / 'launch.json'], 'More than one training attempt under this arm')
    require(pod['status'] == 'passed' and pod['snapshot_id'] == source.name and row['returncode'] == 0
            and checkpoints.sha256(directory / 'result.json') == row['result_sha256']
            and spec['registered_unix'] < pod['start_unix_time'] <= pod['end_unix_time']
            and pod['timeout_seconds'] == spec['training_timeout_seconds'] and pod['reserved_chips'] == 16
            and pod['resume_attempt'] is None and pod['stop_after_turn'] is None, 'Controller identity or schedule differs')
    hosts = []; group_hashes = {}; exports = set(); phase_totals = {}; total_games = 0; total_caps = 0
    for host in range(4):
        artifacts = directory / f'rank-{host}/artifacts'; result = read_json(artifacts / 'result.json')
        rank = result['jax_rank']; entry = next(e for e in descriptor['rank_checkpoints'] if e['jax_rank'] == rank)
        parent_state, parent_arrays, _ = forks.load_rank(root, descriptor, jax_rank=rank, host_rank=host)
        require(result['status'] == 'passed' and result['kind'] == 'continued_selfplay_training'
                and result['snapshot_id'] == source.name and result['config_sha256'] == arm['config_sha256']
                and result['initialization'] == origin and result['phase_start_turn'] == spec['parent_turn']
                and result['forked_from']['parent_group_sha256'] == spec['parent_group_sha256']
                and result['host_rank'] == host and result['world_size'] == 4 and result['turn'] == spec['final_turn']
                and result['native'] == native and len(result['devices']) == 16
                and result['phase_turns'] == spec['final_turn'] - spec['parent_turn'], 'Rank execution differs')
        expected_turns = list(range(spec['parent_turn'] + config['checkpoint_every'], spec['final_turn'] + 1, config['checkpoint_every']))
        require(sorted(p.name for p in (artifacts / 'checkpoints').glob('turn-*') if p.is_dir()) ==
                [f'turn-{t:09d}' for t in expected_turns], 'Checkpoint schedule differs')
        for turn in expected_turns:
            path = artifacts / 'checkpoints' / f'turn-{turn:09d}'; group_path = path.with_suffix('.group.json'); group = read_json(group_path)
            state, arrays, actors = checkpoints.read(path, expected_manifest_sha256=group['rank_manifests'][rank])
            group_hashes.setdefault(turn, []).append(checkpoints.sha256(group_path))
            require(state['snapshot_id'] == group['snapshot_id'] == source.name and state['config_sha256'] == group['config_sha256'] == arm['config_sha256']
                    and state['turn'] == group['turn'] == turn and state['jax_rank'] == rank and state['world_size'] == group['world_size'] == 4
                    and state['initialization'] == origin and group['initialization_sha256'] == hashlib.sha256(canonical_json(origin)).hexdigest()
                    and state['phase_start_counters'] == entry['counters'] and state['phase_start_turn'] == spec['parent_turn']
                    and state['native_sha256'] == native['binary_sha256'] and state['model_schema'] == parent_state['model_schema']
                    and state['counters']['updates'] == group['updates'] == parent_state['counters']['updates'] + turn - spec['parent_turn']
                    and forks.replicated_digest(arrays) == group['replicated_state_sha256'], 'Committed checkpoint differs')
            require(set(arrays) == set(parent_arrays) and state['replay_count'] == config['learner']['replay_capacity']
                    and all(np.isfinite(value).all() for value in arrays.values()) and json.loads(actors), 'Incomplete/nonfinite replay or actors')
            require(np.min(arrays['replay_pi']) >= 0 and np.allclose(arrays['replay_pi'].sum(axis=1), 1., atol=1e-5)
                    and np.max(np.abs(arrays['replay_z'])) <= 1 and np.max(np.abs(arrays['replay_owner'])) <= 1,
                    'Replay targets are invalid')
        require(checkpoints.sha256(path / 'manifest.json') == result['latest_checkpoint']['manifest_sha256']
                and checkpoints.sha256(group_path) == result['latest_checkpoint']['group_sha256'], 'Final result checkpoint differs')
        phase = {k: value - entry['counters'][k] for k, value in forks.work_counters(state['counters']).items()}
        require(phase == result['phase_counters'] and forks.work_counters(state['counters']) == forks.work_counters(result['counters'])
                and phase['real_moves'] * 4 == spec['additional_global_moves_each_arm']
                and phase['updates'] == spec['additional_updates_each_arm'], 'Phase work reset, double-counted or uneven')
        for key, value in phase.items():
            require(type(value) is int and value >= 0, 'Negative/noninteger phase counter')
            if key != 'updates': phase_totals[key] = phase_totals.get(key, 0) + value
        records = sorted((artifacts / 'games').glob('*.json')); complete = caps = 0
        require(len(records) == phase['completed_games'] + phase['truncated_games'], 'Game count differs from phase counters')
        for file in records:
            game = read_json(file)
            require(file.stem == f"{game['game_id']:016x}" and game['size'] == config['actors']['size']
                    and game['scoring'] == config['actors']['scoring'] and file.with_suffix('.sgf').is_file(), 'Game artifact identity differs')
            if game['truncated']:
                require(game['white_score'] is None, 'Truncated game has an assigned outcome'); caps += 1
            else:
                require(game['white_score'] is not None and game['actions'][-2:] == [81, 81], 'Completed game has no terminal outcome'); complete += 1
        require(complete == phase['completed_games'] and caps == phase['truncated_games'], 'Completion counters differ')
        total_games += complete + caps; total_caps += caps
        export = artifacts / 'model_export.npz'; require(checkpoints.sha256(export) == result['model_export_sha256'], 'Export hash differs')
        with np.load(export, allow_pickle=False) as saved:
            require(set(saved.files) == {k for k in arrays if k.startswith('p_')}
                    and all(np.array_equal(saved[k], arrays[k]) for k in saved.files), 'Export is not the final checkpoint')
        exports.add(result['model_export_sha256'])
        change = max(float(np.max(np.abs(arrays[k] - parent_arrays[k]))) for k in arrays if k.startswith('p_'))
        require(change > 0, 'Continuation did not change parameters')
        elapsed = result['elapsed_segment_seconds']
        hosts.append({'host_rank': host, 'jax_rank': rank, 'result_sha256': checkpoints.sha256(artifacts / 'result.json'),
                      'complete_checkpoint_arrays': len(arrays), 'phase_counters': phase, 'elapsed_segment_seconds': elapsed,
                      'host_average_cpu_cores': result['process_cpu_segment_seconds'] / elapsed,
                      'maximum_parameter_change_from_parent': change,
                      'phase_timing_seconds': {k: result['counters'][k] - parent_state['counters'][k]
                                               for k in result['counters'] if k.endswith('_seconds')}})
    require({h['jax_rank'] for h in hosts} == set(range(4)) and len(exports) == 1
            and all(len(set(hashes)) == 1 and len(hashes) == 4 for hashes in group_hashes.values()), 'Replica/group coverage differs')
    return {'attempt': row['attempt'], 'snapshot': source.name, 'model_export_sha256': exports.pop(),
            'pod_result_sha256': row['result_sha256'], 'hosts': hosts, 'phase_global_counters': phase_totals,
            'additional_updates': spec['additional_updates_each_arm'],
            'global_learner_exposures': spec['additional_updates_each_arm'] * config['learner']['batch_size'] * 4,
            'training_cap_fraction': total_caps / total_games, 'training_completion_criterion_met': total_caps / total_games <= spec['criterion']['training_cap_fraction_maximum'],
            'group_sha256': {str(t): hashes[0] for t, hashes in group_hashes.items()},
            'recorded_attempt_chip_hours': pod['reserved_chip_hours'], 'elapsed_attempt_seconds': pod['elapsed_seconds']}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workspace-root', type=Path, required=True); p.add_argument('--spec', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True); a = p.parse_args(); root = a.workspace_root.resolve(); verify(SOURCE)
    spec = read_json(a.spec); execution_path = root / spec['training_output'] / 'result.json'; execution = read_json(execution_path)
    report = {'schema_version': 1, 'kind': 'online_annealing_training_audit', 'status': 'failed', 'analysis_snapshot': SOURCE.name,
              'spec_sha256': checkpoints.sha256(a.spec), 'execution_result_sha256': checkpoints.sha256(execution_path), 'arms': {}}
    try:
        require(execution['status'] == 'passed' and execution['spec_sha256'] == report['spec_sha256']
                and set(execution['arms']) == set(spec['arms']), 'Training execution failed or differs')
        for name in spec['arm_order']: report['arms'][name] = audit_arm(root, spec, execution, name)
        require(all(row['phase_global_counters']['real_moves'] == spec['additional_global_moves_each_arm']
                    and row['global_learner_exposures'] == spec['global_learner_exposures_each_arm'] for row in report['arms'].values()),
                'Registered matched budgets differ')
        report['status'] = 'passed'
    except BaseException as error:
        report['error'] = repr(error); raise
    finally:
        with a.output.open('xb') as stream: stream.write(canonical_json(report))
        print(canonical_json(report).decode(), flush=True)


if __name__ == '__main__':
    main()
