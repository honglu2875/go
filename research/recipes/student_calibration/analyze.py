#!/usr/bin/env python3
"""Independently reconstruct phase and calibration summaries from raw logits."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

sys.dont_write_bytecode = True
SOURCE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(SOURCE / 'packages/gozero/src'))
from gozero.board_sequence_batches import Dataset
from gozero.causal_artifacts import validate as validate_candidate
from gozero.checkpoints import sha256
from gozero.model_artifacts import artifact
from gozero.snapshots import canonical_json, read_json, verify
from diagnose import require, selection, validate


def probabilities(logits):
    x = logits.astype(np.float64); shifted = x - x.max(axis=-1, keepdims=True)
    logp = shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))
    return np.exp(logp), logp


def expert_metrics(logits, value, policy, legal, outcome, c):
    if len(value) == 0:
        return {'positions': 0}
    probability, logp = probabilities(logits); policy = policy.astype(np.float64)
    value = value.astype(np.float64); outcome = outcome.astype(np.float64)
    win = (value + 1.) / 2; target = (outcome + 1.) / 2
    entropy = -(policy * np.log(np.maximum(policy, 1e-30))).sum(axis=-1)
    cross_entropy = -(policy * logp).sum(axis=-1)
    bins = np.clip(np.searchsorted(c['calibration_edges'], win, side='right') - 1, 0, 9)
    calibration = []
    for index, (low, high) in enumerate(zip(c['calibration_edges'], c['calibration_edges'][1:])):
        mask = bins == index; n = int(mask.sum())
        calibration.append({'range': [low, high], 'positions': n,
                            'mean_predicted_win_probability': float(win[mask].mean()) if n else None,
                            'observed_win_fraction': float(target[mask].mean()) if n else None})
    saturated = np.abs(value) >= c['saturation_absolute_value']
    legal_sum = (probability * legal).sum(axis=-1)
    return {'positions': len(value), 'play_loss': float(cross_entropy.mean()), 'target_entropy': float(entropy.mean()),
            'expert_kl': float((cross_entropy - entropy).mean()),
            'play_top1': float((logits.argmax(axis=-1) == policy.argmax(axis=-1)).mean()),
            'illegal_probability': float((probability * ~legal).sum(axis=-1).mean()),
            'raw_top1_illegal_fraction': float((~np.take_along_axis(legal, logits.argmax(axis=-1)[:, None], axis=-1)[:, 0]).mean()),
            'raw_pass_probability': float(probability[:, -1].mean()),
            'legal_normalized_pass_probability': float((probability[:, -1] / legal_sum).mean()),
            'target_pass_probability': float(policy[:, -1].mean()),
            'raw_pass_top1_fraction': float((logits.argmax(axis=-1) == logits.shape[-1] - 1).mean()),
            'target_pass_top1_fraction': float((policy.argmax(axis=-1) == policy.shape[-1] - 1).mean()),
            'value_loss': float(((value - outcome)**2).mean()), 'win_brier_score': float(((win - target)**2).mean()),
            'zero_value_mse': float((outcome**2).mean()), 'observed_win_fraction': float(target.mean()),
            'mean_predicted_win_probability': float(win.mean()),
            'value_expected_calibration_error': float(sum(r['positions'] * abs(r['mean_predicted_win_probability'] - r['observed_win_fraction'])
                                                         for r in calibration if r['positions']) / len(value)),
            'saturated_positions': int(saturated.sum()),
            'saturated_sign_error_fraction': float((value[saturated] * outcome[saturated] < 0).mean()) if saturated.any() else None,
            'calibration': calibration}


def behavior_metrics(logits, observed):
    if not len(observed):
        return {'positions': 0}
    probability, logp = probabilities(logits)
    return {'positions': len(observed), 'behavior_loss': float(-logp[np.arange(len(observed)), observed].mean()),
            'behavior_top1': float((logits.argmax(axis=-1) == observed).mean()),
            'predicted_pass_probability': float(probability[:, -1].mean()),
            'observed_pass_fraction': float((observed == logits.shape[-1] - 1).mean())}


def aggregate(ids, logits, values, data, c):
    policies = np.zeros_like(logits); legal = np.ones_like(logits, bool)
    outcomes = np.zeros(len(ids), np.float32); observed = np.zeros(len(ids), np.int32)
    after_pass = np.zeros(len(ids), bool)
    for role in (0, 1):
        name = 'expert' if role == 0 else 'behavior'
        for shard in np.unique(ids[ids[:, 0] == role, 1]):
            mask = (ids[:, 0] == role) & (ids[:, 1] == shard); positions = np.flatnonzero(mask)
            a = data.shards[int(shard)]; rows = a[name + '_offsets'][ids[mask, 2]] + ids[mask, 3]
            observed[mask] = a[name + '_actions'][rows]
            prior = ids[mask, 3] > 0
            after_pass[positions[prior]] = a[name + '_actions'][rows[prior] - 1] == 81
            if role == 0:
                policies[mask] = a['expert_policies'][rows]; legal[mask] = a['expert_legal'][rows]; outcomes[mask] = a['expert_values'][rows]
    require(np.isin(outcomes[ids[:, 0] == 0], [-1., 0., 1.]).all(), 'Terminal outcome labels differ')
    def subset(mask):
        expert = mask & (ids[:, 0] == 0); behavior = mask & (ids[:, 0] == 1)
        return {'expert': expert_metrics(logits[expert], values[expert], policies[expert], legal[expert], outcomes[expert], c),
                'behavior': behavior_metrics(logits[behavior], observed[behavior])}
    result = {'all': subset(np.ones(len(ids), bool)), 'phase': [], 'after_pass': subset(after_pass),
              'not_after_pass': subset(~after_pass), 'expert_target_pass_bins': [], 'teacher_shards': []}
    for low, high in zip(c['phase_edges'], c['phase_edges'][1:]):
        result['phase'].append({'ply_range_half_open': [low, high], **subset((ids[:, 3] >= low) & (ids[:, 3] < high))})
    for low, high in zip(c['pass_target_edges'], c['pass_target_edges'][1:]):
        selected = (ids[:, 0] == 0) & (policies[:, -1] >= low) & (policies[:, -1] < high)
        result['expert_target_pass_bins'].append({'target_pass_range_half_open': [low, high], **subset(selected)['expert']})
    for shard in range(len(data.shards)):
        selected = ids[:, 1] == shard
        if selected.any():
            teacher = data.parent_manifest['spec']['shards'][shard]['training_snapshot']
            result['teacher_shards'].append({'shard': shard, 'teacher_training_snapshot': teacher, **subset(selected)})
    return result


def audit(root, source, c, directory, protocol):
    require(protocol['maximum_attempts'] == 1, 'Diagnostic attempt budget differs')
    if c['platform'] == 'cpu':
        require(protocol['kind'] == 'student_calibration_cpu_qualification'
                and directory == root / protocol['output'], 'CPU qualification identity differs')
    else:
        require(protocol['kind'] == 'student_calibration_tpu_validation'
                and protocol['expected_hosts'] == c['expected_processes'] == 4
                and protocol['expected_devices'] == c['expected_devices'] == 16, 'TPU diagnostic identity differs')
        for name, expected in protocol['cpu_qualified_code_sha256'].items():
            require(sha256(source / name) == expected, 'Qualified execution code changed')
        prerequisites = [
            ('research/studies/student_calibration/cpu_result.json', protocol['cpu_audit_sha256']),
            ('runs/qualification/student-calibration-metrics-c05cad6a/result.json', protocol['metric_tests_sha256']),
            ('runs/qualification/student-calibration-empty-bundle-stage/result.json', protocol['control_bundle_stage_sha256'])]
        for name, expected in prerequisites:
            path = artifact(root, name)
            require(sha256(path) == expected and read_json(path)['status'] == 'passed', 'Qualification or checkpoint staging prerequisite differs')
    data = Dataset(c['dataset']['path'], c['dataset']['manifest_sha256'], rank=0, world=1, board_mode='exact')
    require([r['id'] for r in data.manifest['shards']] == list(range(len(data.shards))), 'Expected canonical global shard indexing')
    arrays = {arm: {k: [] for k in ('ids', 'logits', 'values')} for arm in ('empty', 'exact')}
    reports = []; ranks = []; files = {}; registered = protocol['registered_unix']
    if c['platform'] == 'tpu':
        launch = read_json(directory / 'launch.json'); pod = read_json(directory / 'result.json')
        require(pod['status'] == 'passed' and launch['snapshot_id'] == pod['snapshot_id'] == source.name
                and registered < launch['start_unix_time'] <= pod['start_unix_time']
                and launch['timeout_seconds'] == protocol['maximum_seconds']
                and launch['reserved_chips'] == 16 and launch['resume_attempt'] is None,
                'Pod execution identity or budget differs')
        attempts = [p for p in (root / 'runs').glob('pod-*/launch.json') if read_json(p)['snapshot_id'] == source.name]
        require(attempts == [directory / 'launch.json'] and protocol['maximum_attempts'] == 1, 'Attempt count differs')
    for host in range(c['expected_processes']):
        base = directory if c['platform'] == 'cpu' else directory / f'rank-{host}/artifacts'
        report = read_json(base / 'result.json'); rank = report['jax_rank']; ranks.append(rank)
        require(report['status'] == 'passed' and report['kind'] == 'fixed_student_calibration' and 'error' not in report
                and report['snapshot_id'] == source.name and report['config_sha256'] == sha256(source / 'resolved_config.json')
                and read_json(base / 'resolved_config.json') == c and report['world_size'] == c['expected_processes']
                and report['host_rank'] == host and report['training_updates'] == report['new_games'] == 0
                and registered < report['started_unix'] <= report['finished_unix']
                and report['finished_unix'] - report['started_unix'] < protocol['maximum_seconds'],
                'Rank identity, scope or deadline differs')
        local_shards = [i for i in range(len(data.shards)) if i % c['expected_processes'] == rank]
        entries = []
        for role in ('expert', 'behavior'):
            chosen = [(role, shard, episode) for shard, episode in data.indices[role, c['split']] if shard in local_shards]
            if c['maximum_games_per_role'] is not None:
                chosen = chosen[:c['maximum_games_per_role']]
            entries.extend(chosen)
        require(report['local_shard_ids'] == local_shards and report['selected_entries'] == [list(e) for e in entries]
                and report['selected_entries_sha256'] == hashlib.sha256(canonical_json(entries)).hexdigest(), 'Position selection differs')
        expected = []
        for role, shard, episode in entries:
            offsets = data.shards[shard][role + '_offsets']; n = int(offsets[episode + 1] - offsets[episode])
            expected.extend((int(role == 'behavior'), shard, episode, ply) for ply in range(n))
        expected = np.asarray(expected, np.int32)
        for arm in ('empty', 'exact'):
            recorded = report['arms'][arm]; descriptor_path = source / c['candidates'][arm]['path']
            descriptor = read_json(descriptor_path); trained = validate_candidate(root, descriptor)
            digest = hashlib.sha256(b''.join(trained['arrays'][f'p_{i:04d}'].tobytes() for i in range(len(trained['arrays'])))).hexdigest()
            require(recorded['descriptor'] == descriptor and recorded['descriptor_sha256'] == c['candidates'][arm]['sha256'] == sha256(descriptor_path)
                    and trained['model_code_sha256'] == recorded['model_code_sha256'] == sha256(source / verify(source)['recipe'] / 'model.py')
                    and recorded['parameter_elements_sha256'] == digest and recorded['execution_dtype'] == trained['config']['model']['dtype'] == 'bfloat16'
                    and sha256(base / (arm + '.hlo.txt')) == recorded['hlo_sha256']
                    and recorded['archive'] == arm + '.npz' and sha256(base / recorded['archive']) == recorded['archive_sha256'],
                    'Loaded model, parameters, HLO or raw archive differs')
            with np.load(base / recorded['archive'], allow_pickle=False) as saved:
                require(set(saved.files) == {'ids', 'logits', 'values'}, 'Unexpected raw prediction schema')
                rows = {k: saved[k] for k in saved.files}
            require(np.array_equal(rows['ids'], expected) and rows['ids'].dtype == np.int32
                    and rows['logits'].shape == (len(expected), 82) and rows['logits'].dtype == np.float32
                    and rows['values'].shape == (len(expected),) and rows['values'].dtype == np.float32
                    and all(np.isfinite(v).all() for v in rows.values()) and np.all(np.abs(rows['values']) <= 1.)
                    and recorded['positions'] == len(expected) and recorded['expert_positions'] == int(np.sum(expected[:, 0] == 0))
                    and recorded['behavior_positions'] == int(np.sum(expected[:, 0] == 1))
                    and recorded['uncompressed_bytes'] == sum(v.nbytes for v in rows.values()) <= c['maximum_output_bytes_per_arm_per_rank'],
                    'Position coverage, raw values or output budget differs')
            for key in arrays[arm]:
                arrays[arm][key].append(rows[key])
        reports.append({'host_rank': host, 'jax_rank': rank, 'positions': len(expected), 'result_sha256': sha256(base / 'result.json'),
                        'arms': {arm: {k: report['arms'][arm][k] for k in ('archive_sha256', 'compile_seconds', 'forward_seconds', 'forward_process_cpu_seconds')}
                                 for arm in ('empty', 'exact')}})
    require(set(ranks) == set(range(c['expected_processes'])), 'Rank coverage differs')
    result = {}; raw_ids = None
    for arm in ('empty', 'exact'):
        combined = {k: np.concatenate(v) for k, v in arrays[arm].items()}
        if raw_ids is not None:
            require(np.array_equal(raw_ids, combined['ids']), 'Paired model positions differ')
        raw_ids = combined['ids']
        require(len(np.unique(raw_ids, axis=0)) == len(raw_ids), 'Repeated validation position')
        result[arm] = aggregate(combined['ids'], combined['logits'], combined['values'], data, c)
    for p in directory.rglob('*'):
        if p.is_file() and p.name != '.operator.lock':
            files[str(p.relative_to(directory))] = sha256(p)
    report = {'arms': result, 'ranks': reports, 'raw_files_sha256': dict(sorted(files.items())),
              'positions_per_arm': len(raw_ids), 'raw_directory': str(directory.relative_to(root))}
    if c['platform'] == 'tpu':
        report['pod_result_sha256'] = sha256(directory / 'result.json')
        report['recorded_attempt_chip_hours'] = read_json(directory / 'result.json')['reserved_chip_hours']
        original_path = artifact(root, protocol['reference_training_audit'])
        require(sha256(original_path) == protocol['reference_training_audit_sha256'], 'Original training audit changed')
        original = read_json(original_path); errors = {}
        for arm in ('empty', 'exact'):
            expected = original['arms'][arm]['final_validation']; actual = result[arm]['all']
            require(actual['expert']['positions'] == expected['expert_tokens'] == expected['value_tokens'] == protocol['expected_expert_positions_per_arm']
                    and actual['behavior']['positions'] == expected['behavior_tokens'] == protocol['expected_behavior_positions_per_arm'],
                    'Whole validation population differs')
            errors[arm] = {}
            for key in ('play_loss', 'target_entropy', 'expert_kl', 'play_top1', 'illegal_probability', 'value_loss', 'behavior_loss', 'behavior_top1'):
                observed = actual['behavior' if key.startswith('behavior') else 'expert'][key]
                errors[arm][key] = observed - expected[key]
                require(abs(observed - expected[key]) <= c['aggregate_reference_tolerance'], 'Held-out aggregate reproduction failed: ' + arm + '/' + key)
        report['original_validation_metric_differences'] = errors
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workspace-root', type=Path, required=True); p.add_argument('--protocol', type=Path, required=True)
    p.add_argument('--expected-protocol-sha256', required=True); p.add_argument('--directory', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True); a = p.parse_args(); verify(SOURCE)
    require(not a.output.exists() and sha256(a.protocol) == a.expected_protocol_sha256, 'Audit output exists or protocol changed')
    protocol = read_json(a.protocol); root = a.workspace_root.resolve()
    source = artifact(root, '.gozero/snapshots/' + protocol['snapshot']); verify(source); c = validate(read_json(source / 'resolved_config.json'))
    require(sha256(source / 'resolved_config.json') == protocol['config_sha256'], 'Registered configuration differs')
    report = {'schema_version': 1, 'kind': 'student_calibration_audit', 'status': 'failed', 'analysis_snapshot': SOURCE.name,
              'execution_snapshot': source.name, 'protocol_sha256': a.expected_protocol_sha256,
              'production_promotion': False, 'claims_rl_sample_efficiency': False, 'claims_strength': False,
              'limitations': protocol['limitations']}
    try:
        report.update(audit(root, source, c, a.directory.resolve(), protocol), status='passed')
        verify(source)
    except BaseException as error:
        report['error'] = repr(error); raise
    finally:
        a.output.parent.mkdir(parents=True, exist_ok=True)
        with a.output.open('xb') as stream:
            stream.write(canonical_json(report))
        a.output.chmod(0o444); verify(SOURCE)
        print(json.dumps({k: v for k, v in report.items() if k not in ('arms', 'ranks', 'raw_files_sha256')}))


if __name__ == '__main__':
    main()
