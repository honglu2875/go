#!/usr/bin/env python3
"""Audit fixed-budget history/spatial expert training, lineage, sampling and prediction gates."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

sys.dont_write_bytecode = True
SOURCE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(SOURCE / 'packages/gozero/src'))
from gozero.causal_artifacts import validate
from gozero.checkpoints import read as read_checkpoint, sha256
from gozero.model_artifacts import artifact
from gozero.snapshots import canonical_json, read_json, verify


def require(value, message):
    if not value:
        raise ValueError(message)


def audit_arm(root, protocol, mode):
    arm = protocol['arms'][mode]; descriptor_path = SOURCE / arm['descriptor']; descriptor = read_json(descriptor_path)
    trained = validate(root, descriptor); source = trained['snapshot']; config = trained['config']
    require(source.name == arm['snapshot_id'] and sha256(source / 'resolved_config.json') == arm['config_sha256'], 'Training source or config differs')
    result_path = artifact(root, descriptor['training_result_path']); directory = result_path.parents[2]
    launch = read_json(directory / 'launch.json'); pod = read_json(directory / 'result.json')
    require(pod['status'] == 'passed' and pod['snapshot_id'] == source.name and pod['start_unix_time'] > protocol['registered_unix']
            and pod['elapsed_seconds'] < protocol['maximum_training_seconds_per_arm']
            and launch['snapshot_id'] == source.name and launch['resume_attempt'] is None and launch['stop_after_turn'] is None,
            'Training failed, preceded registration, resumed unexpectedly or exceeded budget')
    attempts = [p for p in (root / 'runs').glob('pod-*/launch.json') if read_json(p)['snapshot_id'] == source.name]
    require(attempts == [directory / 'launch.json'] and protocol['maximum_training_attempts_per_arm'] == 1, 'Training attempt count differs')
    results = []; groups = []; samplers = {}; ranks = []; identities = []
    for host in range(4):
        base = directory / f'rank-{host}/artifacts'; result = read_json(base / 'result.json')
        checkpoint = base / 'checkpoints' / f'turn-{protocol["steps"]:09d}'
        group = read_json(checkpoint.with_suffix('.group.json')); rank = result['jax_rank']
        state, arrays, _ = read_checkpoint(checkpoint, expected_manifest_sha256=group['rank_manifests'][rank], array_prefix='p_')
        require(result['status'] == 'passed' and result['training_complete'] and result['kind'] == 'state_expert_distillation'
                and result['snapshot_id'] == source.name and result['turn'] == protocol['steps'] and result['world_size'] == 4
                and result['board_mode'] == 'exact' and result['value_objective'] == 'mse' and result['expert_architecture'] == mode
                and result['behavior_training_enabled'] is False and result['fixed_observer'] == trained['training_result']['fixed_observer'],
                'Rank did not complete its declared scientific condition')
        require(state['snapshot_id'] == source.name and state['config_sha256'] == arm['config_sha256'] == group['config_sha256']
                and state['turn'] == group['turn'] == protocol['steps'] and state['jax_rank'] == rank and state['world_size'] == group['world_size'] == 4
                and state['dataset_manifest_sha256'] == protocol['dataset_manifest_sha256'], 'Checkpoint scientific identity differs')
        require(sha256(checkpoint / 'arrays.npz') == group['replicated_arrays_sha256']
                and sha256(base / 'model_export.npz') == result['model_export_sha256'] == descriptor['model_export_sha256'],
                'Replicated checkpoint or export differs')
        require(set(arrays) == set(trained['arrays']) and all(a.dtype == trained['arrays'][k].dtype and a.shape == trained['arrays'][k].shape
                and a.tobytes() == trained['arrays'][k].tobytes() for k, a in arrays.items()), 'Rank parameters differ from committed model')
        for key in ('updates', 'expert_token_exposures'):
            require(state['counters'][key] == result['counters'][key], 'Checkpoint work counter differs')
        samplers[str(rank)] = hashlib.sha256(canonical_json(state['numpy_rng'])).hexdigest()
        results.append(result); ranks.append(rank); groups.append(sha256(checkpoint.with_suffix('.group.json')))
        identities.append(sha256(base / 'result.json'))
    require(set(ranks) == set(range(4)) and len(set(groups)) == 1, 'Rank coverage or checkpoint group differs')
    first = results[0]
    require(first['parameter_count'] == arm['parameter_count']
            and first['initial_parameter_elements_sha256'] == arm['initial_parameter_elements_sha256'], 'Registered parameter count differs')
    for result in results:
        for key in ('initial_parameter_elements_sha256', 'parameter_count', 'initial_validation', 'validation_history', 'test'):
            require(result[key] == first[key], 'Replicated model or validation metrics differ')
        for key in ('updates', 'expert_token_exposures'):
            require(result['counters'][key] == first['counters'][key], 'Replicated global work counts differ')
    require(first['validation_history'][-1]['turn'] == protocol['steps'], 'Fixed final checkpoint is missing')
    files = {str(p.relative_to(directory)): sha256(p) for p in sorted(directory.rglob('*')) if p.is_file() and p.name != '.operator.lock'}
    return {'mode': mode, 'attempt': directory.name, 'snapshot': source.name, 'descriptor_sha256': sha256(descriptor_path),
            'model_export_sha256': descriptor['model_export_sha256'], 'pod_result_sha256': sha256(directory / 'result.json'),
            'rank_results_sha256': identities, 'checkpoint_group_sha256': groups[0],
            'initial_parameter_elements_sha256': first['initial_parameter_elements_sha256'], 'parameter_count': first['parameter_count'],
            'global_exposures': {k: first['counters'][k] for k in ('updates', 'expert_token_exposures')},
            'sampler_state_sha256_by_jax_rank': samplers, 'initial_validation': first['initial_validation'],
            'final_validation': first['validation_history'][-1]['validation'], 'test': first['test'],
            'started_unix': pod['start_unix_time'], 'finished_unix': pod['end_unix_time'],
            'attempt_seconds': pod['elapsed_seconds'], 'recorded_attempt_chip_hours': pod['reserved_chip_hours'],
            'maximum_learner_seconds': max(r['counters']['learner_seconds'] for r in results),
            'maximum_segment_seconds': max(r['elapsed_segment_seconds'] for r in results),
            'fixed_observer': first['fixed_observer'], 'raw_files_sha256': files}, config


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workspace-root', type=Path, required=True); p.add_argument('--protocol', type=Path, required=True)
    p.add_argument('--expected-protocol-sha256', required=True); p.add_argument('--output', type=Path, required=True)
    a = p.parse_args(); verify(SOURCE); root = a.workspace_root.resolve()
    require(not a.output.exists() and sha256(a.protocol) == a.expected_protocol_sha256, 'Output exists or protocol changed')
    protocol = read_json(a.protocol); require(protocol['kind'] == 'state_expert_distillation_pilot', 'Wrong training protocol')
    result = {'schema_version': 1, 'kind': 'state_expert_training_audit', 'status': 'failed', 'analysis_snapshot': SOURCE.name,
              'protocol_sha256': a.expected_protocol_sha256, 'claims_rl_sample_efficiency': False,
              'claims_strength_improvement': False, 'claims_mfu': False, 'production_promotion': False,
              'limitations': protocol['limitations']}
    try:
        for name, field in (('state_expert_distillation/cpu_result.json', 'cpu_qualification_sha256'),
                            ('state_expert_distillation/tpu_qualification_result.json', 'tpu_qualification_sha256'),
                            ('state_expert_distillation/gtp_result.json', 'gtp_qualification_sha256')):
            prerequisite = root / 'research/studies' / name
            require(sha256(prerequisite) == protocol[field] and read_json(prerequisite)['status'] == 'passed',
                    'Registered prerequisite changed or did not pass: ' + name)
        rows = {}; configs = {}
        for mode in ('history', 'state'):
            rows[mode], configs[mode] = audit_arm(root, protocol, mode)
        require(protocol['training_order'] == ['history', 'state'] and rows['history']['finished_unix'] < rows['state']['started_unix'],
                'Registered sequential training order differs')
        expected = copy.deepcopy(configs['history']); expected['model']['architecture'] = 'state'
        require(expected == configs['state'] and configs['history']['seed'] == protocol['seed']
                and configs['history']['steps'] == protocol['steps'], 'Training changed more than the registered objective')
        for name in ('model.py', 'train.py', 'config.py', 'data.py', 'history_model.py', 'spatial_model.py', 'state_features.py'):
            path = 'research/recipes/state_expert_distillation/' + name
            require(sha256(root / '.gozero/snapshots' / rows['history']['snapshot'] / path)
                    == sha256(root / '.gozero/snapshots' / rows['state']['snapshot'] / path), 'Scientific implementation differs between arms')
        for key in ('global_exposures', 'sampler_state_sha256_by_jax_rank'):
            require(rows['history'][key] == rows['state'][key], 'Matched example exposures or sampler state differs')
        require(rows['history']['fixed_observer'] == rows['state']['fixed_observer']
                and rows['history']['fixed_observer']['updated_in_this_run'] is False, 'Fixed observer differs')
        x = rows['history']['final_validation']; y = rows['state']['final_validation']; gate = protocol['primary_prediction_criterion']
        metrics = {'value_validation_mse_ratio': y['value_loss'] / x['value_loss'], 'expert_validation_kl_ratio': y['expert_kl'] / x['expert_kl'],
                   'expert_validation_illegal_probability_increase': y['illegal_probability'] - x['illegal_probability']}
        passed = (metrics['value_validation_mse_ratio'] <= gate['value_validation_mse_ratio_at_most']
                  and metrics['expert_validation_kl_ratio'] <= gate['expert_validation_kl_ratio_at_most']
                  and metrics['expert_validation_illegal_probability_increase'] <= gate['expert_validation_illegal_probability_increase_at_most'])
        result.update(status='passed', arms=rows, comparison=metrics, registered_prediction_criterion_met=passed,
                      external_benchmark_required=True, recorded_pilot_attempt_chip_hours=sum(r['recorded_attempt_chip_hours'] for r in rows.values()),
                      global_exposure_note='Exposure counts are replicated global totals, not totals to sum across ranks.')
    except BaseException as error:
        result['error'] = repr(error); raise
    finally:
        a.output.parent.mkdir(parents=True, exist_ok=True)
        with a.output.open('xb') as stream:
            stream.write(canonical_json(result))
        a.output.chmod(0o444); verify(SOURCE); require(sha256(a.protocol) == a.expected_protocol_sha256, 'Registration changed during audit')
        print(json.dumps({k: v for k, v in result.items() if k != 'arms'}))


if __name__ == '__main__':
    main()
