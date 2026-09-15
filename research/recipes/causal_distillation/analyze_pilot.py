#!/usr/bin/env python3
"""Audit fixed-budget distillation without treating it as a Go-strength result."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.dont_write_bytecode = True
SOURCE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(SOURCE / 'packages/gozero/src'))
from gozero.checkpoints import read as checkpoint_read, sha256
from gozero.snapshots import canonical_json, read_json, verify

REGISTRATION = '7f458e3d3c889d1e47da0408cb8bf9c5b6c67f8a7505ccd9c6cba95995a73a94'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace-root', type=Path, required=True)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--attempt', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); verify(SOURCE)
    root = args.workspace_root.resolve(); protocol = read_json(args.protocol)
    if sha256(args.protocol) != REGISTRATION:
        raise ValueError('Pilot registration changed')
    report = {'schema_version': 1, 'kind': 'causal_distillation_pilot_analysis', 'status': 'failed',
              'protocol_sha256': REGISTRATION, 'analysis_snapshot': SOURCE.name,
              'claims_rl_sample_efficiency': False, 'claims_go_strength': False, 'production_promotion': False}
    try:
        directory = root / 'runs' / args.attempt; pod = read_json(directory / 'result.json')
        source = root / '.gozero/snapshots' / protocol['source_snapshot']; manifest = verify(source)
        config = read_json(source / 'resolved_config.json'); budget = protocol['budget']
        if (pod['status'] != 'passed' or pod['snapshot_id'] != source.name or pod['start_unix_time'] < protocol['registered_unix']
                or pod['timeout_seconds'] != budget['timeout_seconds'] or config['steps'] != budget['updates']
                or config['seed'] != budget['seed'] or config['learner']['games_per_role'] != budget['games_per_role_per_host_per_update']):
            raise ValueError('Training identity, budget or registration order differs')
        attempts = [p for p in (root / 'runs').glob('pod-*/launch.json') if read_json(p)['snapshot_id'] == source.name]
        if len(attempts) != budget['maximum_new_attempts'] or attempts[0].parent.name != args.attempt:
            raise ValueError('Attempt selection or retry budget differs')
        qualification = root / 'research/studies/causal_distillation/tpu_qualification_result.json'
        if sha256(qualification) != protocol['qualification_result_sha256'] or read_json(qualification)['status'] != 'passed':
            raise ValueError('TPU recovery prerequisite differs')
        dataset = Path(config['dataset']['path']); data_manifest = read_json(dataset / 'manifest.json')
        if sha256(dataset / 'manifest.json') != protocol['dataset_manifest_sha256']:
            raise ValueError('Dataset content identity differs')
        teacher_moves = teacher_cost = 0
        for item in data_manifest['spec']['teacher_cost']:
            path = root / 'runs' / item['attempt'] / 'result.json'
            teacher = read_json(path)
            if sha256(path) != item['result_sha256'] or teacher['reserved_chip_hours'] != item['recorded_attempt_chip_hours']:
                raise ValueError('Reused teacher cost differs')
            teacher_moves += item['global_real_moves']; teacher_cost += item['recorded_attempt_chip_hours']
        results, states, hashes, groups = [], [], [], []
        for host in range(4):
            artifacts = directory / f'rank-{host}/artifacts'; result = read_json(artifacts / 'result.json')
            checkpoint = artifacts / 'checkpoints' / f'turn-{config["steps"]:09d}'
            group_path = checkpoint.with_suffix('.group.json'); group = read_json(group_path)
            state, arrays, _ = checkpoint_read(checkpoint, expected_manifest_sha256=group['rank_manifests'][result['jax_rank']])
            if (result['status'] != 'passed' or not result['training_complete'] or result['snapshot_id'] != source.name
                    or result['turn'] != config['steps'] or result['world_size'] != 4 or state['jax_rank'] != result['jax_rank']
                    or state['snapshot_id'] != source.name or state['config_sha256'] != sha256(source / 'resolved_config.json')
                    or state['dataset_manifest_sha256'] != protocol['dataset_manifest_sha256']
                    or result['dataset_manifest_sha256'] != protocol['dataset_manifest_sha256']
                    or result['teacher_cost'] != data_manifest['spec']['teacher_cost']
                    or group['replicated_arrays_sha256'] != sha256(checkpoint / 'arrays.npz')
                    or group['config_sha256'] != state['config_sha256'] or group['snapshot_id'] != source.name
                    or group['turn'] != result['turn'] or group['world_size'] != 4):
                raise ValueError('Rank checkpoint scientific identity differs')
            with np.load(artifacts / 'model_export.npz', allow_pickle=False) as export:
                names = {n for n in arrays if n.startswith('p_')}
                if set(export.files) != names or any(not np.array_equal(export[n], arrays[n]) for n in names):
                    raise ValueError('Export differs from final checkpoint parameters')
            if sha256(artifacts / 'model_export.npz') != result['model_export_sha256']:
                raise ValueError('Export integrity differs')
            for key in ('updates', 'expert_token_exposures', 'behavior_token_exposures'):
                if result['counters'][key] != state['counters'][key]:
                    raise ValueError('Final token exposure accounting differs')
            results.append(result); states.append(state); hashes.append(group['replicated_arrays_sha256']); groups.append(sha256(group_path))
        if ({r['jax_rank'] for r in results} != set(range(4)) or len(set(hashes)) != 1 or len(set(groups)) != 1
                or len({r['model_export_sha256'] for r in results}) != 1):
            raise ValueError('Replicated parameters/optimizer or rank group differ')
        first = results[0]
        expected_turns = list(range(config['eval_every'], config['steps'] + 1, config['eval_every']))
        if [v['turn'] for v in first['validation_history']] != expected_turns:
            raise ValueError('Validation schedule differs')
        for result in results[1:]:
            for key in ('initial_validation', 'validation_history', 'test', 'last_metrics'):
                if result[key] != first[key]:
                    raise ValueError('Global prediction metrics differ across ranks')
            for key in ('updates', 'expert_token_exposures', 'behavior_token_exposures'):
                if result['counters'][key] != first['counters'][key]:
                    raise ValueError('Replicated global exposure counters differ')
        initial, final = first['initial_validation'], first['validation_history'][-1]['validation']
        expert_ratio = final['expert_kl'] / initial['expert_kl']; behavior_ratio = final['behavior_loss'] / initial['behavior_loss']
        gates = {'expert_kl': expert_ratio <= protocol['primary']['expert_validation_kl_fraction_of_initial_maximum'],
                 'behavior_nll': behavior_ratio <= protocol['primary']['behavior_validation_nll_fraction_of_initial_maximum']}
        report.update(status='passed', attempt=args.attempt, source_snapshot=source.name,
                      pod_result_sha256=sha256(directory / 'result.json'),
                      rank_result_sha256=[sha256(directory / f'rank-{h}/artifacts/result.json') for h in range(4)],
                      model_export_sha256=first['model_export_sha256'], replicated_arrays_sha256=hashes[0],
                      parameter_count=first['parameter_count'], initial_validation=initial, validation_history=first['validation_history'],
                      test=first['test'], expert_validation_kl_fraction=expert_ratio, behavior_validation_nll_fraction=behavior_ratio,
                      feasibility_criteria=gates, registered_feasibility_criterion_met=all(gates.values()),
                      global_token_exposures={k: first['counters'][k] for k in ('expert_token_exposures', 'behavior_token_exposures')},
                      updates=config['steps'], padded_input_token_exposures=budget['maximum_padded_input_tokens'],
                      new_attempt_chip_hours=pod['reserved_chip_hours'], new_attempt_seconds=pod['elapsed_seconds'],
                      max_training_seconds=max(r['elapsed_segment_seconds'] for r in results),
                      max_learner_seconds=max(r['counters']['learner_seconds'] for r in results),
                      host_average_cpu_cores=[r['process_cpu_segment_seconds'] / r['elapsed_segment_seconds'] for r in results],
                      reused_teacher_real_moves=teacher_moves, reused_teacher_attempt_chip_hours=teacher_cost,
                      dataset_manifest_sha256=protocol['dataset_manifest_sha256'],
                      limitations=protocol['limitations'])
        verify(SOURCE)
    except BaseException as error:
        report['error'] = repr(error)
        raise
    finally:
        with args.output.open('xb') as stream:
            stream.write(canonical_json(report))
        args.output.chmod(0o444)
        print(json.dumps({k: v for k, v in report.items() if k in ('status', 'error', 'registered_feasibility_criterion_met',
                         'expert_validation_kl_fraction', 'behavior_validation_nll_fraction', 'test', 'new_attempt_chip_hours')}), flush=True)


if __name__ == '__main__':
    main()
