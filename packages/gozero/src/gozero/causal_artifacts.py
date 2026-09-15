"""Validate a causal student's trained weights and committed checkpoint lineage."""
from pathlib import Path

import numpy as np

from .checkpoints import read as checkpoint_read, sha256
from .model_artifacts import artifact
from .snapshots import read_json, verify


def validate(root, descriptor):
    root = Path(root).resolve()
    kinds={'causal_history_policy':'causal_expert_behavior_distillation',
           'board_causal_history_policy':'board_causal_expert_behavior_distillation',
           'state_expert_policy':'state_expert_distillation'}
    if descriptor['schema_version'] != 1 or descriptor['kind'] not in kinds:
        raise ValueError('Unknown causal model descriptor')
    source = artifact(root, '.gozero/snapshots/' + descriptor['training_snapshot'])
    manifest = verify(source); config = read_json(source / 'resolved_config.json')
    weights = artifact(root, descriptor['model_export_path'])
    result_path = artifact(root, descriptor['training_result_path'])
    if sha256(weights) != descriptor['model_export_sha256'] or sha256(result_path) != descriptor['training_result_sha256']:
        raise ValueError('Causal export or completed training result differs')
    result = read_json(result_path)
    if (result['kind'] != kinds[descriptor['kind']] or result['status'] != 'passed'
            or not result['training_complete'] or result['snapshot_id'] != source.name
            or result['turn'] != config['steps'] or result['counters']['updates'] != descriptor['network_version']
            or result['model_export_sha256'] != descriptor['model_export_sha256']):
        raise ValueError('Causal model did not complete its declared training')
    if descriptor['kind']=='board_causal_history_policy' and result['board_mode']!=config['model']['board_mode']:
        raise ValueError('Trained board input condition differs')
    if 'value_objective' in config['model'] and result.get('value_objective') != config['model']['value_objective']:
        raise ValueError('Trained value objective differs')
    if descriptor['kind'] == 'state_expert_policy':
        observer_path = artifact(source, config['observer']['descriptor'])
        if (result.get('expert_architecture') != config['model']['architecture']
                or config['model']['architecture'] not in ('history', 'state')
                or result.get('board_mode') != 'exact' or config['model']['board_mode'] != 'exact'
                or result.get('behavior_training_enabled') is not False
                or sha256(observer_path) != config['observer']['sha256']):
            raise ValueError('Expert architecture, input or fixed observer contract differs')
        observer = read_json(observer_path)
        if observer.get('kind') != 'board_causal_history_policy':
            raise ValueError('Unsupported fixed observer lineage')
        observer_identity = validate(root, observer)
        expected_observer = {'descriptor_sha256': config['observer']['sha256'],
                             'training_snapshot': observer['training_snapshot'],
                             'weights_sha256': observer['model_export_sha256'],
                             'model_code_sha256': observer_identity['model_code_sha256'],
                             'updated_in_this_run': False, 'included_in_trainable_parameters': False}
        if result.get('fixed_observer') != expected_observer:
            raise ValueError('Fixed observer identity differs from training')
    reference = descriptor['checkpoint']; checkpoint = artifact(root, reference['path'])
    group_path = checkpoint.with_suffix('.group.json')
    if sha256(group_path) != reference['group_sha256']:
        raise ValueError('Causal checkpoint group differs')
    group = read_json(group_path)
    state, arrays, _ = checkpoint_read(checkpoint, expected_manifest_sha256=reference['manifest_sha256'], array_prefix='p_')
    if (state['snapshot_id'] != source.name or state['config_sha256'] != sha256(source / 'resolved_config.json')
            or group['snapshot_id'] != source.name or group['config_sha256'] != state['config_sha256']
            or state['turn'] != config['steps'] or group['turn'] != state['turn']
            or state['world_size'] != group['world_size'] or state['jax_rank'] != result['jax_rank']
            or group['rank_manifests'][state['jax_rank']] != reference['manifest_sha256']
            or state['counters']['updates'] != descriptor['network_version']
            or state['dataset_manifest_sha256'] != config['dataset']['manifest_sha256']):
        raise ValueError('Causal checkpoint scientific identity differs')
    schema = read_json(result_path.parent / 'model_schema.json')
    if schema != state['model_schema'] or set(arrays) != {f'p_{i:04d}' for i in range(len(schema))}:
        raise ValueError('Causal parameter schema differs')
    with np.load(weights, allow_pickle=False) as saved:
        if set(saved.files) != set(arrays) or any(not np.array_equal(saved[n], arrays[n]) for n in arrays):
            raise ValueError('Causal export differs from checkpoint parameters')
    for index, item in enumerate(schema):
        value = arrays[f'p_{index:04d}']
        if list(value.shape) != item['shape'] or str(value.dtype) != item['dtype'] or not np.isfinite(value).all():
            raise ValueError('Invalid causal parameter array')
    return {'snapshot': source, 'manifest': manifest, 'config': config, 'weights': weights,
            'arrays': arrays, 'model_schema': schema, 'training_result': result,
            'model_code_sha256': sha256(source / manifest['recipe'] / 'model.py')}
