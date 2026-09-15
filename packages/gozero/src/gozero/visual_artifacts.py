"""Verify visual model parameters directly in committed shared checkpoints."""
import importlib
from pathlib import Path
import sys
import types
import numpy as np
from . import checkpoints
from .model_artifacts import artifact
from .snapshots import read_json, verify


def load_module(snapshot, recipe, module):
    """A private namespace retains the recipe's relative module imports."""
    name = '_visual_recipe_' + snapshot.name
    if name not in sys.modules:
        package = types.ModuleType(name); package.__path__ = [str(snapshot / recipe)]
        sys.modules[name] = package
    return importlib.import_module(name + '.' + module)


def validate(root, descriptor, *, allow_partial=False):
    root = Path(root).resolve()
    if (set(descriptor) != {'schema_version', 'kind', 'training_snapshot', 'training_result',
            'checkpoint', 'network_version', 'training_complete'} or descriptor['schema_version'] != 1
            or descriptor['kind'] != 'visual_causal_checkpoint' or type(descriptor['training_complete']) is not bool
            or (not descriptor['training_complete'] and not allow_partial)):
        raise ValueError('Unsupported or incomplete visual candidate')
    source = artifact(root, '.gozero/snapshots/' + descriptor['training_snapshot'])
    manifest = verify(source); config = read_json(source / 'resolved_config.json')
    result_path = artifact(root, descriptor['training_result']['path'])
    if checkpoints.sha256(result_path) != descriptor['training_result']['sha256']:
        raise ValueError('Training result identity differs')
    result = read_json(result_path); ref = descriptor['checkpoint']; checkpoint = artifact(root, ref['path'])
    group_path = checkpoint.with_suffix('.group.json')
    if checkpoints.sha256(group_path) != ref['group_sha256']:
        raise ValueError('Visual checkpoint group identity differs')
    group = read_json(group_path)
    if (result['kind'] != 'visual_causal_distillation' or result['status'] != 'passed'
            or result['training_complete'] != descriptor['training_complete']
            or result['snapshot_id'] != source.name or result['turn'] != descriptor['network_version']
            or result['host_rank'] != 0 or config['kind'] != 'visual_causal_distillation'
            or result['counters']['updates'] != result['turn']
            or (result['training_complete'] and result['turn'] != config['steps'])
            or group['kind'] != 'visual_replicated_checkpoint_group'
            or group['snapshot_id'] != source.name or group['turn'] != result['turn']
            or group['config_sha256'] != checkpoints.sha256(source / 'resolved_config.json')
            or result['config_sha256'] != group['config_sha256']
            or artifact(root, group['owner_checkpoint_path']) != checkpoint
            or group['host_manifests']['0'] != ref['manifest_sha256']
            or result['latest_checkpoint']['group_sha256'] != ref['group_sha256']):
        raise ValueError('Visual checkpoint and completed training lineage differ')
    state, arrays, _ = checkpoints.read(checkpoint, expected_manifest_sha256=ref['manifest_sha256'], array_prefix='p_')
    if (state['snapshot_id'] != source.name or state['turn'] != result['turn'] or state['host_rank'] != 0
            or not state['owns_replicated_arrays'] or state['config_sha256'] != group['config_sha256']
            or state['model_schema'] != result['model_schema']
            or state['dataset_manifest_sha256'] != config['dataset']['manifest_sha256']):
        raise ValueError('Visual owner scientific state differs')
    schema = state['model_schema']
    if set(arrays) != {f'p_{i:04d}' for i in range(len(schema))}:
        raise ValueError('Visual parameter coverage differs')
    for i, item in enumerate(schema):
        p = arrays[f'p_{i:04d}']
        if list(p.shape) != item['shape'] or str(p.dtype) != item['dtype'] or not np.isfinite(p).all():
            raise ValueError('Invalid visual parameter array')
    dataset_path = artifact(root, str(Path(config['dataset']['path']) / 'manifest.json'))
    if checkpoints.sha256(dataset_path) != config['dataset']['manifest_sha256']:
        raise ValueError('Visual dataset rule identity differs')
    dataset = read_json(dataset_path)
    if dataset['kind'] != 'visual_causal_teacher_dataset':
        raise ValueError('Expected exact visual data lineage')
    return {'snapshot': source, 'manifest': manifest, 'config': config, 'arrays': arrays,
            'model_schema': schema, 'rules': dataset['rules'], 'training_result': result,
            'model_code_sha256': checkpoints.sha256(source / manifest['recipe'] / 'model.py')}
