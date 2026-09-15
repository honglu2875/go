"""Portable, source-bound policy/value weights from complete joint checkpoints.

Exports retain FP32 master parameter bytes for inference. Optimizer arrays and
training-only helper parameters stay in the original recovery checkpoint.
Loading an export requires its pinned manifest, not the original large optimizer
checkpoint. Numerical model compatibility is checked separately by the caller.
"""
import math
from pathlib import Path
import re

import numpy as np

from . import checkpoints
from .joint_resume import RAM_BASES, regular
from .snapshots import read_json, verify

KIND = 'joint_policy_value_parameters'
MODEL_MODULES = ('causal.py', 'compact.py', 'draft_mask.py', 'draft_model.py', 'encoder.py', 'heads.py', 'inference.py',
                 'joint.py', 'katago.py', 'observation_attention.py', 'policy_model.py',
                 'spatial_readout.py')
FIELDS = {'schema_version', 'kind', 'training_snapshot', 'training_config_sha256',
          'training_result_sha256', 'checkpoint_manifest_sha256', 'checkpoint_group_sha256',
          'network_version', 'training_purpose', 'dataset_manifest_sha256', 'input_contract',
          'model', 'value_model', 'model_schema', 'numerical_sources', 'parameter_elements',
          'inference_parameter_elements', 'value_convention'}


def identity(value):
    if not isinstance(value, str) or re.fullmatch('[0-9a-f]{64}', value) is None:
        raise ValueError('Expected a SHA-256 identity')
    return value


def validate_schema(schema):
    if not isinstance(schema, list) or not schema:
        raise ValueError('Missing model schema')
    names = []
    for row in schema:
        if (set(row) != {'path', 'shape', 'dtype', 'elements', 'inference'}
                or not isinstance(row['path'], str) or not row['path']
                or not isinstance(row['shape'], list)
                or any(type(x) is not int or x <= 0 for x in row['shape'])
                or row['dtype'] != 'float32' or type(row['elements']) is not int
                or row['elements'] != math.prod(row['shape'])
                or type(row['inference']) is not bool):
            raise ValueError('Invalid joint parameter schema')
        names.append(row['path'])
    if names != sorted(set(names)):
        raise ValueError('Parameter schema is duplicated or out of order')
    selected = [row for row in schema if row['inference']]
    if not selected or not any(row['path'].startswith('value_head.') for row in selected):
        raise ValueError('Joint export requires its trained value head')
    return schema


def parameters(arrays, schema, *, inference_only):
    validate_schema(schema)
    selected = [(f'p_{i:04d}', row) for i, row in enumerate(schema)
                if not inference_only or row['inference']]
    if set(arrays) != {key for key, _ in selected}:
        raise ValueError('Export parameter coverage differs')
    result = {}
    for key, row in selected:
        value = np.asarray(arrays[key])
        if (list(value.shape) != row['shape'] or value.dtype != np.dtype('float32')
                or not np.isfinite(value).all()):
            raise ValueError('Invalid trained parameter: ' + row['path'])
        result[row['path']] = value
    return result


def validate_metadata(state):
    if set(state) != FIELDS or state['schema_version'] != 1 or state['kind'] != KIND:
        raise ValueError('Unsupported joint parameter export')
    for name in ('training_snapshot', 'training_config_sha256', 'training_result_sha256',
                 'checkpoint_manifest_sha256', 'checkpoint_group_sha256', 'dataset_manifest_sha256'):
        identity(state[name])
    if (type(state['network_version']) is not int or not 0 < state['network_version'] < 2**32
            or state['training_purpose'] not in ('qualification', 'learning')
            or state['value_convention'] != 'player_to_move:softmax(logits)[0]-softmax(logits)[1]'):
        raise ValueError('Invalid joint model version or value convention')
    validate_schema(state['model_schema'])
    for key, selected in (('parameter_elements', state['model_schema']),
                          ('inference_parameter_elements', [r for r in state['model_schema'] if r['inference']])):
        if type(state[key]) is not int or state[key] != sum(row['elements'] for row in selected):
            raise ValueError('Parameter accounting differs')
    if set(state['numerical_sources']) != set(MODEL_MODULES):
        raise ValueError('Incomplete numerical source closure')
    for digest in state['numerical_sources'].values():
        identity(digest)
    contract = state['input_contract']
    if (set(contract) != {'feature_version', 'feature_worker_binary_sha256', 'size', 'komi',
                         'spatial_channels', 'global_channels'}
            or contract['feature_version'] != 7 or contract['spatial_channels'] != 22
            or contract['global_channels'] != 19
            or type(contract['size']) is not int or not 1 <= contract['size'] <= 25
            or not math.isfinite(contract['komi']) or float(np.float32(contract['komi'])) != contract['komi']
            or contract['size'] != state['model']['max_board_size']):
        raise ValueError('Invalid exact V7 input contract')
    identity(contract['feature_worker_binary_sha256'])
    return state


def export(root, source, result_path, result_sha256, output, *, schema):
    """Export a completed owner checkpoint after checking its trained lineage."""
    root = Path(root).resolve(); source = Path(source)
    if source.parent != root / '.gozero/snapshots':
        raise ValueError('Training source must be in the declared snapshot store')
    manifest = verify(source); config = read_json(source / 'resolved_config.json')
    config_sha = checkpoints.sha256(source / 'resolved_config.json')
    result_path = regular(result_path, (root,))
    if checkpoints.sha256(result_path) != identity(result_sha256):
        raise ValueError('Completed training result changed')
    result = read_json(result_path); schema = validate_schema(schema)
    if (result['kind'] != 'fixed_joint_learning' or config['kind'] != result['kind']
            or result['status'] != 'passed' or result['training_complete'] is not True
            or result['snapshot_id'] != source.name or result['host_rank'] != 0
            or result['config_sha256'] != config_sha or result['turn'] != config['steps']
            or result['counters']['updates'] != result['turn']
            or result['model_schema'] != schema
            or result['training_purpose'] != config['training']['purpose']
            or result['world_size'] != config['expected_processes']):
        raise ValueError('Not the complete declared joint learning endpoint')
    latest = result['latest_checkpoint']
    owner = Path(latest['owner_checkpoint_path'])
    group_path = regular(owner.with_suffix('.group.json'), (root, *RAM_BASES))
    if checkpoints.sha256(group_path) != latest['group_sha256']:
        raise ValueError('Owner checkpoint group differs')
    group = read_json(group_path); world = result['world_size']
    mapping = group['host_jax_mapping']
    if (group['kind'] != 'visual_replicated_checkpoint_group'
            or group['snapshot_id'] != source.name or group['config_sha256'] != config_sha
            or group['turn'] != result['turn'] or group['owner_checkpoint_path'] != str(owner)
            or group['host_manifests'].get('0') != latest['manifest_sha256']
            or set(group['host_manifests']) != {str(i) for i in range(world)}
            or mapping != result['host_jax_mapping'] or len(mapping) != world
            or {r['host'] for r in mapping} != set(range(world))
            or {r['jax_rank'] for r in mapping} != set(range(world))
            or group['replicated_arrays_elements_sha256'] != latest['replicated_arrays_elements_sha256']):
        raise ValueError('Checkpoint group and completed result disagree')
    regular(owner / 'manifest.json', (root, *RAM_BASES))
    saved, arrays, _ = checkpoints.read(owner, expected_manifest_sha256=latest['manifest_sha256'], array_prefix='p_')
    if (saved['kind'] != 'visual_replicated_rank_state' or saved['snapshot_id'] != source.name
            or saved['config_sha256'] != config_sha or saved['turn'] != result['turn']
            or saved['host_rank'] != 0 or saved['owns_replicated_arrays'] is not True
            or saved['jax_rank'] != next(r['jax_rank'] for r in mapping if r['host'] == 0)
            or saved['model_schema'] != schema or saved['counters'] != result['counters']
            or saved['dataset_manifest_sha256'] != config['dataset']['manifest_sha256']
            or result['dataset_manifest_sha256'] != saved['dataset_manifest_sha256']):
        raise ValueError('Owner scientific state differs')
    parameters(arrays, schema, inference_only=False)
    dataset_path = regular(Path(config['dataset']['path']) / 'manifest.json',
                           (root, Path('/dev/shm/gozero-datasets')))
    if checkpoints.sha256(dataset_path) != config['dataset']['manifest_sha256']:
        raise ValueError('Input dataset manifest changed')
    data = read_json(dataset_path)
    if (data['kind'] != 'katago_raw_teacher_sequences'
            or data.get('qualification_only', False) and result['training_purpose'] != 'qualification'):
        raise ValueError('Unexpected joint input population')
    contract = {key: data[key] for key in ('feature_version', 'feature_worker_binary_sha256',
                                         'size', 'komi', 'spatial_channels', 'global_channels')}
    state = dict(schema_version=1, kind=KIND, training_snapshot=source.name,
                 training_config_sha256=config_sha, training_result_sha256=result_sha256,
                 checkpoint_manifest_sha256=latest['manifest_sha256'], checkpoint_group_sha256=latest['group_sha256'],
                 network_version=result['turn'], training_purpose=result['training_purpose'],
                 dataset_manifest_sha256=saved['dataset_manifest_sha256'], input_contract=contract,
                 model=config['model'], value_model=config['value_model'], model_schema=schema,
                 numerical_sources={name: checkpoints.sha256(source / manifest['recipe'] / name) for name in MODEL_MODULES},
                 parameter_elements=sum(row['elements'] for row in schema),
                 inference_parameter_elements=sum(row['elements'] for row in schema if row['inference']),
                 value_convention='player_to_move:softmax(logits)[0]-softmax(logits)[1]')
    validate_metadata(state)
    selected = {f'p_{i:04d}': arrays[f'p_{i:04d}'] for i, row in enumerate(schema) if row['inference']}
    output = Path(output)
    digest = checkpoints.write(output, state=state, arrays=selected, actors='{}', compress=False)
    exported, restored = load(output, digest)
    expected = parameters(selected, schema, inference_only=True)
    if exported != state or set(restored) != set(expected):
        raise ValueError('Export read-back structure differs')
    for name, value in expected.items():
        if value.dtype != restored[name].dtype or value.shape != restored[name].shape or value.tobytes() != restored[name].tobytes():
            raise ValueError('Export parameter read-back differs')
    return dict(manifest_sha256=digest, inference_parameters=state['inference_parameter_elements'],
                training_parameters=state['parameter_elements'], arrays=len(selected),
                dropped_helper_arrays=len(arrays)-len(selected), network_version=state['network_version'])


def load(path, manifest_sha256):
    """Load portable weights after full file and typed-parameter verification."""
    identity(manifest_sha256)
    state, arrays, actors = checkpoints.read(Path(path), expected_manifest_sha256=manifest_sha256)
    validate_metadata(state)
    if actors != '{}':
        raise ValueError('Inference export unexpectedly contains actor state')
    return state, parameters(arrays, state['model_schema'], inference_only=True)


def compatible(state, recipe, *, schema):
    """Require the qualified numerical bytes and complete declared shape schema."""
    validate_metadata(state)
    if schema != state['model_schema']:
        raise ValueError('Inference model schema differs from trained weights')
    for name, digest in state['numerical_sources'].items():
        if checkpoints.sha256(Path(recipe) / name) != digest:
            raise ValueError('Inference numerical source differs: ' + name)


def compatible_features(state, inference_binary_sha256, *, evidence=None):
    """Bind a distinct feature wrapper to an explicitly qualified provider pair."""
    validate_metadata(state);identity(inference_binary_sha256)
    contract=state['input_contract'];training=contract['feature_worker_binary_sha256']
    if inference_binary_sha256==training:
        return
    if (not isinstance(evidence,dict) or evidence.get('kind')!='exact_v7_provider_comparison'
            or evidence.get('status')!='passed' or evidence.get('training_provider_sha256')!=training
            or evidence.get('inference_provider_sha256')!=inference_binary_sha256
            or evidence.get('input_contract')!={k:v for k,v in contract.items() if k!='feature_worker_binary_sha256'}
            or evidence.get('shared_objects_identical') is not True
            or any(evidence.get(key) is not True for key in
                   ('all_spatial_bytes_equal','all_global_bytes_equal','all_legal_masks_equal'))
            or type(evidence.get('games')) is not int or evidence['games']<=0
            or type(evidence.get('positions')) is not int or evidence['positions']<=0):
        raise ValueError('Distinct feature providers require pinned compatibility evidence')
    for key in ('operator_sha256','dataset_manifest_sha256','offline_build_receipt_sha256',
                'inference_build_receipt_sha256','shared_objects_sha256'):
        identity(evidence[key])


def validate_descriptor(descriptor):
    fields={'schema_version','kind','parameters','training_snapshot','network_version',
            'training_purpose','input_contract','max_positions'}
    if (not isinstance(descriptor,dict) or set(descriptor)!=fields or descriptor['schema_version']!=1
            or descriptor['kind']!='joint_policy_value_candidate'
            or type(descriptor['network_version']) is not int or not 0<descriptor['network_version']<2**32
            or type(descriptor['max_positions']) is not int or not 1<descriptor['max_positions']<=2048
            or descriptor['training_purpose'] not in ('qualification','learning')
            or set(descriptor['parameters'])!={'path','manifest_sha256'}
            or not isinstance(descriptor['parameters']['path'],str) or not descriptor['parameters']['path']):
        raise ValueError('Invalid joint evaluation candidate')
    identity(descriptor['training_snapshot']);identity(descriptor['parameters']['manifest_sha256'])
    return descriptor


def load_candidate(root, descriptor):
    validate_descriptor(descriptor);root=Path(root).resolve()
    path=root/descriptor['parameters']['path']
    regular(path/'manifest.json',(root,Path('/dev/shm/gozero-parameters')))
    state,params=load(path,descriptor['parameters']['manifest_sha256'])
    if (any(descriptor[key]!=state[key] for key in
            ('training_snapshot','network_version','training_purpose','input_contract'))
            or descriptor['max_positions']!=state['model']['max_positions']):
        raise ValueError('Candidate metadata reassigns the trained model identity')
    return state,params
