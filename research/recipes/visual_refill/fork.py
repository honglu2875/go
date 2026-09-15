"""Explicit Adam-state forks for a new, pinned self-play dataset generation."""
import json
from pathlib import Path
import numpy as np
from gozero import checkpoints
from gozero.model_artifacts import artifact
from gozero.snapshots import read_json
from gozero.visual_artifacts import validate


def load(root, source, c, mapping, host, rank, rules):
    declaration = c['fork']
    if set(declaration) != {'candidate', 'candidate_sha256'}: raise ValueError('Unknown fork field')
    path = artifact(source, declaration['candidate'])
    if checkpoints.sha256(path) != declaration['candidate_sha256']: raise ValueError('Fork candidate changed')
    candidate = read_json(path); checked = validate(root, candidate); parent = checked['config']
    model_path = Path(__file__).with_name('model.py')
    if (checked['model_code_sha256'] != checkpoints.sha256(model_path) or parent['model'] != c['model']
            or checked['rules'] != rules): raise ValueError('Fork must preserve trained model and exact rules')
    child_compare, parent_compare = json.loads(json.dumps(c)), json.loads(json.dumps(parent))
    child_compare.pop('fork'); parent_compare.pop('fork', None)
    for key in ('steps', 'checkpoint_every', 'eval_every', 'log_every', 'evaluation', 'dataset'):
        child_compare.pop(key); parent_compare.pop(key)
    for key in ('learning_rate', 'end_learning_rate', 'warmup_steps'):
        child_compare['learner'].pop(key); parent_compare['learner'].pop(key)
    if child_compare != parent_compare: raise ValueError('Fork changed unapproved optimizer, objectives, initialization seed or topology')
    owner = artifact(root, candidate['checkpoint']['path']); group = read_json(owner.with_suffix('.group.json'))
    if group['host_jax_mapping'] != mapping: raise ValueError('Fork host/JAX mapping differs')
    if not candidate['network_version'] < c['steps']: raise ValueError('Fork must advance the absolute optimizer step')
    del checked['arrays']
    owner_state, arrays, actors = checkpoints.read(owner, expected_manifest_sha256=group['host_manifests']['0'])
    if actors != '{}': raise ValueError('Offline parent unexpectedly has unfinished actors')
    if host == 0: state = owner_state
    else:
        local = owner.parents[3] / f'rank-{host}/artifacts/checkpoints' / owner.name
        state, local_arrays, actors = checkpoints.read(local, expected_manifest_sha256=group['host_manifests'][str(host)])
        if local_arrays or actors != '{}': raise ValueError('Non-owner fork state contains unexpected data')
    schema = checked['model_schema']; required = {f'{kind}_{i:04d}' for kind in ('p', 'm', 'v') for i in range(len(schema))}
    if (set(arrays) != required or state['jax_rank'] != rank or state['host_rank'] != host
            or state['turn'] != candidate['network_version'] or owner_state['turn'] != state['turn']):
        raise ValueError('Fork parent state or Adam array coverage differs')
    for kind in ('p', 'm', 'v'):
        for i, item in enumerate(schema):
            array = arrays[f'{kind}_{i:04d}']
            if list(array.shape) != item['shape'] or str(array.dtype) != item['dtype'] or not np.isfinite(array).all():
                raise ValueError('Invalid parent parameter or moment')
    lineage = {'kind': 'visual_complete_adam_fork', 'parent_candidate_sha256': declaration['candidate_sha256'],
        'parent_snapshot': candidate['training_snapshot'], 'parent_turn': state['turn'],
        'parent_group_sha256': candidate['checkpoint']['group_sha256'],
        'preserved': 'Parameters, both Adam moments, absolute optimizer step, all rank sampler states, augmentation streams and exposure counters.',
        'changed': 'Explicit pinned dataset, local learning-rate schedule and run/evaluation/checkpoint bounds only.'}
    return arrays, state, lineage
