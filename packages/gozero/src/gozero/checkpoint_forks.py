"""Explicit, content-addressed forks of complete synchronous self-play state.

A fork changes provenance and a declared run schedule, never the parent files.
Model, rules, search, replay shape, topology, native binary and dependency lock
remain identical. Ordinary same-source resume does not use this escape hatch.
This module validates data only; it never imports model code or initializes JAX.
"""
from __future__ import annotations

import copy
import hashlib
from pathlib import Path
import re

import numpy as np

from .checkpoints import read as read_checkpoint, sha256
from .model_artifacts import artifact
from .snapshots import canonical_json, read_json, verify


EXTRA_CONFIG_FIELDS = {'native', 'initialization'}
SCHEDULE_FIELDS = {'/selfplay_turns', '/checkpoint_every', '/log_every', '/learner/learning_rate'}


def require(value, message):
    if not value:
        raise ValueError(message)


def base_config(config):
    return {k: copy.deepcopy(v) for k, v in config.items() if k not in EXTRA_CONFIG_FIELDS}


def changes(before, after, prefix=''):
    if isinstance(before, dict) and isinstance(after, dict):
        require(set(before) == set(after), 'Fork cannot change configuration structure')
        return {p: v for key in sorted(before) for p, v in changes(before[key], after[key], prefix + '/' + key).items()}
    return {} if before == after and type(before) is type(after) else {prefix: {'from': before, 'to': after}}


def work_counters(counters):
    return {k: value for k, value in counters.items() if not k.endswith('_seconds')}


def replicated_digest(arrays):
    p = sorted(name for name in arrays if re.fullmatch(r'p_[0-9]{4}', name))
    v = sorted(name for name in arrays if re.fullmatch(r'v_[0-9]{4}', name))
    require(p and len(p) == len(v) and [n[2:] for n in p] == [n[2:] for n in v] and 'key' in arrays,
            'Missing parameter, momentum or augmentation RNG arrays')
    require(set(arrays) == set(p + v) | {'key', 'replay_x', 'replay_pi', 'replay_z', 'replay_meta', 'replay_owner'},
            'Complete self-play replay array tree is required')
    digest = hashlib.sha256()
    for name in [*p, *v, 'key']:
        digest.update(np.ascontiguousarray(arrays[name]).tobytes())
    return digest.hexdigest()


def native_identity(root, reference):
    require(set(reference) == {'receipt', 'receipt_sha256'}, 'Invalid native dependency declaration')
    path = artifact(root, reference['receipt']); receipt = read_json(path)
    require(sha256(path) == reference['receipt_sha256'] and receipt['schema_version'] == 1
            and receipt['filename'] == 'lib_gozero_native.so', 'Native receipt differs')
    source = artifact(root, '.gozero/snapshots/' + receipt['snapshot_id']); verify(source)
    require(sha256(path.parent / receipt['filename']) == receipt['binary_sha256'], 'Native binary differs')
    return path, receipt


def describe(root, parent_checkpoints, target_config, native_reference, *, learning_rate_mode='inherit_constant', reason):
    """Create a portable descriptor after verifying every parent rank.

    parent_checkpoints is an explicit host-rank/path mapping, avoiding an
    assumption that host order and JAX rank order are interchangeable.
    """
    root = Path(root).resolve(); _, native = native_identity(root, native_reference)
    require(learning_rate_mode in ('inherit_constant', 'constant_override') and isinstance(reason, str) and reason.strip(),
            'Fork requires a declared constant schedule and reason')
    entries = []; groups = []; sources = set(); schemas = []; configs = set()
    for host_rank, name in sorted(parent_checkpoints.items()):
        require(type(host_rank) is int and host_rank >= 0, 'Invalid parent host rank')
        checkpoint = artifact(root, name); group_path = checkpoint.with_suffix('.group.json'); group = read_json(group_path)
        require(group['schema_version'] == 1 and type(group['world_size']) is int and group['world_size'] > 0
                and len(group['rank_manifests']) == group['world_size'], 'Incomplete parent group manifest')
        # Read the state's rank before selecting its all-rank committed hash;
        # read_checkpoint then verifies these exact state bytes and all arrays.
        rank = read_json(checkpoint / 'state.json')['jax_rank']
        require(type(rank) is int and 0 <= rank < group['world_size'], 'Invalid parent JAX rank')
        state, arrays, _ = read_checkpoint(checkpoint, expected_manifest_sha256=group['rank_manifests'][rank])
        require(state['snapshot_id'] == group['snapshot_id'] and state['config_sha256'] == group['config_sha256']
                and state['turn'] == group['turn'] and state['counters']['updates'] == group['updates']
                and state['world_size'] == group['world_size'] and state['native_sha256'] == native['binary_sha256']
                and replicated_digest(arrays) == group['replicated_state_sha256'], 'Parent checkpoint group or native state differs')
        entries.append({'host_rank': host_rank, 'jax_rank': rank, 'path': str(checkpoint.relative_to(root)),
                        'manifest_sha256': sha256(checkpoint / 'manifest.json'), 'counters': work_counters(state['counters'])})
        groups.append(sha256(group_path)); sources.add(state['snapshot_id']); configs.add(state['config_sha256']); schemas.append(state['model_schema'])
    require(entries and len(sources) == len(configs) == len(set(groups)) == 1
            and all(s == schemas[0] for s in schemas), 'Parent ranks selected different source, model or groups')
    world = group['world_size']
    require(len(entries) == world and {e['jax_rank'] for e in entries} == set(range(world))
            and {e['host_rank'] for e in entries} == set(range(world)), 'Incomplete or duplicate parent rank set')
    source = artifact(root, '.gozero/snapshots/' + sources.pop()); manifest = verify(source)
    parent_config = read_json(source / 'resolved_config.json')
    require(sha256(source / 'resolved_config.json') == next(iter(configs)), 'Parent source configuration differs')
    target = base_config(target_config); diff = changes(base_config(parent_config), target)
    require(set(diff) <= SCHEDULE_FIELDS and target['selfplay_turns'] > group['turn'], 'Fork changes scientific settings or has no new turns')
    require(learning_rate_mode == 'constant_override' or '/learner/learning_rate' not in diff, 'Inherited learning rate changed')
    require(learning_rate_mode == 'inherit_constant' or '/learner/learning_rate' in diff, 'Override must declare an actual learning-rate change')
    return {'schema_version': 1, 'kind': 'complete_selfplay_checkpoint_fork', 'reason': reason,
            'parent_snapshot': source.name, 'parent_config_sha256': sha256(source / 'resolved_config.json'),
            'parent_group': group, 'parent_group_sha256': groups[0], 'rank_checkpoints': sorted(entries, key=lambda e: e['jax_rank']),
            'native': native_reference, 'native_binary_sha256': native['binary_sha256'],
            'model_code_sha256': sha256(source / manifest['recipe'] / 'model.py'),
            'model_schema_sha256': hashlib.sha256(canonical_json(schemas[0])).hexdigest(),
            'uv_lock_sha256': sha256(source / 'uv.lock'), 'python_version_sha256': sha256(source / '.python-version'),
            'target_base_config_sha256': hashlib.sha256(canonical_json(target)).hexdigest(),
            'declared_changes': diff, 'learning_rate_mode': learning_rate_mode,
            'semantics': 'Complete state continuation on the same topology and native binary; no reset of learner, replay, games or RNGs.'}


def contract(root, source, config):
    """Validate a frozen child without requiring retained parent array files."""
    root, source = Path(root).resolve(), Path(source).resolve(); manifest = verify(source)
    native_path, native = native_identity(root, config['native'])
    init = config['initialization']
    if init == {'kind': 'fresh'}:
        return {'kind': 'fresh'}, None, native_path, native
    require(set(init) == {'kind', 'descriptor', 'sha256'} and init['kind'] == 'fork', 'Unsupported initialization')
    path = artifact(source, init['descriptor']); require(sha256(path) == init['sha256'], 'Fork descriptor changed')
    spec = read_json(path)
    require(spec['schema_version'] == 1 and spec['kind'] == 'complete_selfplay_checkpoint_fork', 'Invalid fork descriptor')
    parent = artifact(root, '.gozero/snapshots/' + spec['parent_snapshot']); parent_manifest = verify(parent)
    parent_config = read_json(parent / 'resolved_config.json'); group = spec['parent_group']
    require(sha256(parent / 'resolved_config.json') == spec['parent_config_sha256'] == group['config_sha256']
            and group['snapshot_id'] == parent.name and hashlib.sha256(canonical_json(group)).hexdigest() == spec['parent_group_sha256'],
            'Parent source or committed group differs')
    require(sha256(source / manifest['recipe'] / 'model.py') == spec['model_code_sha256']
            == sha256(parent / parent_manifest['recipe'] / 'model.py'), 'Fork changes the model implementation')
    for name, field in (('uv.lock', 'uv_lock_sha256'), ('.python-version', 'python_version_sha256')):
        require(sha256(source / name) == spec[field] == sha256(parent / name), 'Fork changes the locked runtime')
    target = base_config(config); diff = changes(base_config(parent_config), target)
    require(hashlib.sha256(canonical_json(target)).hexdigest() == spec['target_base_config_sha256']
            and diff == spec['declared_changes'] and set(diff) <= SCHEDULE_FIELDS
            and target['selfplay_turns'] > group['turn'], 'Undeclared or forbidden fork change')
    require(spec['learning_rate_mode'] in ('inherit_constant', 'constant_override')
            and ('/learner/learning_rate' in diff) == (spec['learning_rate_mode'] == 'constant_override'), 'Schedule declaration differs')
    world = config['expected_processes']; entries = spec['rank_checkpoints']
    require(group['world_size'] == world and len(entries) == len(group['rank_manifests']) == world
            and {e['jax_rank'] for e in entries} == {e['host_rank'] for e in entries} == set(range(world))
            and all(e['manifest_sha256'] == group['rank_manifests'][e['jax_rank']] for e in entries), 'Parent rank coverage differs')
    require(config['native'] == spec['native'] and native['binary_sha256'] == spec['native_binary_sha256'], 'Fork changes the native binary')
    origin = {'kind': 'fork', 'descriptor_sha256': init['sha256'], 'parent_snapshot': parent.name,
              'parent_group_sha256': spec['parent_group_sha256'], 'parent_turn': group['turn'],
              'learning_rate_mode': spec['learning_rate_mode'], 'declared_changes': diff}
    return origin, spec, native_path, native


def load_rank(root, spec, *, jax_rank, host_rank):
    entries = [e for e in spec['rank_checkpoints'] if e['jax_rank'] == jax_rank]
    require(len(entries) == 1 and entries[0]['host_rank'] == host_rank, 'Fork requires unchanged host/JAX rank placement')
    entry = entries[0]; path = artifact(root, entry['path']); group_path = path.with_suffix('.group.json')
    require(sha256(group_path) == spec['parent_group_sha256'] and read_json(group_path) == spec['parent_group'], 'Local parent group differs')
    state, arrays, actors = read_checkpoint(path, expected_manifest_sha256=entry['manifest_sha256'])
    group = spec['parent_group']
    require(state['jax_rank'] == jax_rank and state['world_size'] == group['world_size'] and state['turn'] == group['turn']
            and state['snapshot_id'] == spec['parent_snapshot'] and state['config_sha256'] == spec['parent_config_sha256']
            and state['native_sha256'] == spec['native_binary_sha256']
            and state['counters']['updates'] == group['updates'] and work_counters(state['counters']) == entry['counters']
            and hashlib.sha256(canonical_json(state['model_schema'])).hexdigest() == spec['model_schema_sha256']
            and replicated_digest(arrays) == group['replicated_state_sha256'], 'Loaded parent scientific state differs')
    return state, arrays, actors
