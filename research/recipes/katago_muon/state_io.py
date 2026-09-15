"""Typed checkpoint arrays for Muon momentum and auxiliary Adam moments."""
import re

import numpy as np

MUON_GROUPS = frozenset(('normal', 'normal_attn', 'normal_gab', 'gab_mlp', 'tab_module'))
ADAM_GROUPS = frozenset(('input', 'input_noreg', 'normal_gamma', 'noreg', 'output', 'output_noreg'))


def identity(value):
    if not isinstance(value, str) or not re.fullmatch('[0-9a-f]{64}', value):
        raise ValueError('Expected a SHA-256 identity')


def schema(params, specs):
    if set(params) != set(specs) or not params:
        raise ValueError('Parameter specification coverage differs')
    result = []
    for name, value in sorted(params.items()):
        spec = specs[name]
        if set(spec) != {'group', 'layout'} or spec['group'] not in MUON_GROUPS | ADAM_GROUPS:
            raise ValueError('Unknown optimizer role')
        if spec['group'] in MUON_GROUPS:
            if spec['layout'] not in ('linear', 'conv'):
                raise ValueError('Matrix layout is required')
        elif spec['layout'] is not None:
            raise ValueError('Adam parameters have no Muon matrix layout')
        a = np.asarray(value)
        if a.dtype != np.float32:
            raise ValueError('Expected float32 master parameters')
        result.append(dict(path=name, shape=list(a.shape), dtype='float32', **spec))
    return result


def flatten(params, state, specs, *, configuration_sha256, source_sha256):
    identity(configuration_sha256)
    identity(source_sha256)
    rows = schema(params, specs)
    adam = {r['path'] for r in rows if r['group'] in ADAM_GROUPS}
    if (set(state) != {'first', 'second', 'step'} or set(state['first']) != set(params)
            or set(state['second']) != adam):
        raise ValueError('Optimizer state coverage differs')
    step = np.asarray(state['step'])
    if step.shape != () or step.dtype != np.int32 or int(step) < 0:
        raise ValueError('Invalid optimizer step')
    arrays = dict(step=step)
    for i, row in enumerate(rows):
        key = row['path']
        for prefix, values in [('p', params), ('m', state['first']), ('v', state['second'])]:
            if prefix == 'v' and key not in adam:
                continue
            value = np.asarray(values[key])
            if value.dtype != np.float32 or list(value.shape) != row['shape'] or not np.isfinite(value).all():
                raise ValueError('Parameter or moment schema/values differ')
            if prefix == 'v' and np.any(value < 0):
                raise ValueError('Negative second moment')
            arrays[f'{prefix}_{i:04d}'] = value
    return dict(kind='katago_muon_aux_adam_state', version=1, parameter_schema=rows, step=int(step),
                configuration_sha256=configuration_sha256, source_sha256=source_sha256), arrays


def restore(metadata, arrays, *, expected_schema, configuration_sha256, source_sha256):
    identity(configuration_sha256)
    identity(source_sha256)
    if (metadata.get('kind') != 'katago_muon_aux_adam_state' or metadata.get('version') != 1
            or metadata.get('configuration_sha256') != configuration_sha256
            or metadata.get('source_sha256') != source_sha256
            or metadata.get('parameter_schema') != expected_schema):
        raise ValueError('Checkpoint identity or role/schema differs')
    rows = expected_schema
    if len({r['path'] for r in rows}) != len(rows):
        raise ValueError('Duplicate parameter names')
    expected = {'step'}
    trees = dict(p={}, m={}, v={})
    for i, row in enumerate(rows):
        for prefix in ('p', 'm', 'v'):
            if prefix == 'v' and row['group'] in MUON_GROUPS:
                continue
            name = f'{prefix}_{i:04d}'
            expected.add(name)
            if name not in arrays:
                raise ValueError('Missing parameter or moment')
            trees[prefix][row['path']] = np.asarray(arrays[name])
    if set(arrays) != expected:
        raise ValueError('Unexpected checkpoint arrays')
    state = dict(first=trees['m'], second=trees['v'], step=np.asarray(arrays['step']))
    specs = {r['path']: dict(group=r['group'], layout=r['layout']) for r in rows}
    actual, _ = flatten(trees['p'], state, specs, configuration_sha256=configuration_sha256, source_sha256=source_sha256)
    if actual != metadata:
        raise ValueError('Checkpoint metadata disagrees with arrays')
    return trees['p'], state
