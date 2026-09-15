"""Prepare one isolated scalar intervention; this does not select or launch it."""
import copy
import math
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'packages/gozero/src'))
from gozero.snapshots import canonical_json, freeze, read_json, verify

FIELDS = {'encoder_scale': ('model', 'encoder_layer_scale'),
          'lr_floor': ('learner', 'end_learning_rate')}


def scalar_only(parent, candidate, mechanism):
    if mechanism not in FIELDS:
        raise ValueError('Unknown scalar intervention')
    section, field = FIELDS[mechanism]
    for config in (parent, candidate):
        if (config.get('kind') != 'fixed_policy_learning' or config.get('steps') != 4096
                or config.get('evaluation', {}).get('run_test') is not False
                or config.get('learner', {}).get('warmup_steps') != 64
                or config.get('model', {}).get('architecture') != 'causal_visual_policy'):
            raise ValueError('Retain the full transformer horizon, warmup and closed test split')
        value = config.get(section, {}).get(field)
        if type(value) not in (float, int) or not math.isfinite(value) or value <= 0:
            raise ValueError('Scalar must be finite and positive')
    left, right = copy.deepcopy(parent), copy.deepcopy(candidate)
    a, b = left[section].pop(field), right[section].pop(field)
    if left != right:
        raise ValueError('A setting other than the declared scalar changed')
    if mechanism == 'encoder_scale' and not a < b <= .1:
        raise ValueError('Encoder probe requires a larger initial scale at most 0.1')
    if mechanism == 'lr_floor' and not b < a:
        raise ValueError('Floor probe requires a lower final learning rate')
    return dict(mechanism=mechanism, path=[section, field], parent_value=a, candidate_value=b)


def configuration(parent, mechanism, value):
    if mechanism not in FIELDS:
        raise ValueError('Unknown scalar intervention')
    candidate = copy.deepcopy(parent)
    section, field = FIELDS[mechanism]
    candidate[section][field] = value
    scalar_only(parent, candidate, mechanism)
    return candidate


def identical_source(parent, candidate, mechanism):
    a, b = verify(parent), verify(candidate)
    source = lambda m:{k:v for k,v in m['files'].items() if k != 'resolved_config.json'}
    if a['recipe'] != b['recipe'] or source(a) != source(b):
        raise ValueError('The intervention changed frozen source bytes')
    change = scalar_only(read_json(parent/'resolved_config.json'),
                         read_json(candidate/'resolved_config.json'), mechanism)
    return dict(source_files=len(source(a)), change=change)


def clone(parent, mechanism, value, store):
    parent, store = Path(parent).resolve(), Path(store).resolve()
    manifest = verify(parent)
    config = configuration(read_json(parent/'resolved_config.json'), mechanism, value)
    with tempfile.TemporaryDirectory(prefix='strong9-scalar-config-', dir='/tmp') as folder:
        path = Path(folder) / 'config.json'
        path.write_bytes(canonical_json(config))
        child = freeze(parent, Path(manifest['recipe']), path, store)
    identical_source(parent, child, mechanism)
    return child
