"""Derive an immutable full-horizon LR intervention from a frozen parent.

This module does not register or launch a learning run. Choosing a rate still
requires reviewing the completed two-seed study and freezing its follow-up plan.
"""
import copy
import math
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'packages/gozero/src'))
from gozero.snapshots import canonical_json, freeze, read_json, verify

RATE_FIELDS = frozenset(('learning_rate', 'end_learning_rate'))


def rate_only(parent, candidate):
    """Reject any change beyond the scale of the existing LR schedule."""
    for config in (parent, candidate):
        if config.get('kind') != 'fixed_policy_learning' or config.get('steps') != 4096:
            raise ValueError('This intervention retains the full 4096-update horizon')
        if config.get('evaluation', {}).get('run_test') is not False:
            raise ValueError('Test targets must remain closed')
        learner = config.get('learner', {})
        for field in RATE_FIELDS:
            value = learner.get(field)
            if type(value) not in (float, int) or not math.isfinite(value) or value <= 0:
                raise ValueError('Learning rates must be finite positive scalars')
        if learner.get('warmup_steps') != 64:
            raise ValueError('Warmup must remain 64 updates')
        if not math.isclose(learner['end_learning_rate']/learner['learning_rate'], .3, rel_tol=1e-14):
            raise ValueError('Final/peak learning-rate ratio must remain 0.3')
        if config.get('model', {}).get('architecture') != 'causal_visual_policy':
            raise ValueError('This intervention uses the existing transformer')
    a, b = copy.deepcopy(parent), copy.deepcopy(candidate)
    for config in (a, b):
        for field in RATE_FIELDS:
            del config['learner'][field]
    if a != b:
        raise ValueError('A setting other than the two learning-rate endpoints changed')
    if candidate['learner']['learning_rate'] <= parent['learner']['learning_rate']:
        raise ValueError('The first intervention tests an upward rate scale')


def configuration(parent, peak):
    candidate = copy.deepcopy(parent)
    if type(peak) not in (float, int) or not math.isfinite(peak) or peak <= 0:
        raise ValueError('A finite positive scalar peak is required')
    candidate['learner']['learning_rate'] = peak
    candidate['learner']['end_learning_rate'] = .3 * peak
    rate_only(parent, candidate)
    return candidate


def identical_source(parent, candidate):
    """Check all frozen source bytes, including shared code and operators."""
    a, b = verify(parent), verify(candidate)
    if a['recipe'] != b['recipe']:
        raise ValueError('The recipe identity changed')
    source = lambda manifest: {k:v for k,v in manifest['files'].items() if k != 'resolved_config.json'}
    if source(a) != source(b):
        raise ValueError('The intervention changed frozen source bytes')
    rate_only(read_json(parent/'resolved_config.json'), read_json(candidate/'resolved_config.json'))
    return len(source(a))


def clone(parent, peak, store):
    parent, store = Path(parent).resolve(), Path(store).resolve()
    manifest = verify(parent)
    config = configuration(read_json(parent/'resolved_config.json'), peak)
    with tempfile.TemporaryDirectory(prefix='strong9-lr-config-', dir='/tmp') as folder:
        path = Path(folder)/'config.json'
        path.write_bytes(canonical_json(config))
        # Freeze from the retained parent, never from the mutable workspace.
        child = freeze(parent, Path(manifest['recipe']), path, store)
    identical_source(parent, child)
    return child
