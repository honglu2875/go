"""Explicit architecture and bounded qualification settings; no framework state."""
import math


def validate_model(c):
    fields = ('max_board_size patch_size input_channels width heads kv_heads layers '
              'mlp_hidden max_positions rope_theta norm_epsilon dtype attention_backend '
              'rematerialize behavior_updates_trunk')
    if not set(fields.split()) <= set(c) or set(c) - set(fields.split()) - {'linear_precision'}:
        raise ValueError('Unknown or missing visual-causal model field')
    if c.get('linear_precision', 'highest') not in ('highest', 'default'):
        raise ValueError('Unsupported explicit linear multiplication precision')
    bounds = {'max_board_size': (1, 52), 'patch_size': (1, 8),
              'input_channels': (6, 6), 'width': (8, 8192), 'heads': (1, 128),
              'kv_heads': (1, 128), 'layers': (1, 128), 'mlp_hidden': (8, 32768),
              'max_positions': (1, 2048)}
    for key, (low, high) in bounds.items():
        if type(c[key]) is not int or not low <= c[key] <= high:
            raise ValueError('Invalid ' + key)
    if c['width'] % c['heads'] or c['heads'] % c['kv_heads']:
        raise ValueError('Width/head and query/KV head divisibility required')
    if (c['width'] // c['heads']) % 2:
        raise ValueError('RoPE requires an even head dimension')
    for key in ('rope_theta', 'norm_epsilon'):
        if type(c[key]) not in (float, int) or not math.isfinite(c[key]) or c[key] <= 0:
            raise ValueError('Invalid ' + key)
    if c['dtype'] not in ('float32', 'bfloat16') or c['attention_backend'] not in ('xla', 'splash'):
        raise ValueError('Unsupported compute type or attention backend')
    if any(type(c[key]) is not bool for key in ('rematerialize', 'behavior_updates_trunk')):
        raise ValueError('Explicit rematerialization and gradient-routing flags required')
    if c['attention_backend'] == 'splash' and c['width'] // c['heads'] % 64:
        raise ValueError('Qualified Splash shapes require head dimension divisible by 64')
    return c


def validate(c):
    if set(c) != set(('schema_version platform expected_processes expected_devices seed '
                      'model probe').split()):
        raise ValueError('Unknown or missing qualification field')
    if c['schema_version'] != 1 or c['platform'] not in ('cpu', 'tpu'):
        raise ValueError('Unsupported schema or platform')
    validate_model(c['model'])
    for key, low, high in [('expected_processes', 1, 4), ('expected_devices', 1, 16),
                           ('seed', 0, 2**32 - 1)]:
        if type(c[key]) is not int or not low <= c[key] <= high:
            raise ValueError('Invalid ' + key)
    p = c['probe']
    required = set(('board_size sequences_per_host positions draft_plies repetitions '
                    'gradient_check cache_tolerance').split())
    if not required <= set(p) or set(p) - required - {'cached_backend', 'profile_compute'}:
        raise ValueError('Unknown probe field')
    if p.get('cached_backend', 'xla') not in ('xla', 'ragged_paged') or type(p.get('profile_compute', False)) is not bool:
        raise ValueError('Unsupported cached kernel or trace flag')
    for key, low, high in [('board_size', 1, c['model']['max_board_size']),
                           ('sequences_per_host', 1, 128), ('positions', 1, c['model']['max_positions']),
                           ('draft_plies', 1, 16), ('repetitions', 1, 10)]:
        if type(p[key]) is not int or not low <= p[key] <= high:
            raise ValueError('Invalid probe ' + key)
    if p['positions'] + p['draft_plies'] > c['model']['max_positions']:
        raise ValueError('Probe continuation exceeds context')
    if type(p['gradient_check']) is not bool or not 0 < p['cache_tolerance'] <= .2:
        raise ValueError('Invalid gradient flag or tolerance')
    if c['expected_devices'] % c['expected_processes']:
        raise ValueError('Expected equal device count per host')
    if p['sequences_per_host'] % (c['expected_devices'] // c['expected_processes']):
        raise ValueError('Local batch must divide across local devices')
    return c
