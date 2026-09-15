"""Strict configuration for the offline visual/draft learning intervention."""
import math
from pathlib import Path
if __package__:
    from .config import validate_model
else:
    from config import validate_model


def validate(c):
    fields = ('schema_version kind platform expected_processes expected_devices seed model dataset '
              'steps checkpoint_every eval_every log_every learner exits evaluation')
    if set(c) != set(fields.split()) or c['schema_version'] != 1 or c['kind'] != 'visual_causal_distillation':
        raise ValueError('Unknown or missing visual learning fields')
    if c['platform'] not in ('cpu', 'tpu'):
        raise ValueError('Unsupported learning backend')
    validate_model(c['model'])
    def integer(key, low, high, obj=c):
        if type(obj[key]) is not int or not low <= obj[key] <= high:
            raise ValueError('Invalid ' + key)
    for key, lo, hi in [('expected_processes', 1, 4), ('expected_devices', 1, 16), ('seed', 0, 2**32-1000000),
                        ('steps', 1, 1000000), ('checkpoint_every', 1, 1000000), ('eval_every', 1, 1000000), ('log_every', 1, 1000000)]:
        integer(key, lo, hi)
    if c['expected_devices'] % c['expected_processes']:
        raise ValueError('Topology is not evenly partitioned')
    d = c['dataset']
    if (not {'path', 'manifest_sha256', 'buckets'} <= set(d)
            or set(d) - {'path', 'manifest_sha256', 'buckets', 'bucket_probabilities', 'warmup_buckets'}
            or not Path(d['path']).is_absolute()):
        raise ValueError('An explicit pinned visual dataset is required')
    if len(d['manifest_sha256']) != 64 or any(x not in '0123456789abcdef' for x in d['manifest_sha256']):
        raise ValueError('Invalid dataset hash')
    if (not d['buckets'] or d['buckets'] != sorted(set(d['buckets']))
            or any(type(b) is not int or not 1 <= b <= c['model']['max_positions'] for b in d['buckets'])):
        raise ValueError('Invalid whole-history buckets')
    if 'bucket_probabilities' in d:
        weights = d['bucket_probabilities']
        if (len(weights) != len(d['buckets']) or any(type(v) not in (int, float) or not math.isfinite(v) or v <= 0 for v in weights)
                or not math.isclose(sum(weights), 1., abs_tol=1e-8)):
            raise ValueError('Bucket probabilities must be positive and sum to one')
    if any(type(b) is not int or b not in d['buckets'] for b in d.get('warmup_buckets', [])):
        raise ValueError('Warmup sequence refers to an unknown bucket')
    if len(d.get('warmup_buckets', [])) > c['steps']:
        raise ValueError('Warmup bucket sequence exceeds the run')
    l = c['learner']
    if set(l) != set('games_per_role learning_rate end_learning_rate warmup_steps beta1 beta2 epsilon weight_decay max_grad_norm loss_weights'.split()):
        raise ValueError('Unknown optimizer field')
    integer('games_per_role', 1, 64, l); integer('warmup_steps', 0, c['steps'], l)
    if 2 * l['games_per_role'] % (c['expected_devices'] // c['expected_processes']):
        raise ValueError('Local training batch must divide over devices')
    for key, low, high in [('learning_rate', 0., .1), ('end_learning_rate', 0., .1), ('beta1', 0., 1.),
                           ('beta2', 0., 1.), ('epsilon', 0., 1.), ('weight_decay', 0., 1.), ('max_grad_norm', 0., 1000.)]:
        v = l[key]
        if type(v) not in (int, float) or not math.isfinite(v) or not low <= v <= high:
            raise ValueError('Invalid optimizer ' + key)
    if min(l['learning_rate'], l['epsilon'], l['max_grad_norm']) <= 0 or max(l['beta1'], l['beta2']) >= 1:
        raise ValueError('Invalid optimizer boundary')
    if len(l['loss_weights']) != 5 or any(not math.isfinite(v) or v < 0 for v in l['loss_weights']):
        raise ValueError('Invalid supervised objective weights')
    e = c['exits']
    if set(e) != {'depths', 'loss_weight', 'temperature'} or not e['depths'] or e['depths'] != sorted(set(e['depths'])):
        raise ValueError('Explicit diagnostic exits are required')
    if any(type(x) is not int or not 1 <= x < c['model']['layers'] for x in e['depths']):
        raise ValueError('Exit must precede target depth')
    if any(type(e[k]) not in (int, float) or not math.isfinite(e[k]) for k in ['loss_weight', 'temperature']):
        raise ValueError('Invalid exit loss scalar')
    if not 0 <= e['loss_weight'] <= 10 or not .1 <= e['temperature'] <= 10:
        raise ValueError('Invalid exit loss or temperature')
    e = c['evaluation']
    if set(e) != {'games_per_role_per_bucket', 'run_test'} or type(e['run_test']) is not bool:
        raise ValueError('Invalid held-out contract')
    integer('games_per_role_per_bucket', 1, 10000, e)
    return c
