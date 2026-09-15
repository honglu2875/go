"""Explicit, bounded plain-JAX distillation contract."""
import math


def validate(c):
    if set(c) != set('schema_version platform expected_processes expected_devices seed dataset model learner steps checkpoint_every log_every eval_every'.split()):
        raise ValueError('Unknown or missing causal training configuration')
    if c['schema_version'] != 1 or c['platform'] not in ('cpu', 'tpu'):
        raise ValueError('Unsupported platform or schema')
    for key, lo, hi in [('expected_processes', 1, 4), ('expected_devices', 1, 16), ('seed', 0, 2**32-1),
                        ('steps', 1, 16384), ('checkpoint_every', 1, 16384), ('log_every', 1, 16384), ('eval_every', 1, 16384)]:
        if type(c[key]) is not int or not lo <= c[key] <= hi:
            raise ValueError('Invalid ' + key)
    d, m, l = c['dataset'], c['model'], c['learner']
    if set(d) != {'path', 'manifest_sha256'} or len(d['manifest_sha256']) != 64:
        raise ValueError('Dataset content identity required')
    if set(m) != set('size komi width heads blocks max_tokens dtype expert_temperature behavior_temperature board_width board_blocks board_mode'.split()):
        raise ValueError('Unknown model configuration')
    if m['size'] != 9 or m['komi'] != 7.5 or m['max_tokens'] != 329 or m['dtype'] not in ('float32', 'bfloat16'):
        raise ValueError('Dataset/model rule or context contract differs')
    if any(type(m[k]) is not int or not lo <= m[k] <= hi for k, lo, hi in [('width', 8, 512), ('heads', 1, 32), ('blocks', 1, 16)]):
        raise ValueError('Invalid attention dimensions')
    if type(m['board_width']) is not int or not 8<=m['board_width']<=128 or type(m['board_blocks']) is not int or not 1<=m['board_blocks']<=8 or m['board_mode'] not in ('exact','empty'):
        raise ValueError('Invalid board encoder or input condition')
    if m['width'] % m['heads']:
        raise ValueError('Attention head dimension must divide width')
    if set(l) != set('games_per_role learning_rate warmup_steps beta1 beta2 epsilon weight_decay grad_clip'.split()):
        raise ValueError('Unknown optimizer configuration')
    if type(l['games_per_role']) is not int or not 1 <= l['games_per_role'] <= 128:
        raise ValueError('Invalid sequence batch size')
    if type(l['warmup_steps']) is not int or not 1 <= l['warmup_steps'] <= c['steps']:
        raise ValueError('Invalid warmup')
    for key, lo, hi in [('learning_rate', 1e-8, .1), ('beta1', 0, .999), ('beta2', 0, .99999),
                        ('epsilon', 1e-12, .1), ('weight_decay', 0, 1), ('grad_clip', .001, 100)]:
        if type(l[key]) not in (float, int) or not math.isfinite(l[key]) or not lo <= l[key] <= hi:
            raise ValueError('Invalid optimizer ' + key)
    if 2 * l['games_per_role'] * c['expected_processes'] % c['expected_devices']:
        raise ValueError('Global sequence batch must divide across devices')
    return c
