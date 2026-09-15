"""Explicit CNN versus causal-transformer architecture contract."""
import math
if __package__:
    from .transformer_config import validate_model as validate_transformer
else:
    from transformer_config import validate_model as validate_transformer

def validate_model(c):
    if 'architecture' not in c:
        return validate_transformer(c)
    fields = set('architecture max_board_size input_channels width layers max_positions norm_epsilon dtype rematerialize behavior_updates_trunk history microbatch'.split())
    if set(c) != fields or c['architecture'] != 'residual_cnn':
        raise ValueError('Unknown CNN model contract')
    for key, lo, hi in [('max_board_size',1,52),('input_channels',6,6),('width',8,2048),('layers',2,128),('max_positions',1,2048),('history',1,32),('microbatch',1,256)]:
        if type(c[key]) is not int or not lo <= c[key] <= hi:
            raise ValueError('Invalid CNN '+key)
    if c['dtype'] not in ('float32','bfloat16') or any(type(c[k]) is not bool for k in ('rematerialize','behavior_updates_trunk')):
        raise ValueError('Invalid CNN execution contract')
    if not math.isfinite(c['norm_epsilon']) or c['norm_epsilon'] <= 0:
        raise ValueError('Invalid CNN normalization')
    return c
