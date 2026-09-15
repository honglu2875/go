"""Plain-JAX spatial attention with D4-tied relative geometry and common heads."""
import math
import jax
import jax.numpy as jnp
from state_features import CHANNELS


def initialize(seed, c):
    key = jax.random.fold_in(jax.random.key(seed), 0x57A7E)
    width, hidden = c['state_width'], c['state_value_hidden']
    def weight(shape, scale=1.):
        nonlocal key
        key, draw = jax.random.split(key)
        return jax.random.normal(draw, shape, jnp.float32) * (scale / math.sqrt(shape[0]))
    def norm():
        return {'scale': jnp.ones(width, jnp.float32), 'bias': jnp.zeros(width, jnp.float32)}
    size = c['state_max_board_size']
    return {'stem': weight((CHANNELS, width), math.sqrt(2)), 'stem_bias': jnp.zeros(width, jnp.float32),
            'blocks': [{'n1': norm(), 'qkv': weight((width, 3 * width)), 'out': weight((width, width)),
                        'n2': norm(), 'mlp_in': weight((width, 4 * width), math.sqrt(2)),
                        'mlp_out': weight((4 * width, width))} for _ in range(c['state_blocks'])],
            'relative_bias': jnp.zeros((size * (size + 1) // 2, c['state_heads']), jnp.float32),
            'final_norm': norm(), 'policy': weight((width, 1), .1),
            'pass': weight((2 * width, 1), .1), 'pass_bias': jnp.zeros(1, jnp.float32),
            'value_hidden': weight((2 * width, hidden), math.sqrt(2)), 'value_bias': jnp.zeros(hidden, jnp.float32),
            'value_out': weight((hidden, 1), .1), 'value_out_bias': jnp.zeros(1, jnp.float32)}


def norm(x, p):
    x = x.astype(jnp.float32)
    centered = x - jnp.mean(x, axis=-1, keepdims=True)
    return centered * jax.lax.rsqrt(jnp.mean(centered ** 2, axis=-1, keepdims=True) + 1e-5) * p['scale'] + p['bias']


def relative_indices(size):
    points = jnp.arange(size * size)
    dy = jnp.abs(points[:, None] // size - points[None, :] // size)
    dx = jnp.abs(points[:, None] % size - points[None, :] % size)
    high, low = jnp.maximum(dy, dx), jnp.minimum(dy, dx)
    return high * (high + 1) // 2 + low


def apply(p, features, c):
    if features.ndim < 4 or features.shape[-1] != CHANNELS or features.shape[-3] != features.shape[-2]:
        raise ValueError('Expected square board features with the declared channels')
    size = features.shape[-2]
    if not 1 <= size <= c['state_max_board_size']:
        raise ValueError('Board exceeds spatial model geometry')
    prefix, width, heads = features.shape[:-3], c['state_width'], c['state_heads']
    dtype = {'float32': jnp.float32, 'bfloat16': jnp.bfloat16}[c['dtype']]
    def linear(x, w):
        return jnp.matmul(x.astype(dtype), w.astype(dtype), precision=jax.lax.Precision.HIGHEST,
                          preferred_element_type=jnp.float32)
    x = linear(features.reshape(-1, size * size, CHANNELS), p['stem']) + p['stem_bias']
    bias = jnp.moveaxis(p['relative_bias'][relative_indices(size)], -1, 0)
    scale = 1 / math.sqrt(c['state_blocks'])
    for block in p['blocks']:
        qkv = linear(norm(x, block['n1']), block['qkv']).reshape(x.shape[0], size * size, 3, heads, width // heads).astype(dtype)
        q, k, v = (qkv[:, :, i] for i in range(3))
        scores = jnp.einsum('bthd,bshd->bhts', q, k, precision=jax.lax.Precision.HIGHEST,
                            preferred_element_type=jnp.float32) / math.sqrt(width // heads)
        weights = jax.nn.softmax(scores + bias[None], axis=-1).astype(dtype)
        y = jnp.einsum('bhts,bshd->bthd', weights, v, precision=jax.lax.Precision.HIGHEST,
                       preferred_element_type=jnp.float32).reshape(x.shape)
        x = x + scale * linear(y, block['out'])
        x = x + scale * linear(jax.nn.gelu(linear(norm(x, block['n2']), block['mlp_in'])), block['mlp_out'])
    x = norm(x, p['final_norm'])
    pooled = jnp.concatenate((jnp.mean(x, axis=1), jnp.max(x, axis=1)), axis=-1)
    point_logits = linear(x, p['policy'])[..., 0]
    pass_logit = linear(pooled, p['pass']) + p['pass_bias']
    hidden = jax.nn.gelu(linear(pooled, p['value_hidden']) + p['value_bias'])
    value = jnp.tanh(linear(hidden, p['value_out']) + p['value_out_bias'])[..., 0]
    logits = jnp.concatenate((point_logits, pass_logit), axis=-1)
    return logits.reshape(*prefix, size * size + 1), value.reshape(prefix)
