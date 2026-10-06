"""One learned query over spatial features, shared by full and draft passes.

The query is independent of the board. Its key contraction can therefore move
before the spatial dot product; the linear value projection can move after the
weighted spatial sum. This is ordinary multihead cross-attention in real
arithmetic, with a specified BF16 execution order, not a linear-attention
approximation. No temporal action or future observation enters this module.
"""
import math
import jax
import jax.numpy as jnp


def validate(c):
    kind = c.get('board_pool_kind', 'flat')
    if kind not in ('flat', 'attention'):
        raise ValueError('Unknown board pooling kind')
    fields = ('board_pool_heads', 'board_pool_mlp_hidden')
    if kind == 'flat':
        if any(k in c for k in fields):
            raise ValueError('Flat pooling cannot carry unused attention settings')
        return
    if any(type(c.get(k)) is not int or c[k] < 1 for k in fields):
        raise ValueError('Explicit attention pooling dimensions required')
    if c['width'] % c['board_pool_heads'] or c['board_pool_mlp_hidden'] > 16384:
        raise ValueError('Incompatible board pooling dimensions')


def initialize(seed, c):
    validate(c)
    w, d, h, f, side = (c[k] for k in
        ('encoder_width', 'width', 'board_pool_heads', 'board_pool_mlp_hidden', 'max_board_size'))
    key = jax.random.fold_in(jax.random.key(seed), 620731)
    p = {}
    def weight(name, shape, scale):
        nonlocal key
        key, draw = jax.random.split(key)
        p[name] = jax.random.normal(draw, shape, jnp.float32) * scale
    weight('query', (h, d // h), .02)
    weight('row', (side, w), .02)
    weight('column', (side, w), .02)
    weight('key.weight', (h, w, d // h), 1 / math.sqrt(w))
    weight('value.weight', (h, w, d // h), 1 / math.sqrt(w))
    weight('out.weight', (d, d), 1 / math.sqrt(d))
    weight('mlp.up.weight', (d, f), 1 / math.sqrt(d))
    weight('mlp.down.weight', (f, d), 1 / math.sqrt(f))
    for name, size in (('input', w), ('query', d), ('mlp', d)):
        p[name + '.norm.scale'] = jnp.ones(size, jnp.float32)
        p[name + '.norm.bias'] = jnp.zeros(size, jnp.float32)
    p['mlp.up.bias'] = jnp.zeros(f, jnp.float32)
    p['mlp.down.bias'] = jnp.zeros(d, jnp.float32)
    return {'pool.' + k: v for k, v in p.items()}


def forward(p, features, c, *, reference=False):
    # Import here to keep initialization and shape accounting independent.
    from encoder import norm, linear, dtype
    p = {k[5:]: v for k, v in p.items() if k.startswith('pool.')}
    h, d = c['board_pool_heads'], c['width']
    z = norm(features, p['input.norm.scale'], p['input.norm.bias'], c)
    z = z + p['row'][:, None, :] + p['column'][None, :, :]
    z = z.reshape(*features.shape[:-3], -1, c['encoder_width'])
    q = norm(p['query'].reshape(d), p['query.norm.scale'], p['query.norm.bias'], c).reshape(h, d // h)
    def dot(equation, *args):
        return jnp.einsum(equation, *(x.astype(dtype(c)) for x in args),
                          preferred_element_type=jnp.float32)
    if reference:
        # Transparent projected K/V implementation for FP32 algebra/gradient
        # qualification. BF16 reassociation is deliberately not claimed exact.
        keys = dot('...nc,hce->...nhe', z, p['key.weight'])
        scores = dot('...nhe,he->...nh', keys, q) / math.sqrt(d // h)
        values = dot('...nc,hce->...nhe', z, p['value.weight'])
        weights = jax.nn.softmax(scores, axis=-2)
        attended = dot('...nh,...nhe->...he', weights, values)
    else:
        score_vector = dot('hce,he->hc', p['key.weight'], q)
        scores = dot('...nc,hc->...nh', z, score_vector) / math.sqrt(d // h)
        weights = jax.nn.softmax(scores, axis=-2)
        pooled = dot('...nh,...nc->...hc', weights, z)
        # Explicit head GEMMs also support the CPU BF16 qualification backend,
        # whose batched-dot thunk cannot accumulate this layout into FP32.
        # Every platform executes the same contractions and precision policy.
        attended = jnp.stack([linear(pooled[..., i, :], p['value.weight'][i], c)
                              for i in range(h)], axis=-2)
    x = p['query'].reshape(d) + linear(attended.reshape(*features.shape[:-3], d), p['out.weight'], c)
    y = norm(x, p['mlp.norm.scale'], p['mlp.norm.bias'], c)
    y = jax.nn.gelu(linear(y, p['mlp.up.weight'], c) + p['mlp.up.bias'])
    return x + linear(y, p['mlp.down.weight'], c) + p['mlp.down.bias']


def flops(c, size):
    """Actual dense contractions, multiply+add=2; includes per-call query work.

    Count the query-key contraction once PER BOARD (conservative for batches
    where it is hoisted). Norm, softmax, GELU, bias and positional adds are not
    matmul FLOPs and are reported separately in the experiment protocol.
    """
    w, d, h, f = (c[k] for k in
        ('encoder_width', 'width', 'board_pool_heads', 'board_pool_mlp_hidden'))
    return 4*w*d + 4*size*size*h*w + 2*d*d + 4*d*f
