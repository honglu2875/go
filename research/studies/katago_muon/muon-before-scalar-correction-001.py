"""Functional JAX standard Muon and auxiliary Adam; see the retained licenses.

Arithmetic follows pinned KataGo's python/muon/muon.py, with independent
matrix batches expressed directly in JAX. Rates/decay and gradient preprocessing
belong to the caller. No Torch, Flax or optimizer-library runtime dependency.
"""
import math

import jax
import jax.numpy as jnp

MUON_GROUPS = frozenset(('normal', 'normal_attn', 'normal_gab', 'gab_mlp', 'tab_module'))
ADAM_GROUPS = frozenset(('input', 'input_noreg', 'normal_gamma', 'noreg', 'output', 'output_noreg'))


def matrix(x, layout):
    """Convert [..., H, W, I, O] or [..., I, O] to independent [..., O, K]."""
    if layout == 'linear':
        if x.ndim < 2:
            raise ValueError('Linear weights need two trailing channel axes')
        return jnp.swapaxes(x, -1, -2)
    if layout != 'conv' or x.ndim < 4:
        raise ValueError('Expected explicit linear or HWIO convolution layout')
    n = x.ndim
    transposed = jnp.transpose(x, (*range(n - 4), n - 1, n - 2, n - 4, n - 3))
    return transposed.reshape(*x.shape[:-4], x.shape[-1], math.prod(x.shape[-4:-1]))


def from_matrix(x, shape, layout):
    if layout == 'linear':
        result = jnp.swapaxes(x, -1, -2)
    elif layout == 'conv':
        n = len(shape)
        x = x.reshape(*shape[:-4], shape[-1], shape[-2], shape[-4], shape[-3])
        result = jnp.transpose(x, (*range(n - 4), n - 2, n - 1, n - 3, n - 4))
    else:
        raise ValueError('Unknown matrix layout')
    if result.shape != shape:
        raise ValueError('Matrix layout changed the parameter shape')
    return result


def polar(g):
    """Five BF16 quintic iterations with FP32 reduction/accumulation."""
    if g.ndim < 2:
        raise ValueError('Muon needs independent matrices')
    x = g.astype(jnp.bfloat16)
    transposed = g.shape[-2] > g.shape[-1]
    if transposed:
        x = jnp.swapaxes(x, -1, -2)
    norm = jnp.sqrt(jnp.sum(x.astype(jnp.float32) ** 2, axis=(-2, -1), keepdims=True)).astype(jnp.bfloat16)
    x = x / (norm + jnp.bfloat16(1e-7))

    def mm(a, b):
        return jnp.matmul(a, b, preferred_element_type=jnp.float32).astype(jnp.bfloat16)

    for _ in range(5):
        a = mm(x, jnp.swapaxes(x, -1, -2))
        # Python's left association in the reference computes (c * A) @ A.
        b = jnp.bfloat16(-4.775) * a + mm(jnp.bfloat16(2.0315) * a, a)
        x = jnp.bfloat16(3.4445) * x + mm(b, x)
    return jnp.swapaxes(x, -1, -2) if transposed else x


def validate(params, specifications):
    if set(params) != set(specifications) or not params:
        raise ValueError('Every parameter needs exactly one optimizer specification')
    for name, p in params.items():
        spec = specifications[name]
        if set(spec) != {'group', 'layout'} or spec['group'] not in MUON_GROUPS | ADAM_GROUPS:
            raise ValueError('Unknown optimizer parameter group')
        if p.dtype != jnp.float32:
            raise ValueError('Master parameters must be float32')
        if spec['group'] in MUON_GROUPS:
            matrix(p, spec['layout'])
        elif spec['layout'] is not None:
            raise ValueError('Auxiliary Adam parameters do not have a Muon matrix layout')


def initialize(params, specifications):
    validate(params, specifications)
    return dict(first=jax.tree.map(jnp.zeros_like, params),
                second={k: jnp.zeros_like(p) for k, p in params.items() if specifications[k]['group'] in ADAM_GROUPS},
                step=jnp.asarray(0, jnp.int32))


def apply(params, state, gradient, loss, specifications, rates, decays):
    """Apply already-preprocessed gradients; reject nonfinite updates atomically."""
    validate(params, specifications)
    groups = {s['group'] for s in specifications.values()}
    if set(rates) != groups or set(decays) != groups or set(gradient) != set(params):
        raise ValueError('Explicit group rates/decays and complete gradients required')
    adam_keys = {k for k, s in specifications.items() if s['group'] in ADAM_GROUPS}
    if set(state) != {'first', 'second', 'step'} or set(state['first']) != set(params) or set(state['second']) != adam_keys:
        raise ValueError('Optimizer state coverage differs')
    first, second, candidate, updates = {}, {}, {}, {}
    step = state['step'] + 1
    finite = jnp.isfinite(loss) & (state['step'] >= 0)
    for key, p in params.items():
        g, spec = gradient[key], specifications[key]
        if g.shape != p.shape or g.dtype != jnp.float32 or state['first'][key].shape != p.shape:
            raise ValueError('Gradient or momentum schema differs')
        rate, decay = rates[spec['group']], decays[spec['group']]
        finite &= jnp.isfinite(rate) & (rate >= 0) & jnp.isfinite(decay) & (decay >= 0) & jnp.all(jnp.isfinite(g))
        first[key] = state['first'][key] + .05 * (g - state['first'][key])
        if spec['group'] in MUON_GROUPS:
            nesterov = g + .95 * (first[key] - g)
            m = matrix(nesterov, spec['layout'])
            update = polar(m) * jnp.bfloat16(.2 * math.sqrt(max(m.shape[-2:])))
            update = from_matrix(update, p.shape, spec['layout']).astype(jnp.float32)
        else:
            if state['second'][key].shape != p.shape:
                raise ValueError('Adam second-moment shape differs')
            second[key] = state['second'][key] + .005 * (g * g - state['second'][key])
            update = (first[key] / (1 - .95 ** step)) / (jnp.sqrt(second[key] / (1 - .995 ** step)) + 1e-6)
        updates[key] = update
        candidate[key] = p * (1 - rate * decay) - rate * update
    new_state = dict(first=first, second=second, step=step)
    finite &= jnp.all(jnp.stack([jnp.all(jnp.isfinite(x)) for x in jax.tree.leaves((candidate, new_state))]))
    select = lambda new, old: jnp.where(finite, new, old)
    return (jax.tree.map(select, candidate, params), jax.tree.map(select, new_state, state),
            dict(accepted=finite, updates=updates))
