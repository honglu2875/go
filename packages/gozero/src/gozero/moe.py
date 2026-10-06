"""Pure-JAX, dropless token-choice mixture of feed-forward experts.

Only selected assignments enter grouped matrix multiplications. Routing has no
batch-dependent capacity or token dropping, so a token's output does not depend
on its neighbours, padding or future tokens. The reference backend uses JAX's
ragged-dot primitive; the TPU backend uses JAX's bundled Pallas Megablox kernels.
No expert parameters are gathered into one matrix per token.
"""
from __future__ import annotations

import math
from functools import partial

import jax
import jax.numpy as jnp


def validate(config):
    required = {'experts', 'top_k', 'backend', 'balance_weight', 'z_weight'}
    if not required <= set(config) or set(config) - required - {'tiling', 'remat_activations', 'permutation_vjp', 'scope'}:
        raise ValueError('Explicit MoE fields required: ' + ', '.join(sorted(required)))
    e, k = config['experts'], config['top_k']
    if type(e) is not int or type(k) is not int or not 1 <= k <= e <= 64:
        raise ValueError('Require 1 <= top_k <= experts <= 64')
    if config['backend'] not in ('ragged', 'pallas'):
        raise ValueError('Unknown grouped matrix multiplication backend')
    if 'tiling' in config:
        tile = config['tiling']
        if not isinstance(tile, (tuple, list)) or len(tile) != 3 or any(type(n) is not int or not 128 <= n <= 2048 or n % 128 for n in tile):
            raise ValueError('Invalid TPU grouped-matmul tiling')
    if type(config.get('remat_activations', False)) is not bool:
        raise ValueError('remat_activations must be boolean')
    if type(config.get('permutation_vjp', False)) is not bool:
        raise ValueError('permutation_vjp must be boolean')
    if config.get('scope', 'all') not in ('all', 'temporal'):
        raise ValueError('MoE scope must be all or temporal')
    for name in ('balance_weight', 'z_weight'):
        if type(config[name]) not in (int, float) or not math.isfinite(config[name]) or config[name] < 0:
            raise ValueError('Router loss weights must be finite and nonnegative')
    return config


def initialize(key, *, width, hidden, experts, gated, bias=False, down_scale=1.):
    if min(width, hidden, experts) < 1:
        raise ValueError('Positive expert dimensions required')
    keys = jax.random.split(key, 4)
    params = {
        'router': jax.random.normal(keys[0], (width, experts), jnp.float32) * (.01 / math.sqrt(width)),
        'up': jax.random.normal(keys[1], (experts, width, hidden), jnp.float32) / math.sqrt(width),
        'down': jax.random.normal(keys[2], (experts, hidden, width), jnp.float32) * (down_scale / math.sqrt(hidden)),
    }
    if gated:
        params['gate'] = jax.random.normal(keys[3], (experts, width, hidden), jnp.float32) / math.sqrt(width)
    if bias:
        params['up_bias'] = jnp.zeros((experts, hidden), jnp.float32)
        params['down_bias'] = jnp.zeros((experts, width), jnp.float32)
    return params


def grouped_dot(lhs, rhs, sizes, *, backend, tiling=(128, 128, 128)):
    if backend == 'ragged':
        return jax.lax.ragged_dot(lhs, rhs, sizes, preferred_element_type=jnp.float32)
    if backend != 'pallas':
        raise ValueError('Unknown grouped dot backend')
    from jax.experimental.pallas.ops.tpu import megablox
    # Pallas requires the total row extent to be tile-aligned (individual
    # expert groups may be ragged). Append zero rows to the last expert and
    # slice them away, so neither predictions nor parameter gradients change.
    rows = lhs.shape[0]
    extra = (-rows) % tiling[0]
    if extra:
        lhs = jnp.pad(lhs, ((0, extra), (0, 0)))
        sizes = sizes.at[-1].add(extra)
    return megablox.gmm(lhs, rhs, sizes, preferred_element_type=jnp.float32,
                       tiling=tuple(tiling))[:rows]


@partial(jax.custom_vjp, nondiff_argnums=(3,))
def _dispatch_permutation(tokens, order, inverse, top_k):
    """Pack selected copies; its transpose uses a permutation and a K-reduction."""
    return tokens[order // top_k]


def _dispatch_permutation_forward(tokens, order, inverse, top_k):
    return tokens[order // top_k], inverse


def _dispatch_permutation_backward(top_k, inverse, gradient):
    restored = gradient[inverse]
    return restored.reshape(-1, top_k, *gradient.shape[1:]).sum(axis=1), None, None


_dispatch_permutation.defvjp(_dispatch_permutation_forward, _dispatch_permutation_backward)


@jax.custom_vjp
def _gather_permutation(values, indices, inverse):
    return values[indices]


def _gather_permutation_forward(values, indices, inverse):
    return values[indices], inverse


def _gather_permutation_backward(inverse, gradient):
    return gradient[inverse], None, None


_gather_permutation.defvjp(_gather_permutation_forward, _gather_permutation_backward)


def feed_forward(params, x, *, top_k, dtype, backend='ragged', valid=None, tiling=(128, 128, 128), remat_activations=False, permutation_vjp=False):
    """Return outputs and additive router statistics for one expert layer.

    x has arbitrary token dimensions followed by width. All K assignments are
    retained. Padded tokens have zero outputs/gradients and do not contribute to
    statistics, but their fixed-shape slots can still execute in the kernel.
    The statistics layout is [probability sums E, assignment counts E,
    sum(logsumexp(logits)**2), entropy sum, valid token count].
    """
    shape = x.shape
    width, experts = params['router'].shape
    if shape[-1] != width or not 1 <= top_k <= experts:
        raise ValueError('Invalid router input or top_k')
    if valid is None:
        valid = jnp.ones(shape[:-1], bool)
    if valid.shape != shape[:-1] or valid.dtype != jnp.bool_:
        raise ValueError('One boolean validity flag required per token')
    live = valid.reshape(-1)
    tokens = jnp.where(live[:, None], x.reshape(-1, width), 0.)
    # Router normalization remains FP32 even when expert matmuls use BF16.
    logits = jnp.matmul(tokens.astype(jnp.float32), params['router'].astype(jnp.float32),
                        precision=jax.lax.Precision.HIGHEST)
    log_prob = jax.nn.log_softmax(logits, axis=-1)
    probability = jnp.exp(log_prob)
    selected_logits, selected = jax.lax.top_k(logits, top_k)
    # A normalized singleton is constant and would remove the task gradient
    # to a top-1 router. Use the selected full-softmax probability in that case.
    mixture = (jnp.take_along_axis(probability, selected, axis=-1) if top_k == 1
               else jax.nn.softmax(selected_logits, axis=-1))
    assignment_experts = selected.reshape(-1)
    order = jnp.argsort(assignment_experts, stable=True)
    sorted_experts = assignment_experts[order]
    source_tokens = order // top_k
    sizes = jnp.bincount(assignment_experts, length=experts).astype(jnp.int32)
    if permutation_vjp:
        # argsort is a permutation, even when many expert IDs are equal.
        # Invert only scalar indices, then reuse the gather in combine and
        # reverse-mode dispatch instead of a repeated-index feature scatter.
        inverse = jnp.zeros_like(order).at[order].set(jnp.arange(order.size, dtype=order.dtype), unique_indices=True)
        dispatched = _dispatch_permutation(tokens, order, inverse, top_k).astype(dtype)
    else:
        dispatched = tokens[source_tokens].astype(dtype)
    gated = 'gate' in params
    up = jnp.concatenate((params['gate'], params['up']), axis=-1) if gated else params['up']
    projected = grouped_dot(dispatched, up.astype(dtype), sizes, backend=backend, tiling=tiling)
    if 'up_bias' in params:
        if gated:
            raise ValueError('Gated expert biases are not part of this contract')
        projected = projected + params['up_bias'][sorted_experts]
    def activate(value):
        if gated:
            gate, up_value = jnp.split(value, 2, axis=-1)
            return jax.nn.silu(gate) * up_value
        return jax.nn.gelu(value)
    # Optional execution-only experiment: save activation inputs, recompute
    # cheap nonlinear arithmetic in reverse mode, never recompute expert dots.
    hidden = (jax.checkpoint(activate, prevent_cse=False)(projected)
              if remat_activations else activate(projected))
    result = grouped_dot(hidden.astype(dtype), params['down'].astype(dtype), sizes, backend=backend, tiling=tiling)
    if 'down_bias' in params:
        result = result + params['down_bias'][sorted_experts]
    # Invert the permutation, then combine the K outputs for each input token.
    restored = _gather_permutation(result, inverse, order) if permutation_vjp else jnp.zeros_like(result).at[order].set(result)
    output = jnp.sum(restored.reshape(-1, top_k, width) * mixture[..., None], axis=1)
    output = jnp.where(live[:, None], output, 0.).reshape(shape)
    weights = live.astype(jnp.float32)
    load = jnp.bincount(assignment_experts, weights=jnp.repeat(weights, top_k), length=experts)
    z = jnp.sum(weights * jax.nn.logsumexp(logits, axis=-1) ** 2)
    entropy = -jnp.sum(weights[:, None] * probability * log_prob)
    stats = jnp.concatenate((jnp.sum(weights[:, None] * probability, axis=0),
                             jax.lax.stop_gradient(load), jnp.array([z, entropy, weights.sum()])))
    return output, stats


def router_metrics(statistics, config, *, axis_name=None):
    """Compute global, padding-excluded balancing and router z losses.

    Statistics has shape [layers, 2*experts+3]. Accumulate chunks and shared
    refinement passes before calling this function. Experts in different layers
    remain distinct; balancing is averaged across layers with live tokens.
    """
    validate(config)
    e, k = config['experts'], config['top_k']
    if statistics.ndim != 2 or statistics.shape[-1] != 2 * e + 3:
        raise ValueError('Invalid layer router statistics')
    if axis_name is not None:
        statistics = jax.lax.psum(statistics, axis_name)
    count = statistics[:, -1]
    denominator = jnp.maximum(count, 1.)
    probability = statistics[:, :e] / denominator[:, None]
    fraction = statistics[:, e:2 * e] / (k * denominator[:, None])
    active = (count > 0).astype(jnp.float32)
    layers = jnp.maximum(active.sum(), 1.)
    average = lambda x: jnp.sum(active * x) / layers
    balance = average(e * jnp.sum(probability * fraction, axis=-1))
    z = average(statistics[:, -3] / denominator)
    entropy = average(statistics[:, -2] / denominator)
    loss = config['balance_weight'] * balance + config['z_weight'] * z
    metrics = dict(moe_balance=balance, moe_z=z, moe_entropy=entropy,
                   moe_aux_loss=loss, moe_max_load=jnp.max(fraction),
                   moe_min_load=jnp.min(jnp.where(active[:, None] > 0, fraction, 1.)),
                   moe_load_cv2=average(e*jnp.sum(fraction*fraction,axis=-1)-1.),
                   moe_dead_expert_fraction=average(jnp.mean(fraction==0.,axis=-1)),
                   moe_valid_tokens=count.sum(), moe_dropped_tokens=jnp.asarray(0., jnp.float32))
    return loss, metrics


def active_flops(*, tokens, width, hidden, experts, top_k, gated):
    """Matrix FLOPs only, including the router; FMA=2, no padding hidden here."""
    return 2 * tokens * width * experts + (6 if gated else 4) * tokens * top_k * width * hidden
