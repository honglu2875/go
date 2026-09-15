"""Functional Nesterov SGD in the historical trainer's summed-gradient units.

The public June2019 TensorFlow implementation uses MomentumOptimizer with
use_nesterov=True and l2_coeff * tf.nn.l2_loss on kernel weights. Folded RepVGG
centers keep the current CNN's two-branch data-gradient multiplier; coupled
decay of the two underlying branches sums to decay of the folded weight.
"""
import jax
import jax.numpy as jnp
import katago


def initialize(params):
    return {'momentum': jax.tree.map(jnp.zeros_like, params),
            'step': jnp.asarray(0, jnp.int32), 'samples': jnp.asarray(0, jnp.int32)}


def group(name):
    if name.startswith(('intermediate_', 'norm_intermediate_')): return 'helper'
    if name.startswith(('policy_head.', 'norm_trunkfinal.')): return 'head'
    if name.startswith(('conv_spatial.', 'linear_global.')): return 'input'
    return 'trunk'


def decay(name):
    return name.endswith('.weight')


def rate(samples, settings):
    # tf.piecewise_constant uses its lower interval at the exact boundary.
    return jnp.where(samples <= settings['warmup_positions'],
                     settings['warmup_per_sample_lr'], settings['per_sample_lr'])


def apply_gradient(params, state, gradients, loss, count, settings):
    count = jnp.asarray(count, jnp.int32)
    per_sample_lr = rate(state['samples'], settings)
    raw_norm = jnp.sqrt(sum(jnp.sum(g * g) for g in gradients.values()))
    gradients = katago.repvgg_gradient(gradients)
    norm = jnp.sqrt(sum(jnp.sum(g * g) for g in gradients.values()))
    summed = {k: count * (g + (settings['l2_coefficient'] * params[k] if decay(k) else 0.))
              for k, g in gradients.items()}
    momentum = {k: settings['momentum'] * state['momentum'][k] + g for k, g in summed.items()}
    direction = {k: g + settings['momentum'] * momentum[k] for k, g in summed.items()}
    delta = jax.tree.map(lambda g: per_sample_lr * g, direction)
    candidate = jax.tree.map(lambda p, d: p - d, params, delta)
    finite = jnp.isfinite(loss) & jnp.isfinite(norm) & (count > 0)
    finite &= jnp.all(jnp.stack([jnp.all(jnp.isfinite(x))
                                 for x in jax.tree.leaves((candidate, momentum))]))
    new_state = {'momentum': momentum, 'step': state['step'] + 1,
                 'samples': state['samples'] + count}
    result = jax.tree.map(lambda new, old: jnp.where(finite, new, old), candidate, params)
    result_state = jax.tree.map(lambda new, old: jnp.where(finite, new, old), new_state, state)
    metrics = {'accepted': finite, 'loss': loss, 'raw_grad_norm': raw_norm, 'grad_norm': norm,
               'clip_scale': jnp.asarray(1., jnp.float32), 'per_sample_learning_rate': per_sample_lr,
               'learning_rate': count * per_sample_lr,
               'samples_before': state['samples'], 'samples_after': result_state['samples'],
               'update': result_state['step']}
    for label in sorted(set(map(group, params))):
        metrics['update_norm_' + label] = jnp.sqrt(sum(jnp.sum(d * d) for k, d in delta.items() if group(k) == label))
    return result, result_state, metrics


def scan_updates(objective, settings):
    """Carry weights and momentum once; materialize only small per-step metrics.

    A failed update latches the scan off. Padding steps have no effect, including
    on momentum, sample counters and the schedule.
    """
    def one(carry, batch):
        params, state, healthy = carry
        def calculate(_):
            (loss, metrics), gradient = jax.value_and_grad(objective, has_aux=True)(params, batch)
            p, s, extra = apply_gradient(params, state, gradient, loss, metrics['expert_positions'], settings)
            return p, s, {**metrics, **extra, 'active': jnp.asarray(True)}
        def skip(_):
            # eval_shape is performed at trace time, with no numerical forward.
            shape = jax.eval_shape(calculate, None)[2]
            return params, state, jax.tree.map(lambda x: jnp.zeros(x.shape, x.dtype), shape)
        active = jnp.sum(batch['counts']) > 0
        p, s, metrics = jax.lax.cond(healthy & active, calculate, skip, None)
        healthy &= (~active) | metrics['accepted']
        return (p, s, healthy), metrics
    def block(params, state, batch):
        (p, s, healthy), metrics = jax.lax.scan(one, (params, state, jnp.asarray(True)), batch)
        return p, s, metrics, healthy
    return block
