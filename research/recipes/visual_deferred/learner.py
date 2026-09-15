"""Functional AdamW update; data, schedules and optimizer settings are explicit."""
import jax
import jax.numpy as jnp

if __package__:
    from . import model
else:
    import model


def initialize(params):
    return {'first': jax.tree.map(jnp.zeros_like, params),
            'second': jax.tree.map(jnp.zeros_like, params), 'step': jnp.asarray(0, jnp.int32)}


def update(params, state, batch, config, *, learning_rate, beta1=.9, beta2=.95,
           epsilon=1e-8, weight_decay=.01, max_grad_norm=1., exit_depths=(),
           exit_loss_weight=0., exit_temperature=1.):
    def objective(p):
        return model.losses(p, batch, config, exit_depths=exit_depths,
                            exit_loss_weight=exit_loss_weight, exit_temperature=exit_temperature)
    (loss, metrics), grads = jax.value_and_grad(objective, has_aux=True)(params)
    params, state, extra = apply_gradient(params, state, grads, loss,
        learning_rate=learning_rate, beta1=beta1, beta2=beta2, epsilon=epsilon,
        weight_decay=weight_decay, max_grad_norm=max_grad_norm)
    return params, state, {**metrics, **extra}


def apply_gradient(params, state, grads, loss, *, learning_rate, beta1=.9, beta2=.95,
                   epsilon=1e-8, weight_decay=.01, max_grad_norm=1.):
    """Apply already globally averaged gradients; reject any nonfinite update."""
    grad_norm = jnp.sqrt(sum(jnp.sum(g.astype(jnp.float32) ** 2) for g in jax.tree.leaves(grads)))
    finite = jnp.isfinite(loss) & jnp.isfinite(grad_norm)
    scale = jnp.minimum(1., max_grad_norm / jnp.maximum(grad_norm, 1e-12))
    grads = jax.tree.map(lambda g: g * scale, grads)
    first = jax.tree.map(lambda a, g: beta1 * a + (1 - beta1) * g, state['first'], grads)
    second = jax.tree.map(lambda a, g: beta2 * a + (1 - beta2) * g * g, state['second'], grads)
    step = state['step'] + 1
    def apply(p, m, v):
        m, v = m / (1 - beta1 ** step), v / (1 - beta2 ** step)
        decay = weight_decay * p if p.ndim >= 2 else 0.
        return p - learning_rate * (m / (jnp.sqrt(v) + epsilon) + decay)
    candidate = jax.tree.map(apply, params, first, second)
    finite = finite & jnp.all(jnp.stack([jnp.all(jnp.isfinite(x)) for x in jax.tree.leaves(candidate)]))
    params = jax.tree.map(lambda new, old: jnp.where(finite, new, old), candidate, params)
    candidate_state = {'first': first, 'second': second, 'step': step}
    state = jax.tree.map(lambda new, old: jnp.where(finite, new, old), candidate_state, state)
    return params, state, {'loss': loss, 'grad_norm': grad_norm, 'accepted': finite}
