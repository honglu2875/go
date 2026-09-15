"""Population-weighted held-out policy quality and shallow/deep agreement."""
import jax
import jax.numpy as jnp
if __package__:
    from . import model
else:
    import model


def totals(params, batch, config, exit_depths, *, axis_name=None):
    prediction = model.forward(params, batch['observations'], batch['actions'], batch['counts'], config,
                                  return_exits=exit_depths)
    target, exits = prediction if exit_depths else (prediction, {})
    size = batch['observations'].shape[2]
    live = jnp.arange(batch['actions'].shape[1])[None, :] < batch['counts'][:, None]
    legal = jnp.concatenate([batch['observations'][..., 5].reshape(*live.shape, size*size) > .5,
                             jnp.ones((*live.shape, 1), jnp.bool_)], -1)
    result = {}
    def add(name, values, mask):
        result[name] = jnp.sum(values * mask * live)
    for role in ('expert', 'behavior'):
        mask = batch[role + '_mask']
        add(role + '_count', jnp.ones_like(mask), mask)
        logp = jax.nn.log_softmax(jnp.where(legal, target[role + '_logits'], -1e9), -1)
        probability = jnp.exp(logp)
        if role == 'expert':
            pi = batch['expert_policies']
            add('expert_ce', -jnp.sum(pi * logp, -1), mask)
            add('expert_target_entropy', -jnp.sum(pi * jnp.log(jnp.maximum(pi, 1e-30)), -1), mask)
        else:
            add('behavior_ce', -jnp.take_along_axis(logp, batch['actions'][..., None], -1)[..., 0], mask)
        add(role + '_entropy', -jnp.sum(probability * logp, -1), mask)
        for depth, prediction in exits.items():
            shallow_log = jax.nn.log_softmax(jnp.where(legal, prediction[role + '_logits'], -1e9), -1)
            prefix = f'{role}_exit_{depth}_'
            add(prefix + 'target_kl', jnp.sum(probability * (logp - shallow_log), -1), mask)
            add(prefix + 'greedy_agreement', jnp.argmax(logp, -1) == jnp.argmax(shallow_log, -1), mask)
            # Abstract one-step rejection-sampling acceptance on EXACT shared
            # histories. Approximate-board drafts need separate rollout tests.
            add(prefix + 'distribution_overlap', jnp.minimum(probability, jnp.exp(shallow_log)).sum(-1), mask)
    add('value_count', jnp.ones_like(batch['value_mask']), batch['value_mask'])
    add('value_mse', (target['value'] - batch['outcomes']) ** 2, batch['value_mask'])
    if axis_name is not None:
        result = jax.tree.map(lambda x: jax.lax.psum(x, axis_name), result)
    return result


def averages(raw):
    result = {k: v for k, v in raw.items() if k.endswith('_count')}
    for k, v in raw.items():
        if not k.endswith('_count'):
            result[k] = v / max(1., raw[k.split('_')[0] + '_count'])
    result['expert_kl'] = result['expert_ce'] - result['expert_target_entropy']
    return result
