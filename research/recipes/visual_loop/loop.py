"""Root-anchored policy proposals with explicit retained-cache commit and repair.

The host accepts with maximal coupling. A second graph commits the chosen prefix,
rewinds speculative KV, and fully scores a residual replacement when necessary.
This intentionally pays the repair cost and makes no MCTS-equivalence assertion.
"""
import jax
import jax.numpy as jnp
from gozero import jax_go
import model
import speculate


def select(mask, new, old):
    return jax.tree.map(lambda n, o: jnp.where(mask.reshape((*mask.shape, *((1,) * (o.ndim - mask.ndim)))), n, o), new, old)


def packet(p, cache, root, state, keys, remaining, c, *, horizon, depth, size, komi, version):
    """One graph: exact-Go shallow drafting, root anchor and full verification."""
    shallow = {**cache, 'keys': cache['keys'][:depth], 'values': cache['values'][:depth]}
    candidates = jax.vmap(lambda s: jax_go.candidates(s, size))
    advance = jax.vmap(lambda s, a, ss, l: jax_go.advance(s, a, size, ss, l))
    observe = jax.vmap(lambda s: jax_go.observe(s, size, komi))
    def step(carry, index):
        ca, prediction, current = carry
        successors, legal = candidates(current)
        active = current['valid'] & (current['passes'] < 2) & (index < remaining)
        q = speculate.probability(prediction, legal, current, current['to_play'], 'expert')
        draw = jax.vmap(lambda k, ply: jax.random.gumbel(jax.random.fold_in(k, ply), (size * size + 1,)))(keys, current['ply'])
        action = jnp.argmax(jnp.where(q > 0, jnp.log(q), -jnp.inf) + draw, -1).astype(jnp.int32)
        following = select(active, advance(current, action, successors, legal), current)
        observation = observe(following)
        appended = active & following['valid'] & (following['passes'] < 2)
        prediction, ca = model.score_continuation(p, ca, observation[:, None], action[:, None],
            appended.astype(jnp.int32), c, network_version=version, exit_depth=depth)
        prediction = jax.tree.map(lambda x: x[:, 0], prediction)
        return (ca, prediction, following), {'actions': action, 'q': q, 'active': active, 'legal': legal,
            'appended': appended, 'observations': observation, 'states': following}
    # The first proposal uses the already available full target prediction.
    _, rows = jax.lax.scan(step, (shallow, root, state), jnp.arange(horizon))
    rows = jax.tree.map(lambda x: jnp.swapaxes(x, 0, 1), rows)
    counts = rows['appended'].sum(1, dtype=jnp.int32)
    predictions, verified = model.score_continuation(p, cache, rows['observations'], rows['actions'], counts,
        c, network_version=version)
    logits = jnp.concatenate([root['expert_logits'][:, None], predictions['expert_logits'][:, :-1]], 1)
    target = jax.nn.softmax(jnp.where(rows['legal'], logits, -1e30), -1)
    return {**rows, 'p': target}, verified, predictions


def settle(p, before, root, state, rows, verified, predictions, accepted, replacement, c, *, size, komi, version):
    """Commit accepted prefix, roll back unused cache length, score replacement."""
    batch = accepted.shape[0]; index = jnp.maximum(accepted - 1, 0)
    take = lambda tree: jax.tree.map(lambda x: x[jnp.arange(batch), index], tree)
    committed = select(accepted > 0, take(rows['states']), state)
    prediction = select(accepted > 0, take(predictions), root)
    added = jnp.sum(rows['appended'] & (jnp.arange(rows['appended'].shape[1])[None, :] < accepted[:, None]), 1)
    cache = {**verified, 'lengths': before['lengths'] + added * model.layout(size, c)['stride'], 'valid': before['valid']}
    has_replacement = replacement >= 0
    following = jax.vmap(lambda s, a: jax_go.advance(s, a, size))(committed, jnp.maximum(replacement, 0))
    committed = select(has_replacement, following, committed)
    obs = jax.vmap(lambda s: jax_go.observe(s, size, komi))(committed)
    repair = has_replacement & committed['valid'] & (committed['passes'] < 2)
    # An all-accepted packet may skip the repair executable's neural branch;
    # any live replacement causes a full padded batch, and that cost is charged.
    def repair_cache(args):
        ca, prediction = args
        corrected, ca = model.score_continuation(p, ca, obs[:, None], jnp.maximum(replacement, 0)[:, None],
            repair.astype(jnp.int32), c, network_version=version)
        corrected = jax.tree.map(lambda x: x[:, 0], corrected)
        return ca, select(repair, corrected, prediction)
    cache, prediction = jax.lax.cond(jnp.any(repair), repair_cache, lambda x: x, (cache, prediction))
    return cache, prediction, committed, repair


def serial(p, cache, root, state, keys, remaining, c, *, size, komi, version):
    """One genuine target-policy action, exact board step and retained full KV."""
    successors, legal = jax.vmap(lambda s: jax_go.candidates(s, size))(state)
    active = state['valid'] & (state['passes'] < 2) & (remaining > 0)
    probability = speculate.probability(root, legal, state, state['to_play'], 'expert')
    draw = jax.vmap(lambda k, ply: jax.random.gumbel(jax.random.fold_in(k, ply), (size * size + 1,)))(keys, state['ply'])
    action = jnp.argmax(jnp.where(probability > 0, jnp.log(probability), -jnp.inf) + draw, -1).astype(jnp.int32)
    updated = jax.vmap(lambda s, a, ss, l: jax_go.advance(s, a, size, ss, l))(state, action, successors, legal)
    updated = select(active, updated, state)
    obs = jax.vmap(lambda s: jax_go.observe(s, size, komi))(updated)
    appended = active & updated['valid'] & (updated['passes'] < 2)
    prediction, cache = model.score_continuation(p, cache, obs[:, None], action[:, None], appended.astype(jnp.int32), c, network_version=version)
    prediction = select(appended, jax.tree.map(lambda x: x[:, 0], prediction), root)
    return {'actions': action, 'active': active, 'appended': appended}, cache, prediction, updated
