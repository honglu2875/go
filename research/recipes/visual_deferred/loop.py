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


def packet(p, cache, root, state, keys, remaining, pending, c, *, horizon, depth, size, komi, version):
    """One graph: exact-Go shallow drafting, root anchor and full verification."""
    shallow = {**cache, 'keys': cache['keys'][:depth], 'values': cache['values'][:depth]}
    candidates = jax.vmap(lambda s: jax_go.candidates(s, size))
    advance = jax.vmap(lambda s, a, ss, l: jax_go.advance(s, a, size, ss, l))
    observe = jax.vmap(lambda s: jax_go.observe(s, size, komi))
    # The real replacement is already committed to the exact board, but its
    # observation may still be absent from deep KV. Produce only its cheap
    # draft prediction here; deep verification includes it in the same block
    # as the new proposals, avoiding a separate full repair per packet.
    previous_pending = pending >= 0
    pending_observation = observe(state)
    def repair_shallow(args):
        ca, prediction = args
        updated, ca = model.score_continuation(p, ca, pending_observation[:, None], jnp.maximum(pending, 0)[:, None],
            previous_pending.astype(jnp.int32), c, network_version=version, exit_depth=depth)
        return ca, select(previous_pending, jax.tree.map(lambda x: x[:, 0], updated), prediction)
    shallow, proposal_root = jax.lax.cond(jnp.any(previous_pending), repair_shallow, lambda x: x, (shallow, root))
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
    _, rows = jax.lax.scan(step, (shallow, proposal_root, state), jnp.arange(horizon))
    rows = jax.tree.map(lambda x: jnp.swapaxes(x, 0, 1), rows)
    counts = rows['appended'].sum(1, dtype=jnp.int32) + previous_pending.astype(jnp.int32)
    # Each row has either [pending, draft...] or [draft..., padding]. Packing
    # that leading optional observation preserves the exact causal chronology.
    indices = jnp.minimum(jnp.arange(horizon + 1)[None, :] + (~previous_pending)[:, None], horizon)
    batch_index = jnp.arange(pending.shape[0])[:, None]
    observations = jnp.concatenate([pending_observation[:, None], rows['observations']], 1)[batch_index, indices]
    actions = jnp.concatenate([jnp.maximum(pending, 0)[:, None], rows['actions']], 1)[batch_index, indices]
    all_predictions, verified = model.score_continuation(p, cache, observations, actions, counts,
        c, network_version=version)
    actual_root = select(previous_pending, jax.tree.map(lambda x: x[:, 0], all_predictions), root)
    prediction_indices = previous_pending[:, None].astype(jnp.int32) + jnp.arange(horizon)[None, :]
    predictions = jax.tree.map(lambda x: x[batch_index, prediction_indices], all_predictions)
    logits = jnp.concatenate([actual_root['expert_logits'][:, None], predictions['expert_logits'][:, :-1]], 1)
    target = jax.nn.softmax(jnp.where(rows['legal'], logits, -1e30), -1)
    return {**rows, 'p': target, 'previous_pending': previous_pending, 'actual_root': actual_root}, verified, predictions


def settle(p, before, root, state, rows, verified, predictions, accepted, replacement, c, *, size, komi, version):
    """Commit accepted prefix, roll back unused cache length, score replacement."""
    batch = accepted.shape[0]; index = jnp.maximum(accepted - 1, 0)
    take = lambda tree: jax.tree.map(lambda x: x[jnp.arange(batch), index], tree)
    committed = select(accepted > 0, take(rows['states']), state)
    prediction = select(accepted > 0, take(predictions), rows['actual_root'])
    added = rows['previous_pending'].astype(jnp.int32) + jnp.sum(rows['appended'] & (jnp.arange(rows['appended'].shape[1])[None, :] < accepted[:, None]), 1)
    cache = {**verified, 'lengths': before['lengths'] + added * model.layout(size, c)['stride'], 'valid': before['valid']}
    has_replacement = replacement >= 0
    following = jax.vmap(lambda s, a: jax_go.advance(s, a, size))(committed, jnp.maximum(replacement, 0))
    committed = select(has_replacement, following, committed)
    repair = has_replacement & committed['valid'] & (committed['passes'] < 2)
    return cache, prediction, committed, jnp.where(repair, replacement, -1)


def drain(p, cache, prediction, state, pending, c, *, size, komi, version):
    """Materialize every final pending replacement; the profile charges this."""
    obs = jax.vmap(lambda s: jax_go.observe(s, size, komi))(state)
    repair = pending >= 0
    def finish(args):
        ca, old = args
        updated, ca = model.score_continuation(p, ca, obs[:, None], jnp.maximum(pending, 0)[:, None],
            repair.astype(jnp.int32), c, network_version=version)
        return ca, select(repair, jax.tree.map(lambda x: x[:, 0], updated), old)
    return jax.lax.cond(jnp.any(repair), finish, lambda x: x, (cache, prediction))


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
