"""Early-layer draft and deep block verification with exact device Go dynamics.

A packet is one HLO invocation, including every draft action, board transition,
visual token and final target verification. Acceptance is a separate protocol:
these are policy proposals, never unverified MCTS decisions.
"""
import jax
import jax.numpy as jnp
from gozero import jax_go
if __package__:
    from . import model
else:
    import model


def probability(prediction, legal, state, view, role):
    if role == 'expert':
        logits = prediction['expert_logits']
    elif role == 'behavior':
        logits = prediction['behavior_logits']
    elif role == 'alternating':
        logits = jnp.where((state['to_play'] == view)[:, None],
                           prediction['expert_logits'], prediction['behavior_logits'])
    else:
        raise ValueError('Unknown target/draft policy role')
    # Terminal/inactive rows produce a harmless deterministic pass. They never
    # count as proposed or accepted moves.
    running = state['valid'] & (state['passes'] < 2)
    fallback = jax.nn.one_hot(jnp.full(logits.shape[0], logits.shape[-1] - 1), logits.shape[-1], dtype=jnp.bool_)
    legal = jnp.where(running[:, None], legal, fallback)
    return jax.nn.softmax(jnp.where(legal, logits, -1e30), axis=-1)


def draft(p, cache, root, state, keys, view, c, *, horizon, depth, size, komi,
          network_version, role='alternating'):
    """Autoregressive draft with exact observations; no CPU board round trips."""
    if not 1 <= horizon <= 16 or not 1 <= depth <= c['layers']:
        raise ValueError('Draft exceeds bounded horizon or layer depth')
    candidates = jax.vmap(lambda s: jax_go.candidates(s, size))
    advance = jax.vmap(lambda s, a, successors, legal: jax_go.advance(s, a, size, successors, legal))
    observe = jax.vmap(lambda s: jax_go.observe(s, size, komi))
    def step(carry, index):
        cache, prediction, state = carry
        successors, legal = candidates(state)
        q = probability(prediction, legal, state, view, role)
        draw = jax.vmap(lambda k, ply: jax.random.gumbel(jax.random.fold_in(k, ply), (size * size + 1,)))(keys, state['ply'])
        action = jnp.argmax(jnp.where(q > 0, jnp.log(q), -jnp.inf) + draw, -1).astype(jnp.int32)
        active = state['valid'] & (state['passes'] < 2)
        next_state = advance(state, action, successors, legal)
        next_state = jax.tree.map(lambda old, new: jnp.where(active.reshape((*active.shape, *((1,) * (old.ndim - 1)))), new, old), state, next_state)
        observation = observe(next_state)
        append = active & next_state['valid'] & (next_state['passes'] < 2)
        prediction, cache = model.score_continuation(p, cache, observation[:, None], action[:, None],
            append.astype(jnp.int32), c, network_version=network_version, exit_depth=depth)
        prediction = jax.tree.map(lambda x: x[:, 0], prediction)
        return (cache, prediction, next_state), {'actions': action, 'q': q, 'future_observations': observation,
            'active': active, 'appended': append, 'legal': legal, 'to_play': state['to_play']}
    final, rows = jax.lax.scan(step, (cache, root, state), jnp.arange(horizon))
    return final, jax.tree.map(lambda x: jnp.swapaxes(x, 0, 1), rows)


def packet(p, cache, deep_root, shallow_root, state, keys, view, c, *, horizon,
           draft_depth, size, komi, network_version, draft_role='alternating', target_role='expert'):
    """One shallow scan plus one deep parallel verification of its exact path."""
    shallow = {**cache, 'keys': cache['keys'][:draft_depth], 'values': cache['values'][:draft_depth]}
    (_, _, end_state), rows = draft(p, shallow, shallow_root, state, keys, view, c,
        horizon=horizon, depth=draft_depth, size=size, komi=komi, network_version=network_version, role=draft_role)
    counts = jnp.sum(rows['appended'], axis=1, dtype=jnp.int32)
    prediction, target_cache = model.score_continuation(p, cache, rows['future_observations'], rows['actions'], counts,
        c, network_version=network_version)
    all_prediction = jax.tree.map(lambda root, block: jnp.concatenate([root[:, None], block[:, :-1]], axis=1), deep_root, prediction)
    if target_role == 'expert':
        logits = all_prediction['expert_logits']
    elif target_role == 'alternating':
        logits = jnp.where((rows['to_play'] == view[:, None])[:, :, None],
            all_prediction['expert_logits'], all_prediction['behavior_logits'])
    else:
        raise ValueError('Target must be expert or alternating')
    p_target = jax.nn.softmax(jnp.where(rows['legal'], logits, -1e30), axis=-1)
    return {**rows, 'p': p_target}, target_cache, end_state


def resolve(rows, uniforms, residual_uniforms):
    """Maximal coupling to the target policy, returning the first replacement.

Random streams for proposals, acceptance and residual sampling must be disjoint.
No bonus token is claimed. A caller commits only the contiguous accepted prefix
and the replacement (if any), and discards all subsequent speculative state.
"""
    import numpy as np
    q, p, actions, active = [np.asarray(rows[k]) for k in ('q', 'p', 'actions', 'active')]
    if (q.shape != p.shape or actions.shape != active.shape or q.shape[:-1] != actions.shape
            or uniforms.shape != actions.shape or residual_uniforms.shape != actions.shape
            or not np.isfinite(q).all() or not np.isfinite(p).all()
            or np.any(q < 0) or np.any(p < 0)
            or not np.allclose(q.sum(-1), 1., atol=2e-6) or not np.allclose(p.sum(-1), 1., atol=2e-6)
            or np.any((uniforms < 0) | (uniforms >= 1)) or np.any((residual_uniforms < 0) | (residual_uniforms >= 1))):
        raise ValueError('Malformed speculative probability or randomness packet')
    results = []
    for game in range(actions.shape[0]):
        moves = []; accepted = 0; replacement = None
        for t in range(actions.shape[1]):
            if not active[game, t]:
                break
            a = int(actions[game, t]); qa = float(q[game, t, a]); pa = float(p[game, t, a])
            if qa <= 0:
                raise ValueError('Draft selected a zero-mass action')
            if uniforms[game, t] * qa < pa:
                moves.append(a); accepted += 1
            else:
                residual = np.maximum(p[game, t].astype(np.float64) - q[game, t].astype(np.float64), 0.)
                if residual.sum() <= 0:
                    raise ValueError('Rejected proposal has no positive residual mass')
                cumulative = np.cumsum(residual / residual.sum()); cumulative[-1] = 1.
                replacement = int(np.searchsorted(cumulative, residual_uniforms[game, t], side='right'))
                moves.append(replacement); break
        results.append({'actions': moves, 'accepted_draft_moves': accepted, 'replacement': replacement})
    return results
