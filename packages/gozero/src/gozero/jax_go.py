"""Functional positional-superko Go for compiled research rollouts.

Rust remains the reference engine. This module does no host callbacks: connected
components, captures, multi-stone suicide and exact packed history comparisons
are JAX operations. It deliberately implements only the registered area-rules
move semantics; terminal scoring remains with Rust.
"""
from functools import lru_cache
import numpy as np
import jax
import jax.numpy as jnp


@lru_cache(maxsize=32)
def geometry(size):
    if type(size) is not int or not 1 <= size <= 52:
        raise ValueError('Invalid square board size')
    n = size * size
    neighbors = np.zeros((n, 4), np.int32)
    valid = np.zeros((n, 4), bool)
    for a in range(n):
        r, c = divmod(a, size)
        for k, (x, y) in enumerate(((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))):
            valid[a, k] = 0 <= x < size and 0 <= y < size
            neighbors[a, k] = x * size + y if valid[a, k] else 0
    return neighbors, valid


def pack(stones):
    """Lossless two-bit words, rather than probabilistic position hashes."""
    n = stones.shape[-1]
    x = jnp.pad(stones.astype(jnp.uint32), [(0, 0)] * (stones.ndim - 1) + [(0, (-n) % 16)])
    x = x.reshape(*stones.shape[:-1], -1, 16)
    return jnp.sum(x << (2 * jnp.arange(16, dtype=jnp.uint32)), axis=-1, dtype=jnp.uint32)


def empty(size, history_capacity):
    if type(history_capacity) is not int or history_capacity < 2:
        raise ValueError('History needs a root and at least one move')
    geometry(size)
    n = size * size
    return {'stones': jnp.zeros(n, jnp.uint8), 'to_play': jnp.asarray(1, jnp.int32),
            'passes': jnp.asarray(0, jnp.int32), 'ply': jnp.asarray(0, jnp.int32),
            'history': jnp.zeros((history_capacity, (n + 15) // 16), jnp.uint32),
            'history_count': jnp.asarray(1, jnp.int32), 'valid': jnp.asarray(True)}


def candidates(state, size):
    """Return every point's successor stones and exact legal mask, plus pass."""
    stones, player = state['stones'], state['to_play']
    n = size * size
    adjacent, present = map(jnp.asarray, geometry(size))
    same = present & (stones[:, None] == stones[adjacent]) & (stones[:, None] != 0)
    labels = jnp.where(stones != 0, jnp.arange(n), n)
    def condition(carry):
        labels, old, iterations = carry
        return jnp.any(labels != old) & (iterations < n)
    def propagate(carry):
        labels, _, iterations = carry
        merged = jnp.minimum(labels, jnp.min(jnp.where(same, labels[adjacent], n), axis=1))
        return merged, labels, iterations + 1
    labels, _, _ = jax.lax.while_loop(condition, propagate,
        (labels, jnp.full(n, -1, jnp.int32), jnp.asarray(0, jnp.int32)))
    neighbor_empty = present & (stones[adjacent] == 0)
    liberty_bits = jnp.zeros((n + 1, n), jnp.int32).at[labels[:, None], adjacent].max(
        (neighbor_empty & (stones[:, None] != 0)).astype(jnp.int32))
    liberty_count = liberty_bits.sum(axis=1)
    last_liberty = jnp.argmax(liberty_bits, axis=1)
    point = jnp.arange(n)
    captured = ((stones[None, :] == 3 - player) & (liberty_count[labels][None, :] == 1)
                & (last_liberty[labels][None, :] == point[:, None]))
    own_neighbor = present & (stones[adjacent] == player)
    alive = (neighbor_empty.any(axis=1) | captured.any(axis=1)
             | (own_neighbor & (liberty_count[labels[adjacent]] > 1)).any(axis=1))
    own_joined = ((labels[adjacent, None] == labels[None, None, :])
                  & own_neighbor[:, :, None]).any(axis=1)
    successor = jnp.broadcast_to(stones, (n, n)).at[point, point].set(player.astype(jnp.uint8))
    remove = captured | (~alive[:, None] & (own_joined | jnp.eye(n, dtype=jnp.bool_)))
    successor = jnp.where(remove, jnp.uint8(0), successor)
    packed = pack(successor)
    repeated = (jnp.all(packed[:, None, :] == state['history'][None, :, :], axis=-1)
                & (jnp.arange(state['history'].shape[0])[None, :] < state['history_count'])).any(axis=1)
    running = state['valid'] & (state['passes'] < 2)
    legal = (stones == 0) & ~repeated & running & (state['history_count'] < state['history'].shape[0])
    return successor, jnp.concatenate([legal, running[None]])


def observe(state, size, komi, legal=None):
    if legal is None:
        _, legal = candidates(state, size)
    n = size * size
    black = state['to_play'] == 1
    constants = jnp.ones(n, jnp.float32)
    # Match the canonical host/Rust observation bytes. TPU division may lower
    # to a rounded reciprocal multiplication and differ by one float32 ULP.
    signed_komi = np.float32(float(komi) / n)
    return jnp.stack([(state['stones'] == 1).astype(jnp.float32),
                      (state['stones'] == 2).astype(jnp.float32),
                      constants * black, constants * jnp.where(black, -signed_komi, signed_komi),
                      constants * state['passes'] / 2., legal[:-1].astype(jnp.float32)], -1).reshape(size, size, 6)


def advance(state, action, size, successor=None, legal=None):
    """Invalid requests preserve every board/history field and poison valid."""
    if successor is None or legal is None:
        successor, legal = candidates(state, size)
    n = size * size
    bounded = (action >= 0) & (action <= n)
    index = jnp.clip(action, 0, n)
    good = bounded & legal[index] & state['valid']
    placing = good & (action < n)
    stones = jnp.where(placing, successor[jnp.minimum(index, n - 1)], state['stones'])
    slot = jnp.minimum(state['history_count'], state['history'].shape[0] - 1)
    history = state['history'].at[slot].set(jnp.where(placing, pack(stones), state['history'][slot]))
    return {**state, 'stones': stones, 'to_play': jnp.where(good, 3 - state['to_play'], state['to_play']),
            'passes': jnp.where(good, jnp.where(action == n, state['passes'] + 1, 0), state['passes']),
            'ply': state['ply'] + good.astype(jnp.int32), 'history': history,
            'history_count': state['history_count'] + placing.astype(jnp.int32), 'valid': state['valid'] & good}


def replay(actions, count, size, komi, history_capacity):
    """Build an exact root state in one graph from a padded complete action tape."""
    def step(state, item):
        i, action = item
        return jax.lax.cond(i < count, lambda s: advance(s, action, size), lambda s: s, state), None
    state, _ = jax.lax.scan(step, empty(size, history_capacity), (jnp.arange(actions.shape[0]), actions))
    return state, observe(state, size, komi)
