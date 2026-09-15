"""Exact state inputs shared by offline batches and native search leaves."""
import numpy as np

CHANNELS = 9


def planes(stones, players, legal, previous_pass, size, komi):
    """Relative stones, legality, pass/turn/komi context, edge distance and size."""
    import jax.numpy as jnp
    shape = stones.shape[:-1]
    if stones.shape[-1] != size * size or legal.shape != stones.shape:
        raise ValueError('State geometry differs')
    if players.shape != shape or previous_pass.shape != shape:
        raise ValueError('State context shape differs')
    own = stones == players[..., None]
    other = (stones != 0) & ~own
    def broadcast(value):
        return jnp.broadcast_to(value[..., None], stones.shape)
    points = jnp.arange(size * size)
    edge = jnp.minimum(jnp.minimum(points // size, size - 1 - points // size),
                       jnp.minimum(points % size, size - 1 - points % size)) / max(1, size - 1)
    signed_komi = jnp.where(players == 2, komi, -komi) / (size * size)
    result = jnp.stack((own, other, stones == 0, legal, broadcast(previous_pass),
                        broadcast(players == 1), broadcast(signed_komi),
                        jnp.broadcast_to(edge, stones.shape), jnp.full(stones.shape, size / 19.)), axis=-1)
    return result.astype(jnp.float32).reshape(*shape, size, size, CHANNELS)


def batch_planes(batch, c):
    import jax.numpy as jnp
    players = jnp.broadcast_to(1 + jnp.arange(batch['tokens'].shape[1]) % 2, batch['tokens'].shape)
    previous_pass = batch['tokens'] == c['size'] ** 2
    return planes(batch['stones'], players, batch['legal'][..., :-1], previous_pass, c['size'], c['komi'])


def native_inputs(features, history, *, size, history_planes, komi):
    """Reject inconsistent native/history receipts before constructing NN inputs.

    Native boards and superko legality remain authoritative. This conversion
    applies only to nonterminal pending leaves with an initially empty board.
    """
    f = np.asarray(features, dtype=np.float32).reshape(size * size, 2 * history_planes + 4)
    moves = np.asarray(history, dtype=np.int32)
    if moves.ndim != 1 or np.any(moves < 0) or np.any(moves > size * size) or not np.isfinite(f).all():
        raise ValueError('Invalid native history or features')
    player = 1 + len(moves) % 2
    passed = bool(len(moves) and moves[-1] == size * size)
    signed_komi = np.float32(komi) / np.float32(size * size) * (1 if player == 2 else -1)
    if (not np.isin(f[:, :2], [0., 1.]).all() or np.any(f[:, :2].sum(axis=-1) > 1)
            or not np.isin(f[:, -1], [0., 1.]).all() or not np.all(f[:, -4] == float(player == 1))
            or not np.all(f[:, -2] == .5 * passed)
            or not np.allclose(f[:, -3], signed_komi, rtol=0, atol=1e-7)):
        raise ValueError('Native stones, turn, komi, pass or legal features differ')
    stones = np.where(f[:, 0] > .5, player, np.where(f[:, 1] > .5, 3 - player, 0)).astype(np.uint8)
    return stones[None], np.asarray([player], np.int32), f[None, :, -1] > .5, np.asarray([passed])
