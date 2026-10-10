"""Grouped winner-take-all ReLU dictionaries, expressed in pure JAX.

Inspired by rig's fuzzy TopK; independently expressed with ordinary autodiff.
There are K fixed groups of G features, one positive winner per group. This is
not a global top-k and does not prune the input. The choicewise implementation
issues G dense products: it is numerically sparse, not a sparse FLOP kernel.
"""
import jax
import jax.numpy as jnp


def _linear(x, w, dtype):
    return jnp.matmul(x.astype(dtype), w.astype(dtype), preferred_element_type=jnp.float32)


def _select(x, up, bias, groups, dtype):
    if type(groups) is not int or groups < 1 or up.shape[1] % groups:
        raise ValueError('Feature width must be divisible by group size')
    scores = _linear(x, up, dtype) + bias
    scores = scores.reshape(*scores.shape[:-1], -1, groups)
    winners = jnp.argmax(scores, axis=-1)
    # Gathering by argmax gives a deterministic first-index subgradient at ties.
    values = jnp.take_along_axis(scores, winners[..., None], axis=-1)[..., 0]
    return winners, jax.nn.relu(values)


def choicewise(x, up, up_bias, down, down_bias, *, groups, dtype=jnp.float32, valid=None):
    """Return the residual branch and additive per-feature activity counts.

    Statistics have [winner counts H, positive-winner counts H, live tokens].
    Validity affects only diagnostics, never forward values or gradients. The
    rematerialized loop avoids retaining a [G, tokens, K] activation stack.
    """
    d = x.shape[-1]; h = up.shape[1]; k = h // groups
    if up.shape[0] != d or down.shape != (h, d) or up_bias.shape != (h,) or down_bias.shape != (d,):
        raise ValueError('Fuzzy projection shapes differ')
    flat = x.reshape(-1, d)
    winner, value = _select(flat, up, up_bias, groups, dtype)
    live = jnp.ones(flat.shape[0], bool) if valid is None else valid.reshape(-1)
    if live.shape != (flat.shape[0],):
        raise ValueError('One validity bit per token required')
    weights = down.reshape(k, groups, d)
    counts = jnp.zeros((2, k, groups), jnp.float32)

    def one(choice, state):
        out, count = state
        selected = winner == choice
        active = jnp.where(selected, value, 0.)
        out = out + _linear(active, weights[:, choice, :], dtype)
        hard = jnp.sum((selected & live[:, None]).astype(jnp.float32), axis=0)
        positive = jnp.sum((selected & (value > 0) & live[:, None]).astype(jnp.float32), axis=0)
        count = count.at[0, :, choice].set(hard).at[1, :, choice].set(positive)
        return out, jax.lax.stop_gradient(count)

    y, counts = jax.lax.fori_loop(0, groups, jax.checkpoint(one), (jnp.zeros_like(flat, dtype=jnp.float32), counts))
    stats = jnp.concatenate((counts.reshape(-1), jnp.sum(live.astype(jnp.float32))[None]))
    return (y + down_bias).reshape(x.shape), jax.lax.stop_gradient(stats)


def reference(x, up, up_bias, down, down_bias, *, groups, dtype=jnp.float32):
    """Small-shape literal selected-row oracle; never use on full training shapes."""
    winner, value = _select(x, up, up_bias, groups, dtype)
    index = groups * jnp.arange(up.shape[1] // groups) + winner
    rows = down[index].astype(dtype)
    return jnp.einsum('...k,...kd->...d', value.astype(dtype), rows, preferred_element_type=jnp.float32) + down_bias


def diagnostics(stats, *, groups, selected, axis_name=None):
    """Batch activity summaries over live draft, board, and action tokens.

    Zero-count features are inactive in THIS batch; these are not estimates of
    permanent feature death. No statistic contributes to the training loss.
    """
    if axis_name is not None:
        stats = jax.lax.psum(stats, axis_name)
    h = groups * selected
    count = stats[..., -1]
    hard = stats[..., :h].reshape(-1, selected, groups)
    positive = stats[..., h:2*h].reshape(-1, selected, groups)
    probability = hard / jnp.maximum(count[:, None, None], 1.)
    entropy = -jnp.sum(jnp.where(probability > 0, probability * jnp.log(jnp.maximum(probability, 1e-30)), 0.), -1)
    return dict(fuzzy_positive_group_fraction=jnp.mean(jnp.sum(positive, (-2, -1)) / jnp.maximum(count * selected, 1.)),
                fuzzy_batch_inactive_feature_fraction=jnp.mean((positive == 0).astype(jnp.float32)),
                fuzzy_batch_never_winner_fraction=jnp.mean((hard == 0).astype(jnp.float32)),
                fuzzy_group_entropy=jnp.mean(entropy), fuzzy_live_tokens_per_layer=jnp.mean(count))
