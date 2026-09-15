"""Expert-only architecture comparison; the observed-opponent model is fixed."""
import math
import jax
import jax.numpy as jnp
import history_model
import spatial_model
from state_features import batch_planes, planes


def initialize(seed, c):
    if c['architecture'] == 'history':
        return {k: v for k, v in history_model.initialize(seed, c).items() if not k.startswith('behavior_')}
    if c['architecture'] == 'state':
        return spatial_model.initialize(seed, c)
    raise ValueError('Unknown expert architecture')


def predictions(p, batch, c):
    if c['architecture'] == 'state':
        return spatial_model.apply(p, batch_planes(batch, c), c)
    if c['architecture'] != 'history':
        raise ValueError('Unknown expert architecture')
    h, _ = history_model.prefill(p, batch['tokens'], batch['lengths'], c)
    players = jnp.broadcast_to(1 + jnp.arange(batch['tokens'].shape[1]) % 2, batch['tokens'].shape)
    board = history_model.board_features(p, batch['stones'], players, c)
    return history_heads(p, h, board)


def history_heads(p, h, board):
    pooled = jnp.mean(board, axis=-2)
    query = h @ p['board_query']
    spatial = jnp.sum(board * query[..., None, :], axis=-1) / math.sqrt(board.shape[-1]) + (board @ p['board_point'])[..., 0]
    play = h @ p['play'] + jnp.concatenate((spatial, pooled @ p['board_pass']), axis=-1)
    value = jnp.tanh((h @ p['value'] + pooled @ p['board_value'])[..., 0])
    return play, value


def leaf_predictions(p, tokens, lengths, stones, legal, previous_pass, c):
    """Evaluate only requested native boards; state inference skips history attention."""
    players = 1 + lengths % 2
    if c['architecture'] == 'state':
        return spatial_model.apply(p, planes(stones, players, legal[..., :-1], previous_pass, c['size'], c['komi']), c)
    if c['architecture'] != 'history':
        raise ValueError('Unknown expert architecture')
    h, _ = history_model.prefill(p, tokens, lengths, c)
    last = h[jnp.arange(tokens.shape[0]), lengths]
    board = history_model.board_features(p, stones, players, c)
    return history_heads(p, last, board)


def losses(p, batch, c):
    play, value = predictions(p, batch, c)
    expert_mask, value_mask = batch['expert_mask'], batch['value_mask']
    expert_count, value_count = expert_mask.sum(), value_mask.sum()
    expert = -jnp.sum(expert_mask * jnp.sum(batch['policies'] * jax.nn.log_softmax(play), -1)) / jnp.maximum(1., expert_count)
    value_loss = jnp.sum(value_mask * (value - batch['values']) ** 2) / jnp.maximum(1., value_count)
    entropy = -jnp.sum(batch['policies'] * jnp.log(jnp.maximum(batch['policies'], 1e-30)), -1)
    return expert + value_loss, {'play_loss': expert, 'value_loss': value_loss,
        'expert_tokens': expert_count, 'value_tokens': value_count,
        'target_entropy': jnp.sum(expert_mask * entropy) / jnp.maximum(1., expert_count),
        'play_top1': jnp.sum(expert_mask * (jnp.argmax(play, -1) == jnp.argmax(batch['policies'], -1))) / jnp.maximum(1., expert_count),
        'illegal_probability': jnp.sum(expert_mask * jnp.sum(jax.nn.softmax(play) * ~batch['legal'], -1)) / jnp.maximum(1., expert_count)}
