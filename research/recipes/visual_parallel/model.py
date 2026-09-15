"""Plain-JAX observation/action decoder with continuous visual patch tokens.

The expert output embedding is tied to the action input embedding. The behavior
adapter/output is separate. Observations are supplied exact inputs, never
silently predicted dynamics. score_continuation evaluates a known draft block
in parallel; it does not accept moves or implement MCTS verification.
"""
from functools import lru_cache
import math
import jax
import jax.numpy as jnp

if __package__:
    from .config import validate_model
else:
    from config import validate_model


def layout(size, c):
    if not 1 <= size <= c['max_board_size']:
        raise ValueError('Board exceeds configured maximum')
    side = (size + c['patch_size'] - 1) // c['patch_size']
    patches = side * side
    return {'patch_side': side, 'patches': patches, 'stride': patches + 2,
            'capacity': c['max_positions'] * (patches + 2)}


def initialize(seed, c):
    validate_model(c)
    key = jax.random.key(seed)
    d, hidden = c['width'], c['mlp_hidden']
    kv = c['kv_heads'] * (d // c['heads'])

    def weight(shape, scale=1.):
        nonlocal key
        key, draw = jax.random.split(key)
        return jax.random.normal(draw, shape, jnp.float32) * (scale / math.sqrt(shape[0]))

    def embedding(shape):
        return weight(shape, .02 * math.sqrt(shape[0]))

    blocks = []
    for _ in range(c['layers']):
        blocks.append({'attn_norm': jnp.ones(d, jnp.float32),
                       'q': weight((d, d)), 'kv': weight((d, 2 * kv)),
                       'out': weight((d, d), 1 / math.sqrt(2 * c['layers'])),
                       'mlp_norm': jnp.ones(d, jnp.float32),
                       'gate': weight((d, hidden)), 'up': weight((d, hidden)),
                       'down': weight((hidden, d), 1 / math.sqrt(2 * c['layers']))})
    max_patches = (c['max_board_size'] + c['patch_size'] - 1) // c['patch_size']
    vocab = c['max_board_size'] ** 2 + 1
    return {'patch': weight((c['patch_size'] ** 2 * (c['input_channels'] + 1), d)),
            'patch_bias': jnp.zeros(d, jnp.float32),
            'patch_rows': embedding((max_patches, d)), 'patch_cols': embedding((max_patches, d)),
            'actions': embedding((vocab, d)), 'types': embedding((3, d)),
            'readout': embedding((1, d))[0], 'blocks': blocks, 'norm': jnp.ones(d, jnp.float32),
            'behavior_norm': jnp.ones(d, jnp.float32), 'behavior_adapter': weight((d, d)),
            'behavior_out': weight((d, vocab), .1),
            'value': weight((d, 3), .1), 'score': weight((d, 1), .1),
            'ownership': weight((d, c['patch_size'] ** 2), .1)}


def parameter_schema(c):
    abstract = jax.eval_shape(lambda: initialize(0, c))
    leaves = jax.tree_util.tree_flatten_with_path(abstract)[0]
    return [{'path': jax.tree_util.keystr(path), 'shape': list(a.shape),
             'dtype': str(a.dtype), 'elements': math.prod(a.shape)} for path, a in leaves]


def linear(x, w, c):
    dtype = {'float32': jnp.float32, 'bfloat16': jnp.bfloat16}[c['dtype']]
    return jnp.matmul(x.astype(dtype), w.astype(dtype),
                      precision=jax.lax.Precision.HIGHEST, preferred_element_type=jnp.float32)


def norm(x, scale, c):
    x = x.astype(jnp.float32)
    return x * jax.lax.rsqrt(jnp.mean(x * x, axis=-1, keepdims=True) + c['norm_epsilon']) * scale


def action_indices(size, c):
    points = jnp.arange(size * size)
    return jnp.concatenate([points // size * c['max_board_size'] + points % size,
                            jnp.array([c['max_board_size'] ** 2])])


def observation_tokens(p, observations, c):
    if observations.ndim != 5 or observations.shape[-1] != c['input_channels']:
        raise ValueError('Expected [batch, positions, height, width, six channels]')
    b, t, size, width, _ = observations.shape
    if width != size:
        raise ValueError('Expected square boards')
    spec = layout(size, c)
    side, patch = spec['patch_side'], c['patch_size']
    # The explicit on-board channel distinguishes padding from empty points.
    x = jnp.concatenate([observations, jnp.ones((*observations.shape[:-1], 1), observations.dtype)], -1)
    x = jnp.pad(x, ((0, 0), (0, 0), (0, side * patch - size),
                    (0, side * patch - size), (0, 0)))
    x = x.reshape(b, t, side, patch, side, patch, c['input_channels'] + 1)
    x = x.transpose(0, 1, 2, 4, 3, 5, 6).reshape(b, t, side * side, -1)
    spatial = (p['patch_rows'][:side, None] + p['patch_cols'][None, :side]).reshape(side * side, -1)
    return linear(x, p['patch'], c) + p['patch_bias'] + spatial + p['types'][0]


def action_tokens(p, actions, size, c):
    return p['actions'][action_indices(size, c)[actions]] + p['types'][2]


def pack(p, observations, actions, c):
    patches = observation_tokens(p, observations, c)
    b, t, _, d = patches.shape
    if actions.shape != (b, t) or t > c['max_positions']:
        raise ValueError('Action/observation shapes or context differ')
    readout = jnp.broadcast_to(p['readout'] + p['types'][1], (b, t, 1, d))
    act = action_tokens(p, actions, observations.shape[2], c)[:, :, None, :]
    return jnp.concatenate([patches, readout, act], axis=2).reshape(b, -1, d)


def rope(x, positions, c):
    frequencies = c['rope_theta'] ** (-jnp.arange(0, x.shape[-1], 2, dtype=jnp.float32) / x.shape[-1])
    angle = positions[..., None, None] * frequencies
    even, odd = x[..., 0::2].astype(jnp.float32), x[..., 1::2].astype(jnp.float32)
    rotated = jnp.stack([even * jnp.cos(angle) - odd * jnp.sin(angle),
                         even * jnp.sin(angle) + odd * jnp.cos(angle)], axis=-1)
    return rotated.reshape(x.shape).astype(x.dtype)


def project(x, block, positions, c):
    b, t, _ = x.shape
    h, kh, dim = c['heads'], c['kv_heads'], c['width'] // c['heads']
    dtype = {'float32': jnp.float32, 'bfloat16': jnp.bfloat16}[c['dtype']]
    y = norm(x, block['attn_norm'], c)
    q = linear(y, block['q'], c).reshape(b, t, h, dim).astype(dtype)
    kv = linear(y, block['kv'], c).reshape(b, t, 2, kh, dim).astype(dtype)
    return rope(q, positions, c), rope(kv[:, :, 0], positions, c), kv[:, :, 1]


def dense_attention(q, k, v, query_positions, key_lengths):
    groups = q.shape[2] // k.shape[2]
    qg = q.reshape(q.shape[0], q.shape[1], k.shape[2], groups, q.shape[-1])
    scores = jnp.einsum('btkgd,bskd->bkgts', qg, k,
                        precision=jax.lax.Precision.HIGHEST,
                        preferred_element_type=jnp.float32) / math.sqrt(q.shape[-1])
    keys = jnp.arange(k.shape[1])[None, None, :]
    allowed = (keys <= query_positions[:, :, None]) & (keys < key_lengths[:, None, None])
    # Empty/inactive rows are zeroed after attention; avoid an all -inf softmax.
    weights = jax.nn.softmax(jnp.where(allowed[:, None, None], scores, -1e30), axis=-1).astype(v.dtype)
    out = jnp.einsum('bkgts,bskd->btkgd', weights, v,
                     precision=jax.lax.Precision.HIGHEST, preferred_element_type=jnp.float32)
    return out.reshape(q.shape)


@lru_cache(maxsize=32)
def splash_kernel(length, heads):
    from jax.experimental.pallas.ops.tpu import splash_attention as splash
    mask = splash.MultiHeadMask([splash.CausalMask((length, length)) for _ in range(heads)])
    blocks = splash.BlockSizes(block_q=128, block_kv=128, block_kv_compute=128,
                               block_q_dkv=128, block_kv_dkv=128, block_kv_dkv_compute=128,
                               block_q_dq=128, block_kv_dq=128)
    # The cached object owns JAX mask arrays. Construct them eagerly so a remat
    # or autodiff trace cannot escape into the process-wide kernel cache.
    with jax.ensure_compile_time_eval():
        return splash.make_splash_mha(mask, block_sizes=blocks, head_shards=1, q_seq_shards=1)


def full_attention(q, k, v, c):
    if c['attention_backend'] == 'xla':
        positions = jnp.broadcast_to(jnp.arange(q.shape[1]), q.shape[:2])
        return dense_attention(q, k, v, positions, jnp.full((q.shape[0],), k.shape[1]))
    length = ((q.shape[1] + 127) // 128) * 128
    def pad(a):
        return jnp.pad(a.transpose(0, 2, 1, 3), ((0, 0), (0, 0), (0, length - a.shape[1]), (0, 0)))
    # Splash takes already scaled Q. GQA heads remain compact in K and V.
    result = jax.vmap(splash_kernel(length, c['heads']))(
        pad(q) / math.sqrt(q.shape[-1]), pad(k), pad(v))
    return result.transpose(0, 2, 1, 3)[:, :q.shape[1]].astype(jnp.float32)


def finish_block(x, attended, block, c):
    x = x + linear(attended.reshape(x.shape), block['out'], c)
    y = norm(x, block['mlp_norm'], c)
    gated = jax.nn.silu(linear(y, block['gate'], c)) * linear(y, block['up'], c)
    return x + linear(gated, block['down'], c)


def checked_depth(p, c, depth):
    depth = c['layers'] if depth is None else depth
    if type(depth) is not int or not 1 <= depth <= c['layers'] or len(p['blocks']) != c['layers']:
        raise ValueError('Invalid decoder exit depth or parameter layer count')
    return depth


def forward(p, observations, actions, counts, c, *, with_cache=False, network_version=0,
            cache_positions=None, exit_depth=None, return_exits=()):
    depth = checked_depth(p, c, exit_depth)
    if len(set(return_exits)) != len(return_exits) or any(type(e) is not int or not 1 <= e < depth for e in return_exits):
        raise ValueError('Auxiliary exits must be distinct layers before the target exit')
    x = pack(p, observations, actions, c)
    size = observations.shape[2]
    spec = layout(size, c)
    cache_positions = c['max_positions'] if cache_positions is None else cache_positions
    if with_cache and not observations.shape[1] <= cache_positions <= c['max_positions']:
        raise ValueError('Cache bucket cannot contain the prefix or exceeds context')
    capacity = cache_positions * spec['stride']
    positions = jnp.broadcast_to(jnp.arange(x.shape[1]), x.shape[:2])
    keys, values = [], []
    def block_apply(x, block):
        q, k, v = project(x, block, positions, c)
        return finish_block(x, full_attention(q, k, v, c), block, c), k, v
    run = jax.checkpoint(block_apply) if c['rematerialize'] else block_apply
    exits = {}
    def read_predictions(hidden):
        states = norm(hidden, p['norm'], c).reshape(observations.shape[0], observations.shape[1], spec['stride'], c['width'])
        prediction = heads(p, states[:, :, spec['patches']], states[:, :, :spec['patches']], size, c)
        live = jnp.arange(observations.shape[1])[None, :] < counts[:, None]
        return jax.tree.map(lambda a: jnp.where(live.reshape((*live.shape, *((1,) * (a.ndim - 2)))), a, 0.), prediction)
    for layer, block in enumerate(p['blocks'][:depth], 1):
        x, k, v = run(x, block)
        if with_cache:
            keys.append(k); values.append(v)
        if layer in return_exits:
            exits[layer] = read_predictions(x)
    predictions = read_predictions(x)
    if return_exits:
        predictions = (predictions, exits)
    if not with_cache:
        return predictions
    lengths = counts * spec['stride'] - 1  # Final action is unknown and excluded.
    good = (counts > 0) & (counts <= observations.shape[1])
    lengths = jnp.where(good, lengths, 0)
    def store(a):
        valid_tokens = jnp.arange(a.shape[1])[None, :] < lengths[:, None]
        a = jnp.where(valid_tokens[..., None, None], a, 0)
        return jnp.pad(a, ((0, 0), (0, capacity - a.shape[1]), (0, 0), (0, 0)))
    return predictions, {'keys': tuple(map(store, keys)), 'values': tuple(map(store, values)),
                         'lengths': lengths, 'valid': good,
                         'network_version': jnp.asarray(network_version, jnp.uint32)}


def heads(p, readout, patches, size, c):
    indices = action_indices(size, c)
    expert = linear(readout, p['actions'][indices].T, c)
    behavior = readout if c['behavior_updates_trunk'] else jax.lax.stop_gradient(readout)
    behavior = behavior + jax.nn.silu(linear(norm(behavior, p['behavior_norm'], c), p['behavior_adapter'], c))
    behavior = linear(behavior, p['behavior_out'][:, indices], c)
    value_logits = linear(readout, p['value'], c)
    probability = jax.nn.softmax(value_logits, -1)
    # Ownership retains patch-local detail and receives the whole-board readout.
    owner = linear(patches + readout[..., None, :], p['ownership'], c)
    side, patch = layout(size, c)['patch_side'], c['patch_size']
    owner = owner.reshape(*owner.shape[:-2], side, side, patch, patch)
    lead = tuple(range(owner.ndim - 4))
    owner = owner.transpose(*lead, owner.ndim - 4, owner.ndim - 2, owner.ndim - 3, owner.ndim - 1)
    owner = owner.reshape(*readout.shape[:-1], side * patch, side * patch)[..., :size, :size]
    return {'expert_logits': expert, 'behavior_logits': behavior, 'value_logits': value_logits,
            'value': probability[..., 2] - probability[..., 0],
            'score': jnp.tanh(linear(readout, p['score'], c)[..., 0]), 'ownership_logits': owner}


def append_embeddings(p, cache, embeddings, token_counts, allowed_phase, c, network_version, exit_depth=None, cached_backend='xla'):
    """Bounded append: an invalid request flags the row and preserves its buffers."""
    depth = checked_depth(p, c, exit_depth)
    if cached_backend not in ('xla', 'ragged_paged'):
        raise ValueError('Unknown cached attention implementation')
    if len(cache['keys']) != depth or len(cache['values']) != depth:
        raise ValueError('Cache layer count differs from requested decoder exit')
    b, qlen, _ = embeddings.shape
    capacity = cache['keys'][0].shape[1]
    active = token_counts != 0
    good = (cache['valid'] & allowed_phase & (token_counts >= 0) & (token_counts <= qlen)
            & (cache['lengths'] + token_counts <= capacity)
            & (cache['network_version'] == jnp.asarray(network_version, jnp.uint32)))
    write = active & good
    positions = cache['lengths'][:, None] + jnp.arange(qlen)[None, :]
    new_lengths = cache['lengths'] + jnp.where(write, token_counts, 0)
    update_mask = write[:, None] & (jnp.arange(qlen)[None, :] < token_counts[:, None])
    rows = jnp.arange(b)[:, None]
    slots = jnp.clip(positions, 0, capacity - 1)
    x = embeddings
    keys, values = [], []
    for block, old_k, old_v in zip(p['blocks'][:depth], cache['keys'], cache['values']):
        q, k, v = project(x, block, positions, c)
        k = old_k.at[rows, slots].set(jnp.where(update_mask[..., None, None], k, old_k[rows, slots]))
        v = old_v.at[rows, slots].set(jnp.where(update_mask[..., None, None], v, old_v[rows, slots]))
        if cached_backend == 'xla':
            attended = dense_attention(q, k, v, positions, new_lengths)
        else:
            if __package__:
                from .cached_attention import attention
            else:
                from cached_attention import attention
            attended = attention(q, k, v, positions)
        x = finish_block(x, attended, block, c)
        keys.append(k); values.append(v)
    updated = {**cache, 'keys': tuple(keys), 'values': tuple(values), 'lengths': new_lengths,
               'valid': cache['valid'] & (~active | good)}
    x = norm(x, p['norm'], c)
    x = jnp.where(update_mask[..., None], x, 0)
    return x, updated


def score_continuation(p, cache, future_observations, proposed_actions, counts, c, *, network_version=0, exit_depth=None, cached_backend='xla'):
    """Score supplied draft edges [a_t, obs_(t+1)] as a single causal block.

    Root-action probabilities come from the preceding root prediction. Returned
    rows predict the action AFTER each supplied future observation. No acceptance
    rule is applied here. Each row is an independent game/branch with a prefix.
    """
    patches = observation_tokens(p, future_observations, c)
    b, t, k, d = patches.shape
    size = future_observations.shape[2]
    spec = layout(size, c)
    if proposed_actions.shape != (b, t) or counts.shape != (b,):
        raise ValueError('Draft observations/actions/counts disagree')
    capacity = cache['keys'][0].shape[1]
    if capacity % spec['stride'] or capacity > spec['capacity']:
        raise ValueError('Cache board/context differs')
    act = action_tokens(p, proposed_actions, size, c)[:, :, None, :]
    readout = jnp.broadcast_to(p['readout'] + p['types'][1], (b, t, 1, d))
    x = jnp.concatenate([act, patches, readout], 2).reshape(b, -1, d)
    phase = cache['lengths'] % spec['stride'] == spec['stride'] - 1
    phase = phase & (counts >= 0)
    action_valid = (proposed_actions >= 0) & (proposed_actions <= size * size)
    phase = phase & jnp.all(action_valid | (jnp.arange(t)[None, :] >= counts[:, None]), axis=1)
    x, updated = append_embeddings(p, cache, x, counts * spec['stride'], phase, c, network_version, exit_depth, cached_backend)
    states = x.reshape(b, t, spec['stride'], d)
    out = heads(p, states[:, :, -1], states[:, :, 1:-1], size, c)
    valid = (jnp.arange(t)[None, :] < counts[:, None]) & updated['valid'][:, None]
    out = jax.tree.map(lambda a: jnp.where(valid.reshape((*valid.shape, *((1,) * (a.ndim - 2)))), a, 0.), out)
    return out, updated


def supervised_losses(prediction, batch, c, *, axis_name=None):
    """MCTS policy targets and observed actions remain distinct objectives."""
    size = batch['observations'].shape[2]
    live = jnp.arange(batch['actions'].shape[1])[None, :] < batch['counts'][:, None]
    legal = jnp.concatenate([batch['observations'][..., 5].reshape(*live.shape, size * size) > .5,
                             jnp.ones((*live.shape, 1), jnp.bool_)], -1)
    def average(values, mask):
        mask = mask * live
        numerator, denominator = jnp.sum(values * mask), jnp.sum(mask)
        if axis_name is not None:
            numerator = jax.lax.psum(numerator, axis_name)
            denominator = jax.lax.psum(denominator, axis_name)
        return numerator / jnp.maximum(1., denominator)
    expert_logp = jax.nn.log_softmax(jnp.where(legal, prediction['expert_logits'], -1e9), -1)
    behavior_logp = jax.nn.log_softmax(jnp.where(legal, prediction['behavior_logits'], -1e9), -1)
    expert = average(-jnp.sum(batch['expert_policies'] * expert_logp, -1), batch['expert_mask'])
    behavior = average(-jnp.take_along_axis(behavior_logp, batch['actions'][..., None], -1)[..., 0], batch['behavior_mask'])
    value = average(jnp.square(prediction['value'] - batch['outcomes']), batch['value_mask'])
    score = average(jnp.square(prediction['score'] - batch['scores']), batch['score_mask'])
    owner = prediction['ownership_logits']
    owner_loss = jnp.mean(jax.nn.softplus(2 * owner) - (batch['ownership'] + 1) * owner, (-2, -1))
    ownership = average(owner_loss, batch['ownership_mask'])
    metrics = {'expert_loss': expert, 'behavior_loss': behavior, 'value_loss': value,
               'score_loss': score, 'ownership_loss': ownership}
    total = sum(batch['loss_weights'][i] * metrics[key] for i, key in enumerate(metrics))
    return total, metrics


def exit_distillation(teacher, students, batch, *, axis_name=None, temperature=1.):
    """Stopped deep policies teach the same expert/behavior heads at early exits.

    This is an auxiliary consistency objective, not a Q or advantage estimate.
    Both policies use the exact observed legal mask. No target action is an input.
    """
    if not students or not math.isfinite(temperature) or temperature <= 0:
        raise ValueError('Distillation needs exits and a positive temperature')
    size = batch['observations'].shape[2]
    live = jnp.arange(batch['actions'].shape[1])[None, :] < batch['counts'][:, None]
    legal = jnp.concatenate([batch['observations'][..., 5].reshape(*live.shape, size * size) > .5,
                             jnp.ones((*live.shape, 1), jnp.bool_)], -1)
    metrics = {}
    total = 0.
    for role in ['expert', 'behavior']:
        target_log = jax.lax.stop_gradient(jax.nn.log_softmax(jnp.where(legal, teacher[role + '_logits'] / temperature, -1e9), -1))
        target = jnp.exp(target_log)
        mask = live * batch[role + '_mask']
        denominator = jnp.sum(mask)
        if axis_name is not None:
            denominator = jax.lax.psum(denominator, axis_name)
        for depth, prediction in students.items():
            logp = jax.nn.log_softmax(jnp.where(legal, prediction[role + '_logits'] / temperature, -1e9), -1)
            divergence = jnp.sum(target * (target_log - logp), -1) * temperature ** 2
            numerator = jnp.sum(divergence * mask)
            if axis_name is not None:
                numerator = jax.lax.psum(numerator, axis_name)
            value = numerator / jnp.maximum(1., denominator)
            metrics[f'exit_{depth}_{role}_kl'] = value
            total = total + value / len(students)
    return total, metrics


def losses(p, batch, c, *, axis_name=None, exit_depths=(), exit_loss_weight=0., exit_temperature=1.):
    if not math.isfinite(exit_loss_weight) or exit_loss_weight < 0 or (exit_loss_weight > 0 and not exit_depths):
        raise ValueError('Invalid auxiliary exit loss configuration')
    prediction = forward(p, batch['observations'], batch['actions'], batch['counts'], c,
                         return_exits=exit_depths)
    if not exit_depths:
        return supervised_losses(prediction, batch, c, axis_name=axis_name)
    teacher, students = prediction
    total, metrics = supervised_losses(teacher, batch, c, axis_name=axis_name)
    auxiliary, extra = exit_distillation(teacher, students, batch, axis_name=axis_name, temperature=exit_temperature)
    return total + exit_loss_weight * auxiliary, {**metrics, **extra, 'exit_distillation_loss': auxiliary}
