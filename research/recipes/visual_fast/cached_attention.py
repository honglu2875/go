"""Adapt the pinned JAX ragged paged attention kernel to dense Go KV storage.

The first implementation keeps the existing cache contract and explicitly packs
interleaved pages. Packing and padding cost are part of the benchmark. A future
native paged cache may remove them; this wrapper does not claim to do so.
"""
import math
import jax
import jax.numpy as jnp


def inputs(q, k, v, positions, *, page_size=128, block_queries=128):
    b, queries, heads, dim = q.shape
    if k.shape != v.shape or k.shape[0] != b or k.shape[-1] != dim or positions.shape != (b, queries):
        raise ValueError('Cached attention tensor shapes differ')
    # Every physical query slot is presented, including inactive output slots.
    # Give it its logical causal endpoint. Extra keys are zero padding and
    # cannot enter a live query before its own absolute position. This permits
    # per-row accepted counts while retaining fixed compiled query dimensions.
    capacity = ((k.shape[1] + queries + page_size - 1) // page_size) * page_size
    def pad(x):
        return jnp.pad(x, ((0, 0), (0, capacity - x.shape[1]), (0, 0), (0, 0)))
    # This pinned kernel's accumulator broadcasting requires a head dimension
    # divisible by 128, even though its public validator permits 64. Zero-pad
    # dot operands and keep the ORIGINAL scale; this adds traffic, not features.
    padded_dim = ((dim + 127) // 128) * 128
    def pad_head(x):
        return jnp.pad(x, ((0, 0), (0, 0), (0, 0), (0, padded_dim - dim)))
    pages = jnp.stack([pad_head(pad(k)), pad_head(pad(v))], axis=3).reshape(-1, page_size, 2 * k.shape[2], padded_dim)
    page_indices = jnp.arange(b * (capacity // page_size), dtype=jnp.int32).reshape(b, -1)
    packed = pad_head(q).reshape(b * queries, heads, padded_dim)
    padded_queries = ((b * queries + block_queries - 1) // block_queries) * block_queries
    packed = jnp.pad(packed, ((0, padded_queries - packed.shape[0]), (0, 0), (0, 0)))
    kv_lengths = positions[:, 0].astype(jnp.int32) + queries
    cu_q_lengths = jnp.arange(b + 1, dtype=jnp.int32) * queries
    return packed, pages, kv_lengths, page_indices, cu_q_lengths, jnp.array([b], jnp.int32)


def attention(q, k, v, positions, *, reference=False):
    from jax.experimental.pallas.ops.tpu import ragged_paged_attention as rpa
    arguments = inputs(q, k, v, positions)
    if reference:
        result = rpa.ref_ragged_paged_attention(*arguments, sm_scale=1 / math.sqrt(q.shape[-1]))
    else:
        result = rpa.ragged_paged_attention(*arguments, sm_scale=1 / math.sqrt(q.shape[-1]),
            num_queries_per_block=128, num_kv_pages_per_block=min(4, arguments[3].shape[1]))
    result = result[:q.shape[0] * q.shape[1], :, :q.shape[-1]]
    return result.reshape(q.shape).astype(jnp.float32)
