"""The RPA packing adapter must implement the exact same causal attention."""
import unittest
import jax.numpy as jnp
import numpy as np

if __package__:
    from . import cached_attention, model
else:
    import cached_attention, model


class PagedAttentionTests(unittest.TestCase):
    def test_reference_matches_dense_for_ragged_histories_and_partial_blocks(self):
        random = np.random.default_rng(181)
        q = jnp.asarray(random.normal(size=(3, 11, 4, 8)).astype(np.float32))
        k = jnp.asarray(random.normal(size=(3, 31, 2, 8)).astype(np.float32))
        v = jnp.asarray(random.normal(size=k.shape).astype(np.float32))
        positions = jnp.array([2, 7, 26])[:, None] + jnp.arange(11)[None, :]
        # The third row has only its first five queries inside the old cache.
        # Those live queries must still agree when the physical block is longer.
        out = cached_attention.attention(q, k, v, positions, reference=True)
        expected = model.dense_attention(q, k, v, positions, jnp.array([13, 18, 31]))
        for row, count in enumerate([11, 11, 5]):
            np.testing.assert_allclose(out[row, :count], expected[row, :count], atol=2e-6, rtol=2e-6)


if __name__ == '__main__':
    unittest.main()
