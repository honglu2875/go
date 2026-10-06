import unittest

import jax
import jax.numpy as jnp
import numpy as np

from gozero import moe


def dense_reference(p, x, k, valid):
    """Evaluate every expert independently, used only as a small correctness oracle."""
    x = jnp.where(valid[:, None], x, 0.)
    logits = x @ p['router']
    probabilities = jax.nn.softmax(logits)
    selected = jnp.argsort(-logits, stable=True, axis=-1)[:, :k]
    mask = jnp.sum(jax.nn.one_hot(selected, logits.shape[-1]), axis=1)
    weights = probabilities * mask
    if k > 1:
        weights /= weights.sum(-1, keepdims=True)
    def expert(up, down, index):
        y = x @ up
        if 'up_bias' in p:
            y = y + p['up_bias'][index]
        if 'gate' in p:
            y = jax.nn.silu(x @ p['gate'][index]) * y
        else:
            y = jax.nn.gelu(y)
        y = y @ down
        if 'down_bias' in p:
            y = y + p['down_bias'][index]
        return y
    y = jax.vmap(expert)(p['up'], p['down'], jnp.arange(logits.shape[-1]))
    result = jnp.einsum('ne,end->nd', weights, y)
    return jnp.where(valid[:, None], result, 0.)


class MoETests(unittest.TestCase):
    def test_permutation_dispatch_vjp_matches_default_with_empty_experts_and_padding(self):
        for gated in (False, True):
            for dtype in (jnp.float32, jnp.bfloat16):
                for collapsed in (False, True):
                    with self.subTest(gated=gated, dtype=dtype, collapsed=collapsed):
                        p = moe.initialize(jax.random.key(51), width=8, hidden=12, experts=4, gated=gated, bias=not gated)
                        if collapsed:
                            p['router'] = jnp.zeros_like(p['router'])
                        x = jax.random.normal(jax.random.key(52), (17, 8))
                        x = x.at[-2:].set(jnp.nan)
                        valid = jnp.arange(17) < 15
                        def objective(p, x, optimized):
                            y, stats = moe.feed_forward(p, x, top_k=2, dtype=dtype, valid=valid, permutation_vjp=optimized)
                            return jnp.sum(y * y), (y, stats)
                        results = [jax.jit(jax.value_and_grad(lambda p, x: objective(p, x, flag), (0, 1), has_aux=True))(p, x)
                                   for flag in (False, True)]
                        # Forward and all parameter gradients are exact here.
                        # XLA may reassociate the sum of router and dispatched
                        # input gradients; the unfused paths agree exactly.
                        for actual, reference in zip(jax.tree.leaves((results[1][0], results[1][1][0])),
                                                     jax.tree.leaves((results[0][0], results[0][1][0]))):
                            np.testing.assert_array_equal(actual, reference)
                        np.testing.assert_allclose(results[1][1][1], results[0][1][1], rtol=2e-6, atol=2e-6)

    def test_activation_recomputation_preserves_outputs_and_gradients(self):
        for gated in (False, True):
            for dtype in (jnp.float32, jnp.bfloat16):
                with self.subTest(gated=gated, dtype=dtype):
                    p=moe.initialize(jax.random.key(31),width=16,hidden=24,experts=4,gated=gated,bias=not gated)
                    x=jax.random.normal(jax.random.key(32),(17,16))
                    def objective(p,x,remat):
                        y,s=moe.feed_forward(p,x,top_k=2,dtype=dtype,remat_activations=remat)
                        return jnp.sum(y*y),(y,s)
                    reference=jax.jit(jax.value_and_grad(lambda p,x:objective(p,x,False),(0,1),has_aux=True))(p,x)
                    actual=jax.jit(jax.value_and_grad(lambda p,x:objective(p,x,True),(0,1),has_aux=True))(p,x)
                    for a,b in zip(jax.tree.leaves(actual),jax.tree.leaves(reference)):
                        np.testing.assert_allclose(a,b,rtol=2e-5,atol=3e-6)

    def test_sparse_outputs_and_all_gradients_match_independent_reference(self):
        for gated, bias, k in [(True, False, 2), (False, True, 2), (True, False, 1)]:
            with self.subTest(gated=gated, k=k):
                p = moe.initialize(jax.random.key(3), width=8, hidden=12, experts=4, gated=gated, bias=bias)
                x = jax.random.normal(jax.random.key(4), (13, 8))
                valid = jnp.arange(13) < 10
                probe = jax.random.normal(jax.random.key(5), x.shape)
                sparse = lambda p, x: moe.feed_forward(p, x, top_k=k, dtype=jnp.float32, valid=valid)[0]
                reference = lambda p, x: dense_reference(p, x, k, valid)
                np.testing.assert_allclose(jax.jit(sparse)(p, x), reference(p, x), rtol=2e-5, atol=2e-6)
                gradients = []
                for fn in (sparse, reference):
                    gradients.append(jax.jit(jax.grad(lambda p, x: jnp.sum(fn(p, x) * probe), argnums=(0, 1)))(p, x))
                for actual, expected in zip(jax.tree.leaves(gradients[0]), jax.tree.leaves(gradients[1])):
                    np.testing.assert_allclose(actual, expected, rtol=6e-5, atol=3e-6)
                self.assertGreater(float(jnp.linalg.norm(gradients[0][0]['router'])), 1e-5)
                np.testing.assert_array_equal(np.asarray(gradients[0][1])[10:], 0.)

    def test_collapsed_routing_does_not_drop_tokens_or_fail_on_empty_experts(self):
        p = moe.initialize(jax.random.key(7), width=8, hidden=12, experts=4, gated=False)
        p['router'] = jnp.tile(jnp.array([3., 2., -2., -3.]), (8, 1))
        x = jnp.ones((257, 8))
        valid = jnp.ones((257,), bool)
        y, stats = jax.jit(lambda p, x: moe.feed_forward(p, x, top_k=2, dtype=jnp.float32))(p, x)
        np.testing.assert_allclose(y, dense_reference(p, x, 2, valid), rtol=2e-5, atol=2e-6)
        np.testing.assert_array_equal(stats[4:8], [257, 257, 0, 0])
        gradient = jax.grad(lambda p: moe.feed_forward(p, x, top_k=2, dtype=jnp.float32)[0].sum())(p)
        for name in ['up', 'down']:
            np.testing.assert_array_equal(np.asarray(gradient[name])[2:], 0.)

    def test_batch_composition_and_padding_cannot_change_a_token_output(self):
        p = moe.initialize(jax.random.key(12), width=8, hidden=8, experts=4, gated=True)
        x = jax.random.normal(jax.random.key(13), (5, 8))
        fn = jax.jit(lambda x, v: moe.feed_forward(p, x, top_k=2, dtype=jnp.float32, valid=v))
        alone, _ = fn(x[:1], jnp.ones((1,), bool))
        together, _ = fn(x, jnp.ones((5,), bool))
        padded = jnp.concatenate((x[:1], jnp.full((4, 8), jnp.nan)))
        result, stats = fn(padded, jnp.arange(5) < 1)
        np.testing.assert_allclose(alone[0], together[0], rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(alone[0], result[0], rtol=1e-6, atol=1e-6)
        np.testing.assert_array_equal(np.asarray(result)[1:], 0.)
        self.assertEqual(float(stats[-1]), 1.)
        self.assertEqual(float(stats[4:8].sum()), 2.)
        empty, statistics = fn(padded, jnp.zeros((5,), bool))
        np.testing.assert_array_equal(empty, 0.)
        np.testing.assert_array_equal(statistics, 0.)

    def test_router_losses_are_finite_and_padding_excluded(self):
        c = dict(experts=4, top_k=2, backend='ragged', balance_weight=.01, z_weight=.001)
        p = moe.initialize(jax.random.key(21), width=8, hidden=8, experts=4, gated=True)
        x = jax.random.normal(jax.random.key(22), (9, 8))
        def loss(p):
            _, stats = moe.feed_forward(p, x, top_k=2, dtype=jnp.float32, valid=jnp.arange(9) < 3)
            return moe.router_metrics(stats[None], c)[0]
        result, gradients = jax.value_and_grad(loss)(p)
        self.assertTrue(np.isfinite(result))
        self.assertGreater(float(jnp.linalg.norm(gradients['router'])), 0.)
        zero, metrics = moe.router_metrics(jnp.zeros((2, 11)), c)
        self.assertEqual(float(zero), 0.)
        self.assertEqual(float(metrics['moe_valid_tokens']), 0.)


if __name__ == '__main__':
    unittest.main()
