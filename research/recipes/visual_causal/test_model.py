"""Independent attention arithmetic, causal alignment, cache and gradient tests."""
import json
from pathlib import Path
import unittest
import jax
import jax.numpy as jnp
import numpy as np

if __package__:
    from . import fixtures, learner, model, observations
else:
    import fixtures, learner, model, observations


class VisualCausalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.c = json.loads(Path(__file__).with_name('cpu_qualification.json').read_text())['model']
        cls.p = model.initialize(719, cls.c)
        cls.o, cls.a, cls.n = fixtures.inputs(31, 2, 6, 5)
        cls.o, cls.a, cls.n = map(jnp.asarray, (cls.o, cls.a, cls.n))

    def test_parameter_count_and_shapes_without_allocating_large_model(self):
        c = json.loads(Path(__file__).with_name('tpu_233m_qualification.json').read_text())['model']
        schema = model.parameter_schema(c)
        count = sum(row['elements'] for row in schema)
        self.assertGreater(count, 232_000_000)
        self.assertLess(count, 234_000_000)
        self.assertEqual(model.layout(9, c)['stride'], 11)
        self.assertEqual(model.layout(19, c)['stride'], 51)

    def test_splash_mask_cache_cannot_capture_forward_or_backward_tracers(self):
        c = json.loads(Path(__file__).with_name('tpu_kernel_qualification.json').read_text())['model']
        p = model.initialize(1, c)
        o, a, n = map(jnp.asarray, fixtures.inputs(2, 1, 2, 9))
        batch = jax.tree.map(jnp.asarray, fixtures.loss_batch(np.asarray(o), np.asarray(a), np.asarray(n)))
        model.splash_kernel.cache_clear()
        # Abstract tracing exercises the TPU kernel's mask construction even on
        # CPU. This catches closure leaks before an actual TPU compilation.
        with jax.checking_leaks():
            jax.make_jaxpr(lambda p: model.forward(p, o, a, n, c))(p)
            jax.make_jaxpr(jax.grad(lambda p: model.losses(p, batch, c)[0]))(p)

    def test_weighted_sharded_gradient_matches_global_reference(self):
        from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
        if len(jax.devices()) < 4:
            self.skipTest('Run with four forced CPU devices for this qualification')
        o, a, n = fixtures.inputs(31, 4, 4, 3)
        batch = fixtures.loss_batch(o, a, np.array([1, 2, 3, 4], np.int32))
        batch['behavior_mask'][:2] = 0
        batch['expert_mask'][2:] = 0
        mesh = Mesh(np.asarray(jax.devices()[:4]), ('data',))
        specs = {key: P() if key == 'loss_weights' else P('data') for key in batch}
        params = jax.device_put(self.p, NamedSharding(mesh, P()))
        batch = {key: jax.device_put(value, NamedSharding(mesh, specs[key])) for key, value in batch.items()}
        mapped = jax.shard_map(lambda p, b: model.losses(p, b, self.c, axis_name='data'),
            mesh=mesh, in_specs=(P(), specs), out_specs=P(), check_vma=False)
        actual = jax.jit(jax.value_and_grad(mapped, has_aux=True))(params, batch)
        reference = jax.jit(jax.value_and_grad(lambda p, b: model.losses(p, b, self.c), has_aux=True))(params, batch)
        for x, y in zip(jax.tree.leaves(actual), jax.tree.leaves(reference)):
            np.testing.assert_allclose(x, y, atol=3e-5, rtol=3e-5)

    def test_gqa_matches_independent_expanded_head_attention(self):
        rng = np.random.default_rng(89)
        q = rng.normal(size=(2, 3, 4, 8)).astype(np.float32)
        k = rng.normal(size=(2, 7, 2, 8)).astype(np.float32)
        v = rng.normal(size=k.shape).astype(np.float32)
        positions = np.array([[0, 1, 2], [2, 3, 4]])
        lengths = np.array([3, 5])
        kk, vv = np.repeat(k, 2, axis=2), np.repeat(v, 2, axis=2)
        logits = np.einsum('bthd,bshd->bhts', q, kk) / np.sqrt(8.)
        mask = (np.arange(7)[None, None, :] <= positions[:, :, None]) & (np.arange(7)[None, None, :] < lengths[:, None, None])
        logits = np.where(mask[:, None], logits, -1e30)
        weights = np.exp(logits - logits.max(-1, keepdims=True))
        weights /= weights.sum(-1, keepdims=True)
        reference = np.einsum('bhts,bshd->bthd', weights, vv)
        actual = model.dense_attention(jnp.array(q), jnp.array(k), jnp.array(v), jnp.array(positions), jnp.array(lengths))
        np.testing.assert_allclose(actual, reference, rtol=2e-6, atol=2e-6)

    def test_patch_projection_preserves_order_and_padding_mask(self):
        got = model.observation_tokens(self.p, self.o, self.c)
        ref = np.zeros(got.shape, np.float32)
        raw = np.asarray(self.o)
        for b in range(2):
            for t in range(6):
                for row in range(2):
                    for col in range(2):
                        patch = np.zeros((3, 3, 7), np.float32)
                        for y in range(3):
                            for x in range(3):
                                yy, xx = row * 3 + y, col * 3 + x
                                if yy < 5 and xx < 5:
                                    patch[y, x, :6] = raw[b, t, yy, xx]
                                    patch[y, x, 6] = 1
                        ref[b, t, row * 2 + col] = (patch.reshape(-1) @ np.asarray(self.p['patch'])
                            + np.asarray(self.p['patch_bias']) + np.asarray(self.p['patch_rows'][row])
                            + np.asarray(self.p['patch_cols'][col]) + np.asarray(self.p['types'][0]))
        np.testing.assert_allclose(got, ref, rtol=3e-6, atol=3e-6)

    def test_future_boards_and_current_action_cannot_leak(self):
        run = jax.jit(lambda o, a: model.forward(self.p, o, a, self.n, self.c))
        baseline = run(self.o, self.a)
        changed = run(self.o.at[0, 3:].set(self.o[0, 3:] + .25).at[1].set(0), self.a.at[0, 2:].set(1))
        for key in baseline:
            np.testing.assert_array_equal(baseline[key][0, :3], changed[key][0, :3])
        changed_board = run(self.o.at[0, 2, 4, 4, 0].add(.5), self.a)
        self.assertGreater(float(jnp.max(jnp.abs(baseline['expert_logits'][0, 2] - changed_board['expert_logits'][0, 2]))), 1e-6)

    def test_block_scoring_matches_full_and_sequential_with_ragged_prefixes(self):
        counts = jnp.array([2, 3])
        _, cache = model.forward(self.p, self.o[:, :3], self.a[:, :3], counts, self.c, with_cache=True, network_version=71)
        future = jnp.stack([self.o[0, 2:5], self.o[1, 3:6]])
        actions = jnp.stack([self.a[0, 1:4], self.a[1, 2:5]])
        added = jnp.array([3, 2])
        run = jax.jit(lambda ca, o, a, n: model.score_continuation(self.p, ca, o, a, n, self.c, network_version=71))
        block, final = run(cache, future, actions, added)
        reference, full_cache = model.forward(self.p, self.o, self.a, counts + added, self.c, with_cache=True, network_version=71)
        for key in block:
            for game in range(2):
                n, start = int(added[game]), int(counts[game])
                np.testing.assert_allclose(block[key][game, :n], reference[key][game, start:start+n], atol=2e-5, rtol=2e-5)
        sequential = cache
        for i in range(3):
            out, sequential = run(sequential, future[:, i:i+1], actions[:, i:i+1], (added > i).astype(jnp.int32))
            for key in block:
                np.testing.assert_allclose(out[key][:, 0], block[key][:, i], atol=2e-5, rtol=2e-5)
        for candidate in [sequential, full_cache]:
            for a, b in zip(jax.tree.leaves(final), jax.tree.leaves(candidate)):
                np.testing.assert_allclose(a, b, atol=2e-5, rtol=2e-5)

    def test_invalid_cache_requests_preserve_buffers_and_flag_error(self):
        _, cache = model.forward(self.p, self.o, self.a, self.n, self.c, with_cache=True, network_version=9)
        for count, version, action in [(jnp.array([3, 3]), 9, self.a[:, :3]),
                                        (jnp.array([1, 1]), 10, self.a[:, :3]),
                                        (jnp.array([-1, -1]), 9, self.a[:, :3]),
                                        (jnp.array([1, 1]), 9, self.a[:, :3].at[:, 0].set(26))]:
            _, updated = model.score_continuation(self.p, cache, self.o[:, :3], action, count, self.c, network_version=version)
            np.testing.assert_array_equal(updated['valid'], [False, False])
            for key in ['keys', 'values', 'lengths']:
                for a, b in zip(jax.tree.leaves(cache[key]), jax.tree.leaves(updated[key])):
                    np.testing.assert_array_equal(a, b)

    def test_expert_behavior_gradient_routing_and_real_optimizer_step(self):
        batch = jax.tree.map(jnp.asarray, fixtures.loss_batch(np.asarray(self.o[:, :2]), np.asarray(self.a[:, :2]), np.array([2, 2])))
        isolated = {**self.c, 'behavior_updates_trunk': False}
        gradient = jax.grad(lambda p: model.losses(p, batch, isolated)[1]['behavior_loss'])(self.p)
        for key in ['blocks', 'patch', 'actions', 'value', 'ownership']:
            self.assertTrue(all(np.count_nonzero(a) == 0 for a in jax.tree.leaves(gradient[key])), key)
        self.assertGreater(float(jnp.linalg.norm(gradient['behavior_out'])), 0)
        shared = jax.grad(lambda p: model.losses(p, batch, self.c)[1]['behavior_loss'])(self.p)
        self.assertGreater(float(jnp.linalg.norm(shared['patch'])), 0)
        expert = jax.grad(lambda p: model.losses(p, batch, self.c)[1]['expert_loss'])(self.p)
        self.assertEqual(int(jnp.count_nonzero(expert['behavior_adapter'])), 0)
        state = learner.initialize(self.p)
        updated, state, metrics = jax.jit(lambda p, s: learner.update(p, s, batch, self.c, learning_rate=1e-4))(self.p, state)
        self.assertTrue(bool(metrics['accepted']))
        self.assertEqual(int(state['step']), 1)
        self.assertGreater(float(jnp.linalg.norm(updated['patch'] - self.p['patch'])), 0)

    def test_board_sizes_and_native_boundary(self):
        for size in [3, 9, 19]:
            obs, acts, counts = fixtures.inputs(51, 1, 1, size)
            observations.validate_inputs(obs, acts, counts, self.c)
            out = model.forward(self.p, jnp.array(obs), jnp.array(acts), jnp.array(counts), self.c)
            self.assertEqual(out['expert_logits'].shape, (1, 1, size * size + 1))
            self.assertEqual(out['ownership_logits'].shape, (1, 1, size, size))
        native = np.zeros((2, 3, 3, 12), np.float32)
        native[0, :, :, -4] = 1
        native[:, 1, 1, 0] = 1
        converted = observations.from_native(native)
        self.assertEqual(converted[0, 1, 1, 0], 1)
        self.assertEqual(converted[1, 1, 1, 1], 1)
        with self.assertRaises(ValueError):
            observations.validate_inputs(np.asarray(self.o), np.asarray(self.a).copy() + 1, np.asarray(self.n), self.c)


if __name__ == '__main__':
    unittest.main()
