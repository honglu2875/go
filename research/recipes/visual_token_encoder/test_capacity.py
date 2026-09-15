"""Independent encoder checks and equivalence to the frozen 64-channel control."""
import hashlib
import json
from pathlib import Path
import unittest

import jax
import jax.numpy as jnp
import numpy as np

import causal
import conv64_reference
from test_causal import C, fixture


def reference_stem(p, spatial, xp):
    """Explicit padded spatial loops, independent of lax convolution."""
    x = spatial
    for i in range(2):
        w = p[f'encoder.stem{i}.weight']
        padded = xp.pad(x, ((0, 0), (0, 0), (1, 1), (1, 1), (0, 0)))
        out = xp.zeros((*x.shape[:-1], w.shape[-1]), dtype=xp.float32)
        for y in range(3):
            for z in range(3):
                out = out + padded[:, :, y:y+x.shape[2], z:z+x.shape[3]] @ w[y, z]
        out = out / xp.sqrt(xp.mean(out * out, -1, keepdims=True) + C['norm_epsilon'])
        out = out * p[f'encoder.stem{i}.scale']
        x = out / (1 + xp.exp(-out))
    return x


class CapacityChecks(unittest.TestCase):
    def test_contract(self):
        for channels in (64, 128, 256):
            causal.validate({**C, 'encoder': 'overlap_conv', 'encoder_channels': channels})
        for channels in (True, 0, 7, 17, 1040, 128.0):
            with self.assertRaises(ValueError):
                causal.validate({**C, 'encoder': 'overlap_conv', 'encoder_channels': channels})
        with self.assertRaises(ValueError):
            causal.validate({**C, 'encoder_channels': 64})

    def test_64_control_is_exact(self):
        self.assertEqual(hashlib.sha256(Path(conv64_reference.__file__).read_bytes()).hexdigest(),
                         '44fbb20ca07d1258158014413fca9e2b26fd114ab65d05af24c46684c7eebc6f')
        s, g, a, n = map(jnp.asarray, fixture())
        old = jax.jit(lambda: conv64_reference.initialize(91, C))()
        objective = lambda module, config, p: jnp.sum(module.forward(p, s, g, a, n, config)**2)
        expected = jax.jit(jax.value_and_grad(lambda p: objective(conv64_reference, C, p)))(old)
        for config in (C, {**C, 'encoder': 'overlap_conv', 'encoder_channels': 64}):
            p = jax.jit(lambda: causal.initialize(91, config))()
            for key in old:
                np.testing.assert_array_equal(p[key], old[key], err_msg=key)
            actual = jax.jit(jax.value_and_grad(lambda p: objective(causal, config, p)))(p)
            for x, y in zip(jax.tree.leaves(actual), jax.tree.leaves(expected)):
                np.testing.assert_array_equal(x, y)

    def test_wide_encoder_forward_and_independent_gradient(self):
        s = jnp.asarray(fixture()[0][:1, :1])
        for channels in (128, 256):
            with self.subTest(channels=channels):
                c = {**C, 'encoder': 'overlap_conv', 'encoder_channels': channels}
                p = jax.jit(lambda: causal.initialize(91, c))()
                stem = {k: v for k, v in p.items() if k.startswith('encoder.stem')}
                expected = reference_stem(jax.tree.map(np.asarray, stem), np.asarray(s), np)
                np.testing.assert_allclose(causal.spatial_features(stem, s, c), expected, atol=3e-6, rtol=3e-5)
                weights = jnp.asarray(np.random.default_rng(811).normal(size=expected.shape).astype(np.float32) / np.sqrt(expected.size))
                actual = jax.jit(jax.value_and_grad(lambda p, x: jnp.sum(causal.spatial_features(p, x, c)*weights), argnums=(0, 1)))(stem, s)
                reference = jax.jit(jax.value_and_grad(lambda p, x: jnp.sum(reference_stem(p, x, jnp)*weights), argnums=(0, 1)))(stem, s)
                for x, y in zip(jax.tree.leaves(actual), jax.tree.leaves(reference)):
                    np.testing.assert_allclose(x, y, atol=3e-6, rtol=3e-4)
                for key, gradient in actual[1][0].items():
                    self.assertGreater(float(jnp.linalg.norm(gradient)), 1e-6, key)

    def test_wide_causal_and_cached_predictions(self):
        s, g, a, n = map(jnp.asarray, fixture())
        for channels in (128, 256):
            with self.subTest(channels=channels):
                c = {**C, 'encoder': 'overlap_conv', 'encoder_channels': channels}
                p = jax.jit(lambda: causal.initialize(91, c))()
                run = jax.jit(lambda s, g, a: causal.forward(p, s, g, a, n, c))
                base = run(s, g, a)
                changed = run(s.at[0, 3:].add(.3).at[1].set(0),
                              g.at[0, 3:].add(1).at[1].set(0), a.at[0, 2:].set(0).at[1].set(9))
                np.testing.assert_array_equal(base[0, :3], changed[0, :3])
                counts = jnp.asarray([2, 3])
                _, cache = causal.forward(p, s[:, :3], g[:, :3], a[:, :3], counts, c, with_cache=True)
                got, updated = causal.append_move(p, cache, jnp.asarray([a[0, 1], a[1, 2]]),
                    jnp.stack([s[0, 2], s[1, 3]]), jnp.stack([g[0, 2], g[1, 3]]), c, attention_positions=4)
                full, expected = causal.forward(p, s[:, :4], g[:, :4], a[:, :4], counts+1, c, with_cache=True)
                np.testing.assert_allclose(got, jnp.stack([full[0, 2], full[1, 3]]), atol=2e-5, rtol=2e-5)
                for key in expected:
                    np.testing.assert_allclose(updated[key], expected[key], atol=2e-5, rtol=2e-5, err_msg=key)
                opening, _ = causal.first_move(p, s[:, 0], g[:, 0], c)
                np.testing.assert_allclose(opening, base[:, 0], atol=2e-5, rtol=2e-5)

    def test_parameter_budget_and_unchanged_shapes(self):
        parent = jax.jit(lambda: conv64_reference.initialize(91, C))()
        for channels, parameters, encoder in [(64, 234588416, 290816), (128, 234281728, 610816), (256, 234516224, 1472000)]:
            config = json.loads((Path(__file__).parent / f'c{channels}_lr_3e-4.json').read_text())['model']
            schema = causal.parameter_schema(config)
            self.assertEqual(sum(x['elements'] for x in schema), parameters)
            self.assertEqual(sum(x['elements'] for x in schema if x['path'].startswith('encoder.')), encoder)
            small = {**C, 'encoder': 'overlap_conv', 'encoder_channels': channels, 'mlp_hidden': 40}
            p = jax.jit(lambda: causal.initialize(91, small))()
            for key in parent:
                if p[key].shape == parent[key].shape:
                    np.testing.assert_array_equal(p[key], parent[key], err_msg=key)


if __name__ == '__main__':
    unittest.main()
