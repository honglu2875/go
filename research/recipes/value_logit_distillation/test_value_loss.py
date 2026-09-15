"""Neutral-gradient matching, saturated correction and unchanged MSE control."""
import importlib.util
import hashlib
from pathlib import Path
import unittest

import jax
import jax.numpy as jnp
import numpy as np
import model
import test_model


class ValueObjectiveTest(unittest.TestCase):
    def test_neutral_gradients_match_for_losses_wins_and_draws(self):
        for target in (-1., 0., 1.):
            gradients = [jax.grad(lambda r: model.value_objective(r, target, objective))(jnp.asarray(0.))
                         for objective in ('mse', 'bce2')]
            np.testing.assert_array_equal(*gradients)
            np.testing.assert_array_equal(gradients[1], -2 * target)

    def test_confident_wrong_logits_retain_a_finite_corrective_gradient(self):
        for raw, outcome in ((20., -1.), (-20., 1.)):
            mse = jax.grad(lambda r: model.value_objective(r, outcome, 'mse'))(jnp.asarray(raw))
            bce = jax.grad(lambda r: model.value_objective(r, outcome, 'bce2'))(jnp.asarray(raw))
            self.assertLess(abs(float(mse)), 1e-6)
            self.assertTrue(np.isfinite(bce))
            self.assertGreater(float(bce) * np.sign(raw), 3.9)

    def test_logit_loss_matches_probability_loss_and_derivative(self):
        for raw in (-3., -.4, 0., .7, 3.):
            for outcome in (-1., 0., 1.):
                probability = 1 / (1 + np.exp(-2 * raw)); target = (outcome + 1) / 2
                expected = -2 * (target * np.log(probability) + (1 - target) * np.log1p(-probability))
                actual = model.value_objective(jnp.asarray(raw), outcome, 'bce2')
                np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=1e-6)
                gradient = jax.grad(lambda r: model.value_objective(r, outcome, 'bce2'))(jnp.asarray(raw))
                np.testing.assert_allclose(gradient, 2 * (np.tanh(raw) - outcome), rtol=2e-5, atol=1e-6)

    def test_existing_mse_predictions_loss_and_gradients_are_preserved(self):
        path = Path(__file__).with_name('reference_model.py')
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), '83a769c5c174ad9445d577b0d9544210ab29b9114883dba08036a1a759a25d04')
        spec = importlib.util.spec_from_file_location('legacy_board_model', path)
        legacy = importlib.util.module_from_spec(spec); spec.loader.exec_module(legacy)
        test_model.BoardHistoryTests.setUpClass(); fixture = test_model.BoardHistoryTests(); batch = fixture.batch()
        c = {**fixture.c, 'value_objective': 'mse'}; params = model.initialize(271, c)
        for new, old in zip(jax.tree.leaves(params), jax.tree.leaves(fixture.p)):
            np.testing.assert_array_equal(new, old)
        for new, old in zip(model.predictions(params, batch, c), legacy.predictions(params, batch, fixture.c)):
            np.testing.assert_array_equal(new, old)
        new = jax.value_and_grad(lambda p: model.losses(p, batch, c)[0])(params)
        old = jax.value_and_grad(lambda p: legacy.losses(p, batch, fixture.c)[0])(params)
        for a, b in zip(jax.tree.leaves(new), jax.tree.leaves(old)):
            np.testing.assert_array_equal(a, b)

    def test_bce_behavior_and_padding_cannot_supply_value_targets(self):
        test_model.BoardHistoryTests.setUpClass(); fixture = test_model.BoardHistoryTests(); batch = fixture.batch()
        c = {**fixture.c, 'value_objective': 'bce2'}
        before = model.losses(fixture.p, batch, c)
        batch['values'][batch['value_mask'] == 0] = 100.
        after = model.losses(fixture.p, batch, c)
        for a, b in zip(jax.tree.leaves(before), jax.tree.leaves(after)):
            np.testing.assert_array_equal(a, b)


if __name__ == '__main__':
    unittest.main()
