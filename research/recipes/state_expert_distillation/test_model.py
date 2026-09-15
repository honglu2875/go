"""Expert representation, symmetry, causal isolation and objective-mask checks."""
import os
os.environ['JAX_PLATFORMS'] = 'cpu'
import unittest
import jax
import numpy as np
import model
import history_model
import spatial_model
from state_features import planes, batch_planes


def config(mode='state'):
    return {'architecture': mode, 'size': 3, 'komi': .5, 'width': 16, 'heads': 4, 'blocks': 1,
            'max_tokens': 8, 'dtype': 'float32', 'board_width': 8, 'board_blocks': 1,
            'board_mode': 'exact', 'value_objective': 'mse', 'expert_temperature': 1., 'behavior_temperature': 1.,
            'state_width': 16, 'state_heads': 4, 'state_blocks': 1, 'state_value_hidden': 16, 'state_max_board_size': 25}


def batch():
    tokens = np.array([[10, 0, 1, 9, 11, 11], [10, 8, 1, 9, 11, 11]], np.int32)
    stones = np.zeros((*tokens.shape, 9), np.uint8)
    stones[0, 1:, 0] = 1; stones[0, 2:, 1] = 2
    stones[1, 1:, 8] = 1; stones[1, 2:, 1] = 2
    mask = np.zeros(tokens.shape, np.float32); mask[:, :4] = 1
    return {'tokens': tokens, 'lengths': np.array([3, 3], np.int32), 'stones': stones,
            'policies': np.eye(10, dtype=np.float32)[np.array([[0, 2, 9, 9, 0, 0], [8, 1, 9, 9, 0, 0]])],
            'observed': np.zeros(tokens.shape, np.int32), 'values': np.ones(tokens.shape, np.float32),
            'legal': np.ones((*tokens.shape, 10), bool), 'expert_mask': mask, 'value_mask': mask.copy(),
            'behavior_mask': np.zeros(tokens.shape, np.float32)}


class ExpertTests(unittest.TestCase):
    def test_history_control_reproduces_expert_predictions_loss_and_gradients(self):
        c = config('history'); old = history_model.initialize(71, c); p = model.initialize(71, c); b = batch()
        self.assertFalse(any(k.startswith('behavior_') for k in p))
        for a, expected in zip(jax.tree.leaves(p), jax.tree.leaves({k: v for k, v in old.items() if not k.startswith('behavior_')})):
            np.testing.assert_array_equal(a, expected)
        play, value = model.predictions(p, b, c); op, _, ov = history_model.predictions(old, b, c)
        np.testing.assert_array_equal(play, op); np.testing.assert_array_equal(value, ov)
        a, ga = jax.value_and_grad(lambda q: model.losses(q, b, c)[0])(p)
        expected, ge = jax.value_and_grad(lambda q: history_model.losses(q, b, c)[0])(old)
        np.testing.assert_array_equal(a, expected)
        for actual, expected in zip(jax.tree.leaves(ga), jax.tree.leaves({k: v for k, v in ge.items() if k in p})):
            np.testing.assert_array_equal(actual, expected)

    def test_state_expert_ignores_earlier_actions_given_identical_exact_inputs(self):
        c = config(); p = model.initialize(71, c); b = batch()
        first = model.predictions(p, b, c)
        b['tokens'][:, 1:3] = 5
        for actual, expected in zip(model.predictions(p, b, c), first):
            np.testing.assert_array_equal(actual, expected)
        b['tokens'][0, 3] = 4
        changed = model.predictions(p, b, c)
        self.assertFalse(np.array_equal(changed[0][0, 3], first[0][0, 3]))

    def test_future_positions_and_other_games_do_not_change_a_prefix(self):
        for mode in ('history', 'state'):
            c = config(mode); p = model.initialize(71, c); b = batch(); first = model.predictions(p, b, c)
            b['tokens'][0, 3:] = 7; b['tokens'][1] = 5
            b['stones'][0, 3:] = 2; b['stones'][1] = 1
            b['legal'][0, 3:] = False; b['legal'][1] = False
            for a, expected in zip(model.predictions(p, b, c), first):
                np.testing.assert_array_equal(a[0, :3], expected[0, :3])

    def test_observed_labels_and_masked_targets_cannot_train_expert(self):
        for mode in ('history', 'state'):
            c = config(mode); p = model.initialize(71, c); b = batch()
            first, ga = jax.value_and_grad(lambda q: model.losses(q, b, c)[0])(p)
            b['values'][:, 4:] = 100.; b['policies'][:, 4:] = 100.
            b['observed'][:] = 9; b['behavior_mask'][:] = 1
            second, gb = jax.value_and_grad(lambda q: model.losses(q, b, c)[0])(p)
            np.testing.assert_array_equal(first, second)
            for a, expected in zip(jax.tree.leaves(ga), jax.tree.leaves(gb)):
                np.testing.assert_array_equal(a, expected)
            b['expert_mask'][:] = 0; b['value_mask'][:] = 0
            loss, grad = jax.value_and_grad(lambda q: model.losses(q, b, c)[0])(p)
            self.assertEqual(float(loss), 0.)
            self.assertTrue(all(np.count_nonzero(a) == 0 for a in jax.tree.leaves(grad)))

    def test_spatial_symmetry_and_19x19_use_the_same_parameter_tree(self):
        c = config(); p = model.initialize(71, c); rng = np.random.default_rng(71)
        # Nonzero, nonconstant bias exercises actual D4 geometry rather than only permutation equivariance.
        p['relative_bias'] = jax.numpy.asarray(rng.normal(0, .1, p['relative_bias'].shape), jax.numpy.float32)
        schema = [(a.shape, a.dtype) for a in jax.tree.leaves(p)]
        for size in (3, 9, 19):
            stones = rng.integers(0, 3, size=(2, size * size), dtype=np.uint8)
            f = planes(stones, np.array([1, 2]), stones == 0, np.array([False, True]), size, .5)
            logits, value = spatial_model.apply(p, f, c)
            self.assertEqual(logits.shape, (2, size * size + 1)); self.assertTrue(np.isfinite(logits).all())
            for symmetry in range(8):
                def transform(x):
                    x = np.rot90(x, symmetry % 4, axes=(1, 2))
                    return np.flip(x, axis=2) if symmetry >= 4 else x
                moved, mv = spatial_model.apply(p, transform(np.asarray(f)).copy(), c)
                expected = transform(np.asarray(logits[:, :-1]).reshape(2, size, size)).reshape(2, -1)
                np.testing.assert_allclose(moved[:, :-1], expected, rtol=2e-5, atol=2e-5)
                np.testing.assert_allclose(moved[:, -1], logits[:, -1], rtol=2e-5, atol=2e-5)
                np.testing.assert_allclose(mv, value, rtol=2e-5, atol=2e-5)
        self.assertEqual(schema, [(a.shape, a.dtype) for a in jax.tree.leaves(p)])

    def test_state_features_use_pre_action_pass_turn_komi_and_legal_inputs(self):
        b = batch(); b['legal'][0, 2, 4] = False
        f = np.asarray(batch_planes(b, config()))
        np.testing.assert_array_equal(f[0, 3, :, :, 4], 1.)
        np.testing.assert_array_equal(f[0, 2, :, :, 4], 0.)
        self.assertEqual(f[0, 2, 1, 1, 3], 0.)
        np.testing.assert_array_equal(f[0, 2, :, :, 5], 1.)
        np.testing.assert_allclose(f[0, 2, :, :, 6], -.5 / 9)
        np.testing.assert_allclose(f[0, 3, :, :, 6], .5 / 9)

    def test_one_spatial_block_matches_independent_numpy_attention(self):
        c = config(); p = model.initialize(71, c); f = np.asarray(batch_planes(batch(), c))
        expected_logits, expected_values = spatial_model.apply(p, f, c)
        p = jax.tree.map(np.asarray, p); width, heads, size = c['state_width'], c['state_heads'], c['size']
        def norm(x, n):
            center = x - x.mean(axis=-1, keepdims=True)
            return center / np.sqrt((center ** 2).mean(axis=-1, keepdims=True) + 1e-5) * n['scale'] + n['bias']
        def gelu(x):
            return .5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + .044715 * x ** 3)))
        x = f.reshape(-1, size * size, f.shape[-1]) @ p['stem'] + p['stem_bias']
        b = p['blocks'][0]; z = (norm(x, b['n1']) @ b['qkv']).reshape(x.shape[0], size * size, 3, heads, width // heads)
        q, k, v = [z[:, :, i] for i in range(3)]
        scores = np.einsum('bthd,bshd->bhts', q, k) / np.sqrt(width // heads)
        indices = np.empty((size * size, size * size), np.int32)
        for i in range(size * size):
            for j in range(size * size):
                high, low = sorted([abs(i // size - j // size), abs(i % size - j % size)], reverse=True)
                indices[i, j] = high * (high + 1) // 2 + low
        scores += np.moveaxis(p['relative_bias'][indices], -1, 0)[None]
        weights = np.exp(scores - scores.max(axis=-1, keepdims=True)); weights /= weights.sum(axis=-1, keepdims=True)
        x = x + np.einsum('bhts,bshd->bthd', weights, v).reshape(x.shape) @ b['out']
        x = x + gelu(norm(x, b['n2']) @ b['mlp_in']) @ b['mlp_out']
        x = norm(x, p['final_norm']); pool = np.concatenate([x.mean(axis=1), x.max(axis=1)], axis=-1)
        logits = np.concatenate([(x @ p['policy'])[..., 0], pool @ p['pass'] + p['pass_bias']], axis=-1)
        value = np.tanh(gelu(pool @ p['value_hidden'] + p['value_bias']) @ p['value_out'] + p['value_out_bias'])[..., 0]
        np.testing.assert_allclose(logits.reshape(expected_logits.shape), expected_logits, rtol=3e-5, atol=3e-5)
        np.testing.assert_allclose(value.reshape(expected_values.shape), expected_values, rtol=3e-5, atol=3e-5)

    def test_requested_leaf_matches_training_position_in_both_architectures(self):
        b = batch()
        for mode in ('history', 'state'):
            c = config(mode); p = model.initialize(71, c); all_logits, all_values = model.predictions(p, b, c)
            for turn in (0, 1, 3):
                logits, values = model.leaf_predictions(p, b['tokens'], np.full(2, turn, np.int32),
                    b['stones'][:, turn], b['legal'][:, turn], b['tokens'][:, turn] == 9, c)
                np.testing.assert_allclose(logits, all_logits[:, turn], rtol=2e-5, atol=2e-5)
                np.testing.assert_allclose(values, all_values[:, turn], rtol=2e-5, atol=2e-5)


if __name__ == '__main__':
    unittest.main()
