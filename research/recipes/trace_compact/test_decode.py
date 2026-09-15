"""Independent reconstruction of state inputs, causal logits and sampler draws."""
import os
os.environ['JAX_PLATFORMS'] = 'cpu'
import unittest

import jax
import jax.numpy as jnp
import numpy as np

import model
from decode import decode


class BoardDecodeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.c = {'size': 3, 'komi': .5, 'width': 16, 'heads': 4, 'blocks': 1,
                 'max_tokens': 13, 'dtype': 'float32', 'expert_temperature': 1., 'behavior_temperature': 1.,
                 'board_mode': 'exact', 'board_width': 8, 'board_blocks': 1}
        cls.params = model.initialize(271, cls.c)
        cls.tokens = jnp.array([[10, 0, 1, 2, 11, 11, 11, 11, 11, 11, 11, 11, 11]], jnp.int32)
        cls.lengths = jnp.array([3], jnp.int32)
        cls.stones = jnp.array([[1, 2, 1, 0, 0, 0, 0, 0, 0]], jnp.uint8)
        cls.episodes = jnp.array([2], jnp.uint32)
        cls.key = jax.random.key(51)

    def run_decode(self, **kwargs):
        settings = dict(horizon=4, samples=3)
        settings.update(kwargs)
        return jax.device_get(jax.jit(lambda p, t, l, e, k, s: decode(p, t, l, e, k, s, self.c, **settings))(
            self.params, self.tokens, self.lengths, self.episodes, self.key, self.stones))

    def test_reported_inputs_precede_each_action_and_place_updates_skip_pass(self):
        actions, _, _, states = self.run_decode()
        for view in range(2):
            for sample in range(3):
                expected = np.asarray(self.stones[0]).copy()
                for depth in range(4):
                    np.testing.assert_array_equal(states[0, view, sample, depth], expected)
                    action = actions[0, view, sample, depth]
                    if action != 9:
                        expected[action] = 1 + (3 + depth) % 2
        # Force pass at the root and check that it leaves the next input intact.
        legal = jnp.zeros((1, 10), jnp.bool_).at[:, 9].set(True)
        actions, _, _, states = self.run_decode(root_legal=legal)
        self.assertTrue((actions[..., 0] == 9).all())
        np.testing.assert_array_equal(states[..., 0, :], states[..., 1, :])

    def test_every_own_logit_row_matches_fresh_prefix_and_reported_state(self):
        actions, logits, noise, states = self.run_decode()
        tokens = np.broadcast_to(np.asarray(self.tokens), (6, 13)).copy()
        for depth in range(4):
            lengths = jnp.full(6, 3 + depth, jnp.int32)
            h, _ = model.prefill(self.params, jnp.asarray(tokens), lengths, self.c)
            counts = model.historical_counts(jnp.asarray(tokens), lengths, 10)
            player = (3 + depth) % 2
            boards = model.board_features(self.params, jnp.asarray(states[0, :, :, depth].reshape(6, 9)),
                                          jnp.full(6, player + 1), self.c)
            expected, _, _ = model.head_outputs(self.params, h[:, 3 + depth], counts[:, 3 + depth, player], boards)
            np.testing.assert_allclose(logits[0, :, :, depth].reshape(6, 10), expected, atol=2e-5, rtol=2e-5)
            np.testing.assert_array_equal(actions[0, player, :, depth],
                                          np.argmax(logits[0, player, :, depth] + noise[0, depth], axis=-1))
            tokens[:, 4 + depth] = actions[0, :, :, depth].reshape(-1)

    def test_hold_control_keeps_roots_but_has_identical_initial_policy(self):
        place = self.run_decode(); hold = self.run_decode(board_update='hold')
        expected = np.broadcast_to(np.asarray(self.stones)[:, None, None, None], (1, 2, 3, 4, 9))
        np.testing.assert_array_equal(hold[3], expected)
        np.testing.assert_array_equal(place[0][..., 0], hold[0][..., 0])
        np.testing.assert_array_equal(place[1][..., 0, :], hold[1][..., 0, :])

    def test_actual_known_draws_are_shared_and_do_not_depend_on_packet_boundaries(self):
        actions, _, long_noise, _ = self.run_decode(coupling='shared', oracle_behavior=True)
        np.testing.assert_array_equal(actions[0], np.broadcast_to(actions[0, 0, 0], (2, 3, 4)))
        other = decode(self.params, self.tokens, self.lengths + 2, self.episodes, self.key,
                       self.stones, self.c, horizon=1, samples=1)
        np.testing.assert_array_equal(long_noise[:, 2], np.asarray(other[2])[:, 0])


if __name__ == '__main__':
    unittest.main()
