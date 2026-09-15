"""Independent native board, legality and successor checks for compiled rules."""
import json
import os
from pathlib import Path
import unittest
import jax
import numpy as np
from gozero import jax_go
from gozero.native import load_library
from gozero.visual_history import Replay


class CompiledGoTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = Path(os.environ['GOZERO_TEST_NATIVE_RECEIPT'])
        receipt = json.loads(path.read_text())
        cls.native = load_library(path.parent / receipt['filename'], receipt['binary_sha256'])

    def exercise(self, size, seed, plies):
        komi = 7.5 if size == 9 else .5
        rules = {'size': size, 'komi': komi, 'scoring': 'pass_alive_area'}
        replay = Replay(self.native, rules, 512)
        state = jax_go.empty(size, 512)
        inspect = jax.jit(lambda s: (jax_go.candidates(s, size), jax_go.observe(s, size, komi)))
        step = jax.jit(lambda s, a: jax_go.advance(s, a, size))
        tape = []; random = np.random.default_rng(seed); captures = 0
        for ply in range(plies):
            (children, legal), observation = jax.device_get(inspect(state))
            exact = replay([tape])[0][-1]
            np.testing.assert_array_equal(observation, exact)
            np.testing.assert_array_equal(legal, np.r_[exact[..., 5].reshape(-1) > .5, True])
            options = np.flatnonzero(legal[:-1])
            # Check every legal successor at selected positions, including ko
            # and suicide changes that a chosen trajectory could fail to expose.
            if ply % 7 == 0 and len(options):
                child_obs = replay([tape + [int(a)] for a in options])
                for a, o in zip(options, child_obs):
                    stones = o[-1, ..., 0].astype(np.uint8) + 2 * o[-1, ..., 1].astype(np.uint8)
                    np.testing.assert_array_equal(children[a], stones.reshape(-1))
            action = size * size if ply % 17 == 9 or not len(options) else int(random.choice(options))
            if tape and tape[-1] == size * size and action == size * size:
                break
            before = np.count_nonzero(np.asarray(state['stones']))
            state = step(state, np.int32(action)); tape.append(action)
            after = np.count_nonzero(np.asarray(state['stones']))
            captures += max(0, before + int(action < size * size) - after)
            self.assertTrue(bool(state['valid']))
        return captures

    def test_random_3x3_and_9x9_all_legal_children(self):
        captured = sum(self.exercise(size, 91312301 + seed, plies)
                       for size, plies in [(3, 55), (9, 160)] for seed in range(3))
        self.assertGreater(captured, 20)

    def test_capture_history_and_invalid_requests_preserve_state(self):
        tape = [0, 1, 3, 4, 8, 6, 0, 9]
        state, observation = jax.jit(lambda a: jax_go.replay(a, len(tape), 3, .5, 16))(np.asarray(tape, np.int32))
        exact = Replay(self.native, {'size': 3, 'komi': .5, 'scoring': 'raw_area'}, 16)([tape])[0][-1]
        np.testing.assert_array_equal(observation, exact)
        step = jax.jit(lambda s, a: jax_go.advance(s, a, 3))
        for action in [-1, 10, 0]:
            bad = step(state, np.int32(action))
            self.assertFalse(bool(bad['valid']))
            for key in set(state) - {'valid'}:
                np.testing.assert_array_equal(bad[key], state[key])
        terminal = step(state, np.int32(9))
        self.assertEqual(int(terminal['passes']), 2)
        self.assertFalse(np.asarray(jax_go.candidates(terminal, 3)[1]).any())
        invalid = step(terminal, np.int32(9))
        self.assertFalse(bool(invalid['valid']))

    def test_lossless_pack_across_word_boundaries(self):
        stones = np.random.default_rng(9).integers(0, 3, (5, 361), dtype=np.uint8)
        words = np.asarray(jax_go.pack(stones))
        decoded = ((words[..., None] >> (2 * np.arange(16, dtype=np.uint32))) & 3).reshape(5, -1)[:, :361]
        np.testing.assert_array_equal(decoded, stones)


if __name__ == '__main__':
    unittest.main()
