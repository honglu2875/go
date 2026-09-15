"""Verify exact state inputs across captures, ko, passes and pending leaves."""
import json
import os
from pathlib import Path
import unittest
import numpy as np
from gozero.native import load_library
from state_features import native_inputs, planes, batch_planes


class NativeStateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Qualification must supply a pinned binary: absence is a failure, not a skipped check.
        cls.native = load_library(Path(os.environ['GOZERO_NATIVE_LIBRARY']), os.environ['GOZERO_NATIVE_SHA256'])

    def test_capture_superko_pass_and_search_leaf_inputs_match_bulk_replay(self):
        size = 5; moves = [7, 12, 11, 8, 17, 18, 0, 14, 13, 25, 24, 12]
        c = {'size': size, 'komi': .5, 'scoring': 'pass_alive_area', 'history': 2,
             'simulations': 0, 'cpuct': 0., 'max_search_edges': 1000}
        all_stones, all_legal = [], []
        for count in range(len(moves) + 1):
            prefix = moves[:count]; game = self.native.Game(json.dumps(c))
            for ply, action in enumerate(prefix):
                game.play(1 + ply % 2, action)
            current_stones = np.asarray(game.state()[4], np.uint8)
            legal = np.zeros(size * size + 1, bool); legal[game.legal()] = True
            request, f = game.start(71); history = game.request_history(request)
            self.assertEqual(history.tolist(), prefix)
            inputs = native_inputs(f, history, size=size, history_planes=2, komi=.5)
            np.testing.assert_array_equal(inputs[0][0], current_stones)
            np.testing.assert_array_equal(inputs[2][0], legal[:-1])
            self.assertEqual(int(inputs[1][0]), 1 + count % 2)
            tape = np.array(prefix + [size * size], np.int32)
            replay_config = {k: c[k] for k in ('size', 'komi', 'scoring')}
            raw_stones, raw_legal, _ = self.native.replay_observations(json.dumps(replay_config), tape, np.array([0, len(tape)], np.int64))
            np.testing.assert_array_equal(raw_stones.reshape(-1, size * size)[-1], current_stones)
            np.testing.assert_array_equal(raw_legal.reshape(-1, size * size + 1)[-1], legal)
            all_stones.append(current_stones); all_legal.append(legal)
            tokens = np.array([[size * size + 1] + prefix], np.int32)
            b = {'tokens': tokens, 'stones': np.array([all_stones]), 'legal': np.array([all_legal])}
            np.testing.assert_array_equal(planes(*inputs, size, .5)[0], batch_planes(b, {'size': size, 'komi': .5})[0, -1])
            if count == 9:
                self.assertEqual(current_stones[12], 0); self.assertFalse(legal[12])
            if count == 11:
                self.assertTrue(legal[12])
            if count == 12:
                self.assertEqual(current_stones[13], 0); self.assertFalse(legal[13])

    def test_corrupt_context_or_fractional_legality_is_rejected(self):
        c = {'size': 3, 'komi': .5, 'history': 2, 'simulations': 0, 'cpuct': 0., 'max_search_edges': 1000}
        game = self.native.Game(json.dumps(c)); request, original = game.start(71)
        history = game.request_history(request)
        for column, bad in ((-4, 0.), (-3, 2.), (-2, .5), (-1, .25), (0, float('nan'))):
            f = np.asarray(original).reshape(9, 8).copy(); f[0, column] = bad
            with self.assertRaises(ValueError):
                native_inputs(f, history, size=3, history_planes=2, komi=.5)


if __name__ == '__main__':
    unittest.main()
