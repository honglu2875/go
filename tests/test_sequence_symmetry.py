"""Spatial targets and transformed complete histories obey the same Rust rules."""
import json
import os
from pathlib import Path
import unittest
import numpy as np
from gozero.native import load_library
from gozero.sequence_symmetry import action_map, augment, transform_grid
from gozero.visual_history import Replay


class SequenceSymmetryTests(unittest.TestCase):
    def test_all_eight_capture_pass_histories_match_independent_native_replay(self):
        path = Path(os.environ['GOZERO_TEST_NATIVE_RECEIPT']); receipt = json.loads(path.read_text())
        native = load_library(path.parent / receipt['filename'], receipt['binary_sha256'])
        replay = Replay(native, {'size': 3, 'komi': .5, 'scoring': 'pass_alive_area'}, 16)
        tape = [0, 1, 3, 4, 8, 6, 0, 9]
        exact = replay([tape])[0]; n = len(exact)
        actions = np.asarray([tape + [9]] * 8, np.int32)
        policies = np.eye(10, dtype=np.float32)[actions]
        owner = np.arange(9, dtype=np.float32).reshape(1, 1, 3, 3).repeat(8, 0).repeat(n, 1)
        batch = {'observations': np.repeat(exact[None], 8, 0), 'actions': actions,
                 'expert_policies': policies, 'ownership': owner, 'counts': np.full(8, n, np.int32),
                 'expert_mask': np.ones((8, n), np.float32), 'outcomes': np.ones((8, n), np.float32)}
        result = augment(batch, np.arange(8))
        for code in range(8):
            mapped = action_map(3, code)[tape].tolist()
            np.testing.assert_array_equal(result['observations'][code], replay([mapped])[0])
            np.testing.assert_array_equal(np.argmax(result['expert_policies'][code], -1), result['actions'][code])
            np.testing.assert_array_equal(result['ownership'][code], transform_grid(owner[code], code, (1, 2)))
            np.testing.assert_array_equal(result['expert_policies'][code, :, -1], policies[code, :, -1])
            self.assertEqual(action_map(3, code)[9], 9)
        for key in ('counts', 'expert_mask', 'outcomes'):
            np.testing.assert_array_equal(result[key], batch[key])
        np.testing.assert_array_equal(batch['observations'][0], exact)


if __name__ == '__main__':
    unittest.main()
