"""Suffix rows preserve native legality and full-history visual token bytes."""
import json
import os
from pathlib import Path
import unittest
import numpy as np
from gozero.native import load_library
from gozero.visual_history import Replay


class SuffixTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = Path(os.environ['GOZERO_TEST_NATIVE_RECEIPT']); receipt = json.loads(path.read_text())
        cls.native = load_library(path.parent / receipt['filename'], receipt['binary_sha256'])

    def test_every_suffix_of_capture_pass_and_different_length_histories(self):
        r = Replay(self.native, {'size': 3, 'komi': .5, 'scoring': 'raw_area'}, 16)
        histories = [[0, 1, 3, 4, 8, 6, 0, 9], [9, 0], []]
        full = r(histories)
        for start in range(len(histories[0]) + 1):
            starts = [start, min(start, 2), 0]
            suffix = r.suffix(histories, starts)
            for a, b, s in zip(full, suffix, starts):
                np.testing.assert_array_equal(a[s:], b)

    def test_random_nine_and_nineteen_board_histories_match_every_plane(self):
        rng = np.random.default_rng(912341)
        for size in (9, 19):
            rules = {'size': size, 'komi': 7.5, 'scoring': 'pass_alive_area'}
            game = self.native.Game(json.dumps({**rules, 'history': 1, 'simulations': 0, 'cpuct': 1., 'max_search_edges': 1000}))
            tape = []
            for turn in range(120):
                legal = [int(x) for x in game.legal() if int(x) != size * size]
                if not legal: break
                action = int(rng.choice(legal)); game.play(1 + turn % 2, action); tape.append(action)
            r = Replay(self.native, rules, 256); full = r([tape])[0]
            for start in (0, 1, len(tape) // 2, len(tape) - 1, len(tape)):
                np.testing.assert_array_equal(full[start:], r.suffix([tape], [start])[0])

    def test_skipped_illegal_prefix_and_bad_offsets_are_rejected(self):
        r = Replay(self.native, {'size': 3, 'komi': .5, 'scoring': 'raw_area'}, 16)
        for tape, starts in [([0, 0, 1], [2]), ([9, 9, 1], [2]), ([0], [-1]), ([0], [2]), ([0], []), ([0], [True])]:
            with self.assertRaises(ValueError): r.suffix([tape], starts)
        self.assertEqual(r.suffix([], []), [])


if __name__ == '__main__': unittest.main()
