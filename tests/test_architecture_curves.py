"""Scientific edge cases for paired curves, including unresolved games."""
import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'eval'))
spec = importlib.util.spec_from_file_location('architecture_curves', ROOT / 'eval/analyze_architecture_curves.py')
curves = importlib.util.module_from_spec(spec)
spec.loader.exec_module(curves)


class CurveStatisticsTests(unittest.TestCase):
    def stats(self, values):
        return curves.curve_statistics(values, draws=10000, seed=19450232)

    def test_known_win_and_null_effect(self):
        values = np.zeros((2, 2, 4, 32, 2, 2))
        result = self.stats(values)
        self.assertEqual(result['bootstrap_95_outer_interval'], [0., 0.])
        self.assertFalse(result['positive_effect_criterion'])
        values[:, 1] = 1
        result = self.stats(values)
        self.assertEqual(result['mean_difference_bounds'], [1., 1.])
        self.assertEqual(result['bootstrap_95_outer_interval'], [1., 1.])
        self.assertTrue(result['positive_effect_criterion'])

    def test_entire_opening_unit_is_retained(self):
        values = np.zeros((2, 2, 4, 32, 2, 2))
        # Every opening contains equally many wins and losses for each arm.
        # Resampling individual budgets/colors would invent uncertainty here.
        values[:, 0, :2] = 1
        values[:, 1, 2:] = 1
        result = self.stats(values)
        self.assertEqual(result['bootstrap_95_outer_interval'], [0., 0.])
        self.assertEqual(result['opening_difference_bounds'], [[0., 0.]] * 32)

    def test_caps_have_full_unassigned_range(self):
        values = np.zeros((2, 2, 4, 32, 2, 2))
        values[..., 1] = 1
        result = self.stats(values)
        self.assertEqual(result['mean_difference_bounds'], [-1., 1.])
        self.assertEqual(result['bootstrap_95_outer_interval'], [-1., 1.])
        self.assertFalse(result['positive_effect_criterion'])
        self.assertEqual(curves.points_interval({'status': 'truncated'}), (0., 1.))
        for game in ({'status': 'failed'}, {'status': 'truncated', 'candidate_points': .5}):
            with self.assertRaises(ValueError):
                curves.points_interval(game)

    def test_positive_average_requires_both_seeds(self):
        values = np.zeros((2, 2, 4, 32, 2, 2))
        values[0, 1] = 1
        values[1, 0, 0] = 1
        result = self.stats(values)
        self.assertEqual(result['mean_difference_bounds'], [.375, .375])
        self.assertEqual(result['seed_mean_difference_bounds']['28'], [-.25, -.25])
        self.assertFalse(result['positive_effect_criterion'])

    def test_missing_and_reversed_intervals_are_rejected(self):
        values = np.zeros((2, 2, 4, 32, 2, 2))
        values[0, 0, 0, 0, 0, 0] = np.nan
        with self.assertRaises(ValueError):
            self.stats(values)
        values[0, 0, 0, 0, 0, 0] = 1
        with self.assertRaises(ValueError):
            self.stats(values)


if __name__ == '__main__':
    unittest.main()
