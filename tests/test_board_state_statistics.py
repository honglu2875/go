from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'research/recipes/board_state_distillation'))
from analyze_external import paired_comparison


class BoardStateStatisticsTests(unittest.TestCase):
    def games(self, points=0.):
        return [{'pair': p, 'candidate_color': c, 'opening': [str(p)],
                 'status': 'completed', 'candidate_points': points}
                for p in range(32) for c in ('B', 'W')]

    def test_all_wins_against_all_losses(self):
        result = paired_comparison(self.games(0.), self.games(1.))
        self.assertEqual(result['exact_minus_empty_score_bounds'], [1., 1.])
        self.assertEqual(result['paired_bootstrap_95_outer_interval'], [1., 1.])
        self.assertTrue(result['registered_external_criterion_met'])

    def test_identical_varying_opening_outcomes_have_zero_paired_variance(self):
        control = self.games()
        for g in control:
            g['candidate_points'] = float(g['pair'] % 2)
        result = paired_comparison(control, [dict(g) for g in control])
        self.assertEqual(result['paired_bootstrap_95_outer_interval'], [0., 0.])
        self.assertFalse(result['registered_external_criterion_met'])

    def test_caps_preserve_positive_score_bounds_but_fail_completion(self):
        exact = self.games(1.)
        for g in exact[:4]:
            g.update(status='truncated', candidate_points=None)
        result = paired_comparison(self.games(), exact)
        self.assertEqual(result['exact_minus_empty_score_bounds'], [60 / 64, 1.])
        self.assertTrue(result['positive_lower_endpoint'])
        self.assertFalse(result['minimum_completion_criterion_met'])
        self.assertFalse(result['registered_external_criterion_met'])

    def test_control_caps_widen_lower_difference_endpoint(self):
        control = self.games()
        for g in control[:2]:
            g.update(status='truncated', candidate_points=None)
        result = paired_comparison(control, self.games())
        self.assertEqual(result['exact_minus_empty_score_bounds'], [-2 / 64, 0.])

    def test_reject_missing_duplicate_assigned_caps_and_unpaired_openings(self):
        control = self.games()
        for invalid in (control[:-1], [control[0]] + control[:-1]):
            with self.assertRaises(ValueError):
                paired_comparison(control, invalid)
        invalid = self.games(); invalid[0]['status'] = 'truncated'
        with self.assertRaises(ValueError):
            paired_comparison(control, invalid)
        invalid = self.games(); invalid[0]['opening'] = ['different']
        with self.assertRaises(ValueError):
            paired_comparison(control, invalid)


if __name__ == '__main__':
    unittest.main()
