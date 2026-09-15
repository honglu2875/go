"""Outcome-censoring and paired-unit checks for the registered score audit."""
import unittest

from analyze_matches import paired_comparison


class PairedScoreTest(unittest.TestCase):
    openings = [['A1', 'B2'], ['A2', 'B1']]
    criterion = {'bootstrap_seed': 91268919, 'paired_bootstrap_replicates': 20000,
                 'confidence': .95, 'minimum_off_completion_fraction': .95,
                 'minimum_lower_outer_interval_strictly_above': 0.}

    def games(self, points):
        return [{'pair': i, 'candidate_color': c, 'opening': opening,
                 'status': 'completed' if points is not None else 'truncated',
                 'candidate_points': points}
                for i, opening in enumerate(self.openings) for c in ('B', 'W')]

    def compare(self, on, off):
        return paired_comparison(on, off, self.openings, self.criterion)

    def test_fixed_positive_contrast(self):
        result = self.compare(self.games(0.), self.games(1.))
        self.assertEqual(result['paired_bootstrap_95_outer_interval'], [1., 1.])
        self.assertTrue(result['registered_criterion_met'])

    def test_equal_arms_do_not_pass_strict_gate(self):
        result = self.compare(self.games(1.), self.games(1.))
        self.assertEqual(result['paired_bootstrap_95_outer_interval'], [0., 0.])
        self.assertFalse(result['registered_criterion_met'])

    def test_control_caps_remain_unknown(self):
        result = self.compare(self.games(None), self.games(1.))
        self.assertEqual(result['off_minus_on_score_bounds'], [0., 1.])
        self.assertEqual(result['off_completion_fraction'], 1.)
        self.assertFalse(result['registered_criterion_met'])

    def test_candidate_caps_fail_completion(self):
        result = self.compare(self.games(0.), self.games(None))
        self.assertEqual(result['off_minus_on_score_bounds'], [0., 1.])
        self.assertEqual(result['off_completion_fraction'], 0.)
        self.assertFalse(result['registered_criterion_met'])

    def test_assigned_cap_is_rejected(self):
        on = self.games(0.); on[0]['status'] = 'truncated'
        with self.assertRaises(ValueError):
            self.compare(on, self.games(1.))

    def test_duplicated_color_is_rejected(self):
        on = self.games(0.); on[1] = on[0]
        with self.assertRaises(ValueError):
            self.compare(on, self.games(1.))


if __name__ == '__main__':
    unittest.main()
