"""Scientific guards for unresolved caps, pairing and the registered threshold."""
import unittest
from paired_score import paired_comparison


class PairedScoreTest(unittest.TestCase):
    openings = [[str(i), str(i + 1)] for i in range(32)]
    criterion = {'opening_units': 32, 'bootstrap_seed': 91310317, 'bootstrap_replicates': 20000,
                 'confidence': .95, 'minimum_each_arm_completion_fraction': .98,
                 'minimum_lower_endpoint_strictly_above': 0., 'scope': 'Synthetic fixture'}

    def games(self, points):
        return [{'pair': i, 'candidate_color': color, 'opening': opening, 'status': 'completed', 'candidate_points': points}
                for i, opening in enumerate(self.openings) for color in ('B', 'W')]

    def test_fixed_gain_and_equal_arms(self):
        positive = paired_comparison(self.games(0.), self.games(1.), self.openings, self.criterion)
        self.assertEqual(positive['paired_bootstrap_95_outer_interval'], [1., 1.])
        self.assertTrue(positive['registered_external_criterion_met'])
        null = paired_comparison(self.games(1.), self.games(1.), self.openings, self.criterion)
        self.assertEqual(null['paired_bootstrap_95_outer_interval'], [0., 0.])
        self.assertFalse(null['registered_external_criterion_met'])

    def test_caps_in_either_arm_keep_bounds_and_enforce_completion(self):
        for arm in (0, 1):
            games = [self.games(0.), self.games(1.)]
            for game in games[arm][:2]: game.update(status='truncated', candidate_points=None)
            result = paired_comparison(*games, self.openings, self.criterion)
            self.assertFalse(result['minimum_completion_criterion_met'])
            self.assertFalse(result['registered_external_criterion_met'])
            self.assertGreater(result['anneal_minus_inherit_score_bounds'][1], result['anneal_minus_inherit_score_bounds'][0])

    def test_assigned_caps_and_missing_colors_are_rejected(self):
        control = self.games(0.); candidate = self.games(1.); control[0]['status'] = 'truncated'
        with self.assertRaises(ValueError): paired_comparison(control, candidate, self.openings, self.criterion)
        control = self.games(0.); control[1] = control[0]
        with self.assertRaises(ValueError): paired_comparison(control, candidate, self.openings, self.criterion)


if __name__ == '__main__': unittest.main()
