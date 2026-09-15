import importlib.util
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'eval'))
from analyze_prefetch_learning import paired_statistics


class PrefetchLearningStatisticsTests(unittest.TestCase):
    def games(self, points=.5):
        return [{'pair': p, 'candidate_color': c, 'status': 'completed', 'candidate_points': points}
                for p in range(256) for c in ('B', 'W')]

    def test_balanced_all_draws_pass_registered_margin(self):
        result = paired_statistics(self.games())
        self.assertTrue(result['registered_criterion_met'])
        self.assertEqual(result['scheduled_score_bounds'], [.5, .5])
        self.assertGreater(result['paired_hoeffding_95_outer_interval'][0], .4)

    def test_caps_are_unassigned_and_coverage_gate_is_independent(self):
        games = self.games(1.)
        for g in games[:26]:
            g.update(status='truncated', candidate_points=None)
        result = paired_statistics(games)
        self.assertEqual(result['scheduled_score_bounds'], [486 / 512, 1.])
        self.assertTrue(result['noninferiority_criterion'])
        self.assertFalse(result['minimum_completion_criterion'])
        self.assertFalse(result['registered_criterion_met'])

    def test_missing_duplicate_or_assigned_caps_invalidate_evidence(self):
        games = self.games()
        for corrupted in (games[:-1], [games[0]] + games[:-1]):
            with self.assertRaises(ValueError):
                paired_statistics(corrupted)
        games[0]['status'] = 'truncated'
        with self.assertRaises(ValueError):
            paired_statistics(games)
