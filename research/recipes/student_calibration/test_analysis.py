"""Check calibration endpoints, uncertainty and legal/pass normalization."""
import unittest
import numpy as np
from analyze import expert_metrics, behavior_metrics


class CalibrationMetricsTest(unittest.TestCase):
    c = {'calibration_edges': [i / 10 for i in range(11)], 'saturation_absolute_value': .95}

    def test_certain_correct_values_include_both_probability_endpoints(self):
        logits = np.array([[30., -30.], [-30., 30.]])
        result = expert_metrics(logits, np.array([-1., 1.]), np.eye(2), np.ones((2, 2), bool), np.array([-1., 1.]), self.c)
        self.assertEqual(result['value_loss'], 0.)
        self.assertEqual(result['win_brier_score'], 0.)
        self.assertEqual(result['value_expected_calibration_error'], 0.)
        self.assertEqual(result['calibration'][0]['positions'], 1)
        self.assertEqual(result['calibration'][-1]['positions'], 1)
        self.assertEqual(result['play_top1'], 1.)

    def test_uncertain_balanced_values_can_be_calibrated_at_high_mse(self):
        result = expert_metrics(np.zeros((2, 2)), np.zeros(2), np.eye(2), np.ones((2, 2), bool), np.array([-1., 1.]), self.c)
        self.assertEqual(result['value_loss'], 1.)
        self.assertEqual(result['win_brier_score'], .25)
        self.assertEqual(result['value_expected_calibration_error'], 0.)
        self.assertEqual(result['saturated_positions'], 0)

    def test_raw_illegal_mass_is_not_removed_from_the_policy_metric(self):
        result = expert_metrics(np.zeros((1, 2)), np.zeros(1), np.array([[0., 1.]]), np.array([[False, True]]), np.ones(1), self.c)
        self.assertEqual(result['illegal_probability'], .5)
        self.assertEqual(result['raw_pass_probability'], .5)
        self.assertEqual(result['legal_normalized_pass_probability'], 1.)
        self.assertAlmostEqual(result['expert_kl'], np.log(2))

    def test_behavior_uses_observed_actions_without_value_or_expert_targets(self):
        result = behavior_metrics(np.array([[0., np.log(3.)]]), np.array([1]))
        self.assertAlmostEqual(result['behavior_loss'], -np.log(.75))
        self.assertEqual(result['behavior_top1'], 1.)
        self.assertNotIn('value_loss', result)


if __name__ == '__main__':
    unittest.main()
