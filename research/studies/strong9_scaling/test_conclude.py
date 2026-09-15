"""Decision regressions for conflicting metrics, seeds and time windows."""
import copy
import math
import unittest

import conclude


def fixture(gains=((.01, .01), (.01, .01)), baselines=(.1, .1)):
    output = []
    for seed, gain, baseline in zip(conclude.SEEDS, gains, baselines):
        curve = []
        for turn in range(0, 4097, 256):
            curve.append(dict(turn=turn, cnn=dict(expert_kl=baseline, family_kl=baseline/2),
                              transformer=dict(expert_kl=baseline*(1-gain[0]),
                                               family_kl=baseline/2*(1-gain[1]))))
        output.append(dict(seed=seed, paired_validation=curve))
    return output


class ConclusionTests(unittest.TestCase):
    def test_mean_relative_gain_does_not_weight_by_seed_baseline(self):
        result = conclude.decision(fixture(((.006, .006), (.012, .012)), (.1, 100.)))
        self.assertTrue(result['transformer_improvement_established'])
        self.assertAlmostEqual(result['metrics']['expert_kl']['mean_relative_endpoint_gain'], .009)

    def test_one_seed_cannot_compensate_for_the_other(self):
        result = conclude.decision(fixture(((.1, .1), (-.01, -.01))))
        self.assertFalse(result['transformer_improvement_established'])
        self.assertTrue(result['metrics']['expert_kl']['gates']['mean_relative_gain_at_least_half_percent'])
        self.assertIn('expert_kl.both_seed_endpoints_improved', result['failed_gates'])

    def test_position_gain_cannot_compensate_for_family_regression(self):
        result = conclude.decision(fixture(((.03, -.01), (.03, -.01))))
        self.assertTrue(result['metrics']['expert_kl']['passed'])
        self.assertFalse(result['metrics']['family_kl']['passed'])
        self.assertFalse(result['transformer_improvement_established'])

    def test_small_consistent_gain_does_not_clear_effect_threshold(self):
        result = conclude.decision(fixture(((.004, .004), (.004, .004))))
        self.assertFalse(result['transformer_improvement_established'])
        self.assertTrue(result['metrics']['expert_kl']['gates']['both_seed_endpoints_improved'])
        self.assertIn('expert_kl.mean_relative_gain_at_least_half_percent', result['failed_gates'])

    def test_endpoint_gain_does_not_hide_tail_regression(self):
        evidence = fixture()
        for row in evidence[0]['paired_validation'][-3:-1]:
            row['transformer']['expert_kl'] = .12
        result = conclude.decision(evidence)
        self.assertTrue(result['metrics']['expert_kl']['gates']['both_seed_endpoints_improved'])
        self.assertIn('expert_kl.neither_seed_last_three_mean_regressed', result['failed_gates'])

    def test_best_observed_turn_does_not_replace_endpoint(self):
        evidence = fixture()
        for row in evidence:
            row['paired_validation'][-1]['transformer']['expert_kl'] = .101
            row['relative_endpoint_kl_improvement'] = dict(expert_kl=.99, family_kl=.99)
        result = conclude.decision(evidence)
        self.assertFalse(result['transformer_improvement_established'])
        self.assertLess(result['metrics']['expert_kl']['mean_relative_endpoint_gain'], 0)

    def test_incomplete_or_nonfinite_evidence_is_rejected(self):
        examples = [fixture()[:1]]
        duplicate = fixture();duplicate[1]['seed'] = duplicate[0]['seed'];examples.append(duplicate)
        shortened = fixture();shortened[0]['paired_validation'].pop(4);examples.append(shortened)
        for value in (math.nan, math.inf, -.1):
            corrupt = fixture();corrupt[0]['paired_validation'][3]['cnn']['expert_kl'] = value
            examples.append(corrupt)
        for evidence in examples:
            with self.subTest(evidence=evidence):
                with self.assertRaises(ValueError):
                    conclude.decision(evidence)


if __name__ == '__main__':
    unittest.main()
