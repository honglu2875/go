"""Strict external mask parsing tolerates field order and rejects malformed data."""
import unittest
from qualify_gtp import raw_policy_mask


class RawPolicyTests(unittest.TestCase):
    def response(self):
        return 'policyPass 0.111111 newField 47 symmetry 0 policy 0.111111 0.111111 0.111111 0.111111 NAN 0.111111 0.111111 0.111111 0.111111 whiteWin 0.5'

    def test_order_whitespace_and_nan_mask(self):
        response = '\n\t'.join(self.response().split())
        self.assertEqual(raw_policy_mask(response, 3), [True, True, True, True, False, True, True, True, True, True])

    def test_invalid_probability_or_pass_is_rejected(self):
        for bad in (self.response().replace('NAN', 'inf'), self.response().replace('policyPass 0.111111', 'policyPass NAN'),
                    self.response().replace('0.111111', '0.25')):
            with self.assertRaises(ValueError):
                raw_policy_mask(bad, 3)

    def test_multiple_symmetries_or_missing_fields_are_rejected(self):
        for bad in (self.response() + ' symmetry 1', self.response().replace('policyPass', 'missing'),
                    self.response().replace('symmetry 0', 'symmetry 2')):
            with self.assertRaises(ValueError):
                raw_policy_mask(bad, 3)


if __name__ == '__main__':
    unittest.main()
