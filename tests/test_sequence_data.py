import copy
import unittest

import numpy as np

from gozero.sequence_data import split_game, validate_game, validate_teacher


class SequenceDataTests(unittest.TestCase):
    def setUp(self):
        self.game = {'game_id': 7, 'size': 3, 'komi': .5, 'scoring': 'pass_alive_area',
                     'truncated': False, 'actions': [0, 1, 9, 9], 'networks': [2, 2, 3, 3],
                     'white_score': .5}
        self.x = np.zeros((4, 3, 3, 12), np.float32)
        self.x[..., -1] = 1
        self.pi = np.eye(10, dtype=np.float32)[self.game['actions']]
        self.z = np.array([-1, 1, -1, 1], np.float32)
        self.meta = np.array([[7, n, a, 16, 14, 2] for a, n in
                              zip(self.game['actions'], self.game['networks'])], np.uint64)

    def check(self, **kw):
        return validate_teacher(self.game, self.x, self.pi, self.z, self.meta,
                                size=3, simulations=16, **kw)

    def test_exact_whole_game_and_legal_pass(self):
        validate_game(self.game, game_id=7, size=3, komi=.5, scoring='pass_alive_area')
        full, legal = self.check()
        self.assertTrue(full)
        self.assertEqual(legal.shape, (4, 10))
        self.assertTrue(legal[:, -1].all())

    def test_only_matching_overwritten_suffix_may_be_excluded(self):
        self.x, self.pi, self.z, self.meta = (a[1:] for a in (self.x, self.pi, self.z, self.meta))
        with self.assertRaisesRegex(ValueError, 'Partial'):
            self.check()
        self.assertFalse(self.check(allow_suffix=True)[0])
        self.meta[0, 1] = 1
        with self.assertRaisesRegex(ValueError, 'align'):
            self.check(allow_suffix=True)

    def test_changed_action_or_network_and_wrong_perspective_are_rejected(self):
        for column in (1, 2):
            original = self.meta.copy()
            self.meta[0, column] += 1
            with self.assertRaisesRegex(ValueError, 'align'):
                self.check()
            self.meta = original
        self.z *= -1
        with self.assertRaisesRegex(ValueError, 'side-to-move'):
            self.check()

    def test_policy_mass_on_illegal_action_and_unnormalized_targets_are_rejected(self):
        self.x[0, 0, 0, -1] = 0
        with self.assertRaisesRegex(ValueError, 'legality'):
            self.check()
        self.x[..., -1] = 1
        self.pi *= .5
        with self.assertRaisesRegex(ValueError, 'normalization'):
            self.check()

    def test_capped_observations_are_valid_without_outcome_and_never_expert(self):
        self.game.update(truncated=True, white_score=None, actions=[0, 1, 2, 3])
        validate_game(self.game, game_id=7, size=3, komi=.5, scoring='pass_alive_area')
        with self.assertRaisesRegex(ValueError, 'terminal status'):
            self.check()
        self.game['white_score'] = 0
        with self.assertRaisesRegex(ValueError, 'assigned outcome'):
            validate_game(self.game, game_id=7, size=3, komi=.5, scoring='pass_alive_area')

    def test_complete_game_requires_finite_score_and_two_passes(self):
        original = copy.deepcopy(self.game)
        for changes in ({'white_score': float('nan')}, {'white_score': 100}, {'actions': [0, 1, 2, 9]}):
            self.game = {**original, **changes}
            with self.assertRaisesRegex(ValueError, 'double-pass'):
                validate_game(self.game, game_id=7, size=3, komi=.5, scoring='pass_alive_area')

    def test_whole_game_split_is_shared_by_roles_and_bound_to_source(self):
        # Independent integer implementation checks the byte encoding, rather
        # than statistically testing a pseudorandom split with a flaky bound.
        import hashlib
        for identity in (0, 7, 2**64 - 1):
            source = 'a' * 64
            bucket = int(hashlib.sha256((source + ':' + format(identity, '016x')).encode()).hexdigest()[:16], 16) % 10000
            self.assertEqual(split_game(source, identity), 0 if bucket < 8000 else 1 if bucket < 9000 else 2)
        self.assertNotEqual([split_game('a' * 64, i) for i in range(100)],
                            [split_game('b' * 64, i) for i in range(100)])
