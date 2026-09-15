import importlib.util
from pathlib import Path
import unittest

import numpy as np

from gozero.sequence_batches import Dataset


class CausalBatchTests(unittest.TestCase):
    def dataset(self):
        data = Dataset.__new__(Dataset)
        data.size = 3; data.time = 6
        data.shards = [{'expert_offsets': np.array([0, 4]), 'expert_actions': np.array([0, 1, 9, 9]),
                        'expert_policies': np.eye(10, dtype=np.float32)[[0, 2, 9, 9]],
                        'expert_values': np.array([-1, 1, -1, 1], np.float32),
                        'expert_legal': np.ones((4, 10), bool),
                        'behavior_offsets': np.array([0, 3]), 'behavior_actions': np.array([2, 3, 4])}]
        return data

    def test_teacher_targets_and_observations_are_distinct_and_causally_shifted(self):
        b = self.dataset().batch([('expert', 0, 0), ('behavior', 0, 0), None])
        np.testing.assert_array_equal(b['tokens'], [[10, 0, 1, 9, 11, 11], [10, 2, 3, 11, 11, 11], [10, 11, 11, 11, 11, 11]])
        np.testing.assert_array_equal(b['lengths'], [3, 2, 0])
        np.testing.assert_array_equal(np.argmax(b['policies'][0, :4], -1), [0, 2, 9, 9])
        self.assertEqual(b['observed'][0, 1], 1)  # MCTS target differs from played action.
        self.assertEqual(b['expert_mask'].sum(), 4)
        self.assertEqual(b['behavior_mask'].sum(), 3)
        self.assertEqual(b['value_mask'].sum(), 4)
        self.assertFalse(b['value_mask'][1:].any())
        self.assertFalse(b['expert_mask'][1:].any())
        self.assertFalse(b['behavior_mask'][[0, 2]].any())


class CausalModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import jax
        cls.jax = jax
        path = Path(__file__).resolve().parents[1] / 'research/recipes/causal_distillation/model.py'
        spec = importlib.util.spec_from_file_location('causal_distillation_model', path)
        cls.model = importlib.util.module_from_spec(spec); spec.loader.exec_module(cls.model)
        cls.config = {'size': 3, 'komi': .5, 'width': 8, 'blocks': 1, 'heads': 2, 'max_tokens': 8, 'dtype': 'float32'}
        cls.params = cls.model.initialize(37, cls.config)

    def batch(self):
        return CausalBatchTests().dataset().batch([('expert', 0, 0), ('behavior', 0, 0)])

    def test_future_tokens_cannot_change_prior_predictions(self):
        batch = self.batch()
        first, _ = self.model.prefill(self.params, batch['tokens'], batch['lengths'], self.config)
        changed = batch['tokens'].copy(); changed[:, 2:] = 7
        second, _ = self.model.prefill(self.params, changed, batch['lengths'], self.config)
        np.testing.assert_array_equal(np.asarray(first[:, :2]), np.asarray(second[:, :2]))

    def test_behavior_gradients_are_confined_to_behavior_parameters(self):
        batch = self.batch(); batch['expert_mask'][:] = 0; batch['value_mask'][:] = 0
        grad = self.jax.grad(lambda p: self.model.losses(p, batch, self.config)[0])(self.params)
        self.assertGreater(sum(float(np.abs(a).sum()) for k, v in grad.items() if k.startswith('behavior') for a in self.jax.tree.leaves(v)), 0)
        for key, tree in grad.items():
            if not key.startswith('behavior'):
                self.assertTrue(all(np.count_nonzero(a) == 0 for a in self.jax.tree.leaves(tree)), key)

    def test_expert_gradients_do_not_train_behavior_head_or_use_masked_outcomes(self):
        batch = self.batch(); batch['behavior_mask'][:] = 0
        loss = self.model.losses(self.params, batch, self.config)[0]
        batch['values'][1, :] = 100
        np.testing.assert_array_equal(np.asarray(loss), np.asarray(self.model.losses(self.params, batch, self.config)[0]))
        grad = self.jax.grad(lambda p: self.model.losses(p, batch, self.config)[0])(self.params)
        for key, value in grad.items():
            if key.startswith('behavior'):
                self.assertEqual(np.count_nonzero(value), 0, key)
