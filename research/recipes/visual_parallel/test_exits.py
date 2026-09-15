"""Early exits preserve causality, use their own KV depth and stop teachers."""
import json
from pathlib import Path
import unittest
import jax
import jax.numpy as jnp
import numpy as np

if __package__:
    from . import fixtures, model
else:
    import fixtures, model


class ExitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.c = json.loads(Path(__file__).with_name('cpu_qualification.json').read_text())['model']
        cls.c = {**cls.c, 'layers': 3}
        cls.p = model.initialize(13, cls.c)
        cls.o, cls.a, cls.n = map(jnp.asarray, fixtures.inputs(14, 2, 4, 3))

    def test_intermediate_reads_equal_standalone_exits_and_target_unchanged(self):
        target, exits = jax.jit(lambda p: model.forward(p, self.o, self.a, self.n, self.c, return_exits=(1, 2)))(self.p)
        direct = model.forward(self.p, self.o, self.a, self.n, self.c)
        for key in target:
            np.testing.assert_allclose(target[key], direct[key], atol=2e-5, rtol=2e-5)
        for depth in (1, 2):
            direct = model.forward(self.p, self.o, self.a, self.n, self.c, exit_depth=depth)
            for key in direct:
                np.testing.assert_allclose(exits[depth][key], direct[key], atol=2e-5, rtol=2e-5)
        changed = model.forward(self.p, self.o.at[:, 2:].set(0), self.a.at[:, 1:].set(1), self.n, self.c, exit_depth=1)
        for key in changed:
            np.testing.assert_allclose(changed[key][:, :2], exits[1][key][:, :2], atol=2e-5, rtol=2e-5)

    def test_exit_cache_matches_full_exit_and_rejects_wrong_depth(self):
        n = jnp.array([2, 2])
        _, cache = model.forward(self.p, self.o[:, :2], self.a[:, :2], n, self.c,
                                 exit_depth=1, with_cache=True, network_version=7)
        self.assertEqual(len(cache['keys']), 1)
        out, updated = model.score_continuation(self.p, cache, self.o[:, 2:], self.a[:, 1:3], n,
                                                self.c, exit_depth=1, network_version=7)
        full = model.forward(self.p, self.o, self.a, self.n, self.c, exit_depth=1)
        for key in out:
            np.testing.assert_allclose(out[key], full[key][:, 2:], atol=2e-5, rtol=2e-5)
        np.testing.assert_array_equal(updated['lengths'], self.n * model.layout(3, self.c)['stride'] - 1)
        with self.assertRaisesRegex(ValueError, 'layer count'):
            model.score_continuation(self.p, cache, self.o[:, 2:], self.a[:, 1:3], n, self.c, network_version=7)

    def test_distillation_stops_teacher_and_trains_each_student_head(self):
        batch = jax.tree.map(jnp.asarray, fixtures.loss_batch(np.asarray(self.o), np.asarray(self.a), np.asarray(self.n)))
        teacher, students = model.forward(self.p, self.o, self.a, self.n, self.c, return_exits=(1, 2))
        def loss(t, s):
            return model.exit_distillation(t, s, batch)[0]
        gt, gs = jax.grad(loss, argnums=(0, 1))(teacher, students)
        self.assertTrue(all(np.count_nonzero(x) == 0 for x in jax.tree.leaves(gt)))
        for depth in (1, 2):
            for role in ('expert', 'behavior'):
                self.assertGreater(float(jnp.linalg.norm(gs[depth][role + '_logits'])), 0)
        loss, metrics = jax.jit(lambda p: model.losses(p, batch, self.c, exit_depths=(1, 2), exit_loss_weight=.2))(self.p)
        self.assertTrue(np.isfinite(loss))
        self.assertGreater(float(metrics['exit_distillation_loss']), 0)

    def test_zero_auxiliary_weight_preserves_supervised_gradient(self):
        batch = jax.tree.map(jnp.asarray, fixtures.loss_batch(np.asarray(self.o), np.asarray(self.a), np.asarray(self.n)))
        control = jax.jit(jax.value_and_grad(lambda p: model.losses(p, batch, self.c)[0]))(self.p)
        observed = jax.jit(jax.value_and_grad(lambda p: model.losses(p, batch, self.c, exit_depths=(1, 2), exit_loss_weight=0)[0]))(self.p)
        for x, y in zip(jax.tree.leaves(control), jax.tree.leaves(observed)):
            np.testing.assert_allclose(x, y, atol=3e-5, rtol=3e-5)


if __name__ == '__main__':
    unittest.main()
