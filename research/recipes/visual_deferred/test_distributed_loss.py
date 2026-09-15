"""Globally masked manual-shard gradients equal a concatenated-batch reference."""
import json
from pathlib import Path
import unittest
import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

if __package__:
    from . import fixtures, model
else:
    import fixtures, model


class DistributedLossTests(unittest.TestCase):
    def test_unequal_and_empty_rank_masks_preserve_global_gradient(self):
        self.assertEqual(len(jax.devices()), 4, 'Run with four forced CPU devices')
        c = {**json.loads(Path(__file__).with_name('cpu_qualification.json').read_text())['model'], 'layers': 3}
        params = model.initialize(111, c)
        o, a, n = fixtures.inputs(222, 4, 4, 3)
        n[:] = [1, 4, 2, 3]
        batch = fixtures.loss_batch(o, a, n)
        batch['expert_mask'][:] = 0; batch['expert_mask'][0, :1] = 1; batch['expert_mask'][1] = 1
        batch['behavior_mask'][:] = 0; batch['behavior_mask'][2, :2] = 1; batch['behavior_mask'][3, :1] = 1
        batch['value_mask'] = batch['expert_mask'].copy()
        batch = jax.tree.map(jnp.asarray, batch)
        def loss(p, b, axis=None):
            return model.losses(p, b, c, axis_name=axis, exit_depths=(1, 2), exit_loss_weight=.2)[0]
        reference = jax.jit(jax.value_and_grad(loss))(params, batch)
        mesh = Mesh(np.asarray(jax.devices()), ('data',))
        specs = {k: P() if k == 'loss_weights' else P('data') for k in batch}
        sharded = {k: jax.device_put(v, NamedSharding(mesh, specs[k])) for k, v in batch.items()}
        params = jax.device_put(params, NamedSharding(mesh, P()))
        objective = jax.shard_map(lambda p, b: loss(p, b, 'data'), mesh=mesh,
            in_specs=(P(), specs), out_specs=P(), check_vma=False)
        observed = jax.jit(jax.value_and_grad(objective))(params, sharded)
        for expected, actual in zip(jax.tree.leaves(reference), jax.tree.leaves(observed)):
            np.testing.assert_allclose(actual, expected, atol=2e-5, rtol=2e-5)


if __name__ == '__main__':
    unittest.main()
