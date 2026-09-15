"""Optimizer, distributed-gradient, stream and scan recovery qualification."""
import copy
import json
from pathlib import Path
import sys
import unittest

SOURCE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(SOURCE / 'packages/gozero/src'))
import numpy as np
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec as P
from gozero import checkpoints
from gozero.katago_sequence_batches import augment
import katago
import policy_model as model
import sgd_optimizer as optimizer
import position_stream
import checkpoint_storage

SETTINGS = dict(momentum=.9, l2_coefficient=3e-5, warmup_positions=512,
                warmup_per_sample_lr=2e-5, per_sample_lr=6e-5)
C = dict(architecture='katago_nested_policy', width=16, mid_width=8, gpool_width=2,
         policy_width=4, layers=4, dtype='float32', rematerialize=True, microbatch=4,
         max_board_size=19, max_positions=512, norm_epsilon=1e-4)


class Qualification(unittest.TestCase):
    def test_reserved_checkpoint_preserves_standard_bytes(self):
        import tempfile
        rng = np.random.default_rng(1433)
        arrays = {'p':rng.normal(size=(32,32)).astype(np.float32), 'm':rng.normal(size=(32,32)).astype(np.float32)}
        state = {'samples':533,'step':3}
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder); allocation = folder/'reservation'
            checkpoint_storage.reserve(allocation, 65536)
            ordinary = folder/'ordinary'; reserved = folder/'reserved'
            a = checkpoints.write(ordinary, state=state, arrays=arrays, actors='{}', compress=True)
            b = checkpoint_storage.write(reserved, reservation=allocation, state=state, arrays=arrays, scratch_root=folder)
            self.assertEqual(a,b); self.assertFalse(allocation.exists())
            self.assertEqual(checkpoints.sha256(ordinary/'arrays.npz'),checkpoints.sha256(reserved/'arrays.npz'))
            saved, restored, _ = checkpoints.read(reserved,expected_manifest_sha256=a)
            self.assertEqual(saved,state)
            for k in arrays:np.testing.assert_array_equal(arrays[k],restored[k])

    def test_nesterov_matches_numpy_summed_objective(self):
        params = {'weight.weight': jnp.asarray([.4, -.7]), 'bias.beta': jnp.asarray([.2])}
        state = optimizer.initialize(params)
        weights = {k: np.asarray(v).copy() for k, v in params.items()}
        velocity = {k: np.zeros_like(v) for k, v in weights.items()}
        samples = 0
        for count in (256, 256, 256, 21):
            gradient = {k: jnp.asarray(.3 * v + .1) for k, v in weights.items()}
            lr = np.float32(2e-5 if samples <= 512 else 6e-5)
            for k in weights:
                g = np.float32(count) * (np.asarray(gradient[k]) + (np.float32(3e-5) * weights[k] if k.endswith('.weight') else 0.))
                velocity[k] = np.float32(.9) * velocity[k] + g
                weights[k] -= lr * (g + np.float32(.9) * velocity[k])
            params, state, m = optimizer.apply_gradient(params, state, gradient, 1., count, SETTINGS)
            for k in weights:
                np.testing.assert_allclose(params[k], weights[k], rtol=2e-6, atol=2e-7)
                np.testing.assert_allclose(state['momentum'][k], velocity[k], rtol=2e-6, atol=2e-7)
            samples += count
            self.assertEqual(int(state['samples']), samples)
            self.assertEqual(float(m['per_sample_learning_rate']), float(lr))

    def test_folded_repvgg_matches_two_independent_sgd_branches(self):
        name = 'cycles.a.normactconv.conv.weight'
        a = np.ones((1, 3, 3, 2, 2), np.float32) * .2
        b = np.ones((1, 2, 2), np.float32) * -.05
        combined = a.copy(); combined[:, 1, 1] += b
        p = {name: jnp.asarray(combined)}; s = optimizer.initialize(p)
        va = np.zeros_like(a); vb = np.zeros_like(b)
        for i in range(5):
            # The two original branches have identical center data derivatives.
            g = np.ones_like(a) * np.float32(.1 + .03 * i)
            ga = 256 * (g + np.float32(3e-5) * a)
            gb = 256 * (g[:, 1, 1] + np.float32(3e-5) * b)
            va = np.float32(.9) * va + ga; vb = np.float32(.9) * vb + gb
            lr = np.float32(2e-5 if i*256 <= 512 else 6e-5)
            a -= lr * (ga + np.float32(.9)*va); b -= lr * (gb + np.float32(.9)*vb)
            p, s, _ = optimizer.apply_gradient(p, s, {name: jnp.asarray(g)}, 1., 256, SETTINGS)
            expected = a.copy(); expected[:, 1, 1] += b
            np.testing.assert_allclose(p[name], expected, atol=2e-7, rtol=2e-6)

    def test_scan_padding_failure_and_restored_carry(self):
        def objective(p, b):
            loss = jnp.sum((p['x.weight'] - b['target'][0])**2)
            return loss, {'expert_positions': jnp.sum(b['counts'])}
        p = {'x.weight': jnp.asarray([.2, -.1])}; s = optimizer.initialize(p)
        batch = {'target': jnp.arange(12, dtype=jnp.float32).reshape(6, 1, 2)/10.,
                 'counts': jnp.asarray([[256], [256], [0], [256], [21], [0]])}
        compiled = jax.jit(optimizer.scan_updates(objective, SETTINGS))
        actual, state, metrics, healthy = compiled(p, s, batch)
        pp, ss = p, s
        for i in (0, 1, 3, 4):
            one = jax.tree.map(lambda a: a[i], batch)
            (loss, totals), g = jax.value_and_grad(objective, has_aux=True)(pp, one)
            pp, ss, _ = optimizer.apply_gradient(pp, ss, g, loss, totals['expert_positions'], SETTINGS)
        for x, y in zip(jax.tree.leaves((actual, state)), jax.tree.leaves((pp, ss))):
            np.testing.assert_allclose(x, y, atol=2e-7, rtol=2e-6)
        self.assertTrue(bool(healthy)); self.assertEqual(int(state['step']), 4)
        midp, mids, _, _ = compiled(p, s, jax.tree.map(lambda a: a[:3], batch))
        # Serialize all optimizer state through the production checkpoint reader.
        import tempfile
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'checkpoint'
            identity = checkpoints.write(path, state={'step':int(mids['step']), 'samples':int(mids['samples'])},
                arrays={'p': np.asarray(midp['x.weight']), 'm': np.asarray(mids['momentum']['x.weight'])}, actors='{}', compress=True)
            saved, arrays, _ = checkpoints.read(path, expected_manifest_sha256=identity)
            restored = {'momentum': {'x.weight':jnp.asarray(arrays['m'])},
                        'step':jnp.int32(saved['step']), 'samples':jnp.int32(saved['samples'])}
            rp, rs, _, _ = compiled({'x.weight':jnp.asarray(arrays['p'])}, restored, jax.tree.map(lambda a:a[3:], batch))
            for x, y in zip(jax.tree.leaves((rp, rs)), jax.tree.leaves((actual, state))): np.testing.assert_array_equal(x, y)
        poisoned = {**batch, 'target':batch['target'].at[1].set(jnp.nan)}
        failedp, faileds, rows, healthy = compiled(p, s, poisoned)
        self.assertFalse(bool(healthy)); self.assertEqual(int(faileds['step']), 1)
        self.assertFalse(bool(rows['active'][3])); self.assertTrue(np.isfinite(failedp['x.weight']).all())

    def test_distributed_position_objective_and_microbatch(self):
        self.assertEqual(len(jax.devices()), 4)
        rng = np.random.default_rng(1947); n = 16; size = 3
        b = dict(spatial=rng.integers(0, 2, (n, 1, size, size, 22)).astype(np.float32),
                 global_features=rng.normal(size=(n, 1, 19)).astype(np.float32)*.1,
                 policies=rng.uniform(.1, 1., (n, 1, size*size+1)).astype(np.float32),
                 actions=np.zeros((n, 1), np.int32), legal=np.ones((n, 1, size*size+1), bool),
                 counts=np.asarray([1]*13+[0]*3, np.int32))
        b['spatial'][..., 0] = 1; b['spatial'][13:] = 0; b['global_features'][13:] = 0
        b['policies'] /= b['policies'].sum(-1, keepdims=True)
        p = jax.jit(lambda:katago.initialize(47, C))()
        fn = jax.jit(jax.value_and_grad(lambda p:model.losses(p, b, C)[0]))
        loss, g = fn(p)
        mesh = Mesh(np.asarray(jax.devices()), ('data',))
        obj = jax.shard_map(lambda p, b:model.losses(p, b, C, axis_name='data'), mesh=mesh,
            in_specs=(P(), jax.tree.map(lambda _:P('data'), b)), out_specs=P(), check_vma=False)
        (dloss, _), dg = jax.jit(jax.value_and_grad(obj, has_aux=True))(p, b)
        np.testing.assert_allclose(loss, dloss, atol=2e-6, rtol=3e-5)
        for k in g: np.testing.assert_allclose(g[k], dg[k], atol=3e-6, rtol=8e-4, err_msg=k)
        state = optimizer.initialize(p)
        a = jax.jit(lambda p, s, g:optimizer.apply_gradient(p, s, g, loss, 13, SETTINGS))(p, state, g)
        d = jax.jit(lambda p, s, g:optimizer.apply_gradient(p, s, g, dloss, 13, SETTINGS))(p, state, dg)
        for x, y in zip(jax.tree.leaves(a[:2]), jax.tree.leaves(d[:2])): np.testing.assert_allclose(x, y, atol=5e-5, rtol=1e-3)
        other = {**C, 'microbatch':8}
        self.assertEqual(katago.parameter_schema(C), katago.parameter_schema(other))
        out1 = jax.jit(lambda:katago.forward(p, b['spatial'][:, 0], b['global_features'][:, 0], C, training=True))()
        out2 = jax.jit(lambda:katago.forward(p, b['spatial'][:, 0], b['global_features'][:, 0], other, training=True))()
        for x, y in zip(out1, out2): np.testing.assert_allclose(x, y, atol=2e-6, rtol=3e-5)

    def test_position_gather_matches_full_episode_augmentation(self):
        rng = np.random.default_rng(2749); size = 3; n = 17
        class Data: pass
        data = Data(); data.size = size
        source = {'spatial': rng.integers(0, 2, (n, size, size, 22)).astype(np.uint8),
                  'global_features':rng.normal(size=(n,19)).astype(np.float32),
                  'expert_actions':rng.integers(0, size*size+1, n).astype(np.int32),
                  'expert_policies':rng.random((n,size*size+1)).astype(np.float32),
                  'expert_legal':rng.integers(0,2,(n,size*size+1)).astype(bool)}
        data.shards = [source]
        records = np.column_stack([np.zeros(n,np.int32),np.arange(n,dtype=np.int32),np.arange(n,dtype=np.int32)%8])
        for rank in range(4):
            actual = position_stream.batch(data, records, 0, 3, 8, 4, rank)
            for step in range(3):
                for row in range(2):
                    index = step*8+rank*2+row
                    if index >= n:
                        self.assertEqual(actual['counts'][step,row],0); self.assertEqual(actual['spatial'][step,row].sum(),0)
                        continue
                    b = {k:source[v][index:index+1,None].copy() for k,v in [('spatial','spatial'),('global_features','global_features'),('actions','expert_actions'),('policies','expert_policies'),('legal','expert_legal')]}
                    b['counts'] = np.ones(1,np.int32)
                    expected = augment(b,np.asarray([index%8]))
                    for k in actual: np.testing.assert_array_equal(actual[k][step,row],expected[k][0],err_msg=k)

if __name__ == '__main__': unittest.main()
