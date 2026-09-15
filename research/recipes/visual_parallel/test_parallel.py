"""Real multi-device tensor shards agree with the unsharded causal decoder."""
import json
from pathlib import Path
import unittest
import jax
import numpy as np
if __package__:
    from . import model, parallel, fixtures
else:
    import model, parallel, fixtures


class TensorParallelTests(unittest.TestCase):
    def test_prefill_and_donated_continuations_across_local_meshes(self):
        self.assertEqual(len(jax.local_devices()), 4)
        c = {**json.loads(Path(__file__).with_name('cpu_qualification.json').read_text())['model'],
             'width': 64, 'heads': 4, 'kv_heads': 4, 'mlp_hidden': 128, 'max_positions': 16}
        params = model.initialize(91312411, c); obs, actions, _ = fixtures.inputs(172, 4, 7, 3)
        counts = np.full(4, 4, np.int32)
        full = jax.device_get(jax.jit(lambda p, o, a, n: model.forward(p, o, a, n, c))(
            params, obs, actions, np.full(4, 7, np.int32)))
        for tp in (1, 2, 4):
            layout = parallel.Layout(params, c, tp)
            root, cache = layout.prefill(obs[:, :4], actions[:, :4], counts, capacity=16, version=19)
            for key in full:
                np.testing.assert_allclose(np.asarray(root[key]), full[key][:, :4], atol=2e-5, rtol=2e-5)
            self.assertEqual(cache['keys'][0].addressable_shards[0].data.shape[2], c['kv_heads'] // tp)
            self.assertEqual(layout.params['blocks'][0]['gate'].addressable_shards[0].data.shape[1], c['mlp_hidden'] // tp)
            prefix = layout.put(counts * model.layout(3, c)['stride'] - 1)
            future = layout.put(obs[:, 4:7]); proposed = layout.put(actions[:, 3:6]); added = layout.put(np.full(4, 3, np.int32))
            fn = layout.continuation(version=19)
            compiled = fn.lower(layout.params, cache, future, proposed, added, prefix).compile()
            self.assertGreater(compiled.memory_analysis().alias_size_in_bytes, 0)
            for repetition in range(2):
                result, cache = compiled(layout.params, cache, future, proposed, added, prefix)
                for key in full:
                    np.testing.assert_allclose(np.asarray(result[key]), full[key][:, 4:7], atol=3e-5, rtol=3e-5)
                self.assertTrue(np.asarray(cache['valid']).all())
                np.testing.assert_array_equal(np.asarray(cache['lengths']), (counts + 3) * model.layout(3, c)['stride'] - 1)


if __name__ == '__main__':
    unittest.main()
