"""Fused drafting consumes exact native histories and coupling keeps target law."""
import json
import os
from pathlib import Path
import unittest
import jax
import jax.numpy as jnp
import numpy as np
from gozero import jax_go
from gozero.native import load_library
from gozero.visual_history import Replay
if __package__:
    from . import model, speculate
else:
    import model, speculate


class SpeculationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = Path(os.environ['GOZERO_TEST_NATIVE_RECEIPT']); receipt = json.loads(path.read_text())
        cls.native = load_library(path.parent / receipt['filename'], receipt['binary_sha256'])

    def test_one_graph_exact_draft_and_deep_causal_verification(self):
        c = {**json.loads(Path(__file__).with_name('cpu_qualification.json').read_text())['model'], 'max_positions': 32}
        params = model.initialize(91312341, c); size = 3; capacity = 24; horizon = 4; depth = 1
        tapes = [[0, 1, 3, 4], [0, 1, 3, 4, 8, 6, 0, 9]]
        replay = Replay(self.native, {'size': size, 'komi': .5, 'scoring': 'raw_area'}, capacity)
        exact = replay(tapes); counts = np.asarray([len(t) + 1 for t in tapes], np.int32)
        obs = np.zeros((2, 12, size, size, 6), np.float32); actions = np.zeros((2, 12), np.int32)
        for row, tape in enumerate(tapes):
            obs[row, :len(tape) + 1] = exact[row]; actions[row, :len(tape)] = tape
        (deep, exits), cache = jax.jit(lambda p, o, a, n: model.forward(p, o, a, n, c, with_cache=True,
            network_version=3, cache_positions=capacity, return_exits=(depth,)))(params, obs, actions, counts)
        root = jax.tree.map(lambda x: x[jnp.arange(2), counts - 1], deep)
        shallow = jax.tree.map(lambda x: x[jnp.arange(2), counts - 1], exits[depth])
        state, _ = jax.jit(jax.vmap(lambda a, n: jax_go.replay(a, n, size, .5, capacity)))(actions, counts - 1)
        keys = jax.random.split(jax.random.key(91), 2); view = jnp.asarray([1, 2])
        compiled = jax.jit(lambda p, ca, r, q, s, k, v: speculate.packet(p, ca, r, q, s, k, v, c,
            horizon=horizon, draft_depth=depth, size=size, komi=.5, network_version=3))
        rows, updated, end = jax.device_get(compiled(params, cache, root, shallow, state, keys, view))
        for game, tape in enumerate(tapes):
            for t in range(horizon):
                if not rows['active'][game, t]:
                    break
                current = replay([tape])[0]
                legal = np.r_[current[-1, ..., 5].reshape(-1) > .5, True]
                np.testing.assert_array_equal(rows['legal'][game, t], legal)
                full = model.forward(params, current[None], np.asarray([tape + [9]], np.int32),
                                     np.asarray([len(tape) + 1], np.int32), c)
                probability = np.asarray(jax.nn.softmax(jnp.where(legal, full['expert_logits'][0, -1], -1e30)))
                np.testing.assert_allclose(rows['p'][game, t], probability, atol=3e-5, rtol=3e-5)
                chosen = int(rows['actions'][game, t]); self.assertTrue(legal[chosen]); tape = tape + [chosen]
                if rows['appended'][game, t]:
                    np.testing.assert_array_equal(rows['future_observations'][game, t], replay([tape])[0][-1])
                else:
                    self.assertEqual(tape[-2:], [9, 9])
            self.assertTrue(end['valid'][game])
        expected_lengths = np.asarray(cache['lengths']) + rows['appended'].sum(1) * model.layout(size, c)['stride']
        np.testing.assert_array_equal(updated['lengths'], expected_lengths)
        self.assertTrue(updated['valid'].all())

    def test_maximal_coupling_preserves_target_and_stops_on_replacement(self):
        rng = np.random.default_rng(91312351); trials = 25000
        q = np.asarray([.15, .55, .30]); p = np.asarray([.70, .05, .25])
        chosen = rng.choice(3, size=(trials, 3), p=q).astype(np.int32)
        rows = {'q': np.broadcast_to(q, (trials, 3, 3)), 'p': np.broadcast_to(p, (trials, 3, 3)),
                'actions': chosen, 'active': np.ones((trials, 3), bool)}
        results = speculate.resolve(rows, rng.random((trials, 3)), rng.random((trials, 3)))
        frequency = np.bincount([r['actions'][0] for r in results], minlength=3) / trials
        np.testing.assert_allclose(frequency, p, atol=.012, rtol=0)
        for r in results:
            self.assertEqual(len(r['actions']), r['accepted_draft_moves'] + int(r['replacement'] is not None))
            self.assertGreater(len(r['actions']), 0)
        rows['p'] = rows['q']
        results = speculate.resolve(rows, rng.random((trials, 3)), rng.random((trials, 3)))
        self.assertTrue(all(r['accepted_draft_moves'] == 3 and r['replacement'] is None for r in results))


if __name__ == '__main__':
    unittest.main()
