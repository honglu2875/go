"""Retained speculative caches survive acceptance, rejection and continuation."""
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
import loop
import model
import speculate


class LoopTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = Path(os.environ['GOZERO_TEST_NATIVE_RECEIPT']); receipt = json.loads(path.read_text())
        cls.native = load_library(path.parent / receipt['filename'], receipt['binary_sha256'])

    def test_multiple_packets_and_forced_repair_match_native_and_full_model(self):
        c = {**json.loads(Path(__file__).with_name('cpu_qualification.json').read_text())['model'], 'max_positions': 32}
        params = model.initialize(91312541, c); size = 3; capacity = 24; horizon = 3
        tapes = [[0, 1], [0, 1, 3, 4]]
        replay = Replay(self.native, {'size': 3, 'komi': .5, 'scoring': 'raw_area'}, capacity)
        counts = np.asarray([len(t) + 1 for t in tapes], np.int32)
        obs = np.zeros((2, 5, 3, 3, 6), np.float32); actions = np.zeros((2, 5), np.int32)
        for row, tape in enumerate(tapes):
            obs[row, :len(tape) + 1] = replay([tape])[0]; actions[row, :len(tape)] = tape
        deep, cache = model.forward(params, obs, actions, counts, c, with_cache=True, network_version=3, cache_positions=capacity)
        root = jax.tree.map(lambda x: x[jnp.arange(2), counts - 1], deep)
        state, _ = jax.jit(jax.vmap(lambda a, n: jax_go.replay(a, n, size, .5, capacity)))(actions, counts - 1)
        packet = jax.jit(lambda ca, r, s, k, left: loop.packet(params, ca, r, s, k, left, c,
            horizon=horizon, depth=1, size=3, komi=.5, version=3))
        settle = jax.jit(lambda before, r, s, rows, verified, predicted, a, replacement:
            loop.settle(params, before, r, s, rows, verified, predicted, a, replacement, c, size=3, komi=.5, version=3))
        rng = np.random.default_rng(91312549); transitions = 0; repairs = 0
        for iteration in range(4):
            rows, verified, predicted = packet(cache, root, state, jax.random.split(jax.random.key(100 + iteration), 2), jnp.asarray([3, 2]))
            host_rows = jax.device_get(rows)
            np.testing.assert_array_equal(host_rows['p'][:, 0], host_rows['q'][:, 0])
            decisions = speculate.resolve(host_rows, rng.random((2, horizon)), rng.random((2, horizon)))
            if iteration == 0:
                # Independently force a zero-prefix repair on one row. settle
                # must handle it even though root anchoring normally accepts 1.
                replacement = int(np.flatnonzero(host_rows['legal'][0, 0, :-1])[0])
                decisions[0] = {'actions': [replacement], 'accepted_draft_moves': 0, 'replacement': replacement}
            accepted = jnp.asarray([d['accepted_draft_moves'] for d in decisions], jnp.int32)
            replacements = jnp.asarray([-1 if d['replacement'] is None else d['replacement'] for d in decisions], jnp.int32)
            cache, root, state, repaired = settle(cache, root, state, rows, verified, predicted, accepted, replacements)
            repairs += int(np.asarray(repaired).sum())
            self.assertTrue(np.asarray(cache['valid']).all()); self.assertTrue(np.asarray(state['valid']).all())
            for game, decision in enumerate(decisions):
                tapes[game].extend(decision['actions']); transitions += len(decision['actions'])
                tape = tapes[game]; terminal = tape[-2:] == [9, 9]
                native_actions = np.asarray(tape, np.int32); offsets = np.asarray([0, len(tape)], np.int64)
                _, _, endings = self.native.replay_observations(json.dumps({'size': 3, 'komi': .5, 'scoring': 'raw_area'}), native_actions, offsets)
                self.assertEqual(json.loads(endings)[0]['terminal'], terminal)
                if not terminal:
                    exact = replay([tape])[0]
                    np.testing.assert_array_equal(np.asarray(jax_go.observe(jax.tree.map(lambda x: x[game], state), 3, .5)), exact[-1])
                    full = model.forward(params, exact[None], np.asarray([tape + [9]], np.int32), np.asarray([len(tape) + 1], np.int32), c)
                    for key in root:
                        np.testing.assert_allclose(np.asarray(root[key])[game], np.asarray(full[key])[0, -1], atol=5e-5, rtol=5e-5)
                expected = (len(tape) + 1 - int(terminal)) * model.layout(3, c)['stride'] - 1
                self.assertEqual(int(cache['lengths'][game]), expected)
        self.assertGreater(transitions, 4); self.assertGreater(repairs, 0)


if __name__ == '__main__': unittest.main()
