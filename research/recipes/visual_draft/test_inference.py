"""Search-path KV rewinds agree with full causal inference on exact Rust boards."""
import json
import os
from pathlib import Path
import unittest
import jax
import numpy as np
from gozero.native import load_library
from gozero.visual_history import observation_sha256

if __package__:
    from . import inference, model, observations
else:
    import inference, model, observations


class InferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        receipt_path = Path(os.environ['GOZERO_TEST_NATIVE_RECEIPT'])
        receipt = json.loads(receipt_path.read_text())
        cls.native = load_library(receipt_path.parent / receipt['filename'], receipt['binary_sha256'])
        cls.c = {**json.loads(Path(__file__).with_name('cpu_qualification.json').read_text())['model'],
                 'max_positions': 16}
        cls.params = model.initialize(171, cls.c)
        cls.rules = {'size': 3, 'komi': .5, 'scoring': 'raw_area'}

    def runner(self, depth=None):
        return inference.Runner(self.params, self.c, self.native, self.rules, slots=2,
            cache_positions=16, network_version=19, exit_depth=depth, max_block=4)

    def test_batched_siblings_captures_passes_resets_and_prefix_hits(self):
        runner = self.runner()
        batches = [{0: [0, 1, 3, 4], 1: [9, 0]},
                   {0: [0, 1, 3, 5], 1: [9]},
                   {0: [0, 1, 3, 4, 8, 6], 1: []},
                   {0: [0, 1], 1: [0, 1, 3, 4, 8, 6, 0]},
                   {0: [9, 1]}, {0: [], 1: []}]
        for batch in batches:
            exact = dict(zip(batch, runner.replay(list(batch.values()))))
            request = {slot: (h, observation_sha256(exact[slot][-1])) for slot, h in batch.items()}
            old_paths = list(runner.paths)
            result = runner.score(request)
            for slot, tape in batch.items():
                actions = np.asarray([tape + [9]], np.int32); count = np.asarray([len(tape) + 1], np.int32)
                full = model.forward(self.params, exact[slot][None], actions, count, self.c)
                for key in full:
                    np.testing.assert_allclose(result[slot][key], full[key][0, -1], atol=4e-5, rtol=4e-5)
            for slot in set(range(2)) - set(batch):
                self.assertEqual(runner.paths[slot], old_paths[slot])
        self.assertGreater(runner.stats['prefix_hits'], 0)
        self.assertGreater(runner.stats['dispatches'], 0)
        self.assertGreater(runner.compilation['1']['memory_bytes']['alias_size_in_bytes'], 0)

    def test_invalid_batch_preserves_cache_and_next_valid_request(self):
        runner = self.runner()
        old = jax.device_get(runner.cache)
        for request in [{0: ([0, 0], '0' * 64)}, {0: ([9, 9], '0' * 64)},
                        {0: ([0], '0' * 64)}, {0: ([0] * 16, '0' * 64)},
                        {2: ([], '0' * 64)}]:
            with self.assertRaises(ValueError):
                runner.score(request)
            self.assertEqual(runner.paths, [(), ()])
            for a, b in zip(jax.tree.leaves(old), jax.tree.leaves(jax.device_get(runner.cache))):
                np.testing.assert_array_equal(a, b)
        exact = runner.replay([[0]])[0]
        result = runner.score({0: ([0], observation_sha256(exact[-1]))})
        self.assertTrue(np.isfinite(result[0]['expert_logits']).all())

    def test_replayed_rows_match_independent_game_features(self):
        tape = [0, 1, 3, 4, 8, 6, 0, 9]
        runner = self.runner(depth=1); exact = runner.replay([tape])[0]
        game = self.native.Game(json.dumps({**self.rules, 'history': 1, 'simulations': 0,
                                           'cpuct': 1., 'max_search_edges': 100}))
        for i in range(len(tape) + 1):
            request, features = game.start(19)
            np.testing.assert_array_equal(observations.from_native(features.reshape(3, 3, 6)), exact[i])
            request, _ = game.evaluate(request, np.zeros(10, np.float32), 0.)
            self.assertIsNone(request); game.finish()
            if i < len(tape):
                game.play(1 + i % 2, tape[i])
        result = runner.score({0: (tape, observation_sha256(exact[-1]))})[0]
        full = model.forward(self.params, exact[None], np.asarray([tape + [9]], np.int32),
            np.asarray([len(tape) + 1], np.int32), self.c, exit_depth=1)
        for key in full:
            np.testing.assert_allclose(result[key], full[key][0, -1], atol=4e-5, rtol=4e-5)


if __name__ == '__main__':
    unittest.main()
