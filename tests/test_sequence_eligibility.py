"""Training population replacement must never change held-out populations."""
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from gozero.checkpoints import sha256
from gozero.sequence_batches import Dataset


class Eligibility(unittest.TestCase):
    def build(self, directory, eligibility):
        arrays = {}
        for role in ('expert', 'behavior'):
            arrays.update({role + '_actions': np.zeros(6, np.int32),
                           role + '_offsets': np.arange(7, dtype=np.int64),
                           role + '_splits': np.asarray([0, 0, 1, 1, 2, 2], np.uint8)})
        if eligibility is not None:
            arrays['expert_training_eligible'] = eligibility
        np.savez(directory / 'shard.npz', **arrays)
        (directory / 'evidence.json').write_text('{}')
        manifest = {'kind': 'causal_teacher_dataset', 'spec': {'size': 3, 'max_game_moves': 2},
                    'shards': [{'id': 0, 'arrays': 'shard.npz', 'sha256': sha256(directory / 'shard.npz'),
                                'evidence': 'evidence.json', 'evidence_sha256': sha256(directory / 'evidence.json')}]}
        (directory / 'manifest.json').write_text(json.dumps(manifest))
        return Dataset(directory, sha256(directory / 'manifest.json'))

    def test_only_training_changes(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)
            data = self.build(path, np.asarray([False, True, False, False, False, False]))
            self.assertEqual(data.indices['expert', 0], [(0, 1)])
            self.assertEqual(data.indices['behavior', 0], [(0, 0), (0, 1)])
            self.assertEqual(data.indices['expert', 1], [(0, 2), (0, 3)])
            self.assertEqual(data.indices['expert', 2], [(0, 4), (0, 5)])

    def test_absent_is_backwards_compatible(self):
        with tempfile.TemporaryDirectory() as d:
            data = self.build(Path(d), None)
            self.assertEqual(data.indices['expert', 0], [(0, 0), (0, 1)])

    def test_rejects_malformed_and_empty_training(self):
        for value in (np.ones(6, np.uint8), np.ones(5, np.bool_), np.zeros(6, np.bool_)):
            with self.subTest(value=value), tempfile.TemporaryDirectory() as d:
                with self.assertRaises(ValueError): self.build(Path(d), value)


if __name__ == '__main__': unittest.main()
