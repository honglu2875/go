import json
from pathlib import Path
import tempfile
import unittest
import numpy as np

from gozero import checkpoint_archives, checkpoints


class CheckpointArchiveTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.name = 'runs/pod-test/rank-0/artifacts/checkpoints/turn-000000001/arrays.npz'
        self.path = self.root / self.name
        self.path.parent.parent.mkdir(parents=True)
        self.arrays = {'p': np.arange(256, dtype=np.float32), 'replay': np.zeros((1024, 128), np.float32)}
        self.manifest = checkpoints.write(self.path.parent, state={'turn': 1}, arrays=self.arrays, actors='{"history":[1,2,3]}')

    def tearDown(self):
        self.temporary.cleanup()

    def test_exact_restoration_and_original_manifest_remain_valid(self):
        original = self.path.read_bytes()
        record = checkpoint_archives.archive(self.root, self.name, self.manifest, remove_original=True)
        self.assertFalse(self.path.exists())
        self.assertLess(record['archive_bytes'], len(original) // 10)
        checkpoint_archives.restore(self.root, record['receipt_path'], record['receipt_sha256'])
        self.assertEqual(self.path.read_bytes(), original)
        state, arrays, actors = checkpoints.read(self.path.parent, expected_manifest_sha256=self.manifest)
        self.assertEqual(state, {'turn': 1}); self.assertEqual(json.loads(actors)['history'], [1, 2, 3])
        for key in self.arrays:
            np.testing.assert_array_equal(arrays[key], self.arrays[key])

    def test_corrupt_archive_cannot_remove_or_restore_original(self):
        record = checkpoint_archives.archive(self.root, self.name, self.manifest)
        blob = self.root / record['archive_path']
        blob.chmod(0o644); blob.write_bytes(b'corrupted')
        with self.assertRaises((ValueError, OSError)):
            checkpoint_archives.archive(self.root, self.name, self.manifest, remove_original=True)
        self.assertTrue(self.path.exists())
        with self.assertRaises(ValueError):
            checkpoint_archives.restore(self.root, record['receipt_path'], record['receipt_sha256'], destination='copy.npz')
        self.assertFalse((self.root / 'copy.npz').exists())

    def test_mutable_or_changed_source_is_rejected(self):
        self.path.chmod(0o644)
        with self.assertRaises(ValueError):
            checkpoint_archives.archive(self.root, self.name, self.manifest, remove_original=True)
        self.path.write_bytes(b'changed'); self.path.chmod(0o444)
        with self.assertRaises(ValueError):
            checkpoint_archives.archive(self.root, self.name, self.manifest, remove_original=True)
        self.assertTrue(self.path.exists())


if __name__ == '__main__':
    unittest.main()
