import importlib.util
from pathlib import Path
import tempfile
import unittest

from gozero import checkpoints


@unittest.skipUnless(importlib.util.find_spec('numpy'), 'NumPy is required for numeric checkpoints')
class CheckpointTests(unittest.TestCase):
    def test_compressed_checkpoint_is_exact_deterministic_and_sparse(self):
        import numpy as np
        import zipfile
        with tempfile.TemporaryDirectory() as folder:
            arrays={'features':np.zeros((1024,9,9,12),np.float32),
                    'cursor':np.asarray(123,np.int64),'weights':np.arange(200,dtype=np.float32)}
            paths=[Path(folder)/name for name in ('plain','compressed_a','compressed_b')]
            for i,path in enumerate(paths):
                identity=checkpoints.write(path,state={'step':8},arrays=arrays,actors='{}',compress=i>0)
                state,saved,actors=checkpoints.read(path,expected_manifest_sha256=identity)
                self.assertEqual((state,actors),({'step':8},'{}'))
                for name,value in arrays.items():np.testing.assert_array_equal(saved[name],value)
            self.assertEqual(checkpoints.sha256(paths[1]/'arrays.npz'),checkpoints.sha256(paths[2]/'arrays.npz'))
            self.assertLess((paths[1]/'arrays.npz').stat().st_size,(paths[0]/'arrays.npz').stat().st_size//10)
            with zipfile.ZipFile(paths[1]/'arrays.npz') as archive:
                self.assertTrue(all(m.date_time==(1980,1,1,0,0,0) for m in archive.infolist()))

    def test_published_state_is_exact_and_cannot_be_overwritten(self):
        import numpy as np
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'step-12'
            x = np.arange(12, dtype=np.float32).reshape((3, 4))
            identity = checkpoints.write(path, state={'step': 12, 'cursor': 4}, arrays={'weights': x}, actors='{"rng":127}')
            state, arrays, actors = checkpoints.read(path, expected_manifest_sha256=identity)
            self.assertEqual(state, {'step': 12, 'cursor': 4})
            np.testing.assert_array_equal(arrays['weights'], x)
            self.assertEqual(actors, '{"rng":127}')
            with self.assertRaises(FileExistsError):
                checkpoints.write(path, state={}, arrays={}, actors='{}')
            with self.assertRaisesRegex(ValueError, 'identity'):
                checkpoints.read(path, expected_manifest_sha256='0'*64)

    def test_tampering_extra_files_and_partial_publication_are_rejected(self):
        import numpy as np
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'step-12'
            checkpoints.write(path, state={}, arrays={'x': np.zeros(1)}, actors='{}')
            (path / 'extra').write_text('x')
            with self.assertRaises(ValueError): checkpoints.read(path)
            (path / 'extra').unlink()
            file = path / 'actors.json'; file.chmod(0o644); file.write_text('[]')
            with self.assertRaisesRegex(ValueError, 'integrity'): checkpoints.read(path)
            partial = Path(folder) / '.step-13.partial-xyz'; partial.mkdir()
            with self.assertRaises(ValueError): checkpoints.read(partial)
