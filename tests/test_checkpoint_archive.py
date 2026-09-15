import hashlib
import io
from pathlib import Path
import tempfile
import unittest
import numpy as np
from gozero import checkpoints, checkpoint_archive as archive


class ArchiveTests(unittest.TestCase):
    def setup_case(self, root):
        original = root/'original'
        arrays = {'p': np.arange(3000, dtype=np.float32).reshape(30,100), 'm': np.full(3000, .125, np.float32)}
        identity = checkpoints.write(original, state={'step':17}, arrays=arrays, actors='{}', compress=True)
        spec = {'manifest_sha256':identity, 'token':'a'*32, 'reservation':'.gozero/checkpoint-reservations/test',
                'files':{n:{'sha256':checkpoints.sha256(original/n), 'bytes':(original/n).stat().st_size} for n in archive.FILES},
                'metadata':{n:(original/n).read_text() for n in archive.FILES-{'arrays.npz'}}}
        reservation = root/spec['reservation']; reservation.parent.mkdir(parents=True)
        with reservation.open('xb') as f:
            import os
            os.posix_fallocate(f.fileno(),0,1024*1024)
        return original, arrays, spec

    def test_reserved_transfer_has_identical_standard_checkpoint(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); original, arrays, spec = self.setup_case(root)
            archive.prepare(spec,root)
            with (original/'arrays.npz').open('rb') as f: archive.stream(spec,f,root)
            result = archive.commit(spec,root)
            saved, restored, _ = checkpoints.read(Path(result['path']),expected_manifest_sha256=spec['manifest_sha256'])
            self.assertEqual(saved,{'step':17})
            for k in arrays: np.testing.assert_array_equal(arrays[k],restored[k])
            for n in archive.FILES:self.assertEqual((original/n).read_bytes(),(Path(result['path'])/n).read_bytes())
            self.assertTrue(archive.prepare(spec,root)['already_committed'])
            self.assertFalse((root/spec['reservation']).exists())

    def test_bad_stream_cannot_publish(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);original,_,spec=self.setup_case(root);archive.prepare(spec,root)
            raw=(original/'arrays.npz').read_bytes()
            with self.assertRaises(ValueError):archive.stream(spec,io.BytesIO(raw[:-1]),root)
            with self.assertRaises(ValueError):archive.commit(spec,root)
            self.assertFalse(archive.locations(spec,root)[1].exists())
            self.assertEqual(checkpoints.sha256(original/'arrays.npz'),spec['files']['arrays.npz']['sha256'])

    def test_metadata_and_path_checks_precede_reservation_consumption(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);_,_,spec=self.setup_case(root)
            bad={**spec,'metadata':{**spec['metadata'],'state.json':'wrong'}}
            with self.assertRaises(ValueError):archive.prepare(bad,root)
            self.assertTrue((root/spec['reservation']).is_file())
            with self.assertRaises(ValueError):archive.safe_path(root/'../outside',root)
