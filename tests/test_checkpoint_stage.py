import json
from pathlib import Path
import tempfile
import unittest
import zipfile
from unittest.mock import patch
from types import SimpleNamespace
import numpy as np
from gozero import checkpoints, checkpoint_stage


class TemporaryCheckpointTests(unittest.TestCase):
    def test_identical_regular_checkpoint_and_explicit_persistent_receipt(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); logical=root/'runs/attempt/rank-0/artifacts/checkpoints/turn-000000001'
            state={'snapshot_id':'a'*64,'turn':1}; arrays={'p_0000':np.arange(12,dtype=np.float32).reshape(3,4)}
            with patch.object(checkpoint_stage,'ROOT',root),patch.object(checkpoint_stage.os,'statvfs',return_value=SimpleNamespace(f_bavail=10_000_000_000,f_frsize=1)):
                cache,identity,receipt=checkpoint_stage.write(logical,state=state,arrays=arrays,actors='{}',cache_root=root/'ram')
            ordinary=root/'ordinary'; expected=checkpoints.write(ordinary,state=state,arrays=arrays,actors='{}',compress=True)
            self.assertEqual(identity,expected)
            for name in checkpoint_stage.FILES:
                self.assertEqual((cache/name).read_bytes(),(ordinary/name).read_bytes())
            self.assertFalse(logical.exists()); self.assertEqual(receipt['status'],'temporary')
            saved=json.loads(Path(receipt['receipt']).read_text())
            self.assertEqual(saved['manifest_sha256'],identity)
            self.assertIn('volatile',saved['durability'])
            for name in checkpoint_stage.FILES-{'arrays.npz'}:
                self.assertEqual((logical.with_suffix('.metadata')/name).read_bytes(),(cache/name).read_bytes())
            checkpoints.read(cache,expected_manifest_sha256=identity)

    def test_low_space_fails_before_writing_arrays(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); logical=root/'runs/attempt/rank-0/artifacts/checkpoints/turn-000000001'
            class Full:
                f_bavail=0; f_frsize=4096
            with patch.object(checkpoint_stage,'ROOT',root),patch.object(checkpoint_stage.os,'statvfs',return_value=Full()):
                with self.assertRaises(ValueError):
                    checkpoint_stage.write(logical,state={'snapshot_id':'a'*64},arrays={'x':np.zeros(1)},actors='{}',cache_root=root/'ram')
            self.assertFalse(logical.with_suffix('.temporary.json').exists())
            self.assertFalse((root/'ram/attempt').exists())

    def test_uncompressed_ram_checkpoint_uses_ordinary_reader(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);logical=root/'runs/attempt/rank-0/artifacts/checkpoints/turn-000000007'
            state={'snapshot_id':'b'*64,'turn':7};arrays={'weights':np.random.default_rng(7).normal(size=(128,64)).astype(np.float32)}
            with patch.object(checkpoint_stage,'ROOT',root),patch.object(checkpoint_stage.os,'statvfs',return_value=SimpleNamespace(f_bavail=80*(1<<30),f_frsize=1)):
                cache,identity,receipt=checkpoint_stage.write(logical,state=state,arrays=arrays,actors='{}',
                    cache_root=root/'ram',compress=False,minimum_free_bytes=64*(1<<30))
            recovered,values,actors=checkpoints.read(cache,expected_manifest_sha256=identity)
            self.assertEqual(recovered,state);np.testing.assert_array_equal(values['weights'],arrays['weights'])
            with zipfile.ZipFile(cache/'arrays.npz') as archive:
                self.assertTrue(all(x.compress_type==zipfile.ZIP_STORED for x in archive.infolist()))
            self.assertFalse(receipt['compressed']);self.assertEqual(receipt['minimum_free_bytes'],64*(1<<30))

    def test_custom_reserve_is_enforced_before_publication(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);logical=root/'runs/attempt/rank-0/artifacts/checkpoints/turn-000000001'
            with patch.object(checkpoint_stage,'ROOT',root),patch.object(checkpoint_stage.os,'statvfs',return_value=SimpleNamespace(f_bavail=60*(1<<30),f_frsize=1)):
                with self.assertRaises(ValueError):checkpoint_stage.write(logical,state={'snapshot_id':'a'*64},
                    arrays={'x':np.zeros(1)},actors='{}',cache_root=root/'ram',compress=False,minimum_free_bytes=64*(1<<30))
            self.assertFalse(logical.with_suffix('.temporary.json').exists())


if __name__=='__main__':unittest.main()
