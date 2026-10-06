"""Retention must never remove the only recoverable full training state."""
import json
from pathlib import Path
import tempfile
import unittest

from gozero.checkpoint_disk import retire
from gozero.durable_files import atomic_json


class RetentionTests(unittest.TestCase):
    def test_unmirrored_old_checkpoint_is_not_retired(self):
        with tempfile.TemporaryDirectory() as name:
            parent=Path(name)
            for n in (1,2,3):
                p=parent/f'turn-{n:09d}';p.mkdir();(p/'arrays.npz').write_bytes(b'only state')
            with self.assertRaises(ValueError):retire(parent/'turn-000000003',keep=2)
            self.assertEqual((parent/'turn-000000001/arrays.npz').read_bytes(),b'only state')

    def test_only_old_mirrored_payload_is_retired_and_metadata_survives(self):
        with tempfile.TemporaryDirectory() as name:
            parent=Path(name)
            for n in (1,2,3):
                p=parent/f'turn-{n:09d}';p.mkdir();(p/'arrays.npz').write_bytes(bytes([n]))
                (p/'state.json').write_text('{}')
                atomic_json(p.with_suffix('.disk.json'),dict(status='passed',bundle_sha256=str(n)))
            retire(parent/'turn-000000003',keep=2)
            self.assertFalse((parent/'turn-000000001/arrays.npz').exists())
            self.assertTrue((parent/'turn-000000001/state.json').exists())
            self.assertTrue((parent/'turn-000000002/arrays.npz').exists())
            self.assertTrue((parent/'turn-000000003/arrays.npz').exists())
            receipt=json.loads((parent/'turn-000000001.retired.json').read_text())
            self.assertEqual(receipt['replaced_by'],'turn-000000003')
            retire(parent/'turn-000000003',keep=2)


if __name__=='__main__':unittest.main()
