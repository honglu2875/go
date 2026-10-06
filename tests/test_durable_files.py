import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from gozero.durable_files import atomic_json, copy_verified, require_disk


class DurableFilesTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def test_interrupted_json_commit_preserves_prior_counter(self):
        counter = self.root/'counter.json'
        atomic_json(counter, {'next_sequence': 8})
        with patch('gozero.durable_files.os.replace', side_effect=OSError('power-loss boundary')):
            with self.assertRaises(OSError):
                atomic_json(counter, {'next_sequence': 9})
        self.assertEqual(json.loads(counter.read_text()), {'next_sequence': 8})
        atomic_json(counter, {'next_sequence': 9})
        self.assertEqual(json.loads(counter.read_text()), {'next_sequence': 9})

    def test_wrong_payload_is_never_published(self):
        source, target = self.root/'source', self.root/'nested/target'
        source.write_bytes(b'incomplete checkpoint')
        with self.assertRaises(ValueError):
            copy_verified(source, target, expected_sha256=hashlib.sha256(b'complete checkpoint').hexdigest(),
                          expected_bytes=19)
        self.assertFalse(target.exists())

    def test_verified_copy_survives_loss_of_source_and_rejects_overwrite(self):
        source, target = self.root/'source', self.root/'nested/target'
        raw = b'accepted immutable checkpoint\0' * 100
        digest = hashlib.sha256(raw).hexdigest()
        source.write_bytes(raw)
        copy_verified(source, target, expected_sha256=digest, expected_bytes=len(raw))
        source.unlink()
        self.assertEqual(target.read_bytes(), raw)
        source.write_bytes(b'replacement')
        with self.assertRaises(ValueError):
            copy_verified(source, target, expected_sha256=hashlib.sha256(b'replacement').hexdigest(), expected_bytes=11)
        self.assertEqual(target.read_bytes(), raw)

    def test_ram_filesystem_cannot_count_as_durable(self):
        with self.assertRaises(ValueError):
            require_disk(Path('/dev/shm'))


if __name__ == '__main__':
    unittest.main()
