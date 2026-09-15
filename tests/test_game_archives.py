import json
import os
from pathlib import Path
import tempfile
import unittest

from gozero import game_archives as archive
from gozero.checkpoints import sha256
from gozero.snapshots import canonical_json


class GameArchiveTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.artifacts = self.root / 'artifacts'
        self.games = self.artifacts / 'games'
        self.games.mkdir(parents=True)
        self.store = self.root / 'store'
        (self.artifacts / 'result.json').write_bytes(canonical_json({
            'status': 'passed', 'snapshot_id': 'a' * 64, 'turn': 96}))
        self.originals = {}
        for game in range(3):
            for suffix in ('json', 'sgf'):
                name = f'{game:016x}.{suffix}'
                data = (str(game) + suffix + '\n').encode() * 100
                path = self.games / name
                path.write_bytes(data)
                path.chmod(0o644 if game % 2 else 0o444)
                stamp = 1789129000000000000 + game
                os.utime(path, ns=(stamp, stamp))
                self.originals[name] = (data, path.stat().st_mode & 0o777, stamp)

    def test_exact_pack_evict_restore_and_repeated_operations(self):
        identity = archive.pack(self.artifacts, self.store)
        self.assertEqual(archive.pack(self.artifacts, self.store), identity)
        check = archive.verify(self.artifacts, self.store)
        self.assertEqual(check['materialized_files'], 6)
        self.assertLess(check['archive_bytes'], check['original_bytes'])
        self.assertEqual(archive.evict(self.artifacts, self.store)['evicted_files'], 6)
        self.assertEqual(list(self.games.iterdir()), [])
        self.assertEqual(archive.evict(self.artifacts, self.store)['evicted_files'], 0)
        self.assertEqual(archive.restore(self.artifacts, self.store)['restored_files'], 6)
        self.assertEqual(archive.restore(self.artifacts, self.store)['restored_files'], 0)
        for name, expected in self.originals.items():
            path = self.games / name
            self.assertEqual((path.read_bytes(), path.stat().st_mode & 0o777, path.stat().st_mtime_ns), expected)

    def test_archive_corruption_prevents_any_eviction(self):
        identity = archive.pack(self.artifacts, self.store)
        file = self.store / identity / 'games.zip'
        file.chmod(0o644)
        with file.open('ab') as stream:
            stream.write(b'corruption')
        with self.assertRaisesRegex(ValueError, 'integrity'):
            archive.evict(self.artifacts, self.store)
        self.assertEqual(len(list(self.games.iterdir())), 6)

    def test_last_changed_original_prevents_partial_eviction(self):
        archive.pack(self.artifacts, self.store)
        path = self.games / sorted(self.originals)[-1]
        path.chmod(0o644)
        path.write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'differs'):
            archive.evict(self.artifacts, self.store)
        self.assertEqual(len(list(self.games.iterdir())), 6)

    def test_partial_eviction_and_restoration_can_resume(self):
        archive.pack(self.artifacts, self.store)
        (self.games / sorted(self.originals)[0]).unlink()
        self.assertEqual(archive.evict(self.artifacts, self.store)['evicted_files'], 5)
        # A restored subset is valid input to a new restoration invocation.
        name = sorted(self.originals)[0]
        (self.games / name).write_bytes(self.originals[name][0])
        self.assertEqual(archive.restore(self.artifacts, self.store)['restored_files'], 5)
        self.assertEqual(archive.verify(self.artifacts, self.store)['materialized_files'], 6)

    def test_unknown_files_symlinks_and_unclosed_runs_are_rejected(self):
        path = self.games / sorted(self.originals)[0]
        path.unlink(); path.symlink_to(self.artifacts / 'result.json')
        with self.assertRaisesRegex(ValueError, 'regular'):
            archive.pack(self.artifacts, self.store)
        path.unlink(); path.write_bytes(self.originals[path.name][0])
        (self.games / 'unexpected.txt').write_text('keep')
        with self.assertRaisesRegex(ValueError, 'unexpected'):
            archive.pack(self.artifacts, self.store)
        (self.games / 'unexpected.txt').unlink()
        (self.artifacts / 'result.json').write_text('{}')
        with self.assertRaisesRegex(ValueError, 'completed'):
            archive.pack(self.artifacts, self.store)

    def test_changed_result_and_competing_operator_are_rejected(self):
        archive.pack(self.artifacts, self.store)
        with archive._lock(self.artifacts):
            with self.assertRaises(BlockingIOError):
                archive.evict(self.artifacts, self.store)
        path = self.artifacts / 'result.json'
        result = json.loads(path.read_text()); result['turn'] += 1
        path.write_bytes(canonical_json(result))
        with self.assertRaisesRegex(ValueError, 'different training'):
            archive.evict(self.artifacts, self.store)
        self.assertEqual(len(list(self.games.iterdir())), 6)

    def test_manifest_paths_are_rejected_even_with_a_matching_outer_hash(self):
        identity = archive.pack(self.artifacts, self.store)
        path = self.store / identity / 'manifest.json'
        manifest = json.loads(path.read_text())
        name = sorted(manifest['files'])[0]
        manifest['files']['../escape.json'] = manifest['files'].pop(name)
        path.chmod(0o644); path.write_bytes(canonical_json(manifest))
        with self.assertRaisesRegex(ValueError, 'member'):
            archive.verify_archive(path.parent, expected_id=sha256(path))

    def test_selected_archived_reads_preserve_exact_bytes_without_restoring(self):
        identity = archive.pack(self.artifacts, self.store)
        archive.evict(self.artifacts, self.store)
        with archive.open_records(self.artifacts, self.store) as reader:
            self.assertEqual(reader.archive_id, identity)
            self.assertEqual(set(reader.names), set(self.originals))
            for name, original in self.originals.items():
                self.assertEqual(reader.read(name), original[0])
            with self.assertRaisesRegex(ValueError, 'name'):
                reader.read('../result.json')
            with self.assertRaises(KeyError):
                reader.read('ffffffffffffffff.json')
            with self.assertRaises(BlockingIOError):
                archive.restore(self.artifacts, self.store)
        self.assertEqual(list(self.games.iterdir()), [])
        with self.assertRaisesRegex(ValueError, 'closed'):
            reader.read(next(iter(self.originals)))

    def test_corruption_is_rejected_before_exposing_any_selected_record(self):
        identity = archive.pack(self.artifacts, self.store)
        file = self.store / identity / 'games.zip'
        file.chmod(0o644)
        with file.open('ab') as stream:
            stream.write(b'corrupt even if this member was not requested')
        entered = False
        with self.assertRaisesRegex(ValueError, 'integrity'):
            with archive.open_records(self.artifacts, self.store):
                entered = True
        self.assertFalse(entered)

    def test_changed_training_result_invalidates_read_context_before_publication(self):
        archive.pack(self.artifacts, self.store)
        with self.assertRaisesRegex(ValueError, 'changed during reading'):
            with archive.open_records(self.artifacts, self.store) as reader:
                reader.read(next(iter(self.originals)))
                (self.artifacts / 'result.json').write_bytes(canonical_json({
                    'status': 'passed', 'snapshot_id': 'a' * 64, 'turn': 97}))
