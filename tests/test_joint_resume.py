import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from gozero import checkpoints, joint_resume
from gozero.snapshots import canonical_json


class JointResumeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='gozero-joint-resume-')
        self.base = Path(self.tmp.name)
        self.root = self.base / 'workspace'; self.root.mkdir()
        self.ram = self.base / 'ram'
        self.mock = patch.object(joint_resume, 'RAM_BASES', (self.ram / 'staged', self.ram / 'archived'))
        self.mock.start()
        self.attempt = self.root / 'runs/pod-20260915T000000Z-aabbccdd'
        self.snapshot = 'a' * 64

    def tearDown(self):
        self.mock.stop(); self.tmp.cleanup()

    def write(self, path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            path.chmod(0o644)
        path.write_bytes(canonical_json(value)); path.chmod(0o444)

    def fixture(self, ram=False):
        self.logical = joint_resume.logical(self.root, self.attempt, 0, 2)
        owner = self.ram / 'staged' / self.logical.relative_to(self.root / 'runs') if ram else self.logical
        mapping = [dict(host=h, jax_rank=r) for h, r in enumerate((1, 3, 0, 2))]
        group = dict(kind='visual_replicated_checkpoint_group', schema_version=1, snapshot_id=self.snapshot,
            turn=2, config_sha256='c'*64, owner_checkpoint_path=str(owner), host_jax_mapping=mapping,
            host_manifests={}, replicated_arrays_elements_sha256='e'*64)
        directories = []
        for host in range(4):
            directory = owner if host == 0 else joint_resume.logical(self.root, self.attempt, host, 2)
            state = dict(kind='visual_replicated_rank_state', snapshot_id=self.snapshot, turn=2,
                host_rank=host, jax_rank=mapping[host]['jax_rank'], config_sha256='c'*64,
                dataset_manifest_sha256='d'*64, model_schema=[dict(path='weight', shape=[7], dtype='float32')],
                optimizer_metadata=dict(step=2), counters=dict(updates=2, expert_positions=512),
                owns_replicated_arrays=host == 0)
            arrays = dict(weight=np.arange(7, dtype=np.float32), step=np.asarray(2, np.int32)) if host == 0 else {}
            group['host_manifests'][str(host)] = checkpoints.write(directory, state=state, arrays=arrays, actors='{}')
            directories.append(directory)
        self.group_path = self.logical.with_suffix('.group.json'); self.group = group; self.owner = owner
        for directory in directories:
            self.write(directory.with_suffix('.group.json'), group)
        self.write(self.group_path, group)
        if ram:
            self.write(self.logical.with_suffix('.temporary.json'), dict(kind='temporary_ram_checkpoint',
                cache_path=str(owner), logical_path=str(self.logical), manifest_sha256=group['host_manifests']['0']))
        return directories

    def collect(self):
        return joint_resume.collect(self.root, self.group_path, checkpoints.sha256(self.group_path), snapshot=self.snapshot)

    def test_regular_checkpoint_covers_all_rank_state_and_arrays(self):
        directories = self.fixture()
        files = self.collect()
        expected = {directory / name for directory in directories for name in joint_resume.MEMBERS}
        expected |= {directory.with_suffix('.group.json') for directory in directories}
        self.assertEqual(set(files), expected)
        for path, record in files.items():
            self.assertEqual(record, dict(bytes=path.stat().st_size, sha256=checkpoints.sha256(path)))

    def test_ram_owner_and_each_nonowner_resolve_to_actual_readable_checkpoints(self):
        self.fixture(ram=True)
        files = self.collect()
        self.assertIn(self.owner / 'arrays.npz', files)
        self.assertNotIn(self.logical / 'arrays.npz', files)
        for host in range(4):
            chosen = joint_resume.resume_path(self.root, self.attempt, host, 2, self.snapshot)
            state, arrays, _ = checkpoints.read(chosen, expected_manifest_sha256=self.group['host_manifests'][str(host)])
            self.assertEqual(state['host_rank'], host)
            self.assertEqual(set(arrays), {'weight', 'step'} if host == 0 else set())

    def test_missing_owner_moments_archive_is_rejected_before_launch(self):
        self.fixture(ram=True)
        (self.owner / 'arrays.npz').unlink()
        with self.assertRaisesRegex(ValueError, 'regular files'):
            self.collect()

    def test_corrupt_nonowner_state_is_rejected_before_launch(self):
        directories = self.fixture(ram=True)
        path = directories[2] / 'state.json'; path.chmod(0o644); path.write_text('{}')
        with self.assertRaisesRegex(ValueError, 'identity differs'):
            self.collect()

    def test_foreign_owner_and_wrong_group_source_are_rejected(self):
        self.fixture(ram=True)
        original = copy.deepcopy(self.group)
        for change in (dict(owner_checkpoint_path=str(self.ram / 'staged/foreign')),
                       dict(snapshot_id='f'*64)):
            self.group_path.chmod(0o644); self.write(self.group_path, {**original, **change})
            with self.assertRaises(ValueError):
                self.collect()

    def test_symlink_payload_is_rejected(self):
        self.fixture(ram=True)
        path = self.owner / 'arrays.npz'; moved = self.owner / 'saved.npz'; path.rename(moved); path.symlink_to(moved)
        with self.assertRaisesRegex(ValueError, 'regular files'):
            self.collect()

    def test_inconsistent_ram_receipt_is_rejected(self):
        self.fixture(ram=True)
        path = self.logical.with_suffix('.temporary.json'); value = json.loads(path.read_text())
        path.chmod(0o644); self.write(path, {**value, 'manifest_sha256': 'f'*64})
        with self.assertRaisesRegex(ValueError, 'RAM owner receipt differs'):
            self.collect()


if __name__ == '__main__':
    unittest.main()
