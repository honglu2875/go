"""Fork provenance must not silently reset or substitute scientific state."""
import copy
import hashlib
from pathlib import Path
import tempfile
import unittest

import numpy as np

from gozero import checkpoint_forks as forks, checkpoints
from gozero.snapshots import canonical_json, freeze


class ForkContractTest(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(); self.addCleanup(temp.cleanup); self.root = Path(temp.name)
        (self.root / 'uv.lock').write_text('version = 1\n')
        (self.root / '.python-version').write_text('3.12.13\n')
        self.base = {'schema_version': 1, 'platform': 'cpu', 'expected_processes': 1, 'expected_devices': 1,
                     'seed': 27, 'selfplay_turns': 96, 'checkpoint_every': 48, 'log_every': 16,
                     'model': {'width': 2}, 'actors': {'simulations': 8}, 'learner': {'learning_rate': .02}}
        self.parent_recipe = self.root / 'research/recipes/parent'; self.parent_recipe.mkdir(parents=True)
        (self.parent_recipe / 'recipe.json').write_bytes(canonical_json({'id': 'parent'}))
        (self.parent_recipe / 'train.py').write_text('# Data-only checkpoint fixture.\n')
        (self.parent_recipe / 'model.py').write_text('FIXTURE_MODEL = 1\n')
        (self.parent_recipe / 'config.json').write_bytes(canonical_json(self.base))
        self.parent = freeze(self.root, self.parent_recipe, self.parent_recipe / 'config.json', self.root / '.gozero/snapshots')
        native = self.root / '.gozero/native' / self.parent.name; native.mkdir(parents=True)
        (native / 'lib_gozero_native.so').write_bytes(b'identity fixture, never loaded as code')
        binary_sha = checkpoints.sha256(native / 'lib_gozero_native.so')
        receipt = {'schema_version': 1, 'snapshot_id': self.parent.name, 'filename': 'lib_gozero_native.so', 'binary_sha256': binary_sha}
        (native / 'receipt.json').write_bytes(canonical_json(receipt))
        self.native = {'receipt': str((native / 'receipt.json').relative_to(self.root)), 'receipt_sha256': checkpoints.sha256(native / 'receipt.json')}
        self.arrays = {'p_0000': np.arange(4, dtype=np.float32).reshape(2, 2), 'v_0000': np.full((2, 2), .25, np.float32),
                       'key': np.array([27, 29], np.uint32), 'replay_x': np.arange(6, dtype=np.float32).reshape(2, 3),
                       'replay_pi': np.array([[.2, .8], [.7, .3]], np.float32), 'replay_z': np.array([1., -1.], np.float32),
                       'replay_owner': np.array([[1., 0.], [-1., 0.]], np.float32), 'replay_meta': np.arange(12, dtype=np.uint64).reshape(2, 6)}
        self.actors = '{"pending_game":{"actions":[0,1,9],"rng":1234}}'
        self.checkpoint = self.root / 'runs/parent/rank-0/artifacts/checkpoints/turn-000000048'
        schema = [{'path': "['w']", 'shape': [2, 2], 'dtype': 'float32'}]
        self.state = {'schema_version': 1, 'snapshot_id': self.parent.name,
                      'config_sha256': checkpoints.sha256(self.parent / 'resolved_config.json'), 'jax_rank': 0, 'world_size': 1,
                      'turn': 48, 'native_sha256': binary_sha, 'model_schema': schema, 'replay_count': 2, 'replay_cursor': 2,
                      'learner_ready': True, 'numpy_rng': np.random.default_rng(28).bit_generator.state,
                      'last_metrics': {'loss': 1.25}, 'counters': {'real_moves': 384, 'updates': 40, 'inference_seconds': 1.}}
        manifest = checkpoints.write(self.checkpoint, state=self.state, arrays=self.arrays, actors=self.actors, compress=True)
        group = {'schema_version': 1, 'snapshot_id': self.parent.name, 'config_sha256': self.state['config_sha256'],
                 'world_size': 1, 'turn': 48, 'updates': 40, 'rank_manifests': [manifest],
                 'replicated_state_sha256': forks.replicated_digest(self.arrays)}
        self.checkpoint.with_suffix('.group.json').write_bytes(canonical_json(group))
        self.mapping = {0: str(self.checkpoint.relative_to(self.root))}
        self.child_recipe = self.root / 'research/recipes/child'; (self.child_recipe / 'forks').mkdir(parents=True)
        (self.child_recipe / 'recipe.json').write_bytes(canonical_json({'id': 'child'}))
        (self.child_recipe / 'train.py').write_text('# Data-only checkpoint fixture.\n')
        (self.child_recipe / 'model.py').write_bytes((self.parent_recipe / 'model.py').read_bytes())

    def build(self, target=None, mode='inherit_constant', descriptor=None):
        target = copy.deepcopy(self.base if target is None else target)
        spec = forks.describe(self.root, self.mapping, target, self.native, learning_rate_mode=mode, reason='Fixture continuation') if descriptor is None else descriptor
        path = self.child_recipe / 'forks/parent.json'; path.write_bytes(canonical_json(spec))
        c = {**target, 'native': self.native, 'initialization': {'kind': 'fork', 'descriptor': str(path.relative_to(self.root)), 'sha256': checkpoints.sha256(path)}}
        (self.child_recipe / 'config.json').write_bytes(canonical_json(c))
        source = freeze(self.root, self.child_recipe, self.child_recipe / 'config.json', self.root / '.gozero/snapshots')
        return source, c, spec

    def test_complete_state_roundtrip_keeps_parent_identity_and_bytes(self):
        before = {p.name: checkpoints.sha256(p) for p in self.checkpoint.iterdir()}
        source, c, _ = self.build(); origin, spec, _, _ = forks.contract(self.root, source, c)
        state, arrays, actors = forks.load_rank(self.root, spec, jax_rank=0, host_rank=0)
        self.assertEqual(state, self.state); self.assertEqual(actors, self.actors)
        for key, value in self.arrays.items():
            np.testing.assert_array_equal(arrays[key], value)
        self.assertEqual(origin['parent_snapshot'], self.parent.name)
        self.assertEqual({p.name: checkpoints.sha256(p) for p in self.checkpoint.iterdir()}, before)
        # A later ordinary child resume can verify its contract after old replay
        # retention has ended; initializing a new fork still requires that replay.
        self.checkpoint.rename(self.checkpoint.with_name('retained-elsewhere'))
        self.assertEqual(forks.contract(self.root, source, c)[0], origin)
        with self.assertRaises(FileNotFoundError):
            forks.load_rank(self.root, spec, jax_rank=0, host_rank=0)

    def test_learning_rate_change_requires_explicit_override(self):
        target = copy.deepcopy(self.base); target['learner']['learning_rate'] = .005
        with self.assertRaisesRegex(ValueError, 'Inherited learning rate'):
            self.build(target)
        source, c, _ = self.build(target, 'constant_override')
        origin = forks.contract(self.root, source, c)[0]
        self.assertEqual(origin['declared_changes'], {'/learner/learning_rate': {'from': .02, 'to': .005}})
        with self.assertRaisesRegex(ValueError, 'actual learning-rate change'):
            self.build(mode='constant_override')

    def test_search_or_topology_change_cannot_be_authorized_as_schedule(self):
        for key in ('actors', 'expected_processes'):
            target = copy.deepcopy(self.base)
            if key == 'actors': target[key]['simulations'] = 16
            else: target[key] = 4
            with self.assertRaisesRegex(ValueError, 'scientific settings'):
                self.build(target)
        target = copy.deepcopy(self.base); target['actors']['simulations'] = 16
        _, _, spec = self.build()
        spec['declared_changes'] = forks.changes(self.base, target)
        spec['target_base_config_sha256'] = hashlib.sha256(canonical_json(target)).hexdigest()
        source, c, _ = self.build(target, descriptor=spec)
        with self.assertRaisesRegex(ValueError, 'forbidden fork change'):
            forks.contract(self.root, source, c)

    def test_model_native_and_parent_rank_substitution_are_rejected(self):
        (self.child_recipe / 'model.py').write_text('FIXTURE_MODEL = 2\n')
        source, c, spec = self.build()
        with self.assertRaisesRegex(ValueError, 'model implementation'):
            forks.contract(self.root, source, c)
        with self.assertRaisesRegex(ValueError, 'placement'):
            forks.load_rank(self.root, spec, jax_rank=0, host_rank=1)
        with self.assertRaisesRegex(ValueError, 'rank set'):
            forks.describe(self.root, {0: self.mapping[0], 1: self.mapping[0]}, self.base, self.native, reason='Duplicate')
        path = self.root / self.native['receipt']; receipt = __import__('json').loads(path.read_text())
        (path.parent / receipt['filename']).write_bytes(b'a different native binary')
        with self.assertRaisesRegex(ValueError, 'Native binary'):
            forks.describe(self.root, self.mapping, self.base, self.native, reason='Substitution')

    def test_replay_corruption_is_checked_even_when_parameters_are_unchanged(self):
        _, _, spec = self.build(); path = self.checkpoint / 'arrays.npz'
        data = bytearray(path.read_bytes()); data[len(data) // 2] ^= 1
        path.chmod(0o644); path.write_bytes(data)
        with self.assertRaisesRegex(ValueError, 'integrity failure'):
            forks.load_rank(self.root, spec, jax_rank=0, host_rank=0)

    def test_evaluation_uses_exact_declared_native_and_initialization(self):
        from gozero.model_artifacts import validate_candidate
        source, c, spec = self.build(); origin = forks.contract(self.root, source, c)[0]
        state = {**self.state, 'snapshot_id': source.name,
                 'config_sha256': checkpoints.sha256(source / 'resolved_config.json'), 'initialization': origin}
        path = self.root / 'runs/child/checkpoints/turn-000000048'
        manifest = checkpoints.write(path, state=state, arrays=self.arrays, actors=self.actors, compress=True)
        group = {**spec['parent_group'], 'snapshot_id': source.name, 'config_sha256': state['config_sha256'],
                 'rank_manifests': [manifest], 'initialization_sha256': hashlib.sha256(canonical_json(origin)).hexdigest()}
        path.with_suffix('.group.json').write_bytes(canonical_json(group))
        weights = self.root / 'runs/child/model_export.npz'; np.savez(weights, p_0000=self.arrays['p_0000'])
        candidate = {'schema_version': 2, 'training_snapshot': source.name, 'network_version': 40,
                     'model_export_path': str(weights.relative_to(self.root)), 'model_export_sha256': checkpoints.sha256(weights),
                     'checkpoint': {'path': str(path.relative_to(self.root)), 'manifest_sha256': manifest,
                                    'group_sha256': checkpoints.sha256(path.with_suffix('.group.json'))}}
        identity = validate_candidate(self.root, candidate)
        self.assertEqual(identity['native_receipt'], self.root / self.native['receipt'])
        self.assertNotEqual(identity['snapshot'].name, self.parent.name)
        result = self.root / 'runs/child/result.json'
        record = {**state, 'status': 'passed', 'model_export_sha256': candidate['model_export_sha256'],
                  'native': {'binary_sha256': self.state['native_sha256']}}
        result.write_bytes(canonical_json(record))
        final = {**candidate, 'schema_version': 1, 'training_result_path': str(result.relative_to(self.root)),
                 'training_result_sha256': checkpoints.sha256(result)}; final.pop('checkpoint')
        self.assertEqual(validate_candidate(self.root, final)['native_receipt'], identity['native_receipt'])
        record['initialization'] = {'kind': 'fresh'}; result.write_bytes(canonical_json(record))
        final['training_result_sha256'] = checkpoints.sha256(result)
        with self.assertRaisesRegex(ValueError, 'Candidate initialization'):
            validate_candidate(self.root, final)


if __name__ == '__main__':
    unittest.main()
