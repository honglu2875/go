from pathlib import Path
import tempfile
import unittest

from gozero.evaluation_inputs import collect, verify
from gozero.snapshots import canonical_json


class EvaluationInputTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(); self.addCleanup(temp.cleanup)
        self.root = Path(temp.name); (self.root / 'eval').mkdir()
        self.write('eval/panel.json', {'matches': [{'spec': 'eval/match.json'}]})
        self.write('eval/match.json', {'candidate': 'eval/future.json', 'katago_config': 'eval/kata.cfg', 'visits': 1})
        self.write('eval/katago_build.json', {'binary_sha256': 'a' * 64})
        self.write('eval/katago_9x9.json', {'sha256': 'b' * 64})
        (self.root / 'eval/kata.cfg').write_text('maxVisits = 1\n')

    def write(self, name, content):
        (self.root / name).write_bytes(canonical_json(content))

    def manifest(self):
        return collect(self.root, ['eval/panel.json'], deferred_models=['eval/future.json'])

    def test_changed_child_is_detected_with_unchanged_parent_panel(self):
        manifest = self.manifest()
        self.assertEqual(len(manifest['files']), 5)
        verify(self.root, manifest)
        self.write('eval/match.json', {'candidate': 'eval/future.json', 'katago_config': 'eval/kata.cfg', 'visits': 100})
        with self.assertRaisesRegex(ValueError, 'closure differs'):
            verify(self.root, manifest)

    def test_future_model_requires_explicit_declaration_but_may_then_be_published(self):
        with self.assertRaises(FileNotFoundError):
            collect(self.root, ['eval/panel.json'])
        manifest = self.manifest()
        self.write('eval/future.json', {'training_snapshot': 'c' * 64})
        self.assertEqual(verify(self.root, manifest), manifest)
        committed = collect(self.root, ['eval/panel.json'])
        self.assertIn('eval/future.json', committed['files'])

    def test_configuration_and_referee_changes_are_bound(self):
        manifest = self.manifest()
        (self.root / 'eval/kata.cfg').write_text('maxVisits = 100\n')
        with self.assertRaises(ValueError):
            verify(self.root, manifest)

    def test_causal_search_and_context_contract_is_part_of_input_closure(self):
        self.write('eval/match.json', {'candidate': 'eval/future.json', 'causal_inference': 'eval/search.json'})
        self.write('eval/search.json', {'simulations':16,'max_game_moves':256})
        manifest=self.manifest()
        self.assertIn('eval/search.json',manifest['files'])
        self.write('eval/search.json', {'simulations':16,'max_game_moves':324})
        with self.assertRaisesRegex(ValueError,'closure differs'):
            verify(self.root,manifest)

    def test_unknown_deferral_and_escaping_references_are_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Unused'):
            collect(self.root, ['eval/panel.json'], deferred_models=['eval/future.json', 'eval/absent.json'])
        self.write('eval/panel.json', {'matches': [{'spec': '../outside.json'}]})
        with self.assertRaisesRegex(ValueError, 'escapes'):
            self.manifest()
