import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from gozero.snapshots import SnapshotError, canonical_json, clone_recipe, digest, freeze, verify


class SnapshotTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.repo = Path(self.temporary.name) / "repo"
        self.recipe = self.repo / "research/recipes/example"
        self.library = self.repo / "packages/shared.py"
        self.recipe.mkdir(parents=True)
        self.library.parent.mkdir()
        self.library.write_text("VALUE = 27\n")
        (self.repo / "uv.lock").write_text("version = 1\n")
        (self.repo / "pyproject.toml").write_text("[project]\nname = 'fixture'\n")
        (self.recipe / "recipe.json").write_text('{"id":"example"}')
        (self.recipe / "train.py").write_text("from shared import VALUE\nprint(VALUE)\n")
        self.config = self.recipe / "smoke.json"
        self.config.write_text('{"seed":27}')
        self.store = Path(self.temporary.name) / "snapshots"

    def freeze(self):
        return freeze(self.repo, self.recipe, self.config, self.store)

    def test_identical_inputs_have_identical_address(self):
        first = self.freeze()
        second = self.freeze()
        self.assertEqual(first, second)
        self.assertEqual(verify(first)["snapshot_id"], first.name)

    def test_frozen_recipe_executes_original_library_after_live_edits(self):
        snapshot = self.freeze()
        self.library.write_text("VALUE = 99\n")
        (self.recipe / "train.py").write_text("raise RuntimeError('live code ran')\n")
        self.config.write_text('{"seed":99}')
        environment = {**os.environ, "PYTHONPATH": str(snapshot / "packages"), "PYTHONDONTWRITEBYTECODE": "1"}
        result = subprocess.run([sys.executable, str(snapshot / "research/recipes/example/train.py")],
                                cwd=snapshot, env=environment, capture_output=True, text=True, check=True)
        self.assertEqual(result.stdout, "27\n")
        self.assertEqual(json.loads((snapshot / "resolved_config.json").read_text()), {"seed": 27})
        verify(snapshot)
        self.assertNotEqual(self.freeze(), snapshot)

    def test_library_and_lock_changes_each_change_address(self):
        original = self.freeze()
        self.library.write_text("VALUE = 28\n")
        changed_library = self.freeze()
        (self.repo / "uv.lock").write_text("version = 2\n")
        changed_lock = self.freeze()
        self.assertEqual(len({original, changed_library, changed_lock}), 3)

    def test_external_engine_configs_and_sgf_fixtures_are_frozen(self):
        evaluation = self.repo / "eval"
        evaluation.mkdir()
        config = evaluation / "engine.cfg"
        config.write_text("maxVisits = 16\n")
        fixture = evaluation / "ko.sgf"
        fixture.write_text("(;GM[1]SZ[9];B[aa])\n")
        first = self.freeze()
        self.assertEqual((first / "eval/engine.cfg").read_bytes(), config.read_bytes())
        self.assertEqual((first / "eval/ko.sgf").read_bytes(), fixture.read_bytes())
        config.write_text("maxVisits = 32\n")
        self.assertNotEqual(self.freeze(), first)

    def test_detects_modified_removed_and_added_sources(self):
        for mutation in ("modify", "remove", "add"):
            with self.subTest(mutation=mutation):
                snapshot = freeze(self.repo, self.recipe, self.config, self.store / mutation)
                path = snapshot / "packages/shared.py"
                if mutation == "modify":
                    path.chmod(0o644)
                    path.write_text("VALUE = 0\n")
                elif mutation == "remove":
                    path.unlink()
                else:
                    (snapshot / "extra.py").write_text("print('extra')\n")
                with self.assertRaises(SnapshotError):
                    verify(snapshot)

    def test_rewritten_manifest_cannot_reuse_old_address(self):
        snapshot = self.freeze()
        path = snapshot / "manifest.json"
        manifest = json.loads(path.read_text())
        manifest["recipe"] = "research/recipes/different"
        manifest["snapshot_id"] = digest(canonical_json({k: v for k, v in manifest.items() if k != "snapshot_id"}))
        path.chmod(0o644)
        path.write_bytes(canonical_json(manifest))
        with self.assertRaisesRegex(SnapshotError, "content address"):
            verify(snapshot)

    def test_source_and_snapshot_symlinks_are_rejected(self):
        (self.recipe / "linked.py").symlink_to(self.library)
        with self.assertRaises(SnapshotError):
            self.freeze()
        (self.recipe / "linked.py").unlink()
        snapshot = self.freeze()
        (snapshot / "extra.py").symlink_to(self.library)
        with self.assertRaises(SnapshotError):
            verify(snapshot)

    def test_duplicate_keys_and_missing_locks_are_rejected(self):
        self.config.write_text('{"seed":1,"seed":2}')
        with self.assertRaisesRegex(SnapshotError, "Duplicate"):
            self.freeze()
        self.config.write_text('{"seed":1}')
        (self.repo / "uv.lock").unlink()
        with self.assertRaisesRegex(SnapshotError, "uv.lock"):
            self.freeze()

    def test_clone_is_independent_and_records_parent(self):
        helper = self.recipe / "helper.sh"
        helper.write_text("#!/bin/sh\nexit 0\n")
        helper.chmod(0o755)
        destination = clone_recipe(self.repo, self.recipe, "variant")
        metadata = json.loads((destination / "recipe.json").read_text())
        self.assertEqual(metadata["id"], "variant")
        self.assertEqual(metadata["parent"]["recipe"], "research/recipes/example")
        self.assertTrue(os.access(destination / "helper.sh", os.X_OK))
        (destination / "train.py").write_text("print('variant')\n")
        self.assertNotEqual((self.recipe / "train.py").read_text(), (destination / "train.py").read_text())
        with self.assertRaises(SnapshotError):
            clone_recipe(self.repo, self.recipe, "variant")
        with self.assertRaises(SnapshotError):
            clone_recipe(self.repo, self.recipe, "../escape")

    def test_snapshot_store_cannot_be_inside_source(self):
        with self.assertRaises(SnapshotError):
            freeze(self.repo, self.recipe, self.config, self.recipe / "snapshots")


if __name__ == "__main__":
    unittest.main()
