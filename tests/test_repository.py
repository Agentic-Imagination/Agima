"""Offline checks for source, repository hygiene, and environment registration."""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("repository_checks", ROOT / "tools/check_repo.py")
CHECKS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CHECKS)


class RepositoryChecksTests(unittest.TestCase):
    def test_archive_downloads_do_not_hide_source_or_get_read_as_text(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name in ("models", "datasets", "data"):
                (root / name).mkdir()
            (root / "models/remote_model.py").write_text("not valid Python!\n")
            (root / "datasets/download.md").write_text("[Remote file](not-in-source.md)\n")
            (root / "data/broken.py").write_text("def broken(:\n")
            weight = root / "download.safetensors"
            weight.touch()
            original_read = Path.read_bytes

            def read_source_only(path):
                if path == weight:
                    raise AssertionError("Binary weights must be skipped before reading their bytes")
                return original_read(path)

            with patch.object(Path, "read_bytes", read_source_only):
                self.assertEqual(CHECKS.check_text_hygiene(root), (1, []))
                count, errors = CHECKS.check_python(root)
            self.assertEqual(count, 1)
            self.assertIn("data/broken.py", errors[0])
            self.assertEqual(CHECKS.check_markdown_links(root), (0, []))

    @unittest.skipUnless(shutil.which("git"), "Git is needed to reproduce parent-repository discovery")
    def test_source_archive_ignores_unrelated_parent_git_rules(self):
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary)
            subprocess.run(["git", "init", "-q", str(parent)], check=True)
            (parent / ".gitignore").write_text("source/\n")
            root = parent / "source"
            root.mkdir()
            (root / "broken.py").write_text("def broken(:\n")
            (root / "README.md").write_text("[Missing](removed.py)\n")
            count, errors = CHECKS.check_python(root)
            self.assertEqual(count, 1)
            self.assertEqual(len(errors), 1)
            self.assertIn("broken.py", errors[0])
            self.assertEqual(CHECKS.check_markdown_links(root)[0], 1)
            self.assertIn("missing local link target", CHECKS.check_markdown_links(root)[1][0])
            self.assertEqual(CHECKS.check_tracked_files(root), (None, []))

            # An actual checkout still honors its own ignore rules.
            subprocess.run(["git", "init", "-q", str(root)], check=True)
            (root / ".gitignore").write_text("broken.py\n")
            self.assertEqual(CHECKS.check_python(root), (0, []))

    def test_syntax_check_does_not_execute_source(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "gpu.py").write_text(
                "import a_dependency_that_is_not_installed\n"
                "raise RuntimeError('source must not execute')\n"
            )
            count, errors = CHECKS.check_python(root)
        self.assertEqual(count, 1)
        self.assertEqual(errors, [])

    def test_reports_bad_source_but_ignores_local_environment(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "broken.py").write_text("def broken(:\n")
            (root / ".venv").mkdir()
            (root / ".venv/ignored.py").write_text("not valid python!\n")
            count, errors = CHECKS.check_python(root)
        self.assertEqual(count, 1)
        self.assertEqual(len(errors), 1)
        self.assertIn("broken.py", errors[0])

    def test_private_hygiene_covers_all_retained_source_and_configs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name in CHECKS.REQUIRED_FILES:
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("")
            (root / ".env.example").write_text("MODEL=/path/to/model\n")
            self.assertEqual(CHECKS.check_text_hygiene(root)[1], [])
            private_path = "/" + "/".join(("home", "private-account", "model"))
            (root / "legacy.py").write_text(f"MODEL = {private_path!r}\n")
            (root / "config.yaml").write_text(f"model: {private_path}\n")
            (root / "notes.md").write_text("Owner: private-account-label")
            errors = CHECKS.check_text_hygiene(root, ("private-account-label",))[1]
        self.assertEqual(len(errors), 3)
        self.assertTrue(any("legacy.py:1: machine-specific path" in error for error in errors))
        self.assertTrue(any("config.yaml:1: machine-specific path" in error for error in errors))
        self.assertTrue(any("notes.md:1: private account identifier" in error for error in errors))

    def test_local_import_check_detects_pruned_modules(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "evaluation").mkdir()
            (root / "evaluation/base.py").write_text("VALUE = 1\n")
            (root / "evaluation/runner.py").write_text(
                "from . import base\nimport evaluation.base\nimport evaluation.removed\n"
            )
            errors = CHECKS.check_local_imports(root)
        self.assertEqual(errors, ["evaluation/runner.py:3: missing local module evaluation.removed"])

    def test_license_manifest_detects_changed_missing_and_unrecorded_copies(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            directory = root / "LICENSES"
            directory.mkdir()
            license_file = directory / "example.txt"
            content = b"License A\n"
            license_file.write_bytes(content)
            (directory / "sources.json").write_text(json.dumps({"sources": [{
                "file": "example.txt", "sha256": hashlib.sha256(content).hexdigest(), "bytes": len(content),
            }]}))
            self.assertEqual(CHECKS.check_license_copies(root), (1, []))
            license_file.write_bytes(b"License B\n")  # Same length; only the hash catches the edit.
            self.assertIn("differs from its recorded bytes/SHA-256", CHECKS.check_license_copies(root)[1][0])
            license_file.unlink()
            self.assertIn("cannot read license copy", CHECKS.check_license_copies(root)[1][0])
            license_file.write_bytes(content)
            (directory / "unrecorded.txt").write_bytes(content)
            self.assertIn("missing from sources.json", CHECKS.check_license_copies(root)[1][0])

    def test_license_manifest_cannot_reference_outside_its_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "LICENSES").mkdir()
            (root / "outside.txt").write_bytes(b"text")
            (root / "LICENSES/sources.json").write_text(json.dumps({"sources": [{
                "file": "../outside.txt", "sha256": hashlib.sha256(b"text").hexdigest(), "bytes": 4,
            }]}))
            count, errors = CHECKS.check_license_copies(root)
        self.assertEqual(count, 0)
        self.assertEqual(len(errors), 1)
        self.assertIn("file must remain inside LICENSES", errors[0])

    def test_config_references_are_checked_without_importing_training(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name in ("training", "scripts", "configs/task", "tests"):
                (root / name).mkdir(parents=True)
            (root / "configs/task/train.yaml").write_text("dataset: {}\n")
            (root / "training/runner.py").write_text(
                "raise RuntimeError('must not execute training')\n"
                "DEFAULT = 'configs/task/train.yaml'\n"
            )
            (root / "scripts/train.py").write_text("CONFIGS = {'task': 'task/removed.yaml'}\n")
            (root / "tests/fixture.py").write_text("EXAMPLE = 'configs/fixture.yaml'\n")
            count, errors = CHECKS.check_config_references(root)
        self.assertEqual(count, 2)
        self.assertEqual(errors, ["scripts/train.py:1: missing local configuration configs/task/removed.yaml"])

    def test_registered_tasks_resolve_to_the_four_retained_families(self):
        entries = {}
        for source in ("gymnasium/envs/__init__.py", "gymnasium/envs/registration.py"):
            tree = ast.parse((ROOT / source).read_bytes())
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name) or node.func.id != "register":
                    continue
                keywords = {keyword.arg: keyword.value for keyword in node.keywords}
                if "id" not in keywords or not isinstance(keywords["id"], ast.Constant):
                    continue
                entry = keywords.get("entry_point")
                self.assertIsInstance(entry, ast.Constant)
                identifier = keywords["id"].value
                module, symbol = entry.value.split(":")
                self.assertTrue(CHECKS.module_exists(ROOT, module), module)
                entries[identifier] = (module, symbol)
        self.assertEqual(set(entries), {"sokoban/easy", "sokoban/hard", "pacman_2d/easy",
                                       "pacman_2d/hard", "maze_2d/easy", "maze_2d/hard", "pusht/easy"})


if __name__ == "__main__":
    unittest.main()
