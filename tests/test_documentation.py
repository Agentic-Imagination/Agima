"""Offline documentation contracts; never install packages or run GPU workloads."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("documentation_checks", ROOT / "tools/check_repo.py")
CHECKS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CHECKS)


class DocumentationTests(unittest.TestCase):
    def test_links_resolve_relative_paths_references_and_duplicate_headings(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "docs").mkdir()
            (root / "docs/guide with spaces.md").write_text("# Run `model`\n\n## Run model\n")
            (root / "README.md").write_text(
                "[Guide](<docs/guide with spaces.md#run-model>)\n"
                "[Duplicate][guide]\n[guide]: docs/guide%20with%20spaces.md#run-model-1\n"
                "`[Not a link](missing.py)`\n<!-- [Hidden](missing.py) -->\n"
                "```text\n[Example](missing.py)\n```\n"
                "[Remote](https://example.invalid/not-fetched)\n"
            )
            count, errors = CHECKS.check_markdown_links(root)
        self.assertEqual(count, 2)
        self.assertEqual(errors, [])

    def test_missing_files_anchors_and_references_are_reported_with_locations(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "docs").mkdir()
            (root / "README.md").write_text("# Available\n")
            (root / "docs/guide.md").write_text(
                "[Missing](../removed.py)\n[Heading](../README.md#removed)\n[Unknown][label]\n"
            )
            errors = CHECKS.check_markdown_links(root)[1]
        self.assertEqual(len(errors), 3)
        for line, reason in ((1, "missing local link target"), (2, "missing Markdown heading"),
                             (3, "undefined Markdown link reference")):
            self.assertTrue(any(f"docs/guide.md:{line}: {reason}" in error for error in errors), errors)

    def test_readme_entrypoints_and_option_names_are_checked_without_execution(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "launcher.py").write_text(
                "raise RuntimeError('must not execute during static checks')\n"
                "parser.add_argument('--dry-run', action='store_true')\n"
            )
            (root / "README.md").write_text(
                "```bash\npython3 launcher.py --dry-run\n"
                "python3 removed.py --help\npython3 launcher.py --typo\n```\n"
            )
            count, errors = CHECKS.check_readme_commands(root)
        self.assertEqual(count, 3)
        self.assertEqual(len(errors), 2)
        self.assertTrue(any("missing Python entrypoint removed.py" in error for error in errors))
        self.assertTrue(any("unknown option --typo" in error for error in errors))

    def test_readme_help_and_dry_run_examples_work_without_site_packages(self):
        checked = []
        allowed = {"scripts/train.py", "scripts/evaluate.py"}
        for line, command in CHECKS.readme_python_commands(ROOT):
            if len(command) < 2 or command[1] not in allowed:
                continue
            if not {"--help", "--dry-run"}.intersection(command[2:]):
                continue  # Real training/download/install commands are never executed.
            with self.subTest(line=line, entrypoint=command[1]):
                result = subprocess.run(
                    [sys.executable, "-B", "-S", str(ROOT / command[1]), *command[2:]],
                    cwd=ROOT, capture_output=True, text=True, timeout=15,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("usage:" if "--help" in command else "Dry run:", result.stdout)
                checked.append(line)
        self.assertTrue(checked, "README should include an executable --help or --dry-run launcher example")

    def test_broken_documented_entrypoint_is_reported_without_aborting_other_checks(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "broken.py").write_text("def broken(:\n")
            (root / "README.md").write_text(
                "```bash\npython broken.py --help\npython missing.py --help\n```\n"
            )
            count, errors = CHECKS.check_readme_commands(root)
        self.assertEqual(count, 2)
        self.assertEqual(len(errors), 2)
        self.assertIn("cannot parse Python entrypoint broken.py", errors[0])
        self.assertIn("missing Python entrypoint missing.py", errors[1])


if __name__ == "__main__":
    unittest.main()
