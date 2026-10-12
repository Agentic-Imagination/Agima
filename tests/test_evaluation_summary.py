"""Protect benchmark tallies from silent corruption and incomplete denominators."""

import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

from scripts.summarize_evaluation import main, retally_run


class EvaluationSummaryTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)

    def record(self, name, **record):
        path = self.root / name
        path.mkdir()
        (path / "episode_stats.json").write_text(json.dumps(record))

    def test_matching_duplicates_and_legacy_success_are_counted_once(self):
        self.record("first", global_index=0, stats={"success": True})
        self.record("cached", global_index=0, stats={"success": True})
        self.record("legacy", global_index=1, history=[{"info": {"is_success": False}}])
        result = retally_run(str(self.root), expected_episodes=2)
        self.assertEqual((result["unique_episodes"], result["successes"], result["success_rate"]), (2, 1, 0.5))
        self.assertEqual(result["duplicate_records"], 1)
        self.assertEqual(result["issues"], [])

    def test_conflicting_outcomes_or_episode_identities_invalidate_the_rate(self):
        for conflict in ({"success": False, "seed": 1}, {"success": True, "seed": 2}):
            with self.subTest(conflict=conflict):
                for p in list(self.root.rglob("episode_stats.json")):
                    p.unlink()
                    p.parent.rmdir()
                self.record("first", global_index=0, seed=1, stats={"success": True})
                self.record("second", global_index=0, seed=conflict["seed"], stats={"success": conflict["success"]})
                result = retally_run(str(self.root))
                self.assertIsNone(result["success_rate"])
                self.assertIn("conflicting", result["issues"][0])

    def test_corrupt_records_and_missing_episodes_fail_with_diagnostics(self):
        self.record("valid", global_index=0, stats={"success": True})
        (self.root / "bad").mkdir()
        (self.root / "bad/episode_stats.json").write_text("{truncated")
        stdout, stderr = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            code = main([str(self.root), "--expected-episodes", "3", "--json"])
        self.assertEqual(code, 1)
        self.assertIsNone(json.loads(stdout.getvalue())[0]["success_rate"])
        self.assertIn("bad/episode_stats.json", stderr.getvalue())
        self.assertIn("expected 3 unique episodes, found 1", stderr.getvalue())

    def test_invalid_success_flags_cannot_turn_strings_into_successes(self):
        self.record("wrong", global_index=0, stats={"success": "false"})
        result = retally_run(str(self.root))
        self.assertEqual(result["successes"], 0)
        self.assertIsNone(result["success_rate"])
        self.assertTrue(result["issues"])

    def test_empty_run_has_no_rate_and_nonexistent_pattern_is_an_error(self):
        self.assertIsNone(retally_run(str(self.root))["success_rate"])
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as exc:
            main([str(self.root), str(self.root / "missing")])
        self.assertEqual(exc.exception.code, 2)

    def test_empty_results_and_nonfinite_json_cannot_create_a_valid_rate(self):
        self.record("empty")
        self.record("nonfinite", stats={"reward": float("nan")})
        result = retally_run(str(self.root))
        self.assertEqual(result["unique_episodes"], 0)
        self.assertIsNone(result["success_rate"])
        self.assertTrue(any("no success flag" in issue for issue in result["issues"]))
        self.assertTrue(any("non-finite JSON" in issue for issue in result["issues"]))

    def test_invalid_cached_summary_invalidates_rate_without_breaking_json_output(self):
        self.record("valid", stats={"success": True})
        for number in ("NaN", "1e309"):
            with self.subTest(number=number):
                (self.root / "summary.json").write_text('{"success_rate": ' + number + '}')
                stdout = io.StringIO()
                with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(io.StringIO()):
                    self.assertEqual(main([str(self.root), "--json"]), 1)
                result = json.loads(stdout.getvalue())[0]
                self.assertEqual(result["summary_json"], "unreadable")
                self.assertIsNone(result["success_rate"])
                self.assertTrue(result["issues"])

    def test_corrupt_summary_cannot_hide_the_planned_count_or_errors(self):
        self.record("valid", global_index=0, stats={"success": True})
        for content in ('{"episodes": 2,', '{"episodes": 2, "errors": 1, "errors": 0}',
                        '{"episodes": "2"}', '{"errors": -1}'):
            with self.subTest(content=content):
                (self.root / "summary.json").write_text(content)
                result = retally_run(str(self.root))
                self.assertIsNone(result["success_rate"])
                self.assertTrue(any("summary.json is invalid" in issue for issue in result["issues"]))

    def test_parent_directory_cannot_merge_different_run_contracts(self):
        for name in ("checkpoint-a", "checkpoint-b"):
            self.record(name, global_index=0, env_id="sokoban/easy", seed=1, stats={"success": True})
            (self.root / name / "run-contract.json").write_text(json.dumps({"checkpoint": name}))
        result = retally_run(str(self.root), expected_episodes=1)
        self.assertIsNone(result["success_rate"])
        self.assertTrue(any("nested runs" in issue for issue in result["issues"]))
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            self.assertEqual(main([str(self.root / "checkpoint-*"), "--expected-episodes", "1", "--json"]), 0)
        self.assertEqual(len(json.loads(stdout.getvalue())), 2)

    def test_failed_episodes_in_summary_cannot_disappear_from_the_denominator(self):
        self.record("valid", global_index=0, stats={"success": True})
        (self.root / "summary.json").write_text(json.dumps({
            "episodes": 2, "completed": 1, "skipped": 0, "errors": 1,
            "successes": 1, "success_rate": None, "status": "incomplete",
        }))
        result = retally_run(str(self.root))
        self.assertIsNone(result["success_rate"])
        self.assertTrue(any("episode errors" in issue for issue in result["issues"]))
        self.assertTrue(any("planned 2 episodes" in issue for issue in result["issues"]))

    def test_duplicate_json_keys_cannot_override_an_outcome(self):
        self.record("duplicate", stats={"success": False})
        (self.root / "duplicate/episode_stats.json").write_text(
            '{"stats": {"success": false, "success": true}}')
        result = retally_run(str(self.root))
        self.assertIsNone(result["success_rate"])
        self.assertTrue(any("duplicate JSON key" in issue for issue in result["issues"]))


if __name__ == "__main__":
    unittest.main()
