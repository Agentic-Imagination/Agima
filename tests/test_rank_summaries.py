"""Prevent resumed evaluations from aggregating an earlier worker group's results."""

import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from evaluation import rank_summaries as summaries


class RankSummaryTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        patch = mock.patch.dict(os.environ, {
            "TORCHELASTIC_RUN_ID": "current-job", "TORCHELASTIC_RESTART_COUNT": "0",
            "WORLD_SIZE": "2",
        })
        patch.start()
        self.addCleanup(patch.stop)

    def write(self, rank, **fields):
        summaries.write_rank_summary(self.root, rank, fields)

    def test_previous_job_or_restart_cannot_satisfy_completion(self):
        for old in ({"TORCHELASTIC_RUN_ID": "old-job"}, {"TORCHELASTIC_RESTART_COUNT": "1"}):
            with self.subTest(old=old):
                self.write(0, episodes=1, completed=1, successes=1)
                with mock.patch.dict(os.environ, old):
                    self.write(1, episodes=1, completed=1, successes=1)
                with self.assertRaises(TimeoutError):
                    summaries.wait_for_rank_summaries(self.root, 2, timeout_sec=0)
                with self.assertRaisesRegex(ValueError, "rank 1"):
                    summaries.aggregate_summaries(self.root, 2)
                self.assertFalse((self.root / "summary.json").exists())

    def test_current_group_counts_completed_and_resumed_episodes(self):
        self.write(0, episodes=2, completed=1, skipped=1, successes=2, errors=0)
        self.write(1, episodes=2, completed=1, skipped=0, successes=0, errors=1)
        summaries.wait_for_rank_summaries(self.root, 2, timeout_sec=0)
        result = summaries.aggregate_summaries(self.root, 2)
        self.assertEqual((result["episodes"], result["done"], result["successes"], result["errors"]),
                         (4, 3, 2, 1))
        self.assertIsNone(result["success_rate"])
        self.assertEqual(result["status"], "incomplete")
        self.assertEqual(len({row["session_id"] for row in result["rank_summaries"]}), 1)
        self.assertFalse(list((self.root / "_rank_summaries").glob("*.tmp")))

    def test_complete_and_empty_pools_have_distinct_completion_states(self):
        self.write(0, episodes=2, completed=1, skipped=1, successes=1, errors=0)
        self.write(1, episodes=1, completed=1, skipped=0, successes=0, errors=0)
        complete = summaries.aggregate_summaries(self.root, 2)
        self.assertEqual(complete["status"], "complete")
        self.assertAlmostEqual(complete["success_rate"], 1 / 3)
        for rank in range(2):
            self.write(rank, episodes=0, completed=0, skipped=0, successes=0, errors=0)
        empty = summaries.aggregate_summaries(self.root, 2)
        self.assertEqual(empty["status"], "incomplete")
        self.assertIsNone(empty["success_rate"])

    def test_partial_legacy_and_wrong_rank_records_are_not_current_results(self):
        self.write(0, completed=1)
        path = self.root / "_rank_summaries/rank_00001.json"
        for content in ("{unfinished", "[]", '{"rank": 1}', json.dumps({
                "rank": 0, "session_id": summaries.summary_session_id()})):
            with self.subTest(content=content):
                path.write_text(content)
                with self.assertRaises(TimeoutError):
                    summaries.wait_for_rank_summaries(self.root, 2, timeout_sec=0)

    def test_distributed_jobs_require_a_shared_unique_rendezvous_id(self):
        for run_id in ("", "none"):
            with self.subTest(run_id=run_id), mock.patch.dict(os.environ, {"TORCHELASTIC_RUN_ID": run_id}):
                with self.assertRaisesRegex(ValueError, "rendezvous ID"):
                    summaries.summary_session_id()
                with mock.patch.dict(os.environ, {"WORLD_SIZE": "1"}):
                    self.write(0, episodes=1, completed=1, successes=1)
                    summaries.wait_for_rank_summaries(self.root, 1, timeout_sec=0)
                    self.assertEqual(summaries.aggregate_summaries(self.root, 1)["success_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()
