"""Ablation tables must report supplied measurements without fixed reference scores."""

import contextlib
import csv
import gzip
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments.image_ablation import make_combined_paper_table as combined
from experiments.image_ablation import preflight
from experiments.image_ablation import preflight_maze_pusht
from experiments.image_ablation import summarize as sokoban_pacman
from experiments.image_ablation import summarize_maze_pusht as maze_pusht


def episode_rows(task, condition, successes, *, images=True):
    return [{
        "episode_id": f"{task}:{index:07d}:{index + 1000}",
        "task": task, "condition": condition, "success": index < successes,
        "global_index": index, "episode_seed": index + 1000,
        "parse_failure": False, "num_invalid_actions": 0,
        "num_executed_actions": 2, "stop_emitted": True,
        "termination_reason": "solved" if index < successes else "budget_exhausted",
        "n_imagined": int(images),
        "pixel_mse": ([{"pixel_mse_0_255": 1.0, "pixel_mse_0_1": 1.0 / 65025}]
                      if images else []),
    } for index in range(100)]


def archive_rows(successes):
    return [{"global_index": index, "episode_seed": index + 1000,
             "stats": {"success": index < successes},
             "history": [] if index % 2 else [{"action": "('stop', 'stop')"}]}
            for index in range(100)]


def read_csv(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


class AblationReportTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)

    def test_sokoban_pacman_reports_follow_changed_measurements_and_keep_archive_sources(self):
        for original, oracle in ((12, 12), (31, 7)):
            with self.subTest(original=original, oracle=oracle):
                out = self.root / f"report-{original}"
                out.mkdir()
                for task in ("sokoban", "pacman"):
                    for run_name in sokoban_pacman.RUN_NAMES.values():
                        (out / f"{task}_{run_name}_first10_audit.json").write_text(json.dumps([
                            {"passed": True, "imagined_frames": 1, "real_boundaries": 2}
                            for _ in range(10)
                        ]))
                counts = {"Original": original, "Blank": 18, "Stale": 24, "Oracle GT": oracle}

                def rows(task, condition):
                    successes = 9 if condition == "No-Imagine" else counts[condition]
                    return episode_rows(task, condition, successes)

                def archive(task, arm):
                    self.assertEqual(arm, "cot")
                    return archive_rows(16)

                with patch.object(sokoban_pacman, "read_rows", side_effect=rows), \
                        patch.object(sokoban_pacman, "archived_stats", side_effect=archive), \
                        contextlib.redirect_stdout(io.StringIO()):
                    sokoban_pacman.summarize(("sokoban", "pacman"), out)

                measured = {(r["task"], r["condition"]): r for r in read_csv(out / "summary.csv")}
                self.assertEqual(int(measured[("pacman", "Original")]["successes"]), original)
                self.assertEqual(int(measured[("pacman", "Oracle GT")]["successes"]), oracle)
                display = {r["condition"]: r for r in read_csv(out / "paper_table.csv")}
                self.assertEqual(display["Original"]["pacman_success_pct"], str(original))
                self.assertEqual(display["Original"]["source"], "this_ablation")
                self.assertEqual(display["No-Imagine"]["pacman_success_pct"], "9")
                self.assertEqual(display["No-Imagine"]["source"], "replayed_baseline")
                self.assertEqual(display["Oracle GT"]["pacman_success_pct"], str(oracle))
                self.assertEqual(display["Oracle GT"]["source"], "this_ablation")
                tests = json.loads((out / "mcnemar.json").read_text())
                result = next(r for r in tests if r["task"] == "pacman" and r["condition"] == "Oracle GT")
                self.assertEqual((result["b_original_only"], result["c_other_only"]),
                                 (original - oracle, 0))
                self.assertEqual(result["p_exact_two_sided"], 1.0 if original == oracle else 2 ** -23)
                report = (out / "report.md").read_text()
                self.assertIn(f"pacman / Oracle GT: {oracle}/100 versus measured Original {original}/100", report)
                self.assertIn(f"difference {oracle - original:+.1f} percentage points", report)

    def test_maze_pusht_report_uses_input_counts_without_historical_diagnostics(self):
        for original in (18, 6):
            with self.subTest(original=original):
                root = self.root / str(original)
                out = root / "output"
                out.mkdir(parents=True)
                # Exercise the existing large-difference action comparison without any move pairs.
                for record in archive_rows(original):
                    path = root / "runs/pusht_original" / str(record["global_index"])
                    path.mkdir(parents=True)
                    (path / "episode_stats.json").write_text(json.dumps(record))

                def rows(task, condition):
                    count = {"No-Imagine": 5, "Original": original, "Blank": 12,
                             "Stale": 8, "Oracle GT": 10}[condition]
                    return episode_rows(task, condition, count, images=False)

                def archive(path):
                    self.assertIn(path, [spec["cot"] for spec in maze_pusht.CONFIG.values()])
                    return archive_rows(18)

                with patch.object(maze_pusht, "ROOT", root), patch.object(maze_pusht, "OUT", out), \
                        patch.object(maze_pusht, "_rows", side_effect=rows), \
                        patch.object(maze_pusht, "archived_stats", side_effect=archive), \
                        contextlib.redirect_stdout(io.StringIO()):
                    maze_pusht.main()

                report = (out / "maze_pusht_report.md").read_text()
                self.assertIn(f"pusht: measured {original}/100; archive 18/100", report)
                self.assertIn(f"PushT measured Original differs from the archive by {original - 18:+d}", report)
                self.assertIn(f"pusht / Oracle GT: 10/100 versus measured Original {original}/100", report)
                self.assertIn("pusht: no imagined frames", report)
                if original == 6:
                    self.assertIn("not available (no paired initial move actions)", report)
                # No generated frames should produce an empty CSV with the documented schema.
                mse = out / "maze_pusht_world_model_pixel_mse.csv"
                self.assertEqual(read_csv(mse), [])
                self.assertIn("pixel_mse_0_255", mse.read_text())
                counts = {(r["task"], r["condition"]): r for r in read_csv(out / "maze_pusht_summary.csv")}
                self.assertEqual(int(counts[("pusht", "Original")]["successes"]), original)

    def test_combined_table_uses_measured_counts_for_every_condition(self):
        for original in (6, 18, 60):
            with self.subTest(original=original):
                out = self.root / f"combined-{original}"
                out.mkdir()
                for filename, tasks in (("summary.csv", ("sokoban", "pacman")),
                                        ("maze_pusht_summary.csv", ("maze", "pusht"))):
                    with (out / filename).open("w", newline="") as stream:
                        writer = csv.DictWriter(stream, fieldnames=["task", "condition", "successes", "n"])
                        writer.writeheader()
                        writer.writerows({"task": task, "condition": condition,
                                          "successes": original if condition == "Original" else 10, "n": 100}
                                         for task in tasks for condition in combined.CONDITIONS)
                with patch.object(combined, "OUT", out), contextlib.redirect_stdout(io.StringIO()):
                    combined.main()
                caption = (out / "table.tex").read_text()
                self.assertIn("All rates are computed from the supplied episode records", caption)
                self.assertIn(f"{original}\\%", caption)
                display = {r["condition"]: r for r in read_csv(out / "paper_table.csv")}
                self.assertEqual(display["Original"]["pusht_success_pct"], str(original))
                self.assertEqual(display["Original"]["source"], "this_ablation")
                self.assertEqual(display["Blank"]["pusht_success_pct"], "10")
                self.assertEqual(display["Blank"]["source"], "this_ablation")
                for condition in combined.CONDITIONS:
                    for task in combined.TASKS:
                        expected = original if condition == "Original" else 10
                        self.assertEqual(int(display[condition][f"{task}_success_pct"]), expected)

    def test_combined_table_rejects_wrong_denominators_and_duplicate_cells(self):
        path = self.root / "summary.csv"
        for n, successes in ((50, 10), (100, -1), (100, 101)):
            with self.subTest(n=n, successes=successes):
                path.write_text(f"task,condition,successes,n\nsokoban,Original,{successes},{n}\n")
                with self.assertRaises(ValueError):
                    combined._read(path)
        path.write_text("task,condition,successes,n\nsokoban,Original,10,100\n"
                        "sokoban,Original,20,100\n")
        with self.assertRaisesRegex(ValueError, "duplicate"):
            combined._read(path)

    def test_sokoban_preflight_accepts_new_scores_but_requires_episode_pairing(self):
        episodes = [{"seed": i, "episode_seed": i + 1000, "num_boxes": 4,
                     "env_kwargs": {"max_steps": 40}} for i in range(100)]
        runs = {arm: [dict(row, seed=i, chunk_k=5, instruction_max_steps=40,
                          n_executed=2, n_chunks=1)
                      for i, row in enumerate(archive_rows(count))]
                for arm, count in (("cot", 100), ("noncot", 0))}
        with patch.object(preflight, "rows_for", return_value=(
                episodes, preflight.EXPECTED["sokoban"]["sha256"])), \
                patch.object(preflight, "archived_stats", side_effect=lambda task, arm: runs[arm]):
            summary, baseline = preflight.check_task("sokoban")
            self.assertEqual(summary["archive_original_successes"], 100)
            self.assertEqual(summary["archive_noncot_successes"], 0)
            self.assertFalse(any(row["success"] for row in baseline))
            runs["cot"][0]["episode_seed"] = -1
            with self.assertRaisesRegex(AssertionError, "order mismatch"):
                preflight.check_task("sokoban")

    def test_maze_pusht_preflight_counts_staged_runs_and_checks_pairing(self):
        pool = self.root / "test.jsonl.gz"
        episodes = [{"episode_seed": i + 1000, "env_kwargs": {"max_steps": 30}}
                    for i in range(100)]
        with gzip.open(pool, "wt") as stream:
            stream.write("\n".join(json.dumps(row) for row in episodes))
        spec = {"eval": pool, "cot": "cot", "noncot": "noncot", "budget": 30}
        runs = {arm: [dict(row, seed=i, chunk_k=5, instruction_max_steps=30)
                      for i, row in enumerate(archive_rows(count))]
                for arm, count in (("cot", 0), ("noncot", 100))}
        with patch.object(preflight_maze_pusht, "ROOT", self.root), \
                patch.object(preflight_maze_pusht, "CONFIG", {"pusht": spec}), \
                patch.object(preflight_maze_pusht, "archived_stats", side_effect=runs.__getitem__), \
                contextlib.redirect_stdout(io.StringIO()):
            preflight_maze_pusht.main()
            summary = json.loads((self.root / "prepared/maze_pusht_preflight.json").read_text())[0]
            self.assertEqual(summary["archive_original_successes"], 0)
            self.assertEqual(summary["archive_noncot_successes"], 100)
            runs["noncot"][0]["episode_seed"] = -1
            with self.assertRaises(AssertionError):
                preflight_maze_pusht.main()


if __name__ == "__main__":
    unittest.main()
