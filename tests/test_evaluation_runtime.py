"""Exercise episode I/O and job completion without loading model dependencies."""

import ast
import contextlib
import datetime as dt
import gzip
import io
import json
import os
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest import mock

from evaluation import rank_summaries


ROOT = Path(__file__).resolve().parents[1]


def runtime_functions():
    """Load the real stdlib-only functions, excluding CUDA imports and model classes."""
    names = {"is_maze2d_env_id", "_open_jsonl", "_compact_episode", "scan_compact_episodes",
             "_cache_manifest", "wait_for_shard_manifest", "wait_for_path", "prepare_shard_cache", "main"}
    tree = ast.parse((ROOT / "evaluation/runtime.py").read_text())
    definitions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    base_tree = ast.parse((ROOT / "evaluation/base.py").read_text())
    definitions += [node for node in base_tree.body
                    if isinstance(node, ast.FunctionDef) and node.name == "iter_jsonl_files"]
    namespace = dict(Path=Path, dt=dt, gzip=gzip, json=json, time=time, os=os)
    for name in ("summary_session_id", "write_rank_summary", "wait_for_rank_summaries",
                 "aggregate_summaries", "write_summary"):
        namespace[name] = getattr(rank_summaries, name)
    future = ast.parse("from __future__ import annotations").body
    exec(compile(ast.Module(body=future + definitions, type_ignores=[]), "runtime-io", "exec"), namespace)
    namespace["base"] = SimpleNamespace(iter_jsonl_files=namespace["iter_jsonl_files"])
    return namespace


class EvaluationRuntimeTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.pool = self.root / "episodes.jsonl"
        self.api = runtime_functions()
        self.args = SimpleNamespace(
            episodes_root=str(self.pool), outdir=str(self.root / "run"), limit=None,
            env_filter=None, profile="sokoban_v8_res256", sokoban_action_mode_override="move_only",
            sokoban_success_mode="stop_then_solved", shard_cache_dir=None,
            rebuild_shard_cache=False, shard_cache_timeout_sec=1,
            prompt_sanity_only=True, summary_wait_timeout_sec=1,
        )
        patch = mock.patch.dict(os.environ, {"TORCHELASTIC_RUN_ID": "job-a", "WORLD_SIZE": "2",
                                            "TORCHELASTIC_RESTART_COUNT": "0"})
        patch.start()
        self.addCleanup(patch.stop)

    def write_pool(self, seeds):
        self.pool.write_text("\n".join(json.dumps({
            "env_id": "sokoban/easy", "seed": seed, "init_state": {"seed": seed}
        }) for seed in seeds) + "\n")

    def test_malformed_or_empty_input_is_not_silently_dropped(self):
        for content in ("{broken\n", "[]\n", '{"seed": 1}\n', "\n"):
            with self.subTest(content=content):
                self.pool.write_text(content)
                with self.assertRaises(ValueError):
                    self.api["scan_compact_episodes"](str(self.pool), None, None)

    def test_new_job_rebuilds_changed_input_and_shards_preserve_pool_order(self):
        self.write_pool([10, 20, 30])
        first = self.api["prepare_shard_cache"](self.args, 0, 2)
        second = self.api["prepare_shard_cache"](self.args, 1, 2)
        self.assertEqual([ep["seed"] for ep in first], [10, 30])
        self.assertEqual([ep["seed"] for ep in second], [20])
        self.write_pool([40, 50, 60])
        with mock.patch.dict(os.environ, {"TORCHELASTIC_RUN_ID": "job-b"}):
            first = self.api["prepare_shard_cache"](self.args, 0, 2)
            second = self.api["prepare_shard_cache"](self.args, 1, 2)
        self.assertEqual([ep["seed"] for ep in first], [40, 60])
        self.assertEqual([ep["seed"] for ep in second], [50])

    def test_late_worker_accepts_current_job_cache_after_forced_rebuild(self):
        self.args.rebuild_shard_cache = True
        self.write_pool([10, 20])
        self.api["prepare_shard_cache"](self.args, 0, 2)
        manifest = Path(self.args.outdir) / "_episode_shards/manifest.json"
        # Wall-clock freshness is unrelated to whether this is the current job.
        os.utime(manifest, (1, 1))
        second = self.api["prepare_shard_cache"](self.args, 1, 2)
        self.assertEqual([ep["seed"] for ep in second], [20])

    def test_main_reports_errors_as_failure_after_publishing_diagnostics(self):
        self.api.update(parse_args=lambda: self.args,
                        init_distributed=lambda **kwargs: (0, 2, 0),
                        cleanup_distributed=mock.Mock())
        outdir = Path(self.args.outdir)
        outdir.mkdir()
        (outdir / "summary.json").write_text('{"success_rate": 1.0}')

        def run_shard(*args):
            self.assertFalse((outdir / "summary.json").exists())
            for rank, errors in ((0, 0), (1, 1)):
                rank_summaries.write_rank_summary(outdir, rank, {
                    "episodes": 1, "completed": 1 - errors, "skipped": 0,
                    "successes": 1 - errors, "errors": errors,
                })
            return {"errors": 0}

        self.api["run_shard"] = run_shard
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(RuntimeError, "Evaluation incomplete"):
            self.api["main"]()
        summary = json.loads((outdir / "summary.json").read_text())
        self.assertEqual(summary["errors"], 1)
        self.assertEqual(summary["status"], "incomplete")
        self.assertIsNone(summary["success_rate"])
        self.api["cleanup_distributed"].assert_called_once()


if __name__ == "__main__":
    unittest.main()
