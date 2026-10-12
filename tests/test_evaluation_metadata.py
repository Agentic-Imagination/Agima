"""Exercise result metadata and intervention resuming without model dependencies."""

import ast
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]


def source_functions(relative_path, names, namespace):
    path = ROOT / relative_path
    tree = ast.parse(path.read_text())
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    if {node.name for node in nodes} != set(names):
        raise AssertionError(f"Missing functions in {relative_path}")
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


class SokobanMetadataTests(unittest.TestCase):
    def test_open_loop_and_feedback_modes_match_the_executed_protocol(self):
        path = ROOT / "evaluation/sokoban.py"
        function = next(node for node in ast.parse(path.read_text()).body
                        if isinstance(node, ast.FunctionDef) and node.name == "run_cot_episode")
        # Run the real serialization expressions; generation and simulation are irrelevant here.
        nodes = [node for node in function.body if isinstance(node, ast.Assign)
                 and any(isinstance(target, ast.Name) and target.id in {"feedback_mode", "result"}
                         for target in node.targets)]
        code = compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec")
        for is_inf, no_reground, mode, label, chunk in (
                (False, False, "gt_frame", "Action executed.", 5),
                (False, True, "text_only_noreground", "Continue.", 5),
                (True, False, "none_singlegen", None, "inf"),
                (True, True, "none_singlegen", None, "inf")):
            with self.subTest(is_inf=is_inf, no_reground=no_reground):
                namespace = dict(
                    os=os, _IS_INF=is_inf, _NO_REGROUND=no_reground, _CHUNK_K=0 if is_inf else 5,
                    _DECODER="perseg", CFG_TEXT=1.0, CFG_IMG=1.0,
                    ENV_FEEDBACK_LABEL="Action executed.", NO_REGROUND_LABEL="Continue.",
                    env_id="sokoban/hard", seed=7, episode_seed=7, ep={"global_index": 0}, rank=0,
                    args=SimpleNamespace(profile="sokoban_v8_res256"), max_steps=40,
                    chunk_texts=["('stop', 'stop')"], imagined=[], hist=[], cum_reward=0.0,
                    terminated=True, truncated=False, success=False, final_action="('stop', 'stop')",
                )
                exec(code, namespace)
                result = namespace["result"]
                self.assertEqual(result["feedback_mode"], mode)
                self.assertEqual(result["feedback_label"], label)
                self.assertEqual(result["chunk_k"], chunk)
                self.assertEqual(result["chunk_eval_mode"], f"cot_multiturn_envfb_{mode}")
                self.assertIs(result["is_inf"], is_inf)
                self.assertIs(result["no_reground"], no_reground and not is_inf)
                self.assertEqual(result["reset_source"], "init_state")


class AblationResumeTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.ep = {"env_id": "maze_2d/hard", "global_index": 10, "seed": 2, "episode_seed": 3}
        self.args = SimpleNamespace(outdir=self.temporary.name, profile="maze2d_native256",
                                    image_quality=95, image_format="JPEG", overwrite=False,
                                    render=True, save_generated_images=True)
        self.calls = []
        self.env = SimpleNamespace(step=lambda action: None)
        self.native_generate = lambda *args, **kwargs: ("", [])
        self.inferencer = SimpleNamespace(gen_cot_chunk_span=self.native_generate)
        self.vlm = SimpleNamespace(inferencer=self.inferencer)
        production = SimpleNamespace(run_maze2d_cot_episode=self.production_runner)
        self.namespace = source_functions(
            "experiments/image_ablation/run_maze_pusht.py",
            {"_outdir", "_read_json", "_completed_result", "run_episode"},
            dict(json=json, Path=Path, TASK="maze", CONDITION="original", QUALITY=95,
                 NATIVE_SIZE=(256, 256), IMAGINED_CODEC="raw", EXPECTED_DECODER="span",
                 base=SimpleNamespace(_env_slug=lambda value: value.replace("/", "_"),
                                      _json_safe=lambda value: value),
                 G=SimpleNamespace(_is_stop_action=lambda value: value == "('stop', 'stop')"),
                 production=production),
        )

    def directory(self):
        return self.namespace["_outdir"](self.ep, self.args)

    def stats_path(self):
        return self.directory() / "episode_stats.json"

    def production_record(self):
        return {
            "env_id": self.ep["env_id"], "global_index": self.ep["global_index"],
            "seed": 2, "episode_seed": 3, "profile": self.args.profile,
            "decoder": self.namespace["EXPECTED_DECODER"],
            "instruction_max_steps": 40, "history": [],
            "stats": {"success": False, "final_action": None, "truncated": False},
        }

    def production_runner(self, env, vlm, extra_state_key, ep, args, rank):
        self.calls.append(args.overwrite)
        self.directory().mkdir(parents=True, exist_ok=True)
        if self.stats_path().exists() and not args.overwrite:
            return {"skipped": True, "success": False}
        self.stats_path().write_text(json.dumps(self.production_record()))
        return {"success": False}

    def run_episode(self):
        return self.namespace["run_episode"](self.env, self.vlm, None, self.ep, self.args, 0)

    def test_complete_matching_result_skips_without_calling_production(self):
        self.assertEqual(self.run_episode(), {"success": False})
        self.assertEqual(self.run_episode(), {"skipped": True, "success": False})
        self.assertEqual(self.calls, [True])
        self.assertFalse(self.args.overwrite)
        self.assertTrue(self.args.render)
        self.assertTrue(self.args.save_generated_images)
        self.assertIs(self.inferencer.gen_cot_chunk_span, self.native_generate)

    def test_pusht_resume_restores_the_production_chunk_hook(self):
        native_chunk = lambda *args, **kwargs: None
        self.namespace.update(TASK="pusht", QUALITY=90, EXPECTED_DECODER="perseg")
        self.namespace["production"]._gen_one_chunk = native_chunk
        self.namespace["production"].run_pusht_cot_episode = self.production_runner
        self.ep["env_id"] = "pusht/easy"
        self.args.profile = "pusht_norm4_native"
        self.args.image_quality = 90
        self.assertEqual(self.run_episode(), {"success": False})
        self.assertIs(self.namespace["production"]._gen_one_chunk, native_chunk)
        self.assertEqual(self.run_episode(), {"skipped": True, "success": False})
        self.assertEqual(self.calls, [True])

    def test_missing_corrupt_and_partial_results_are_rerun(self):
        self.directory().mkdir(parents=True)
        for contents in (json.dumps(self.production_record()), "{invalid", "[]",
                         json.dumps({**self.production_record(), "result": {"success": False}})):
            with self.subTest(contents=contents[:40]):
                self.stats_path().write_text(contents)
                before = len(self.calls)
                self.assertEqual(self.run_episode(), {"success": False})
                self.assertEqual(len(self.calls), before + 1)
                data = json.loads(self.stats_path().read_text())
                self.assertEqual(data["result"]["condition"], "Original")
                self.assertEqual(data["runner"], "agima_image_ablation_maze_pusht")
                self.assertFalse(self.args.overwrite)

    def test_different_completed_condition_is_preserved_even_with_overwrite(self):
        self.run_episode()
        original = self.stats_path().read_bytes()
        self.namespace["CONDITION"] = "blank"
        for overwrite in (False, True):
            with self.subTest(overwrite=overwrite):
                self.args.overwrite = overwrite
                with self.assertRaisesRegex(ValueError, "different settings.*condition"):
                    self.run_episode()
                self.assertEqual(self.stats_path().read_bytes(), original)
                self.assertEqual(self.calls, [True])

    def test_different_codec_or_episode_identity_is_not_silently_resumed(self):
        self.run_episode()
        original = self.stats_path().read_bytes()
        for key, value in (("imagined_codec", "jpeg"), ("episode_id", "maze:0000010:99")):
            with self.subTest(key=key):
                data = json.loads(original)
                data["result"][key] = value
                self.stats_path().write_text(json.dumps(data))
                before = self.stats_path().read_bytes()
                with self.assertRaisesRegex(ValueError, key):
                    self.run_episode()
                self.assertEqual(self.stats_path().read_bytes(), before)

    def test_explicit_overwrite_reruns_the_matching_condition(self):
        self.run_episode()
        self.args.overwrite = True
        self.assertEqual(self.run_episode(), {"success": False})
        self.assertEqual(self.calls, [True, True])
        self.assertTrue(self.args.overwrite)

    def test_missing_required_trace_recomputes_the_episode(self):
        self.ep["global_index"] = 0
        self.run_episode()
        (self.directory() / "chunks.json").unlink()
        self.assertEqual(self.run_episode(), {"success": False})
        self.assertEqual(self.calls, [True, True])
        self.assertEqual(self.run_episode(), {"skipped": True, "success": False})

    def test_trace_write_failure_does_not_publish_a_completed_result(self):
        self.ep["global_index"] = 0
        original_write = Path.write_text

        def fail_chunk_write(path, contents, *args, **kwargs):
            if path.name == "chunks.json":
                raise OSError("interrupted while writing traces")
            return original_write(path, contents, *args, **kwargs)

        with patch.object(Path, "write_text", fail_chunk_write):
            with self.assertRaisesRegex(OSError, "interrupted"):
                self.run_episode()
        self.assertNotIn("result", json.loads(self.stats_path().read_text()))
        self.assertEqual(self.run_episode(), {"success": False})
        self.assertEqual(self.calls, [True, True])
        self.assertFalse(self.stats_path().with_suffix(".json.tmp").exists())

    def test_production_exception_restores_wrapper_hooks_and_arguments(self):
        def fail(*args, **kwargs):
            raise RuntimeError("production failed")

        original_step = self.env.step
        self.namespace["production"].run_maze2d_cot_episode = fail
        with self.assertRaisesRegex(RuntimeError, "production failed"):
            self.run_episode()
        self.assertFalse(self.args.overwrite)
        self.assertTrue(self.args.render)
        self.assertTrue(self.args.save_generated_images)
        self.assertIs(self.inferencer.gen_cot_chunk_span, self.native_generate)
        self.assertIs(self.env.step, original_step)


if __name__ == "__main__":
    unittest.main()
