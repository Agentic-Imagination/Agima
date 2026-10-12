"""CPU checks for public launchers; no model imports or asset downloads."""

from __future__ import annotations

import ast
import contextlib
import gzip
import importlib
import importlib.util
import io
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
COMMON_SPEC = importlib.util.spec_from_file_location(
    "launcher_common", ROOT / "scripts/_common.py"
)
COMMON = importlib.util.module_from_spec(COMMON_SPEC)
COMMON_SPEC.loader.exec_module(COMMON)


class LauncherTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.scratch = Path(self.temporary.name)
        self.base = ["--task", "maze2d", "--variant", "cot", "--chunk-size", "5",
                     "--model-path", str(self.scratch / "missing model"),
                     "--output-dir", str(self.scratch / "new output")]

    def run_launcher(self, launcher, extra=(), *, dry_run=True, env=None, base=None, module=False):
        arguments = list(self.base if base is None else base)
        if launcher == "train":
            arguments += ["--data-dir", str(self.scratch / "missing data")]
        else:
            arguments += ["--checkpoint", str(self.scratch / "missing checkpoint"),
                          "--episodes-root", str(self.scratch / "missing episodes")]
        arguments += list(extra)
        if dry_run:
            arguments.append("--dry-run")
        entrypoint = [str(ROOT / "scripts" / (launcher + ".py"))]
        if module:
            entrypoint = ["-m", "scripts." + launcher]
            env = dict(os.environ if env is None else env, PYTHONPATH=str(ROOT))
        # -S proves help/dry-run/validation work without site-packages installed.
        return subprocess.run(
            [sys.executable, "-B", "-S", *entrypoint, *arguments],
            cwd=self.scratch, env=env, text=True, capture_output=True,
            timeout=10,
        )

    def dry_run_result(self, result):
        self.assertEqual(result.returncode, 0, result.stderr)
        lines = result.stdout.splitlines()
        self.assertIn("Dry run:", lines[-1])
        self.assertFalse((self.scratch / "new output").exists())
        environment = json.loads(next(line.removeprefix("Environment: ")
                                     for line in lines if line.startswith("Environment: ")))
        command = shlex.split(next(line for line in lines if " -m torch.distributed.run " in line))
        return environment, command

    def evaluation_source(self, command):
        module = next(value for value in command if value.startswith("evaluation."))
        index = command.index(module)
        self.assertEqual(command[index - 1], "-m")
        source = ROOT.joinpath(*module.split(".")).with_suffix(".py")
        self.assertTrue(source.is_file())
        return source

    def test_help_requires_no_dependencies(self):
        for launcher in ("train", "evaluate"):
            with self.subTest(launcher=launcher):
                result = self.run_launcher(launcher, ["--help"])
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("--dry-run", result.stdout)

    def test_module_and_file_invocation_build_identical_commands(self):
        for launcher, options in (("train", []), ("evaluate", ["--decoder", "span"])):
            with self.subTest(launcher=launcher):
                _, direct = self.dry_run_result(self.run_launcher(launcher, options))
                _, module = self.dry_run_result(self.run_launcher(launcher, options, module=True))
                self.assertEqual(module, direct)

    def test_train_override_paths_and_parallelism(self):
        _, command = self.dry_run_result(self.run_launcher(
            "train", ["--gpus", "2", "--token-budget", "32000", "--total-steps", "25"]
        ))
        self.assertIn("--nproc_per_node=2", command)
        for key, value in (("--num_shard", "2"), ("--max_num_tokens", "32000"),
                           ("--expected_num_tokens", "32000"), ("--total_steps", "25"),
                           ("--eval_every", "0"), ("--viz_every", "0")):
            self.assertEqual(command[command.index(key) + 1], value)
        self.assertEqual(command[command.index("--model_path") + 1],
                         str((self.scratch / "missing model").resolve()))
        self.assertNotIn("--eval_data_dir", command)

    def test_maze_cot_k1_uses_documented_token_budget(self):
        _, command = self.dry_run_result(self.run_launcher("train", ["--chunk-size", "1"]))
        self.assertEqual(command[command.index("--max_num_tokens") + 1], "24000")

    def test_custom_training_endpoint_satisfies_the_preserved_save_condition(self):
        tree = ast.parse((ROOT / "training/pretrain_unified_navit.py").read_text())
        save_condition = next(node.test for node in ast.walk(tree)
                              if isinstance(node, ast.If)
                              and "curr_step % training_args.save_every" in ast.unparse(node.test))
        condition = compile(ast.Expression(save_condition), "trainer-save-condition", "eval")
        for final_step in (10, 500, 1500):
            with self.subTest(final_step=final_step):
                _, command = self.dry_run_result(self.run_launcher(
                    "train", ["--total-steps", str(final_step), "--save-every", "1000"]
                ))
                self.assertEqual(command[command.index("--total_steps") + 1], str(final_step))
                extra = set(map(int, command[command.index("--extra_save_steps") + 1].split(",")))
                namespace = {"training_args": SimpleNamespace(save_every=1000),
                             "_extra_save": extra, "curr_step": final_step}
                self.assertTrue(eval(condition, namespace))
                namespace["curr_step"] = 500
                self.assertTrue(eval(condition, namespace))

    def test_real_cold_starts_get_distinct_wandb_ids_without_resuming(self):
        trainer = importlib.import_module("scripts.train")
        common = importlib.import_module("scripts._common")
        model = self.scratch / "base"
        model.mkdir()
        for name in ("llm_config.json", "vit_config.json", "tokenizer_config.json", "vocab.json"):
            (model / name).write_text("{}")
        for name in ("ae.safetensors", "ema.safetensors", "merges.txt"):
            (model / name).write_text("nonempty fixture")
        data = self.scratch / "data"
        data.mkdir()
        row = {"messages": [{"role": role, "content": [{"type": "text", "text": "fixture"}]}
                            for role in ("user", "assistant")]}
        (data / "train.jsonl").write_text(json.dumps(row) + "\n")
        identifiers = []
        for name in ("first", "second"):
            output = self.scratch / name
            arguments = ["train.py", "--task", "sokoban", "--variant", "cot", "--chunk-size", "5",
                         "--model-path", str(model), "--data-dir", str(data), "--output-dir", str(output),
                         "--gpus", "1", "--run-name", "shared-display-name", "--wandb-online"]
            with mock.patch.object(sys, "argv", arguments), \
                    mock.patch.object(common, "validate_runtime", return_value={"python": "fixture", "torch": "fixture"}), \
                    mock.patch.object(common, "source_provenance", return_value={"runtime_source_sha256": "fixture"}), \
                    mock.patch.object(common, "run_command", return_value=0) as child, \
                    contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(trainer.main(), 0)
            command = child.call_args.args[0]
            identifier = command[command.index("--wandb_runid") + 1]
            self.assertRegex(identifier, r"^[0-9a-f]{32}$")
            identifiers.append(identifier)
            self.assertEqual(command[command.index("--wandb_name") + 1], "shared-display-name")
            self.assertEqual(command[command.index("--wandb_resume") + 1], "never")
            self.assertEqual(command[command.index("--wandb_offline") + 1], "False")
            manifest = json.loads(next(output.glob("launch-*.json")).read_text())
            self.assertEqual(manifest["command"], command)
        self.assertNotEqual(*identifiers)

    def test_evaluation_routes_tasks_and_clears_inherited_experiment_settings(self):
        expected = {
            "sokoban": "evaluation.sokoban",
            "pusht": "evaluation.pusht",
            "maze2d": "evaluation.maze2d",
        }
        environment = dict(os.environ, BAGEL_NO_REGROUND="1", BAGEL_UNKNOWN_EXPERIMENT="1")
        for task, runner in expected.items():
            with self.subTest(task=task):
                env, command = self.dry_run_result(self.run_launcher(
                    "evaluate", ["--task", task, "--decoder", "span", "--limit", "3"],
                    env=environment,
                ))
                self.assertIn(runner, command)
                self.evaluation_source(command)
                self.assertEqual(command[command.index("--limit") + 1], "3")
                self.assertEqual(env["BAGEL_DECODER"], "span")
                self.assertEqual(env["BAGEL_NO_REGROUND"], "0")
                self.assertNotIn("BAGEL_UNKNOWN_EXPERIMENT", env)

    def test_noncot_training_and_evaluation_share_envfb_prompt_contract(self):
        # Run the real, stdlib-only data remappers on a tiny history. Inspect the
        # selected evaluator's shared prompt imports without importing GPU code.
        # The old plain Sokoban evaluator uses a different counter/feedback prompt.
        row = {"metadata": {"max_steps": 40}, "env_kwargs": {"max_steps": 40},
               "history": [{"prompt": "Move to the target.", "action": "('move', 0)",
                            "image_prev": "initial-frame", "image_next": f"frame-{i}"}
                           for i in range(2)]}
        for task in ("sokoban", "pusht", "maze2d"):
            with self.subTest(task=task):
                options = ["--task", task, "--variant", "noncot", "--chunk-size", "1"]
                env, command = self.dry_run_result(self.run_launcher(
                    "evaluate", options + ["--decoder", "perseg"]
                ))
                runner = self.evaluation_source(command)
                tree = ast.parse(runner.read_bytes())
                prompt_import = next(node for node in ast.walk(tree)
                                     if isinstance(node, ast.ImportFrom)
                                     and any(alias.name == "cot_instruction" for alias in node.names))
                prompt_module = importlib.import_module(prompt_import.module)
                remap = importlib.import_module(f"scripts.data.remap_{task}_noncot_chunk_k_envfb")
                with mock.patch.object(remap, "ENV_FEEDBACK_LABEL", env["BAGEL_ENV_FEEDBACK_LABEL"]):
                    transformed = remap.transform_row(row, 1)
                self.assertEqual(transformed["messages"][0]["content"][0]["text"],
                                 prompt_module.cot_instruction("Move to the target.", 40, 1))
                self.assertEqual(transformed["messages"][2]["content"], [
                    {"type": "text", "text": env["BAGEL_ENV_FEEDBACK_LABEL"]},
                    {"type": "image", "image": "frame-0", "loss": False},
                ])
                self.assertEqual(transformed["n_imagined_frames"], 0)
                self.assertEqual(env["BAGEL_NO_REGROUND"], "0")
                self.assertEqual(env["BAGEL_DECODER"], "perseg")
                _, train_command = self.dry_run_result(self.run_launcher("train", options))
                config = Path(train_command[train_command.index("--dataset_config_file") + 1])
                groups = [line for line in config.read_text().splitlines()
                          if line and not line[0].isspace() and not line.startswith("#")]
                self.assertTrue(groups)
                self.assertTrue(all(line.endswith("_interleaved_cot_envfb:") for line in groups))

    def test_sokoban_noncot_inf_uses_matching_open_loop_instruction(self):
        env, command = self.dry_run_result(self.run_launcher(
            "evaluate", ["--task", "sokoban", "--variant", "noncot",
                         "--chunk-size", "inf", "--decoder", "span"]
        ))
        runner = self.evaluation_source(command)
        tree = ast.parse(runner.read_bytes())
        self.assertTrue(any(isinstance(node, ast.ImportFrom)
                            and node.module == "scripts.data.remap_sokoban_cot_chunk_inf"
                            and any(alias.name == "cot_inf_instruction" for alias in node.names)
                            for node in ast.walk(tree)))
        remap = importlib.import_module("scripts.data.remap_sokoban_noncot_chunk_k_envfb")
        prompt = importlib.import_module("scripts.data.remap_sokoban_cot_chunk_inf")
        self.assertEqual(remap._instruction("Move to the target.", 40, "inf"),
                         prompt.cot_inf_instruction("Move to the target.", 40))
        self.assertEqual(env["BAGEL_CHUNK_K"], "inf")

    def test_decoder_is_explicit_for_every_task_and_variant(self):
        for task in ("sokoban", "pusht", "maze2d"):
            for variant in ("cot", "noncot"):
                with self.subTest(task=task, variant=variant):
                    result = self.run_launcher("evaluate", ["--task", task, "--variant", variant])
                    self.assertEqual(result.returncode, 2)
                    self.assertIn("--decoder", result.stderr)

    def test_maze_budget_uses_metadata_and_rejects_ignored_cli_override(self):
        result = self.run_launcher("evaluate", ["--decoder", "span", "--max-steps", "17"])
        self.assertEqual(result.returncode, 2)
        self.assertIn("unsupported for Maze2D", result.stderr)
        _, command = self.dry_run_result(self.run_launcher("evaluate", ["--decoder", "span"]))
        self.assertNotIn("--max-steps", command)
        remap = importlib.import_module("scripts.data.remap_maze2d_cot_chunk_k_envfb")
        self.assertEqual(remap._max_steps({"metadata": {"max_steps": 17}}), 17)
        self.assertEqual(remap._max_steps({"env_kwargs": {"max_steps": 17}}), 40)

    def test_real_launch_rejects_missing_assets_before_creating_output(self):
        for launcher, options in (("train", []), ("evaluate", ["--decoder", "span"])):
            with self.subTest(launcher=launcher):
                result = self.run_launcher(launcher, options, dry_run=False)
                self.assertEqual(result.returncode, 2)
                self.assertIn("Required file not found", result.stderr)
                self.assertNotIn("ModuleNotFoundError", result.stderr)
                self.assertFalse((self.scratch / "new output").exists())

    def test_nonpositive_gpu_count_is_rejected(self):
        result = self.run_launcher("train", ["--gpus", "0"])
        self.assertEqual(result.returncode, 2)
        self.assertIn("positive integer", result.stderr)

    def test_invalid_learning_rates_are_rejected_without_importing_training(self):
        for rate in ("nan", "inf", "-inf", "0", "-0.01", "1e309"):
            with self.subTest(rate=rate):
                result = self.run_launcher("train", ["--lr=" + rate])
                self.assertEqual(result.returncode, 2)
                self.assertIn("finite positive number", result.stderr)
                self.assertNotIn("Traceback", result.stderr)

    def test_missing_torch_preflight_creates_no_run_files(self):
        output = self.scratch / "new output"
        with mock.patch.object(COMMON.importlib.util, "find_spec", return_value=None), \
                mock.patch.object(COMMON, "run_command") as child, \
                contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(ValueError, "PyTorch is not installed"):
                COMMON.launch([sys.executable, "trainer.py"], {}, output, False, {}, gpus=1)
        self.assertFalse(output.exists())
        child.assert_not_called()

    def test_cuda_preflight_failures_create_no_run_files(self):
        cases = (
            ({"cuda_available": False, "visible_gpus": 0}, 0, "", "0 are available"),
            ({"cuda_available": True, "visible_gpus": 2}, 0, "", "2 are available"),
            (None, 1, "ImportError: CUDA library unavailable", "CUDA library unavailable"),
        )
        for runtime, returncode, error, message in cases:
            with self.subTest(message=message):
                output = self.scratch / "new output"
                result = subprocess.CompletedProcess([], returncode, json.dumps(runtime), error)
                with mock.patch.object(COMMON.importlib.util, "find_spec", return_value=object()), \
                        mock.patch.object(COMMON.subprocess, "run", return_value=result), \
                        mock.patch.object(COMMON, "run_command") as child, \
                        contextlib.redirect_stdout(io.StringIO()):
                    with self.assertRaisesRegex(ValueError, message):
                        COMMON.launch([sys.executable, "trainer.py"], {}, output, False, {}, gpus=8)
                self.assertFalse(output.exists())
                child.assert_not_called()

    def test_successful_preflight_records_runtime_then_launches(self):
        output = self.scratch / "new output"
        runtime = {"python": "3.10.0", "torch": "2.5.1", "cuda_available": True, "visible_gpus": 8}
        result = subprocess.CompletedProcess([], 0, json.dumps(runtime), "")
        command = [sys.executable, "trainer.py"]
        environment = {"CUDA_VISIBLE_DEVICES": "0,1,2,3,4,5,6,7"}
        with mock.patch.object(COMMON.importlib.util, "find_spec", return_value=object()), \
                mock.patch.object(COMMON.subprocess, "run", return_value=result) as probe, \
                mock.patch.object(COMMON, "source_provenance", return_value={"runtime_source_sha256": "source"}), \
                mock.patch.object(COMMON, "run_command", return_value=0) as child, \
                contextlib.redirect_stdout(io.StringIO()):
            code = COMMON.launch(command, environment, output, False, {"task": "sokoban"}, gpus=8)
        self.assertEqual(code, 0)
        self.assertEqual(probe.call_args.kwargs["env"], environment)
        child.assert_called_once_with(command, environment)
        manifests = list(output.glob("launch-*.json"))
        self.assertEqual(len(manifests), 1)
        manifest = json.loads(manifests[0].read_text())
        self.assertEqual(manifest["runtime"], runtime)
        self.assertEqual(manifest["returncode"], 0)
        self.assertEqual(manifest["status"], "succeeded")
        self.assertIn("finished_at", manifest)
        self.assertEqual(manifest["environment"]["CUDA_VISIBLE_DEVICES"], environment["CUDA_VISIBLE_DEVICES"])
        self.assertFalse((output / ".launch.lock").exists())

    def test_output_file_is_rejected_before_runtime_probe(self):
        output = self.scratch / "output-file"
        output.write_text("existing content")
        with mock.patch.object(COMMON, "validate_runtime") as probe, \
                contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(ValueError, "must be a directory"):
                COMMON.launch([sys.executable, "trainer.py"], {}, output, False, {}, gpus=1)
        probe.assert_not_called()
        self.assertEqual(output.read_text(), "existing content")

    def test_sharded_checkpoint_requires_all_local_shards(self):
        root = self.scratch.resolve()
        index = root / "ema.safetensors.index.json"
        index.write_text(json.dumps({"weight_map": {"layer": "part1.safetensors"}}))
        with self.assertRaisesRegex(ValueError, "Required file not found"):
            COMMON.validate_checkpoint(root)
        (root / "part1.safetensors").write_bytes(b"nonempty weight fixture")
        COMMON.validate_checkpoint(root)
        index.write_text(json.dumps({"weight_map": {"layer": "../outside.safetensors"}}))
        with self.assertRaisesRegex(ValueError, "outside its directory"):
            COMMON.validate_checkpoint(root)

    def test_checkpoint_rejects_empty_lfs_and_malformed_indexes(self):
        checkpoint = self.scratch / "checkpoint"
        checkpoint.mkdir()
        weight = checkpoint / "part.safetensors"
        index = checkpoint / "ema.safetensors.index.json"
        index.write_text(json.dumps({"weight_map": {"layer": weight.name}}))
        weight.touch()
        with self.assertRaisesRegex(ValueError, "empty"):
            COMMON.validate_checkpoint(checkpoint)
        weight.write_text("version https://git-lfs.github.com/spec/v1\noid sha256:example\n")
        with self.assertRaisesRegex(ValueError, "Git LFS pointer"):
            COMMON.validate_checkpoint(checkpoint)
        weight.write_bytes(b"nonempty weight fixture")
        invalid = (
            [], {"weight_map": {}}, {"weight_map": {"layer": []}},
            {"weight_map": {"layer": ""}}, {"weight_map": {"": weight.name}},
            {"weight_map": {"layer": str(weight)}},
            {"weight_map": {"layer": "nested/../part.safetensors"}},
            {"weight_map": {"layer": "part.json"}},
        )
        for value in invalid:
            with self.subTest(value=value):
                index.write_text(json.dumps(value))
                with self.assertRaises(ValueError):
                    COMMON.validate_checkpoint(checkpoint)
        index.write_text('{"weight_map": {"layer": "part.safetensors", "layer": "other.safetensors"}}')
        with self.assertRaisesRegex(ValueError, "duplicate JSON key"):
            COMMON.validate_checkpoint(checkpoint)
        for token in ("NaN", "Infinity", "-Infinity", "1e309", "-1e309"):
            index.write_text('{"metadata": {"total_size": ' + token +
                             '}, "weight_map": {"layer": "part.safetensors"}}')
            with self.subTest(token=token), self.assertRaisesRegex(ValueError, "non-finite JSON"):
                COMMON.validate_checkpoint(checkpoint)

    def test_checkpoint_supports_huggingface_snapshot_symlinks(self):
        checkpoint = self.scratch / "snapshot"
        checkpoint.mkdir()
        blob = self.scratch / "cache-blob"
        blob.write_bytes(b"nonempty weight fixture")
        shard = checkpoint / "part.safetensors"
        shard.symlink_to(blob)
        index = checkpoint / "ema.safetensors.index.json"
        index.write_text(json.dumps({"weight_map": {"layer": shard.name}}))
        self.assertEqual(COMMON.validate_checkpoint(checkpoint), [index, shard])

    def test_training_directory_discovery_matches_real_loader(self):
        data = self.scratch / "dataset"
        nested = data / "training"
        nested.mkdir(parents=True)
        shard = nested / "part.jsonl"
        shard.write_text('{}\n')
        with self.assertRaisesRegex(ValueError, "direct children"):
            COMMON.validate_episodes(data, directory_only=True)
        self.assertEqual(COMMON.validate_episodes(data), [shard])
        direct = data / "direct.jsonl.gz"
        with gzip.open(direct, "wt") as stream:
            stream.write('{}\n')
        tree = ast.parse((ROOT / "data/interleave_datasets/interleave_t2i_dataset.py").read_text())
        loader = next(node for node in tree.body if isinstance(node, ast.ClassDef)
                      and node.name == "JsonLIterableDataset")
        expand = next(node for node in loader.body if isinstance(node, ast.FunctionDef)
                      and node.name == "_expand_jsonl_paths")
        namespace = {"os": os}
        exec(compile(ast.Module(body=[expand], type_ignores=[]), "loader-discovery", "exec"), namespace)
        self.assertEqual(COMMON.validate_episodes(data, directory_only=True),
                         [Path(name) for name in namespace["_expand_jsonl_paths"](None, [str(data)])])

    def test_row_checks_reject_wrong_schema_empty_pool_and_too_few_training_rows(self):
        path = self.scratch / "data.jsonl"
        row = {"messages": [{"role": role, "content": [{"type": "text", "text": "example"}]}
                            for role in ("user", "assistant")]}
        path.write_text(json.dumps(row) + "\n")
        self.assertEqual(COMMON.validate_rows([path]), 1)
        with self.assertRaisesRegex(ValueError, "at least 2"):
            COMMON.validate_rows([path], minimum=2)
        with self.assertRaisesRegex(ValueError, "evaluation episode"):
            COMMON.validate_rows([path], task="sokoban", limit=5)
        path.write_text(json.dumps({"env_id": "sokoban/easy", "init_state": {"board": [[0]]}}) + "\n")
        self.assertEqual(COMMON.validate_rows([path], task="sokoban", limit=5), 1)
        with self.assertRaisesRegex(ValueError, "prepared messages"):
            COMMON.validate_rows([path])
        with self.assertRaisesRegex(ValueError, "maze2d evaluation episode"):
            COMMON.validate_rows([path], task="maze2d", limit=5)
        path.write_text('{"env_id": "sokoban/easy"}\n')
        with self.assertRaisesRegex(ValueError, "init_state"):
            COMMON.validate_rows([path], task="sokoban")
        path.write_text("\n")
        with self.assertRaisesRegex(ValueError, "Found only 0"):
            COMMON.validate_rows([path], task="sokoban")
        path.write_text('{"env_id": "sokoban/easy", "init_state": {"value": 1e309}}\n')
        with self.assertRaisesRegex(ValueError, "non-finite JSON"):
            COMMON.validate_rows([path], task="sokoban")

    def test_output_cannot_overlap_inputs_or_pollute_recursive_episode_scan(self):
        data = self.scratch / "episodes"
        data.mkdir()
        episode = data / "test.jsonl"
        episode.write_text('{}\n')
        for output, inputs in ((data, [data]), (data / "results", [data]),
                               (self.scratch, [episode]), (episode, [episode])):
            with self.subTest(output=output, inputs=inputs), self.assertRaisesRegex(ValueError, "overlap"):
                COMMON.validate_output_location(output, inputs)
        COMMON.validate_output_location(self.scratch / "results", [data])

    def test_asset_fingerprints_hash_episode_contents_without_hashing_weights(self):
        episode = self.scratch / "test.jsonl"
        episode.write_text('{"value": 1}\n')
        weight = self.scratch / "ema.safetensors"
        weight.write_bytes(b"nonempty weight fixture")
        before = COMMON.file_fingerprints([episode], hash_all=True)
        timestamp = episode.stat().st_mtime_ns
        episode.write_text('{"value": 2}\n')
        os.utime(episode, ns=(timestamp, timestamp))
        after = COMMON.file_fingerprints([episode], hash_all=True)
        self.assertEqual(before[0]["size"], after[0]["size"])
        self.assertEqual(before[0]["mtime_ns"], after[0]["mtime_ns"])
        self.assertNotEqual(before[0]["sha256"], after[0]["sha256"])
        self.assertNotIn("sha256", COMMON.file_fingerprints([weight])[0])

    def evaluation_launch(self, *, command=None, assets=None, source=None, env=None, returncode=0):
        runtime = {"python": "3.10.0", "torch": "2.5.1", "cuda_available": True, "visible_gpus": 1}
        source = source or {"runtime_source_sha256": "original-source", "git_commit": "one", "git_dirty": False}
        with mock.patch.object(COMMON, "validate_runtime", return_value=runtime), \
                mock.patch.object(COMMON, "source_provenance", return_value=source), \
                mock.patch.object(COMMON, "run_command", return_value=returncode) as child, \
                contextlib.redirect_stdout(io.StringIO()):
            code = COMMON.launch(command or [sys.executable, "evaluator.py", "--limit", "5"],
                                 env or {}, self.scratch / "evaluation", False,
                                 {"task": "sokoban", "decoder": "perseg"}, gpus=1,
                                 assets=assets or {"checkpoint": "original-weights"}, evaluation=True)
        return code, child

    def test_evaluation_resume_accepts_same_contract_and_clears_stale_rank_barriers(self):
        self.evaluation_launch()
        output = self.scratch / "evaluation"
        summaries = output / "_rank_summaries"
        summaries.mkdir()
        stale = summaries / "rank_00000.json"
        stale.write_text('{"completed": 100}')
        (output / "summary.json").write_text('{"success_rate": 1.0}')
        episode = output / "episode_stats.json"
        episode.write_text('{"success": true}')
        self.evaluation_launch(command=[sys.executable, "evaluator.py", "--limit", "5", "--overwrite"],
                               source={"runtime_source_sha256": "original-source", "git_commit": "documentation-commit", "git_dirty": True})
        self.assertFalse(stale.exists())
        self.assertFalse((output / "summary.json").exists())
        self.assertTrue(episode.exists())
        self.assertEqual(len(list(output.glob("launch-*.json"))), 2)
        self.assertTrue((output / "run-contract.json").is_file())

    def test_evaluation_refuses_changed_contract_even_with_overwrite(self):
        self.evaluation_launch()
        cases = (
            {"command": [sys.executable, "evaluator.py", "--limit", "100", "--overwrite"]},
            {"assets": {"checkpoint": "replacement-weights"}},
            {"source": {"runtime_source_sha256": "changed-runtime-source"}},
            {"env": {"CUDA_VISIBLE_DEVICES": "1"}},
        )
        for changed in cases:
            with self.subTest(changed=changed), self.assertRaisesRegex(ValueError, "different run"):
                self.evaluation_launch(**changed)
        self.assertEqual(len(list((self.scratch / "evaluation").glob("launch-*.json"))), 1)

    def test_evaluation_rejects_legacy_output_and_locked_runs(self):
        output = self.scratch / "evaluation"
        output.mkdir()
        legacy = output / "episode_stats.json"
        legacy.write_text('{"success": true}')
        with self.assertRaisesRegex(ValueError, "no run-contract.json"):
            self.evaluation_launch()
        legacy.unlink()
        self.evaluation_launch()
        lock = output / ".launch.lock"
        lock.write_text('{"pid": 123}')
        with self.assertRaisesRegex(ValueError, "verify its job has stopped"):
            self.evaluation_launch()
        self.assertTrue(lock.is_file())

    def test_failed_launch_is_recorded_without_logging_credentials(self):
        secret = "fixture-secret-not-for-launch-records"
        self.evaluation_launch(returncode=7, env={"HF_TOKEN": secret, "BAGEL_API_KEY": secret,
                                                 "WANDB_API_KEY": secret, "CUDA_VISIBLE_DEVICES": "0"})
        path = next((self.scratch / "evaluation").glob("launch-*.json"))
        manifest = json.loads(path.read_text())
        self.assertEqual(manifest["status"], "failed")
        self.assertEqual(manifest["returncode"], 7)
        self.assertEqual(manifest["environment"], {"CUDA_VISIBLE_DEVICES": "0"})
        self.assertNotIn(secret, path.read_text())
        self.assertFalse((path.parent / ".launch.lock").exists())

    @unittest.skipUnless(os.name == "posix", "POSIX session/signal regression")
    def test_signaled_supervisor_retains_lock_while_independent_worker_survives(self):
        output = self.scratch / "signaled-run"
        worker_pid = self.scratch / "worker.pid"
        supervisor = (
            "import os, signal, subprocess, sys; from pathlib import Path; "
            "worker = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'], "
            "start_new_session=True); "
            f"Path({str(worker_pid)!r}).write_text(str(worker.pid)); "
            "os.kill(os.getpid(), signal.SIGKILL)"
        )
        command = [sys.executable, "-B", "-S", "-c", supervisor]
        try:
            with mock.patch.object(COMMON, "validate_runtime", return_value={"python": "fixture", "torch": "fixture"}), \
                    mock.patch.object(COMMON, "source_provenance", return_value={"runtime_source_sha256": "fixture"}), \
                    contextlib.redirect_stdout(io.StringIO()), \
                    contextlib.redirect_stderr(io.StringIO()):
                code = COMMON.launch(command, dict(os.environ), output, False, {}, gpus=1, evaluation=True)
                self.assertEqual(code, -signal.SIGKILL)
                os.kill(int(worker_pid.read_text()), 0)
                self.assertTrue((output / ".launch.lock").is_file())
                manifest = json.loads(next(output.glob("launch-*.json")).read_text())
                self.assertEqual(manifest["status"], "interrupted")
                self.assertEqual(manifest["returncode"], -signal.SIGKILL)
                with mock.patch.object(COMMON, "run_command") as retry:
                    with self.assertRaisesRegex(ValueError, "Output is locked"):
                        COMMON.launch(command, dict(os.environ), output, False, {}, gpus=1, evaluation=True)
                    retry.assert_not_called()
        finally:
            if worker_pid.exists():
                try:
                    os.kill(int(worker_pid.read_text()), signal.SIGKILL)
                except ProcessLookupError:
                    pass

    def test_training_rechecks_empty_output_after_preflight_and_lock(self):
        output = self.scratch / "training"
        marker = output / "other-launcher-result.json"

        def concurrent_launch_finishes(*_):
            # Simulate another launcher completing after this invocation's
            # initial empty-directory check, before it acquires the lock.
            output.mkdir()
            marker.write_text('{"completed": true}')
            return {"python": "3.10", "torch": "2.5.1"}

        with mock.patch.object(COMMON, "validate_runtime", side_effect=concurrent_launch_finishes), \
                mock.patch.object(COMMON, "source_provenance", return_value={"runtime_source_sha256": "source"}), \
                mock.patch.object(COMMON, "run_command") as child, \
                contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(ValueError, "no longer empty"):
                COMMON.launch([sys.executable, "trainer.py"], {}, output, False, {}, gpus=1)
        child.assert_not_called()
        self.assertEqual(marker.read_text(), '{"completed": true}')
        self.assertFalse((output / ".launch.lock").exists())
        self.assertFalse(list(output.glob("launch-*.json")))

    def test_runtime_source_fingerprint_excludes_documentation_edits(self):
        (self.scratch / "inferencer.py").write_text("VALUE = 1\n")
        (self.scratch / "requirements.txt").write_text("example==1\n")
        documentation = self.scratch / "README.md"
        documentation.write_text("First version\n")
        git = subprocess.CompletedProcess([], 1, "", "not a git repository")
        with mock.patch.object(COMMON, "ROOT", self.scratch), \
                mock.patch.object(COMMON.subprocess, "run", return_value=git):
            first = COMMON.source_provenance()
            documentation.write_text("Documentation improved\n")
            second = COMMON.source_provenance()
            self.assertEqual(first, second)
            (self.scratch / "inferencer.py").write_text("VALUE = 2\n")
            self.assertNotEqual(first["runtime_source_sha256"], COMMON.source_provenance()["runtime_source_sha256"])
            assets = self.scratch / "gymnasium/assets"
            assets.mkdir(parents=True)
            frame = assets / "sprite.png"
            frame.write_bytes(b"first fixture pixels")
            before = COMMON.source_provenance()["runtime_source_sha256"]
            frame.write_bytes(b"other fixture pixels")
            self.assertNotEqual(before, COMMON.source_provenance()["runtime_source_sha256"])

    def test_source_archive_does_not_claim_an_unrelated_parent_commit(self):
        subprocess.run(["git", "init", "-q", str(self.scratch)], check=True)
        subprocess.run(["git", "-C", str(self.scratch), "-c", "user.name=Fixture",
                        "-c", "user.email=fixture@example.invalid", "-c", "commit.gpgsign=false",
                        "commit", "--allow-empty", "-qm", "Unrelated project"], check=True)
        archive = self.scratch / "source"
        archive.mkdir()
        (archive / "inferencer.py").write_text("VALUE = 1\n")
        (archive / "requirements.txt").write_text("example==1\n")
        with mock.patch.object(COMMON, "ROOT", archive):
            result = COMMON.source_provenance()
        self.assertIsNone(result["git_commit"])
        self.assertIsNone(result["git_dirty"])
        self.assertEqual(len(result["runtime_source_sha256"]), 64)

    def test_interrupt_cleanup_has_bounded_escalation(self):
        process = mock.Mock()
        process.poll.return_value = None
        process.wait.side_effect = [KeyboardInterrupt(),
                                    subprocess.TimeoutExpired("torchrun", 30),
                                    subprocess.TimeoutExpired("torchrun", 10),
                                    subprocess.TimeoutExpired("torchrun", 5)]
        output = self.scratch / "forced-interruption"
        with mock.patch.object(COMMON.subprocess, "Popen", return_value=process) as popen, \
                mock.patch.object(COMMON, "validate_runtime", return_value={"python": "fixture", "torch": "fixture"}), \
                mock.patch.object(COMMON, "source_provenance", return_value={"runtime_source_sha256": "fixture"}), \
                contextlib.redirect_stdout(io.StringIO()), \
                contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(KeyboardInterrupt):
                COMMON.launch([sys.executable, "torchrun-fixture.py"], {}, output, False,
                              {}, gpus=1, evaluation=True)
        popen.assert_called_once_with([sys.executable, "torchrun-fixture.py"],
                                      cwd=ROOT, env={}, start_new_session=True)
        self.assertEqual(process.send_signal.call_args_list,
                         [mock.call(signal.SIGINT), mock.call(signal.SIGTERM)])
        process.kill.assert_called_once_with()
        self.assertEqual(process.wait.call_args_list,
                         [mock.call(), mock.call(timeout=30), mock.call(timeout=10), mock.call(timeout=5)])
        self.assertTrue((output / ".launch.lock").exists())
        manifest = json.loads(next(output.glob("launch-*.json")).read_text())
        self.assertEqual(manifest["status"], "interrupted")

    def test_unexpected_process_errors_retain_lock_and_record_interruption(self):
        for failure in (OSError("spawn failed"), RuntimeError("wait failed")):
            output = self.scratch / type(failure).__name__
            with self.subTest(failure=failure), \
                    mock.patch.object(COMMON, "run_command", side_effect=failure), \
                    mock.patch.object(COMMON, "validate_runtime", return_value={"python": "fixture", "torch": "fixture"}), \
                    mock.patch.object(COMMON, "source_provenance", return_value={"runtime_source_sha256": "fixture"}), \
                    contextlib.redirect_stdout(io.StringIO()), \
                    contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(type(failure)):
                    COMMON.launch([sys.executable, "trainer.py"], {}, output, False, {}, gpus=1)
            self.assertTrue((output / ".launch.lock").exists())
            manifest = json.loads(next(output.glob("launch-*.json")).read_text())
            self.assertEqual(manifest["status"], "interrupted")

    @unittest.skipUnless(os.name == "posix", "POSIX session/signal regression")
    def test_ctrl_c_waits_for_independent_worker_cleanup_and_retains_lock(self):
        # Match torchrun's topology: its worker starts in an independent session.
        # Cleanup deliberately takes > subprocess.call's 0.25-second interrupt
        # grace, reproducing the old orphan-worker / prematurely removed lock bug.
        ready = self.scratch / "ready"
        worker_pid = self.scratch / "worker.pid"
        cleaned = self.scratch / "worker-cleaned"
        output = self.scratch / "interrupted-run"
        supervisor = self.scratch / "supervisor.py"
        worker_code = ("import os, time; from pathlib import Path; "
                       f"Path({str(worker_pid)!r}).write_text(str(os.getpid())); time.sleep(30)")
        supervisor.write_text(
            "import signal, subprocess, sys, time\nfrom pathlib import Path\n"
            f"worker = subprocess.Popen([sys.executable, '-c', {worker_code!r}], start_new_session=True)\n"
            "def stop(signum, frame):\n"
            "    signal.signal(signal.SIGINT, signal.SIG_IGN)\n"
            "    time.sleep(0.7)\n"
            "    worker.terminate()\n"
            "    worker.wait(timeout=2)\n"
            f"    Path({str(cleaned)!r}).write_text('worker stopped')\n"
            "    raise SystemExit(0)\n"
            "signal.signal(signal.SIGINT, stop)\n"
            f"Path({str(ready)!r}).write_text('ready')\n"
            "while True:\n    time.sleep(1)\n"
        )
        harness = self.scratch / "harness.py"
        harness.write_text(
            "import importlib.util, os, sys\nfrom pathlib import Path\n"
            f"spec = importlib.util.spec_from_file_location('common', {str(ROOT / 'scripts/_common.py')!r})\n"
            "common = importlib.util.module_from_spec(spec)\nspec.loader.exec_module(common)\n"
            "common.validate_runtime = lambda *_: {'python': 'fixture', 'torch': 'fixture'}\n"
            "common.source_provenance = lambda: {'runtime_source_sha256': 'fixture'}\n"
            "common.INTERRUPT_GRACE_SECONDS = (3, 1, 1)\n"
            "try:\n"
            f"    common.launch([sys.executable, {str(supervisor)!r}], dict(os.environ), "
            f"Path({str(output)!r}), False, {{}}, gpus=1, evaluation=True)\n"
            "except KeyboardInterrupt:\n    raise SystemExit(130)\n"
        )
        log_path = self.scratch / "harness.log"
        process = None
        try:
            with log_path.open("w") as log:
                process = subprocess.Popen([sys.executable, "-B", "-S", str(harness)],
                                           stdout=log, stderr=log, start_new_session=True)
                deadline = time.monotonic() + 5
                while not (ready.exists() and worker_pid.exists()) and time.monotonic() < deadline:
                    self.assertIsNone(process.poll(), log_path.read_text())
                    time.sleep(0.02)
                self.assertTrue(ready.exists() and worker_pid.exists(), log_path.read_text())
                os.killpg(process.pid, signal.SIGINT)
                self.assertEqual(process.wait(timeout=8), 130, log_path.read_text())
            self.assertTrue(cleaned.exists(), log_path.read_text())
            with self.assertRaises(ProcessLookupError):
                os.kill(int(worker_pid.read_text()), 0)
            self.assertTrue((output / ".launch.lock").is_file())
            manifest = json.loads(next(output.glob("launch-*.json")).read_text())
            self.assertEqual(manifest["status"], "interrupted")
            self.assertIsNone(manifest["returncode"])
            self.assertIn("finished_at", manifest)
            self.assertIn("Output lock retained", log_path.read_text())
        finally:
            if process is not None and process.poll() is None:
                process.kill()
                process.wait(timeout=3)
            if worker_pid.exists():
                try:
                    os.kill(int(worker_pid.read_text()), signal.SIGKILL)
                except ProcessLookupError:
                    pass


if __name__ == "__main__":
    unittest.main()
