"""Pixel-exact Oracle checks against the production Maze and PushT simulators."""

import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

import evaluation.runtime as G
from evaluation.environments import _reset_preserving_init_state
from experiments.image_ablation.oracle import render_rgb
from experiments.image_ablation.preflight_maze_pusht import CONFIG, archived_stats


ROOT = "experiments/image_ablation/assets"


def _load_runner(task):
    """Load the task's configuration independently of test order and prior imports."""
    settings = {"AGIMA_TASK": task, "AGIMA_CONDITION": "oracle",
                "BAGEL_DECODER": "span" if task == "maze" else "perseg",
                "BAGEL_CHUNK_K": "5"}
    path = Path(__file__).with_name("run_maze_pusht.py")
    spec = importlib.util.spec_from_file_location(f"oracle_test_runner_{task}", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load Oracle runner from {path}")
    runner = importlib.util.module_from_spec(spec)
    with patch.dict(os.environ, settings):
        spec.loader.exec_module(runner)
    return runner


class OraclePixelTest(unittest.TestCase):
    def _run(self, task, path, profile, actions):
        A = _load_runner(task)
        args = SimpleNamespace(profile=profile, sim2real=False,
                               action_preset=None, action_name_map=None)
        for ep in G.scan_compact_episodes(path, 3, None):
            env, _ = G.make_profiled_env(ep, args)
            seed = int(ep.get("episode_seed") or ep.get("seed") or
                       (ep.get("alignment_index") or {}).get("seed") or 0)
            _reset_preserving_init_state(env, seed=seed, init_state=ep["init_state"])
            past_actions = []
            for _ in range(2):
                replica = A._Replica(env, ep, args, past_actions)
                self.assertTrue(np.array_equal(render_rgb(env), replica.last_frame))
                for action in actions:
                    _, _, terminated, truncated, _ = env.step(action)
                    past_actions.append(action)
                    frame = replica.step(action)
                    self.assertTrue(np.array_equal(render_rgb(env), frame),
                                    f"{task} ep{ep['global_index']} action {action}")
                    if terminated or truncated:
                        break
                replica.close()
            env.close()

    def test_maze(self):
        self._run("maze", f"{ROOT}/maze_hard_fixedstart/testing/shard_00000.jsonl.gz",
                  "maze2d_native256", ["('move', 0)", "('move', 1)", "('move', 2)"])

    def test_pusht(self):
        self._run("pusht", f"{ROOT}/pusht_in_dist/data/test/batch_000000.jsonl.gz",
                  "pusht_norm4_native", ["('move', (0.5, 0.5))", "('move', (0.4, 0.4))",
                                          "('move', (0.3, 0.6))"])

    def test_published_actions_across_chunks(self):
        # Exercise genuine model plans, including PushT block contacts, rather than
        # only synthetic actions that may leave the block untouched.
        for task, spec in CONFIG.items():
            A = _load_runner(task)
            path = str(spec["eval"])
            profile = "maze2d_native256" if task == "maze" else "pusht_norm4_native"
            args = SimpleNamespace(profile=profile, sim2real=False,
                                   action_preset=None, action_name_map=None)
            episodes = G.scan_compact_episodes(path, 3, None)
            traces = archived_stats(spec["cot"])[:3]
            for ep, trace in zip(episodes, traces):
                env, _ = G.make_profiled_env(ep, args)
                seed = int(trace["episode_seed"])
                _reset_preserving_init_state(env, seed=seed, init_state=ep["init_state"])
                history = trace["history"]
                past_actions = []
                for chunk_index in sorted(set(int(item["chunk"]) for item in history)):
                    replica = A._Replica(env, ep, args, past_actions)
                    for item in history:
                        if int(item["chunk"]) != chunk_index:
                            continue
                        _, _, terminated, truncated, _ = env.step(item["action"])
                        past_actions.append(item["action"])
                        frame = replica.step(item["action"])
                        self.assertTrue(np.array_equal(render_rgb(env), frame),
                                        f"{task} ep{ep['global_index']} chunk{chunk_index} {item['action']}")
                        if terminated or truncated:
                            break
                    replica.close()
                env.close()


if __name__ == "__main__":
    unittest.main()
