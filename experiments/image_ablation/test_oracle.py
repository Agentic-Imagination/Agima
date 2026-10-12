#!/usr/bin/env python3
"""Pixel-exact real-vs-cloned-simulator test, with a noninitial Pacman chunk."""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import evaluation.runtime as G
from evaluation.environments import _reset_preserving_init_state
from experiments.image_ablation.oracle import OracleReplica, render_rgb


ASSETS = Path(__file__).resolve().parent / "assets"


def first_row(task: str) -> dict:
    with gzip.open(ASSETS / f"{task}_hard_test.jsonl.gz", "rt") as stream:
        return json.loads(next(stream))


def run_test(task: str) -> None:
    row = first_row(task)
    args = SimpleNamespace(
        profile="sokoban_v8_res256" if task == "sokoban" else "pacman_res256",
        sokoban_action_mode_override="move_only", sokoban_success_mode="stop_then_solved",
        sim2real=False, action_preset=None, action_name_map=None,
    )
    ep = G.prepare_sokoban_episode(row, args)
    env, _ = G.make_profiled_env(ep, args)
    try:
        _reset_preserving_init_state(env, seed=ep["episode_seed"], init_state=ep["init_state"])
        # Make the clone at a noninitial chunk boundary. These moves are a fixed simulator test,
        # never inputs to model inference.
        for action in ["('move', 0)", "('move', 1)"]:
            env.step(action)
        clone = OracleReplica(env)
        try:
            assert np.array_equal(render_rgb(env), clone.last_frame)
            real_rng = getattr(env.unwrapped, "np_random", None)
            clone_rng = getattr(clone.env.unwrapped, "np_random", None)
            if real_rng is not None and clone_rng is not None:
                assert real_rng.bit_generator.state == clone_rng.bit_generator.state
            checks = 0
            for action in ["('move', 2)", "('move', 0)", "('move', 3)",
                           "('move', 1)", "('move', 1)", "('move', 0)"]:
                _, _, terminated, truncated, _ = env.step(action)
                assert np.array_equal(render_rgb(env), clone.step(action)), (task, action)
                checks += 1
                if terminated or truncated:
                    terminal = clone.step("('move', 0)")
                    assert np.array_equal(terminal, clone.last_frame)
                    break
            print(f"{task}: {checks} real/Oracle frames identical, including RNG state")
        finally:
            clone.close()
    finally:
        env.close()


if __name__ == "__main__":
    run_test("sokoban")
    run_test("pacman")
