#!/usr/bin/env python3
# Modified for Agima: portable paths and explicit dataset arguments.
"""Checkpoint-free PushT candidate-action coverage scorer (EXP2 branching substrate).

Given a PushT env spec (`env_kwargs`, exactly the dict baked into every source row) and a
physics `state` snapshot (the `init_state` dict of a row, or any `get_init_state()` dict),
score what coverage a candidate action -- or a whole action list -- *produces*, by actually
cloning the deterministic PushT physics env and stepping it. No PPO checkpoint, no learned
policy, no reward model: the ground-truth simulator IS the scorer.

Public API
----------
    score_action(env_kwargs, state, action_str) -> float
        Resulting coverage after executing ONE action string from `state`.
    score_rollout(env_kwargs, state, action_list) -> list[float]
        Coverage after EACH action (len == len(action_list)); env reset to `state` once,
        then the actions are applied in order (physics carries over between actions).

Both take the norm4 action serialization used verbatim by the source/remap rows, e.g.
"('move', (0.4025, 0.7464))" and "('stop', 'stop')" -- step() parses it via ast.literal_eval
and decodes with the row's own `action_codec` (norm4). `state` is a dict matching
PushTEnv.get_init_state() / the row `init_state` (agent_position, agent_velocity,
block_position, block_velocity, block_angle, block_angvel, goal_pose, episode_step); a bare
5-vector [ax, ay, bx, by, angle] is also accepted by reset(init_state=...).

This deliberately does NOT use the env's own stale planning helpers
(`_make_planning_env` / `_planning_score`), which crash on the main import path. It uses only
the stable public surface: PushTEnv(**kwargs) -> reset(init_state=state) -> step(action)
-> info['coverage'] (gymnasium/envs/pusht/pusht.py: get_init_state 224-234,
reset(init_state=) 236-262, step 264-324, _get_coverage 588-592, info['coverage'] 701).

Run directly for a self-test that replays a recorded expert episode from the CoT-inf source
(remap_pusht_cot_chunk_inf.DEFAULT_SRC) and checks the reproduced coverage matches the row's
recorded success/coverage:

    PYTHONPATH=. python scripts/data/pusht_coverage_score.py --src-root /path/to/source
"""
from __future__ import annotations

import inspect
import os
from typing import Dict, List, Sequence, Union

# Headless physics/render: PushT.step() unconditionally re-renders an observation via pygame,
# so a dummy SDL video driver is required on a display-less node (matches the dataset
# generators, e.g. scripts/pusht_gen/generate_pusht_ppo_success_dataset.py).
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("XDG_RUNTIME_DIR", "/tmp")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

from gymnasium.envs.pusht.pusht import PushTEnv  # noqa: E402  (import after SDL setup)

State = Union[Dict, Sequence[float]]

# Cache the accepted constructor parameters once so we can pass a row's env_kwargs verbatim
# (dropping any stray keys) without ever touching the env's stale planning helpers.
_CTOR_PARAMS = frozenset(inspect.signature(PushTEnv.__init__).parameters) - {"self"}


def make_env(env_kwargs: Dict) -> PushTEnv:
    """Instantiate a fresh PushTEnv from a row's `env_kwargs`.

    Only keys accepted by PushTEnv.__init__ are forwarded (e.g. action_codec, control_hz,
    success_threshold, init_distribution, agent_render_radius, prompt_mode). render_mode is
    pinned to 'rgb_array' so no interactive window is ever opened.
    """
    kwargs = {k: v for k, v in dict(env_kwargs or {}).items() if k in _CTOR_PARAMS}
    kwargs["render_mode"] = "rgb_array"
    return PushTEnv(**kwargs)


def score_rollout(
    env_kwargs: Dict,
    state: State,
    action_list: Sequence[str],
    *,
    stop_on_terminal: bool = False,
) -> List[float]:
    """Coverage after each action in `action_list`, replayed from `state`.

    Returns one float per action (len == len(action_list)) unless stop_on_terminal=True, in
    which case replay halts (and the returned list is shorter) once the env reports
    terminated/truncated. Coverage is info['coverage'] == overlap(goal T, block T) / area(goal T),
    the same quantity `is_success` (coverage >= success_threshold) is thresholded on.
    """
    env = make_env(env_kwargs)
    try:
        env.reset(init_state=state)
        coverages: List[float] = []
        for action in action_list:
            _obs, _reward, terminated, truncated, info = env.step(str(action))
            coverages.append(float(info["coverage"]))
            if stop_on_terminal and (terminated or truncated):
                break
        return coverages
    finally:
        env.close()


def score_action(env_kwargs: Dict, state: State, action_str: str) -> float:
    """Resulting coverage after executing ONE action string from `state` (float)."""
    return score_rollout(env_kwargs, state, [action_str])[0]


# --------------------------------------------------------------------------------------- #
# Self-test: replay a recorded expert episode and confirm we reproduce its coverage.
# --------------------------------------------------------------------------------------- #
def _load_first_source_episode(src_root: str):
    import glob
    import gzip
    import json

    shards = sorted(glob.glob(f"{src_root}/data/train/*.jsonl.gz"))
    if not shards:
        raise FileNotFoundError(f"no shards under {src_root}/data/train")
    with gzip.open(shards[0], "rt") as f:
        for line in f:
            if line.strip():
                return json.loads(line)
    raise RuntimeError(f"no rows in {shards[0]}")


def _self_test(src_root: str) -> int:
    row = _load_first_source_episode(src_root)
    env_kwargs = row["env_kwargs"]
    state = row["init_state"]
    actions = [str(h.get("vlm_output") or h.get("action")) for h in row["history"]]
    rec_stats = row.get("stats", {})
    rec_success = bool(rec_stats.get("success"))
    rec_coverage = float(rec_stats.get("coverage", float("nan")))
    thr = float(env_kwargs.get("success_threshold", 0.9))

    print(f"[self-test] src={src_root}")
    print(f"[self-test] episode order_key={row.get('__bagel_order_key')} n_actions={len(actions)} "
          f"success_threshold={thr}")

    coverages = score_rollout(env_kwargs, state, actions)

    init_cov = float(score_action(env_kwargs, state, actions[0]))  # sanity: single-action path
    final_cov = coverages[-1]
    # "monotonic-ish": coverage should trend up; count non-decreasing transitions and the net rise.
    non_dec = sum(1 for a, b in zip(coverages, coverages[1:]) if b >= a - 1e-6)
    frac_non_dec = non_dec / max(1, len(coverages) - 1)

    print("[self-test] per-action coverage:")
    for i, (a, c) in enumerate(zip(actions, coverages)):
        print(f"    {i:2d} {a:34s} -> coverage={c:.4f}")
    print(f"[self-test] first-action coverage (score_action path) = {init_cov:.4f} "
          f"(matches rollout[0]={coverages[0]:.4f}: {abs(init_cov - coverages[0]) < 1e-9})")
    print(f"[self-test] final replayed coverage = {final_cov:.4f}")
    print(f"[self-test] recorded stats coverage = {rec_coverage:.4f}  success={rec_success}")
    print(f"[self-test] monotonic-ish: {non_dec}/{len(coverages) - 1} non-decreasing "
          f"({frac_non_dec:.0%}); net rise {coverages[0]:.4f} -> {final_cov:.4f}")

    checks = {
        "final_success_ge_threshold": final_cov >= thr,
        "recorded_was_success": rec_success,
        "final_matches_recorded(|d|<0.02)": abs(final_cov - rec_coverage) < 0.02,
        "single==rollout[0]": abs(init_cov - coverages[0]) < 1e-9,
        "final_ge_0.9": final_cov >= 0.9,
        "trends_up(final>first)": final_cov > coverages[0],
    }
    print("[self-test] CHECKS:")
    for name, ok in checks.items():
        print(f"    [{'PASS' if ok else 'FAIL'}] {name}")
    ok_all = all(checks.values())
    print(f"[self-test] RESULT: {'PASS' if ok_all else 'FAIL'}")
    return 0 if ok_all else 1


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Replay a recorded PushT expert episode.")
    parser.add_argument("--src-root", required=True)
    args = parser.parse_args()
    raise SystemExit(_self_test(args.src_root))
