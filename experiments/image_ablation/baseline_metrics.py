#!/usr/bin/env python3
"""Replay staged No-Imagine actions for invalid-move and termination metadata."""

from __future__ import annotations

import argparse
import json
import tarfile
from pathlib import Path
from types import SimpleNamespace

import evaluation.base as base
import evaluation.runtime as G
from evaluation.environments import _reset_preserving_init_state
from evaluation.actions import parse_action_chunk
from experiments.image_ablation.preflight import ASSETS, ROOT, archived_stats, rows_for
from experiments.image_ablation.oracle import invalid_action, physical_state


def last_chunk_text(task: str) -> dict[int, str]:
    path = ASSETS / f"{task}_k5_noncot.tar.gz"
    result = {}
    with tarfile.open(path, "r:gz") as archive:
        for member in archive:
            if not member.isfile() or not member.name.endswith("/cot_full_text.txt"):
                continue
            episode = member.name.split("/")[-2]
            index = int(episode.split("_seed")[0].removeprefix("idx"))
            text = archive.extractfile(member).read().decode()
            result[index] = text.split("----- CHUNK -----")[-1]
    return result


def evaluate(task: str, limit: int) -> list[dict]:
    rows, _ = rows_for(task)
    stats = archived_stats(task, "noncot")
    texts = last_chunk_text(task)
    args = SimpleNamespace(
        profile="sokoban_v8_res256" if task == "sokoban" else "pacman_res256",
        sokoban_action_mode_override="move_only", sokoban_success_mode="stop_then_solved",
        sim2real=False, action_preset=None, action_name_map=None,
    )
    results = []
    for index, (row, published) in enumerate(zip(rows[:limit], stats[:limit])):
        ep = G.prepare_sokoban_episode(row, args)
        seed = int(ep["episode_seed"])
        base.set_global_seed(int(ep["seed"]))
        env, _ = G.make_profiled_env(ep, args)
        try:
            _reset_preserving_init_state(env, seed=seed, init_state=ep["init_state"])
            invalid = 0
            last_info = {}
            terminated = truncated = False
            for item in published["history"]:
                before = physical_state(env, task)
                _, _, terminated, truncated, info = env.step(item["action"])
                last_info = info or {}
                after = physical_state(env, task)
                invalid += int(invalid_action(task, before, after, last_info,
                                              G._is_stop_action(item["action"])))
                if bool(terminated) != bool(item["terminated"]) or bool(truncated) != bool(item["truncated"]):
                    raise AssertionError((task, index, "simulator/release transition mismatch", item["action"]))
            success = bool(published["stats"]["success"])
            final_action = published["stats"].get("final_action")
            parse_failure = not parse_action_chunk(texts.get(index, ""))
            if success:
                reason = "solved"
            elif task == "pacman" and last_info.get("alive") is False:
                reason = "caught_by_ghost"
            elif parse_failure:
                reason = "parse_failure"
            elif truncated or len(published["history"]) >= row["env_kwargs"]["max_steps"]:
                reason = "budget_exhausted"
            else:
                reason = "other"
            results.append({
                "episode_id": f"{task}:{index:07d}:{seed}", "task": task,
                "condition": "No-Imagine", "success": success,
                "termination_reason": reason,
                "num_executed_actions": len(published["history"]),
                "num_chunks": published["n_chunks"], "parse_failure": parse_failure,
                "num_invalid_actions": invalid, "stop_emitted": G._is_stop_action(final_action),
                # Every archived action-only chunk is at most 66 characters, well below
                # the published 600-token generation cap.
                "truncated": False, "committed_matches_imagined": None,
                "global_index": index, "seed": published["seed"],
                "episode_seed": seed, "source": "staged_k5_noncot_action_replay",
            })
        finally:
            env.close()
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--output", type=Path, default=ROOT / "prepared" / "baseline_metrics.jsonl")
    args = parser.parse_args()
    output = evaluate("sokoban", args.limit) + evaluate("pacman", args.limit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as stream:
        for row in output:
            stream.write(json.dumps(row, separators=(",", ":")) + "\n")
    print(f"Wrote {len(output)} paired No-Imagine metadata rows to {args.output}")
