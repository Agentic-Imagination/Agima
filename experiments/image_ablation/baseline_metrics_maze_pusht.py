"""Replay staged Maze/PushT No-Imagine K=5 actions for paired diagnostics."""

import gzip
import json
import tarfile
from pathlib import Path
from types import SimpleNamespace

import evaluation.base as base
import evaluation.runtime as G
from evaluation.environments import _reset_preserving_init_state
from evaluation.actions import parse_action_chunk
from experiments.image_ablation.preflight_maze_pusht import CONFIG, ROOT, archived_stats


def _physical(env, task):
    state = env.unwrapped.get_init_state()
    keys = ("agent_location",) if task == "maze" else (
        "agent_position", "agent_velocity", "block_position", "block_velocity", "block_angle", "block_angvel")
    return tuple(json.dumps(state.get(key), sort_keys=True) for key in keys)


def _texts(path):
    texts = {}
    with tarfile.open(path) as archive:
        for member in archive:
            if member.isfile() and member.name.endswith("/cot_full_text.txt"):
                epdir = member.name.split("/")[-2]
                idx = int(epdir.split("_seed")[0].removeprefix("idx"))
                texts[idx] = archive.extractfile(member).read().decode()
    return texts


def evaluate(task, spec):
    with gzip.open(spec["eval"], "rt") as stream:
        episodes = [json.loads(line) for line in stream if line.strip()]
    stats = archived_stats(spec["noncot"])
    texts = _texts(spec["noncot"])
    compact = G.scan_compact_episodes(str(spec["eval"]), None, None)
    args = SimpleNamespace(profile="maze2d_native256" if task == "maze" else "pusht_norm4_native",
                           sim2real=False, action_preset=None, action_name_map=None)
    results = []
    for index, (row, published) in enumerate(zip(episodes, stats)):
        ep = compact[index]
        seed = int(published["seed"])
        episode_seed = int(published["episode_seed"])
        base.set_global_seed(seed)
        env, _ = G.make_profiled_env(ep, args)
        try:
            _reset_preserving_init_state(env, seed=episode_seed, init_state=ep["init_state"])
            invalid = 0
            terminated = truncated = False
            for action in published["history"]:
                before = _physical(env, task) if not G._is_stop_action(action["action"]) else None
                _, _, terminated, truncated, _ = env.step(action["action"])
                if before is not None and before == _physical(env, task):
                    invalid += 1
                if bool(terminated) != bool(action["terminated"]) or bool(truncated) != bool(action["truncated"]):
                    raise AssertionError((task, index, "simulator transition mismatch"))
            success = bool(published["stats"]["success"])
            final_action = published["stats"].get("final_action")
            parts = texts[index].split("----- CHUNK -----")
            parse_failure = any(not parse_action_chunk(part) for part in parts)
            if success:
                reason = "solved"
            elif parse_failure:
                reason = "parse_failure"
            elif truncated or len(published["history"]) >= spec["budget"]:
                reason = "budget_exhausted"
            else:
                reason = "other"
            results.append({"episode_id": f"{task}:{index:07d}:{episode_seed}",
                            "task": task, "condition": "No-Imagine", "success": success,
                            "termination_reason": reason, "num_executed_actions": len(published["history"]),
                            "num_chunks": published["n_chunks"], "parse_failure": parse_failure,
                            "num_invalid_actions": invalid, "stop_emitted": G._is_stop_action(final_action),
                            "truncated": False, "committed_matches_imagined": None,
                            "global_index": index, "seed": seed, "episode_seed": episode_seed,
                            "source": "staged_k5_noncot_action_replay"})
        finally:
            env.close()
    return results


if __name__ == "__main__":
    path = ROOT / "prepared" / "maze_pusht_baseline_metrics.jsonl"
    results = [row for task, spec in CONFIG.items() for row in evaluate(task, spec)]
    with path.open("w") as stream:
        for row in results:
            stream.write(json.dumps(row, separators=(",", ":")) + "\n")
    print(f"Wrote {len(results)} paired No-Imagine rows to {path}")
