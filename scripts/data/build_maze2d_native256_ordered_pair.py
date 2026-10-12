#!/usr/bin/env python3
# Modified for Agima: portable paths and explicit dataset arguments.
"""Build aligned native-256 Maze2D plain and CoT SFT datasets."""

from __future__ import annotations

import argparse
import base64
import gzip
import hashlib
import io
import json
import multiprocessing as mp
import os
import random
import shutil
import subprocess
import sys
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any, BinaryIO

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
from PIL import Image

from gymnasium.envs.maze_2d.maze_2d import Maze2DEnv

try:
    import orjson
except Exception:  # pragma: no cover
    orjson = None


ORDER_KEY = "__bagel_order_key"
MANIFEST_VERSION = "maze2d_native256_stopreq_stepmsg_randstartgoal_plain_cot_ordered100k_v1"
POLICY_VERSION = "maze2d_native256_stopreq_stepmsg_randstartgoal_cot_lastbranch_v1"
MAX_STEPS = 40

ACTION_TO_DELTA = {
    0: (0, 1),   # right
    1: (-1, 0),  # up
    2: (0, -1),  # left
    3: (1, 0),   # down
}
SPECS = {
    "easy": {
        "env_id": "maze_2d/easy",
        "maze_width": 11,
        "maze_height": 11,
        "target_in_center": False,
        "render_size": 256,
        "train_seed_start": 1_100_000,
        "test_seed_start": 9_100_000,
    },
    "hard": {
        "env_id": "maze_2d/hard",
        "maze_width": 15,
        "maze_height": 15,
        "target_in_center": False,
        "render_size": 256,
        "train_seed_start": 2_100_000,
        "test_seed_start": 9_200_000,
    },
}


class PigzWriter:
    def __init__(self, path: Path, threads: int, compresslevel: int):
        self.path = path
        self.threads = max(1, threads)
        self.compresslevel = compresslevel
        self.proc: subprocess.Popen[bytes] | None = None
        self.handle: BinaryIO | None = None

    def __enter__(self) -> BinaryIO:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.path.open("wb")
        self.proc = subprocess.Popen(
            ["pigz", f"-{self.compresslevel}", "-p", str(self.threads), "-c"],
            stdin=subprocess.PIPE,
            stdout=self.handle,
        )
        assert self.proc.stdin is not None
        return self.proc.stdin

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        assert self.proc is not None and self.proc.stdin is not None and self.handle is not None
        self.proc.stdin.close()
        rc = self.proc.wait()
        self.handle.close()
        if rc != 0 and exc is None:
            raise RuntimeError(f"pigz failed with exit code {rc}")


def gzip_writer(path: Path, threads: int, compresslevel: int):
    if shutil.which("pigz"):
        return PigzWriter(path, threads=threads, compresslevel=compresslevel)
    path.parent.mkdir(parents=True, exist_ok=True)
    return gzip.open(path, "wb", compresslevel=compresslevel)


def dumps_record(record: dict[str, Any]) -> bytes:
    if orjson is not None:
        return orjson.dumps(record, option=orjson.OPT_APPEND_NEWLINE)
    return (json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")


def stable_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def canonical_hash(spec: dict[str, Any], init_state: dict[str, Any]) -> str:
    return stable_hash(
        {
            "env_id": spec["env_id"],
            "maze_width": spec["maze_width"],
            "maze_height": spec["maze_height"],
            "target_in_center": spec["target_in_center"],
            "maze_map": init_state["maze_map"],
            "agent_location": init_state["agent_location"],
            "target_location": init_state["target_location"],
        }
    )


def make_env(spec: dict[str, Any], seed: int | None = None) -> Maze2DEnv:
    return Maze2DEnv(
        maze_width=spec["maze_width"],
        maze_height=spec["maze_height"],
        target_in_center=spec["target_in_center"],
        render_size=spec["render_size"],
        seed=seed,
    )


def encode_jpeg(array: np.ndarray, quality: int) -> str:
    image = Image.fromarray(array.astype(np.uint8))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def simulate_move(
    maze_map: np.ndarray,
    location: tuple[int, int],
    action: int,
) -> tuple[tuple[int, int], bool]:
    dr, dc = ACTION_TO_DELTA[action]
    nxt = (location[0] + dr, location[1] + dc)
    rows, cols = maze_map.shape
    if not (0 <= nxt[0] < rows and 0 <= nxt[1] < cols):
        return location, True
    if int(maze_map[nxt[0], nxt[1]]) == 1:
        return location, True
    return nxt, False


def shortest_path(
    maze_map: np.ndarray,
    start: tuple[int, int],
    target: tuple[int, int],
) -> list[tuple[int, int]]:
    from collections import deque

    queue: deque[tuple[tuple[int, int], list[tuple[int, int]]]] = deque([(start, [start])])
    visited = {start}
    while queue:
        current, path = queue.popleft()
        if current == target:
            return path
        for action_id in (0, 1, 2, 3):
            nxt, blocked = simulate_move(maze_map, current, action_id)
            if blocked or nxt in visited:
                continue
            visited.add(nxt)
            queue.append((nxt, path + [nxt]))
    return []


def bfs_distance(
    maze_map: np.ndarray,
    start: tuple[int, int],
    target: tuple[int, int],
) -> int | None:
    path = shortest_path(maze_map, start, target)
    if not path:
        return None
    return len(path) - 1


def actions_from_path(path: list[tuple[int, int]]) -> list[str]:
    delta_to_action = {delta: action_id for action_id, delta in ACTION_TO_DELTA.items()}
    actions: list[str] = []
    for current, nxt in zip(path, path[1:]):
        delta = (nxt[0] - current[0], nxt[1] - current[1])
        actions.append(f"('move', {delta_to_action[delta]})")
    actions.append("('stop', 'stop')")
    return actions


def action_id(action: str) -> int:
    return int(action.split(",")[1].split(")")[0].strip())


def deterministic_candidate_order(init_hash: str, step_idx: int) -> list[int]:
    seed_material = f"{init_hash}{step_idx}{POLICY_VERSION}"
    digest = hashlib.sha256(seed_material.encode("utf-8")).hexdigest()
    rng = random.Random(int(digest[:16], 16))
    order = [0, 1, 2, 3]
    rng.shuffle(order)
    return order


def branch_stats(
    maze_map: np.ndarray,
    location: tuple[int, int],
    target: tuple[int, int],
) -> list[dict[str, Any]]:
    stats = []
    for action in (0, 1, 2, 3):
        nxt, blocked = simulate_move(maze_map, location, action)
        stats.append(
            {
                "action": action,
                "action_text": f"('move', {action})",
                "cell": list(nxt),
                "blocked": blocked,
                "distance": None if blocked else bfs_distance(maze_map, nxt, target),
            }
        )
    return stats


def select_key_step(
    maze_map: np.ndarray,
    path: list[tuple[int, int]],
    actions: list[str],
    target: tuple[int, int],
    init_hash: str,
) -> dict[str, Any] | None:
    # KEY-STEP RULE (lastbranch v3): the key step is the LAST decisive fork BEFORE the goal
    # -- the highest-index move step with >=2 valid moves where the oracle strictly beats the
    # best alternative (margin > 0). Same candidate dict shape as before; only WHICH step is
    # chosen changes (was argmax-margin, which front-loaded to step 0/1 on this fixed-start
    # corner layout -> ~88%). Every row accepted under v2 had >=1 such fork, so this never
    # drops a row (remap-safe / preserves the retained set + __bagel_order_key alignment).
    last: dict[str, Any] | None = None
    for step_idx, action_text in enumerate(actions[:-1]):
        oracle_action = action_id(action_text)
        current = path[step_idx]
        stats = branch_stats(maze_map, current, target)
        valid = [item for item in stats if not item["blocked"] and item["distance"] is not None]
        if len(valid) < 2:
            continue
        oracle = next(item for item in valid if item["action"] == oracle_action)
        alternatives = [item for item in valid if item["action"] != oracle_action]
        if not alternatives:
            continue
        best_alt_distance = min(int(item["distance"]) for item in alternatives)
        margin = best_alt_distance - int(oracle["distance"])
        if margin <= 0:
            continue
        order = deterministic_candidate_order(init_hash, step_idx)
        chosen_label = "ABCD"[order.index(oracle_action)]
        candidate_map = {
            "ABCD"[idx]: next(item for item in stats if item["action"] == action)
            for idx, action in enumerate(order)
        }
        candidate = {
            "step_idx": step_idx,
            "oracle_action": oracle_action,
            "oracle_action_text": action_text,
            "margin": margin,
            "candidate_order": order,
            "chosen_label": chosen_label,
            "candidates": candidate_map,
        }
        last = candidate  # keep overwriting -> ends as the LAST qualifying (decisive fork) step
    return last


def generate_candidate(task: tuple[str, str, int]) -> dict[str, Any] | None:
    difficulty, split, seed = task
    spec = SPECS[difficulty]
    env = make_env(spec, seed=seed)
    env.reset(seed=seed)
    init_state = env.get_init_state()
    maze_map = np.array(init_state["maze_map"], dtype=int)
    start = tuple(int(x) for x in init_state["agent_location"])
    target = tuple(int(x) for x in init_state["target_location"])
    path = shortest_path(maze_map, start, target)
    if len(path) < 3:
        return None
    actions = actions_from_path(path)
    if len(actions) > MAX_STEPS:
        return None
    init_hash = canonical_hash(spec, init_state)
    key_step = select_key_step(maze_map, path, actions, target, init_hash)
    if key_step is None:
        return None
    return {
        "manifest_version": MANIFEST_VERSION,
        "policy_version": POLICY_VERSION,
        "difficulty": difficulty,
        "split": split,
        "seed": seed,
        "env_id": spec["env_id"],
        "maze_width": spec["maze_width"],
        "maze_height": spec["maze_height"],
        "target_in_center": spec["target_in_center"],
        "render_size": spec["render_size"],
        "hash": init_hash,
        "init_state": init_state,
        "path": [list(cell) for cell in path],
        "actions": actions,
        "key_step": key_step,
    }


def collect_accepted(
    *,
    difficulty: str,
    split: str,
    seed_start: int,
    surplus: int,
    workers: int,
    chunksize: int,
) -> list[dict[str, Any]]:
    accepted: list[dict[str, Any]] = []
    next_seed = seed_start
    batch_size = max(workers * chunksize * 20, 2000)
    print(
        f"[{difficulty}/{split}] collecting {surplus} accepted candidates from seed_start={seed_start}",
        flush=True,
    )

    def tasks(seed0: int) -> Iterable[tuple[str, str, int]]:
        for seed in range(seed0, seed0 + batch_size):
            yield (difficulty, split, seed)

    if workers <= 1:
        while len(accepted) < surplus:
            for task in tasks(next_seed):
                candidate = generate_candidate(task)
                if candidate is not None:
                    candidate["accepted_index"] = len(accepted)
                    accepted.append(candidate)
                    if len(accepted) % 1000 == 0 or len(accepted) == surplus:
                        print(f"[{difficulty}/{split}] accepted={len(accepted)} seed={task[2]}", flush=True)
                    if len(accepted) >= surplus:
                        break
            next_seed += batch_size
        return accepted

    with mp.Pool(processes=workers) as pool:
        while len(accepted) < surplus:
            for candidate in pool.imap(generate_candidate, tasks(next_seed), chunksize=chunksize):
                if candidate is not None:
                    candidate["accepted_index"] = len(accepted)
                    accepted.append(candidate)
                    if len(accepted) % 1000 == 0 or len(accepted) == surplus:
                        print(
                            f"[{difficulty}/{split}] accepted={len(accepted)} seed={candidate['seed']}",
                            flush=True,
                        )
                    if len(accepted) >= surplus:
                        break
            next_seed += batch_size
    return accepted


def retain_unique(
    accepted: list[dict[str, Any]],
    *,
    keep: int,
    excluded_hashes: set[str],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    seen: set[str] = set()
    retained: list[dict[str, Any]] = []
    stats = {"accepted": len(accepted), "internal_duplicates": 0, "excluded_overlaps": 0}
    for candidate in accepted:
        init_hash = str(candidate["hash"])
        if init_hash in seen:
            stats["internal_duplicates"] += 1
            continue
        seen.add(init_hash)
        if init_hash in excluded_hashes:
            stats["excluded_overlaps"] += 1
            continue
        candidate = dict(candidate)
        candidate[ORDER_KEY] = f"{len(retained):012d}"
        candidate["global_index"] = len(retained)
        retained.append(candidate)
        if len(retained) == keep:
            break
    stats["retained"] = len(retained)
    if len(retained) != keep:
        raise RuntimeError(f"retained {len(retained)} unique rows, expected {keep}; stats={stats}")
    return retained, stats


def row_prompt(env: Maze2DEnv, step_idx: int, max_steps: int = MAX_STEPS) -> str:
    if step_idx == 0:
        return env.get_prompt()
    remaining = max(0, max_steps - step_idx)
    return f"This is step {step_idx}. You are allowed to take {remaining} more steps."


def trivial_cot(action: str) -> str:
    return (
        "<think>This step is straightforward, so I can skip extra rollout.</think>\n"
        f"{action}"
    )


def stop_cot() -> str:
    return (
        "<think>The blue circle is on the red target square. Stop is required for success, "
        "so I should end the episode now.</think>\n"
        "('stop', 'stop')"
    )


def render_location_b64(
    spec: dict[str, Any],
    maze_map: np.ndarray,
    agent_location: tuple[int, int],
    target_location: tuple[int, int],
    quality: int,
) -> str:
    env = make_env(spec)
    env.reset(
        init_state={
            "maze_map": maze_map.tolist(),
            "agent_location": list(agent_location),
            "target_location": list(target_location),
        }
    )
    return encode_jpeg(env.render(), quality)


def key_cot_content(
    manifest_entry: dict[str, Any],
    spec: dict[str, Any],
    maze_map: np.ndarray,
    target: tuple[int, int],
    quality: int,
) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = [
        {
            "type": "text",
            # F5 fix: open the key-step <think> here (the closing </think> is in final_text below),
            # so the emitted key-step assistant turn is balanced <think>...</think> like the
            # trivial_cot / stop_cot turns. Previously the opener was missing (only the storyboard
            # model_text added it), leaving a dangling </think> in every CoT row.
            "text": (
                "<think>This state has several plausible directions, so I should look one step deeper "
                "before choosing. I will compare the possible branches from the current cell.\n\n"
            ),
        }
    ]
    key = manifest_entry["key_step"]
    for label in "ABCD":
        candidate = key["candidates"][label]
        if candidate["blocked"]:
            text = (
                f"Candidate {label}: `{candidate['action_text']}` is blocked by a wall, "
                "so the state does not advance.\n"
            )
        else:
            text = (
                f"Candidate {label}: `{candidate['action_text']}` reaches cell "
                f"({candidate['cell'][0]}, {candidate['cell'][1]}) with remaining "
                f"shortest-path distance {candidate['distance']}.\n"
            )
        content.append({"type": "text", "text": text})
        content.append(
            {
                "type": "image",
                "image_b64": render_location_b64(
                    spec,
                    maze_map,
                    tuple(candidate["cell"]),
                    target,
                    quality,
                ),
                "loss": True,
            }
        )
    final_text = (
        f"\nCandidate {key['chosen_label']} has the best valid remaining distance, "
        f"so I will take `{key['oracle_action_text']}`.</think>\n"
        f"{key['oracle_action_text']}"
    )
    content.append({"type": "text", "text": final_text})
    return content


def build_pair_payload(task: tuple[dict[str, Any], int]) -> tuple[bytes, bytes]:
    entry, quality = task
    spec = {
        "env_id": entry["env_id"],
        "maze_width": entry["maze_width"],
        "maze_height": entry["maze_height"],
        "target_in_center": entry["target_in_center"],
        "render_size": entry["render_size"],
    }
    env = make_env(spec)
    env.reset(init_state=entry["init_state"])
    maze_map = np.array(entry["init_state"]["maze_map"], dtype=int)
    target = tuple(int(x) for x in entry["init_state"]["target_location"])

    history: list[dict[str, Any]] = []
    messages: list[dict[str, Any]] = []
    for step_idx, action in enumerate(entry["actions"]):
        prompt = row_prompt(env, step_idx)
        current_b64 = encode_jpeg(env.render(), quality)
        messages.append(
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image", "image_b64": current_b64, "loss": False},
                ],
            }
        )
        if step_idx == entry["key_step"]["step_idx"]:
            assistant_content = key_cot_content(entry, spec, maze_map, target, quality)
        elif action == "('stop', 'stop')":
            assistant_content = [{"type": "text", "text": stop_cot()}]
        else:
            assistant_content = [{"type": "text", "text": trivial_cot(action)}]
        messages.append({"role": "assistant", "content": assistant_content})

        obs, _, terminated, _, _ = env.step(action)
        next_b64 = encode_jpeg(obs, quality)
        history.append(
            {
                "step_idx": step_idx,
                "prompt": prompt,
                "image": current_b64,
                "image_prev": current_b64,
                "image_next": next_b64,
                "vlm_output": action,
                "action": action,
                "terminated": terminated,
            }
        )

    common = {
        ORDER_KEY: entry[ORDER_KEY],
        "task": f"maze2d_{entry['difficulty']}",
        "env_id": entry["env_id"],
        "hash": entry["hash"],
        "split": entry["split"],
        "alignment_index": {
            "manifest_version": MANIFEST_VERSION,
            "global_index": entry["global_index"],
            "seed": entry["seed"],
            "init_state_hash": entry["hash"],
        },
        "metadata": {
            "policy_version": POLICY_VERSION,
            "prompt_contract": "step0_full_stepgt0_step_count_only_v2",
            "max_steps": MAX_STEPS,
            "difficulty": entry["difficulty"],
            "seed": entry["seed"],
            "maze_width": entry["maze_width"],
            "maze_height": entry["maze_height"],
            "target_in_center": entry["target_in_center"],
            "render_size": entry["render_size"],
            "init_state_hash": entry["hash"],
            "key_step": entry["key_step"],
        },
        "init_state": entry["init_state"],
        "trajectory_steps": [
            {"step_idx": step_idx, "action": action}
            for step_idx, action in enumerate(entry["actions"])
        ],
    }
    plain_row = {
        **common,
        "schema": "maze2d_native256_stopreq_stepmsg_randstartgoal_plain_v1",
        "history": history,
    }
    cot_row = {
        **common,
        "schema": "maze2d_native256_stopreq_stepmsg_randstartgoal_cot_lastbranch_v1",
        "messages": messages,
        "interleaved_cot": {
            "policy_version": POLICY_VERSION,
            "selected_step": entry["key_step"]["step_idx"],
            "chosen_label": entry["key_step"]["chosen_label"],
            "candidate_order": entry["key_step"]["candidate_order"],
            "generated_image_count": 4,
        },
    }
    return dumps_record(plain_row), dumps_record(cot_row)


def shard_slices(total: int, shards: int) -> list[tuple[int, int]]:
    rows_per_shard = (total + shards - 1) // shards
    return [
        (start, min(start + rows_per_shard, total))
        for start in range(0, total, rows_per_shard)
    ]


def write_manifest(path: Path, manifest: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for entry in manifest:
            handle.write(json.dumps(entry, ensure_ascii=False, separators=(",", ":")) + "\n")


def write_split_rows(
    *,
    manifest: list[dict[str, Any]],
    plain_dir: Path,
    cot_dir: Path,
    split: str,
    shards: int,
    workers: int,
    chunksize: int,
    gzip_threads: int,
    compresslevel: int,
    jpeg_quality: int,
) -> None:
    plain_dir.mkdir(parents=True, exist_ok=True)
    cot_dir.mkdir(parents=True, exist_ok=True)
    slices = shard_slices(len(manifest), shards)
    for shard_idx, (start, end) in enumerate(slices):
        shard_entries = manifest[start:end]
        if split == "train":
            name = f"shard_{shard_idx:05d}.jsonl.gz"
        else:
            name = "shard_00000.jsonl.gz"
        plain_path = plain_dir / name
        cot_path = cot_dir / name
        print(f"[{manifest[0]['difficulty']}/{split}] writing {name}: rows {start}..{end - 1}", flush=True)
        tasks = ((entry, jpeg_quality) for entry in shard_entries)
        with gzip_writer(plain_path, gzip_threads, compresslevel) as plain_handle, gzip_writer(
            cot_path, gzip_threads, compresslevel
        ) as cot_handle:
            if workers <= 1:
                iterator = map(build_pair_payload, tasks)
                for plain_payload, cot_payload in iterator:
                    plain_handle.write(plain_payload)
                    cot_handle.write(cot_payload)
            else:
                with mp.Pool(processes=workers) as pool:
                    for plain_payload, cot_payload in pool.imap(build_pair_payload, tasks, chunksize=chunksize):
                        plain_handle.write(plain_payload)
                        cot_handle.write(cot_payload)


def symlink_latest(target: Path, link: Path) -> None:
    tmp = link.with_name(link.name + ".tmp")
    if tmp.exists() or tmp.is_symlink():
        tmp.unlink()
    if link.exists() and not link.is_symlink():
        raise RuntimeError(f"refusing to replace non-symlink path: {link}")
    tmp.symlink_to(target, target_is_directory=True)
    tmp.replace(link)


def build_difficulty(args: argparse.Namespace, difficulty: str) -> dict[str, Any]:
    spec = SPECS[difficulty]
    output_base = args.output_base.resolve()
    plain_name = f"maze2d_{difficulty}_native256_stopreq_stepmsg_randstartgoal_ordered"
    cot_name = f"maze2d_{difficulty}_native256_stopreq_stepmsg_randstartgoal_cot_ordered"
    plain_root = output_base / f"{plain_name}_{args.date_tag}"
    cot_root = output_base / f"{cot_name}_{args.date_tag}"
    for root in (plain_root, cot_root):
        if root.exists() and any(root.iterdir()) and not args.overwrite:
            raise SystemExit(f"output root exists and is non-empty: {root}")
        (root / "metadata").mkdir(parents=True, exist_ok=True)
        (root / "training").mkdir(parents=True, exist_ok=True)
        (root / "testing").mkdir(parents=True, exist_ok=True)

    split_plan = {
        "train": {
            "surplus": args.train_surplus,
            "keep": args.train_keep,
            "seed_start": args.train_seed_start or spec["train_seed_start"],
            "shards": args.train_shards,
        },
        "test": {
            "surplus": args.test_surplus,
            "keep": args.test_keep,
            "seed_start": args.test_seed_start or spec["test_seed_start"],
            "shards": 1,
        },
    }

    summary: dict[str, Any] = {
        "ok": False,
        "manifest_version": MANIFEST_VERSION,
        "policy_version": POLICY_VERSION,
        "prompt_contract": "step0_full_stepgt0_step_count_only_v2",
        "max_steps": MAX_STEPS,
        "difficulty": difficulty,
        "spec": {k: v for k, v in spec.items() if k.endswith("size") or k in {"env_id", "maze_width", "maze_height", "target_in_center", "render_size"}},
        "plain_root": str(plain_root),
        "cot_root": str(cot_root),
        "splits": {},
    }
    retained_by_split: dict[str, list[dict[str, Any]]] = {}
    train_hashes: set[str] = set()
    for split in ("train", "test"):
        plan = split_plan[split]
        accepted = collect_accepted(
            difficulty=difficulty,
            split=split,
            seed_start=plan["seed_start"],
            surplus=plan["surplus"],
            workers=args.workers,
            chunksize=args.chunksize,
        )
        retained, stats = retain_unique(
            accepted,
            keep=plan["keep"],
            excluded_hashes=train_hashes if split == "test" else set(),
        )
        retained_by_split[split] = retained
        if split == "train":
            train_hashes = {entry["hash"] for entry in retained}
        manifest_name = f"{split}_manifest.jsonl"
        for root in (plain_root, cot_root):
            write_manifest(root / "metadata" / manifest_name, retained)
        summary["splits"][split] = {
            "seed_start": plan["seed_start"],
            "surplus": plan["surplus"],
            "keep": plan["keep"],
            "stats": stats,
            "manifest": str(plain_root / "metadata" / manifest_name),
        }

    for split in ("train", "test"):
        plan = split_plan[split]
        write_split_rows(
            manifest=retained_by_split[split],
            plain_dir=plain_root / ("training" if split == "train" else "testing"),
            cot_dir=cot_root / ("training" if split == "train" else "testing"),
            split=split,
            shards=plan["shards"],
            workers=args.workers,
            chunksize=args.chunksize,
            gzip_threads=args.gzip_threads,
            compresslevel=args.compresslevel,
            jpeg_quality=args.jpeg_quality,
        )

    readme = (
        f"# {plain_name} / {cot_name}\n\n"
        f"Built by `{Path(__file__).name}` at `{args.date_tag}`.\n"
        f"Manifest version: `{MANIFEST_VERSION}`. CoT policy: `{POLICY_VERSION}`.\n"
        "Plain and CoT rows share the same retained episode manifests and order for each split.\n"
    )
    (plain_root / "README.md").write_text(readme, encoding="utf-8")
    (cot_root / "README.md").write_text(readme, encoding="utf-8")
    summary["ok"] = True
    for root in (plain_root, cot_root):
        (root / "metadata" / "build_summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    if args.update_latest:
        symlink_latest(plain_root, output_base / plain_name)
        symlink_latest(cot_root, output_base / cot_name)
        symlink_latest(plain_root, output_base / f"{plain_name}_latest")
        symlink_latest(cot_root, output_base / f"{cot_name}_latest")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--difficulty", choices=("easy", "hard", "both"), default="both")
    parser.add_argument("--output-base", type=Path, required=True)
    parser.add_argument("--date-tag", default=time.strftime("%Y%m%d_%H%M%S", time.gmtime()))
    parser.add_argument("--train-surplus", type=int, default=120000)
    parser.add_argument("--test-surplus", type=int, default=120)
    parser.add_argument("--train-keep", type=int, default=100000)
    parser.add_argument("--test-keep", type=int, default=100)
    parser.add_argument("--train-shards", type=int, default=8)
    parser.add_argument("--train-seed-start", type=int)
    parser.add_argument("--test-seed-start", type=int)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 1) - 2))
    parser.add_argument("--chunksize", type=int, default=16)
    parser.add_argument("--gzip-threads", type=int, default=8)
    parser.add_argument("--compresslevel", type=int, default=1)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--update-latest", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_base.mkdir(parents=True, exist_ok=True)
    difficulties = ("easy", "hard") if args.difficulty == "both" else (args.difficulty,)
    summary = {
        "ok": True,
        "date_tag": args.date_tag,
        "difficulties": {},
    }
    for difficulty in difficulties:
        summary["difficulties"][difficulty] = build_difficulty(args, difficulty)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
