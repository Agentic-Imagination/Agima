#!/usr/bin/env python3
"""4-box sokoban_hard forward-planning level scanner (heuristic — Exp3 extension).

Companion to `sokoban_forward_level_gen.py` (2-box sokoban_easy scanner with full BFS
certification). This one targets `sokoban/hard` (num_boxes=4) with a much cheaper
heuristic — corner-deadlock after first push — because full 4-box unsolvability
certification (BFS over (player, box_1..4) states) is exponential and impractical.

Qualifying property (heuristic): among the 4 possible "first-box" candidates, at least
one first-push lands its box on an immediate corner-deadlock (2 adjacent walls, not on
target) AND at least one first-push does NOT — so a forward-thinking model that
imagined both orders would visibly reject one and prefer another.

Weaker than 2-box "provably unsolvable"; that's the intended trade-off given the
smoke-only scope. Downstream 4-box hard curated-eval pool consumes the same row schema
as the 2-box path so eval scripts work unchanged.

Row schema (mirrors sokoban_forward_level_gen 2-box output where semantics carry over;
non-applicable fields dropped / renamed):
    {source, seed, env_id, env_kwargs, init_state, init_state_schema, init_state_hash,
     init_frame_b64, near_box, far_box,
     n_deadlock_first_pushes_heuristic,   # (2-box's n_dead_states_certified counterpart)
     checks: {near_first_push_deadlock: True, far_first_push_deadlock: False,
              first_push_dir_near: (dr,dc), first_push_dir_far: (dr,dc)}}

Smoke usage (~5000 seeds → aim ≥ 10 qualifiers):
    python scripts/data/sokoban_forward_level_gen_4box.py \
        --n-seeds 5000 --target-qualifiers 20 --workers 8 \
        --out /tmp/fwd4box_smoke
"""
from __future__ import annotations

import argparse, base64, glob, gzip, hashlib, io, json, os, sys, time
from collections import deque
from multiprocessing import Pool
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import gymnasium
from gymnasium.envs.sokoban.sokoban import FLOOR, TARGET, WALL  # 1, 2, 0

ENV_ID = "sokoban/hard"
ENV_KWARGS: Dict[str, Any] = {"max_steps": 120, "action_mode": "move_only"}

# 4 push directions on the (row, col) grid — same convention as the 2-box scanner.
DIRS: Tuple[Tuple[int, int], ...] = ((-1, 0), (1, 0), (0, -1), (0, 1))

_ENV_U: Any = None  # set per worker


def _init_worker() -> None:
    global _ENV_U
    env = gymnasium.make(ENV_ID)
    _ENV_U = env.unwrapped


def stable_hash(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def encode_frame_b64(env_u: Any) -> str:
    arr = env_u.render()
    if arr is None:
        return ""
    if arr.ndim == 3 and arr.shape[-1] == 4:
        arr = arr[..., :3]
    im = Image.fromarray(np.asarray(arr).astype(np.uint8))
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=95)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def is_corner_deadlock(room_fixed: np.ndarray, box_pos: Tuple[int, int]) -> bool:
    """Classic sokoban deadlock: box on a non-target cell with 2 adjacent walls forming a corner."""
    r, c = box_pos
    if room_fixed[r, c] == TARGET:
        return False  # box on a target is not a deadlock even if in a corner
    up_wall    = room_fixed[r - 1, c] == WALL if r > 0 else True
    down_wall  = room_fixed[r + 1, c] == WALL if r < room_fixed.shape[0] - 1 else True
    left_wall  = room_fixed[r, c - 1] == WALL if c > 0 else True
    right_wall = room_fixed[r, c + 1] == WALL if c < room_fixed.shape[1] - 1 else True
    # any orthogonal corner = 2 adjacent walls at right angle
    return (up_wall and left_wall) or (up_wall and right_wall) or \
           (down_wall and left_wall) or (down_wall and right_wall)


def player_can_reach(room_fixed: np.ndarray, box_positions: List[Tuple[int, int]],
                     player: Tuple[int, int], goal: Tuple[int, int]) -> bool:
    """BFS on the free floor (walls + boxes are obstacles) — can player walk from player→goal?"""
    box_set = {tuple(b) for b in box_positions}
    if goal in box_set or room_fixed[goal] == WALL:
        return False
    if player == goal:
        return True
    seen = {player}
    q = deque([player])
    while q:
        r, c = q.popleft()
        for dr, dc in DIRS:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < room_fixed.shape[0] and 0 <= nc < room_fixed.shape[1]):
                continue
            if (nr, nc) in seen or room_fixed[nr, nc] == WALL or (nr, nc) in box_set:
                continue
            if (nr, nc) == goal:
                return True
            seen.add((nr, nc))
            q.append((nr, nc))
    return False


def simulate_first_push(room_fixed: np.ndarray, box_positions: List[Tuple[int, int]],
                        player: Tuple[int, int], box_idx: int,
                        push_dir: Tuple[int, int]) -> Optional[Dict[str, Any]]:
    """Try pushing box_idx in push_dir: player must be able to walk to the cell behind the box,
    box target cell must be a walkable floor. Returns new (box_positions, player) or None if infeasible."""
    box = tuple(box_positions[box_idx])
    dr, dc = push_dir
    dest = (box[0] + dr, box[1] + dc)
    stand = (box[0] - dr, box[1] - dc)   # player stands on the opposite side to push
    H, W = room_fixed.shape
    if not (0 <= dest[0] < H and 0 <= dest[1] < W):
        return None
    if not (0 <= stand[0] < H and 0 <= stand[1] < W):
        return None
    if room_fixed[dest] == WALL:
        return None
    if room_fixed[stand] == WALL:
        return None
    # dest can't have another box; stand can't have another box either
    others = {tuple(b) for i, b in enumerate(box_positions) if i != box_idx}
    if dest in others or stand in others:
        return None
    # player must reach 'stand' without crossing walls or boxes (this box treated as obstacle)
    if not player_can_reach(room_fixed, box_positions, player, stand):
        return None
    # simulate
    new_boxes = list(box_positions)
    new_boxes[box_idx] = list(dest)
    return {"boxes": new_boxes, "player": list(box), "box_new": list(dest), "push_dir": [dr, dc]}


def scan_seed(seed: int) -> Optional[Dict[str, Any]]:
    global _ENV_U
    env_u = _ENV_U
    env_u.reset(seed=seed)
    init = env_u.get_init_state()
    boxes = init.get("box_positions") or []
    if len(boxes) != 4:
        return {"seed": seed, "drop": "wrong_num_boxes"}
    room_fixed = np.array(init["room_fixed"], dtype=int)
    player = tuple(init["player_position"])

    # Enumerate 4 boxes × 4 dirs = 16 candidate first-pushes; keep the *feasible* ones (~one per box).
    per_box_first_pushes: Dict[int, List[Dict[str, Any]]] = {}
    for bi, b in enumerate(boxes):
        for d in DIRS:
            r = simulate_first_push(room_fixed, [tuple(x) for x in boxes], player, bi, d)
            if r is not None:
                per_box_first_pushes.setdefault(bi, []).append(r)

    if len(per_box_first_pushes) < 2:
        return {"seed": seed, "drop": "fewer_than_2_pushable_boxes"}

    # For each box that has ANY feasible first push, classify as "deadlock-first" iff
    # *every* feasible first-push of that box lands on a corner-deadlock (non-target).
    box_verdicts: Dict[int, bool] = {}   # box_idx -> True means "first-push of this box is a trap"
    for bi, pushes in per_box_first_pushes.items():
        all_dead = all(is_corner_deadlock(room_fixed, tuple(p["box_new"])) for p in pushes)
        box_verdicts[bi] = all_dead

    near_boxes = [bi for bi, dead in box_verdicts.items() if dead]
    far_boxes  = [bi for bi, dead in box_verdicts.items() if not dead]
    if not near_boxes or not far_boxes:
        return {"seed": seed, "drop": "no_split"}

    # Pick the first near and first far (deterministic under seed).
    near_bi = near_boxes[0]
    far_bi  = far_boxes[0]
    near_push = per_box_first_pushes[near_bi][0]
    far_push  = per_box_first_pushes[far_bi][0]

    # emit row
    frame_b64 = encode_frame_b64(env_u)
    row = {
        "source": "generated_4box",
        "seed": int(seed),
        "env_id": ENV_ID,
        "env_kwargs": dict(ENV_KWARGS),
        "init_state": init,
        "init_state_schema": "sokoban_hard_v8_4box_forward_v1",
        "init_state_hash": stable_hash({"room_fixed": init["room_fixed"], "player": list(player),
                                        "boxes": sorted(map(list, boxes))}),
        "init_frame_b64": frame_b64,
        "near_box": list(boxes[near_bi]),
        "far_box": list(boxes[far_bi]),
        "n_deadlock_first_pushes_heuristic": sum(1 for v in box_verdicts.values() if v),
        "checks": {
            "near_first_push_deadlock": True,
            "far_first_push_deadlock": False,
            "first_push_dir_near": near_push["push_dir"],
            "first_push_dir_far": far_push["push_dir"],
            "near_box_new_pos": near_push["box_new"],
            "far_box_new_pos": far_push["box_new"],
            "certifier": "corner_deadlock_heuristic_v1",
        },
    }
    return row


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-seeds", type=int, default=5000)
    ap.add_argument("--base-seed", type=int, default=40_000_000)
    ap.add_argument("--target-qualifiers", type=int, default=20)
    ap.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    os.makedirs(f"{args.out}/levels", exist_ok=True)
    seeds = [args.base_seed + i for i in range(args.n_seeds)]

    t0 = time.time()
    qualifiers: List[Dict[str, Any]] = []
    drop_reasons: Dict[str, int] = {}
    with Pool(args.workers, initializer=_init_worker) as pool:
        for res in pool.imap_unordered(scan_seed, seeds, chunksize=8):
            if res is None:
                continue
            if "drop" in res:
                drop_reasons[res["drop"]] = drop_reasons.get(res["drop"], 0) + 1
                continue
            qualifiers.append(res)
            if len(qualifiers) >= args.target_qualifiers:
                pool.terminate()
                break

    dt = time.time() - t0
    out_path = f"{args.out}/levels/shard_00000.jsonl.gz"
    with gzip.open(out_path, "wt") as f:
        for row in qualifiers:
            f.write(json.dumps(row) + "\n")

    summary = {
        "generator": "scripts/data/sokoban_forward_level_gen_4box.py",
        "env_id": ENV_ID,
        "env_kwargs": ENV_KWARGS,
        "seeds_tried": len(seeds),
        "qualifiers_written": len(qualifiers),
        "target_qualifiers": args.target_qualifiers,
        "drop_reasons": drop_reasons,
        "elapsed_sec": round(dt, 2),
        "certifier": "corner_deadlock_heuristic_v1",
        "notes": "Weaker guarantee than 2-box scanner: near_box first-push lands in a corner "
                 "deadlock (2 adj walls, non-target); far_box first-push does not. Does NOT "
                 "prove full unsolvability -- only immediate box-dead. Suitable for smoke.",
    }
    with open(f"{args.out}/summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[4box scanner] {len(qualifiers)}/{args.target_qualifiers} qualifiers in "
          f"{dt:.1f}s (seeds tried: {len(seeds)}, drops: {drop_reasons}) -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
