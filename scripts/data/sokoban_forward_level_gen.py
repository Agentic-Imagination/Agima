#!/usr/bin/env python3
# Modified for Agima: portable paths and explicit dataset arguments.
"""FWD1 -- Exp3 "forward-thinking" Sokoban qualifying-episode generator.

Finds 2-box sokoban_easy levels with the FORWARD-PLANNING property and emits an
intermediate JSONL dataset with BOTH the doomed near-first rollout and the expert
far-first solution (actions + rendered frames), for the Exp3 CoT remap to consume.

The property (validated on natural episodes: ~2.7 % of the v8-easy train set)
--------------------------------------------------------------------------------
A 2-box level QUALIFIES iff solving the NEAR box first is PLAN-LEVEL FATAL:
every reachable way to put the near box on a target while the far box is
untouched leaves a board that is BFS-provably unsolvable -- while the far-first
solution exists. NEAR = the box with the smaller BFS walking distance from the
player start to a box-adjacent square (both boxes treated as obstacles; distance
ties are skipped).

Certificate (two phases, mirroring the validated reference implementation)
--------------------------------------------------------------------------
  phase 1: BFS over (player, near_box) with the far box FROZEN as an obstacle
           (touching the far box is exactly what "near-first" excludes), which
           enumerates every reachable near-box-on-target state.
  phase 2: each such dead-candidate board is certified with
           scripts/data/sokoban_bfs_certify.certify (state_limit=200000).
           QUALIFIES iff ALL verdicts are 'unsolvable'. 'solvable' means the
           near-first line is recoverable; 'truncated' is not a proof -- both
           disqualify.
  HARDENING vs the reference scan: the reference certified one player-witness
  per on-target box CELL. Solvability of the certified board also depends on
  which walkable COMPONENT the player ends in (the parked near box + far box can
  split the floor), so we certify one representative per distinct
  (box_cell, player-component) class over ALL hit states. Strictly sounder,
  and at most a couple of extra certify calls per level (2 target cells max).

Level sources
-------------
  --source generated (primary): fresh seeds through the SAME env construction
      the v8-easy training rows carry: gym.make("sokoban/easy",
      **{"max_steps": 20, "action_mode": "move_only"}) + env.reset(seed=S)
      (gymnasium/envs/sokoban/sokoban.py:288 reset -> :513 _generate_level ->
      cheap_generator.generate_level). Levels are a pure deterministic function
      of the seed. NOTE (distribution caveat, verified): the natural training
      episodes were ORIGINALLY generated in the VisGym_3d repo
      (the recorded source dataset; recorded init_state
      num_gen_steps=21, stream-level seed 900000 on every row), so per-level
      seed replay is impossible anywhere and this repo's cheap native generator
      (num_gen_steps default 23) is the sanctioned in-repo "easy" distribution
      (it is what visgym_training/sokoban_dataset_pipeline.py samples via the
      same env id).
  --source natural (bonus): stream the v8-easy ordered source dataset and keep
      the free ~2.7 % qualifiers.

Per qualifier we construct and verify BOTH plans:
  far_plan : env.solve(strategy='bfs') from init (sokoban.py:476; qualifiers are
             solvable so it returns ["('move', P)", ..., "('stop', 'stop')"] --
             already the training action-text format, str(("move", P)) with
             move_only implicit-push semantics). Replayed step-by-step through
             env.step and ASSERTED solved; the first pushed box is ASSERTED to
             be the far box (drop+count on violation -- the property makes this
             overwhelmingly likely but does not literally force the optimal
             plan's first push).
  near_plan: the BFS-shortest phase-1 action sequence that walks to and pushes
             the near box onto its nearest reachable target (parent-pointer
             reconstruction; ends in one of the certified-dead states).
             Replayed through env.step, end state ASSERTED to match the BFS
             prediction and RE-CERTIFIED 'unsolvable'.
Frames for the init board and every post-action board of both plans are rendered
with scripts/data/sokoban_branch_render.render_branch_frame (256px q95 base64,
byte-compatible with the v8 training frames).

Output layout (production):  <out>/levels/shard_XXXX.jsonl.gz (>=8 shards unless
fewer parts have rows), <out>/summary.json. One JSON row per qualifier:

  {source, seed, env_id, env_kwargs, init_state, init_state_schema,
   init_state_hash, init_frame_b64,
   far_plan: {actions, frames_b64},
   near_plan: {actions, frames_b64, end_state},
   near_box, far_box, near_walk_dist, far_walk_dist,
   n_dead_states_certified,
   certificates: {phase1_states, n_hit_states, per_state_verdicts_count},
   checks, natural_provenance}

Parallelism (Exp3 item 1): the driver Pools over MANY CONTIGUOUS SEED RANGES
(generated) or per-shard row ranges (natural) -- workers*tasks_per_worker tasks,
NEVER a Pool bounded by the shard count. Each worker writes an independent
gzip part file (renamed into place only when complete); completed parts are
hash-deduped and concatenated in range order (gzip members concatenate legally)
into the final shards. TARGET_QUALIFIERS stops dispatching early.

Self-test (writes nothing):
  python scripts/data/sokoban_forward_level_gen.py --self-test 2000
  python scripts/data/sokoban_forward_level_gen.py --source natural --self-test 300 \
      --natural-src /path/to/source

NEW file; does not modify any existing script/env/loader.
"""
from __future__ import annotations

import argparse
import base64
import glob
import gzip
import hashlib
import io
import json
import multiprocessing as mp
import os
import shutil
import subprocess
import sys
import time
from collections import Counter, deque
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parents[1]
for _p in (str(_REPO_ROOT), str(_THIS_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Robust to both run-as-script (scripts/data on sys.path) and import-as-module
# (scripts.data.<mod>) contexts, matching sokoban_irrev_detect.py.
try:
    from sokoban_bfs_certify import certify
    from sokoban_branch_render import render_branch_frame
except ImportError:  # pragma: no cover - import-path shim
    from scripts.data.sokoban_bfs_certify import certify
    from scripts.data.sokoban_branch_render import render_branch_frame

import gymnasium as gym
from gymnasium.envs.sokoban.sokoban import FLOOR, TARGET, WALL  # 1, 2, 0

# ---------------------------------------------------------------------------
# Constants: the EXACT env construction the v8-easy source rows carry.
# Every row of the ordered 100k source has env_id="sokoban/easy" and
# env_kwargs={"max_steps": 20, "action_mode": "move_only"} (verified).
# ---------------------------------------------------------------------------
ENV_ID = "sokoban/easy"
ENV_KWARGS: Dict[str, Any] = {"max_steps": 20, "action_mode": "move_only"}

DEFAULT_NATURAL_SRC = None

FLOORS = frozenset({FLOOR, TARGET})  # {1, 2}: walkable / pushable-onto cells
# direction -> move_only payload, matching CHANGE_COORDINATES+DIRECTION_TO_PAYLOAD
DIRS: Tuple[Tuple[int, int, int], ...] = ((-1, 0, 1), (1, 0, 2), (0, -1, 3), (0, 1, 4))

STOP_ACTION = str(("stop", "stop"))

# Drop reasons that fire AFTER the property certificate has already passed
# (used to report the pure property-qualifier rate alongside the written rate).
PLAN_GATE_REASONS = frozenset({
    "far_solve_failed", "far_plan_too_long", "no_far_first_solution",
    "far_first_plan_too_long", "far_replay_error", "far_replay_not_solved",
    "far_plan_no_push", "first_push_not_far", "near_plan_empty",
    "near_replay_error", "near_replay_mismatch", "near_end_not_dead",
    "frame_count_mismatch",
})

# Per-worker globals (set in the Pool initializer)
_CFG: Dict[str, Any] = {}
_ENV_U: Any = None


# ---------------------------------------------------------------------------
# Small pure helpers
# ---------------------------------------------------------------------------
def canonical_level_hash(room_fixed: List[List[int]], player: Tuple[int, int],
                         boxes: Iterable[Tuple[int, int]]) -> str:
    payload = json.dumps(
        [
            [[int(v) for v in row] for row in room_fixed],
            [int(player[0]), int(player[1])],
            sorted([int(b[0]), int(b[1])] for b in boxes),
        ],
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def boxes_from_init_state(state: Dict[str, Any]) -> List[Tuple[int, int]]:
    """Dual-schema box extraction, mirroring BaseSokobanEnv.reset()/SK1."""
    if state.get("box_positions"):
        raw = state["box_positions"]
    elif state.get("box_mapping"):
        raw = list(state["box_mapping"].values())
    else:
        raise KeyError("init_state must contain 'box_positions' or 'box_mapping'")
    return [tuple(int(v) for v in pos) for pos in raw]


def walking_dists(room: List[List[int]], player: Tuple[int, int],
                  boxes: Tuple[Tuple[int, int], ...]) -> List[float]:
    """BFS walking distance from the player start to a box-adjacent square, for
    each box, treating BOTH boxes as obstacles. inf if no adjacent square of a
    box is reachable."""
    h, w = len(room), len(room[0])
    box_set = set(boxes)
    dist: Dict[Tuple[int, int], int] = {player: 0}
    q: deque = deque([player])
    while q:
        r, c = q.popleft()
        d = dist[(r, c)]
        for dr, dc, _pay in DIRS:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < h and 0 <= nc < w):
                continue
            if room[nr][nc] not in FLOORS or (nr, nc) in box_set:
                continue
            if (nr, nc) in dist:
                continue
            dist[(nr, nc)] = d + 1
            q.append((nr, nc))
    out: List[float] = []
    for b in boxes:
        best = float("inf")
        for dr, dc, _pay in DIRS:
            adj = (b[0] + dr, b[1] + dc)
            if adj in dist and dist[adj] < best:
                best = dist[adj]
        out.append(best)
    return out


def phase1_near_solved_states(
    room: List[List[int]],
    player: Tuple[int, int],
    near: Tuple[int, int],
    far: Tuple[int, int],
    limit: int = 60000,
) -> Dict[str, Any]:
    """Frozen-far BFS over (player, near_box) states (the validated reference
    logic), extended with parent pointers and full hit collection.

    Returns dict with:
      hits_by_box : {box_cell -> set(player)} over every VISITED state whose
                    near box sits on a target (collected at pop time, like the
                    reference; the queue drains fully so popped == discovered)
      first_hit   : the first such state popped (BFS pop order is nondecreasing
                    distance => shortest near-first failure line), or None
      parents     : {state -> (parent_state|None, action_str|None)}
      seen        : number of states discovered
      limit_hit   : True if the state budget stopped the search (incomplete
                    certificate => caller must disqualify)
    """
    h, w = len(room), len(room[0])
    start = (player, near)
    seen = {start}
    parents: Dict[Tuple, Tuple[Optional[Tuple], Optional[str]]] = {start: (None, None)}
    q: deque = deque([start])
    hits_by_box: Dict[Tuple[int, int], set] = {}
    first_hit: Optional[Tuple] = None
    limit_hit = False
    while q:
        if len(seen) >= limit and q:
            limit_hit = True
            break
        pl, bx = q.popleft()
        if room[bx[0]][bx[1]] == TARGET:
            hits_by_box.setdefault(bx, set()).add(pl)
            if first_hit is None:
                first_hit = (pl, bx)
        for dr, dc, pay in DIRS:
            npl = (pl[0] + dr, pl[1] + dc)
            if not (0 <= npl[0] < h and 0 <= npl[1] < w):
                continue
            if room[npl[0]][npl[1]] not in FLOORS or npl == far:
                continue
            if npl == bx:
                nbx = (bx[0] + dr, bx[1] + dc)
                if not (0 <= nbx[0] < h and 0 <= nbx[1] < w):
                    continue
                if room[nbx[0]][nbx[1]] not in FLOORS or nbx == far:
                    continue
                s = (npl, nbx)
            else:
                s = (npl, bx)
            if s not in seen:
                seen.add(s)
                parents[s] = ((pl, bx), str(("move", pay)))
                q.append(s)
    return {
        "hits_by_box": hits_by_box,
        "first_hit": first_hit,
        "parents": parents,
        "seen": len(seen),
        "limit_hit": limit_hit,
    }


def pick_component_reps(
    room: List[List[int]],
    far: Tuple[int, int],
    hits_by_box: Dict[Tuple[int, int], set],
) -> Tuple[List[Tuple[Tuple[int, int], Tuple[int, int]]], int]:
    """One representative (player, box) per distinct (box_cell, player-walk-
    component) class. Two hit states with the player in the same component
    (obstacles = walls + {box_cell, far}) have identical certify verdicts, so
    certifying one per class is exact and cheap."""
    h, w = len(room), len(room[0])
    reps: List[Tuple[Tuple[int, int], Tuple[int, int]]] = []
    n_hit_states = 0
    for bx in sorted(hits_by_box):
        players = sorted(hits_by_box[bx])
        n_hit_states += len(players)
        labeled: Dict[Tuple[int, int], int] = {}
        next_cid = 0
        obstacles = {bx, far}
        for pl in players:
            if pl in labeled:
                continue
            # flood fill pl's component
            comp_q: deque = deque([pl])
            labeled[pl] = next_cid
            reps.append((pl, bx))
            while comp_q:
                r, c = comp_q.popleft()
                for dr, dc, _pay in DIRS:
                    nr, nc = r + dr, c + dc
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if room[nr][nc] not in FLOORS or (nr, nc) in obstacles:
                        continue
                    if (nr, nc) in labeled:
                        continue
                    labeled[(nr, nc)] = next_cid
                    comp_q.append((nr, nc))
            next_cid += 1
    return reps, n_hit_states


def reconstruct_actions(parents: Dict, end_state: Tuple) -> List[str]:
    out: List[str] = []
    cur = end_state
    while True:
        parent, action = parents[cur]
        if parent is None or action is None:
            break
        out.append(action)
        cur = parent
    out.reverse()
    return out


def certify_state_dict(room_fixed: List[List[int]], player: Tuple[int, int],
                       boxes: Iterable[Tuple[int, int]]) -> Dict[str, Any]:
    return {
        "room_fixed": room_fixed,
        "player_position": [int(player[0]), int(player[1])],
        "box_positions": sorted([int(b[0]), int(b[1])] for b in boxes),
    }


def solve_far_first(
    room: List[List[int]],
    player: Tuple[int, int],
    near: Tuple[int, int],
    far: Tuple[int, int],
    state_limit: int = 200000,
) -> Optional[List[str]]:
    """Shortest move_only plan whose FIRST box push moves the FAR box.

    env.solve(strategy='bfs') returns *a* shortest plan; on a majority of
    natural property-qualifiers (verified: 6 of the 8 in the first 300 source
    rows) that arbitrary tie-break pushes the near box first -- a mere
    REPOSITIONING push, which the property does not forbid (only near
    COMPLETION while far is untouched is fatal). Forward-thinking data needs an
    expert demo that literally acts far-first, so when the unconstrained plan
    violates the first-push assert we take the shortest plan constrained to
    never push the near box until the far box has been pushed at least once.
    Boxes carry identity in the search state ((player, near, far, far_pushed));
    dynamics mirror the env's move_only forward model. Returns the action list
    (incl the trailing stop) or None if no such plan exists within state_limit.
    """
    h, w = len(room), len(room[0])
    start = (player, near, far, False)
    parents: Dict[Tuple, Tuple[Optional[Tuple], Optional[str]]] = {start: (None, None)}
    seen = {start}
    q: deque = deque([start])
    expanded = 0
    while q:
        state = q.popleft()
        expanded += 1
        if expanded > state_limit:
            return None
        pl, nb, fb, far_pushed = state
        for dr, dc, pay in DIRS:
            npl = (pl[0] + dr, pl[1] + dc)
            if not (0 <= npl[0] < h and 0 <= npl[1] < w):
                continue
            if room[npl[0]][npl[1]] not in FLOORS:
                continue
            nnb, nfb, nfp = nb, fb, far_pushed
            if npl == nb or npl == fb:
                dest = (npl[0] + dr, npl[1] + dc)
                if not (0 <= dest[0] < h and 0 <= dest[1] < w):
                    continue
                if room[dest[0]][dest[1]] not in FLOORS:
                    continue
                other = fb if npl == nb else nb
                if dest == other:
                    continue
                if npl == nb:
                    if not far_pushed:
                        continue  # the forbidden near-before-far push
                    nnb = dest
                else:
                    nfb = dest
                    nfp = True
            s = (npl, nnb, nfb, nfp)
            if s in seen:
                continue
            seen.add(s)
            parents[s] = (state, str(("move", pay)))
            if room[nnb[0]][nnb[1]] == TARGET and room[nfb[0]][nfb[1]] == TARGET:
                actions = reconstruct_actions(parents, s)
                actions.append(STOP_ACTION)
                return actions
            q.append(s)
    return None


# ---------------------------------------------------------------------------
# Env plumbing (worker-local)
# ---------------------------------------------------------------------------
def build_env_unwrapped() -> Any:
    """gym.make with the exact source env_kwargs; env rendering is no-opped on
    THIS INSTANCE ONLY (render_utils reloads PNGs on every call and the level is
    fully generated before reset()/step() render -- no RNG touched)."""
    env = gym.make(ENV_ID, **ENV_KWARGS)
    u = env.unwrapped
    u.render = lambda *a, **k: None  # instance attribute shadows the method
    return u


def restore_env_state(env_u: Any, player: Tuple[int, int],
                      boxes: Tuple[Tuple[int, int], ...]) -> None:
    """Reset the mutable episode fields to a known state (room_fixed is never
    mutated by step())."""
    env_u.player_position = tuple(int(v) for v in player)
    env_u.box_positions = tuple(sorted(tuple(int(v) for v in b) for b in boxes))
    env_u.step_count = 0
    env_u.env_feedback = ""
    env_u._last_is_success = False
    env_u._cached_solution_actions = None


def replay_actions(env_u: Any, actions: List[str]) -> Dict[str, Any]:
    """Replay action strings through env.step (the real move_only transition,
    ast-parsed exactly like training/eval). Returns per-step board snapshots and
    verification facts."""
    snapshots: List[Tuple[Tuple[int, int], Tuple[Tuple[int, int], ...]]] = []
    first_pushed: Optional[Tuple[int, int]] = None
    solved_final = False
    ok = True
    note = ""
    for act in actions:
        prev_player = tuple(env_u.player_position)
        prev_boxes = tuple(sorted(env_u.box_positions))
        _obs, _reward, terminated, truncated, info = env_u.step(act)
        cur_player = tuple(env_u.player_position)
        cur_boxes = tuple(sorted(env_u.box_positions))
        snapshots.append((cur_player, cur_boxes))
        if cur_boxes != prev_boxes and first_pushed is None:
            moved_from = set(prev_boxes) - set(cur_boxes)
            if len(moved_from) != 1:
                ok, note = False, f"multi-box change on {act!r}"
                break
            first_pushed = next(iter(moved_from))
        if act == STOP_ACTION:
            solved_final = bool(info.get("is_success"))
        elif cur_player == prev_player and cur_boxes == prev_boxes:
            ok, note = False, f"no-op replay step {act!r}: {info.get('env_feedback')!r}"
            break
        if truncated:
            ok, note = False, f"truncated during replay at {act!r}"
            break
        if terminated and act != STOP_ACTION:
            ok, note = False, f"early termination at {act!r}"
            break
    return {
        "snapshots": snapshots,
        "first_pushed": first_pushed,
        "solved_final": solved_final,
        "ok": ok,
        "note": note,
    }


# ---------------------------------------------------------------------------
# The per-level pipeline
# ---------------------------------------------------------------------------
def evaluate_level(
    env_u: Any,
    cfg: Dict[str, Any],
    *,
    source: str,
    seed: Optional[int],
    stored_init_state: Optional[Dict[str, Any]],
    init_state_schema: str,
    natural_provenance: Optional[Dict[str, Any]],
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Run the full property pipeline on env_u's CURRENT (init) board.
    Returns (row, None) for a qualifier or (None, drop_reason).
    stored_init_state=None (generated mode) snapshots get_init_state() lazily,
    only for qualifiers, while the env still sits at the init board."""
    room_np = env_u.room_fixed
    room = [[int(v) for v in row] for row in room_np]
    player0 = tuple(int(v) for v in env_u.player_position)
    boxes0 = tuple(sorted(tuple(int(v) for v in b) for b in env_u.box_positions))

    if len(boxes0) != 2:
        return None, "not_2_boxes"
    for row_vals in room:
        for v in row_vals:
            if v not in (WALL, FLOOR, TARGET):
                return None, "bad_room_codes"
    if any(room[b[0]][b[1]] == TARGET for b in boxes0):
        return None, "box_starts_on_target"

    # near/far by BFS walking distance to a box-adjacent square
    dists = walking_dists(room, player0, boxes0)
    if dists[0] == dists[1]:
        return None, "near_far_tie"
    near_i = 0 if dists[0] < dists[1] else 1
    near, far = boxes0[near_i], boxes0[1 - near_i]
    near_d, far_d = dists[near_i], dists[1 - near_i]
    if near_d == float("inf"):
        return None, "no_reachable_box_adjacency"

    # phase 1: frozen-far BFS
    p1 = phase1_near_solved_states(room, player0, near, far, limit=cfg["phase1_limit"])
    if p1["limit_hit"]:
        return None, "phase1_truncated"
    if not p1["hits_by_box"]:
        return None, "near_cannot_reach_target"

    # phase 2: certify every (box_cell, player-component) dead candidate
    reps, n_hit_states = pick_component_reps(room, far, p1["hits_by_box"])
    verdict_counts: Counter = Counter()
    for pl, bx in reps:
        v = certify(certify_state_dict(room, pl, (bx, far)),
                    state_limit=cfg["certify_state_limit"])
        verdict_counts[v] += 1
        if v == "solvable":
            return None, "near_first_recoverable"
        if v == "truncated":
            return None, "certify_truncated"

    # qualifier: snapshot the init_state while the env is still at init (the
    # replays below mutate it), then build the far-first expert plan
    if stored_init_state is None:
        stored_init_state = stored_init_state_generated(env_u)
    try:
        far_actions = env_u.solve(strategy="bfs", state_limit=cfg["solve_state_limit"])
    except (RuntimeError, ValueError):
        return None, "far_solve_failed"
    if len(far_actions) > cfg["max_plan_len"]:
        return None, "far_plan_too_long"

    far_constrained = False
    far_rep = replay_actions(env_u, far_actions)
    if (far_rep["ok"] and far_rep["first_pushed"] is not None
            and tuple(far_rep["first_pushed"]) != far):
        # BFS tie-break repositioned the near box first; construct the shortest
        # genuinely far-first plan instead (see solve_far_first docstring).
        far_actions = solve_far_first(room, player0, near, far,
                                      state_limit=cfg["solve_state_limit"])
        if far_actions is None:
            return None, "no_far_first_solution"
        if len(far_actions) > cfg["max_plan_len"]:
            return None, "far_first_plan_too_long"
        far_constrained = True
        restore_env_state(env_u, player0, boxes0)
        far_rep = replay_actions(env_u, far_actions)
    if not far_rep["ok"]:
        return None, "far_replay_error"
    if not far_rep["solved_final"]:
        return None, "far_replay_not_solved"
    if far_rep["first_pushed"] is None:
        return None, "far_plan_no_push"
    if tuple(far_rep["first_pushed"]) != far:
        return None, "first_push_not_far"

    # near-first failure rollout (shortest, ends in a certified-dead state)
    near_actions = reconstruct_actions(p1["parents"], p1["first_hit"])
    if not near_actions:
        return None, "near_plan_empty"
    restore_env_state(env_u, player0, boxes0)
    near_rep = replay_actions(env_u, near_actions)
    if not near_rep["ok"]:
        return None, "near_replay_error"
    end_pl, end_boxes = near_rep["snapshots"][-1]
    exp_pl, exp_bx = p1["first_hit"]
    if end_pl != exp_pl or set(end_boxes) != {exp_bx, far}:
        return None, "near_replay_mismatch"
    end_verdict = certify(certify_state_dict(room, end_pl, end_boxes),
                          state_limit=cfg["certify_state_limit"])
    if end_verdict != "unsolvable":
        return None, "near_end_not_dead"

    # frames: init + every post-action board of both plans (byte-compatible)
    init_frame = render_branch_frame(player0, boxes0, room, action_mode="move_only")
    far_frames = [
        render_branch_frame(pl, bxs, room, action_mode="move_only")
        for pl, bxs in far_rep["snapshots"]
    ]
    near_frames = [
        render_branch_frame(pl, bxs, room, action_mode="move_only")
        for pl, bxs in near_rep["snapshots"]
    ]
    if len(far_frames) != len(far_actions) or len(near_frames) != len(near_actions):
        return None, "frame_count_mismatch"

    row = {
        "source": source,
        "seed": None if seed is None else int(seed),
        "env_id": ENV_ID,
        "env_kwargs": dict(ENV_KWARGS),
        "init_state": stored_init_state,
        "init_state_schema": init_state_schema,
        "init_state_hash": canonical_level_hash(room, player0, boxes0),
        "init_frame_b64": init_frame,
        "far_plan": {"actions": list(far_actions), "frames_b64": far_frames},
        "near_plan": {
            "actions": list(near_actions),
            "frames_b64": near_frames,
            "end_state": {
                "player_position": [int(end_pl[0]), int(end_pl[1])],
                "box_positions": sorted([int(b[0]), int(b[1])] for b in end_boxes),
            },
        },
        "near_box": [int(near[0]), int(near[1])],
        "far_box": [int(far[0]), int(far[1])],
        "near_walk_dist": int(near_d),
        "far_walk_dist": (None if far_d == float("inf") else int(far_d)),
        "n_dead_states_certified": len(reps),
        "certificates": {
            "phase1_states": int(p1["seen"]),
            "n_hit_states": int(n_hit_states),
            "per_state_verdicts_count": dict(verdict_counts),
        },
        "checks": {
            "far_replay_solved": True,
            "first_pushed_box_is_far": True,
            "far_plan_constrained": far_constrained,
            "near_replay_matches_bfs": True,
            "near_end_recertified": end_verdict,
        },
        "natural_provenance": natural_provenance,
    }
    return row, None


def stored_init_state_generated(env_u: Any) -> Dict[str, Any]:
    """This repo's get_init_state() schema (reset(init_state=...)-compatible).
    constructive_solution is nulled: it is a generator-internal cached plan in
    legacy move_push ids; the shipped far_plan is the BFS solution."""
    st = env_u.get_init_state()
    st["constructive_solution"] = None
    return st


# ---------------------------------------------------------------------------
# Pool workers
# ---------------------------------------------------------------------------
def _pool_init(cfg: Dict[str, Any]) -> None:
    global _CFG, _ENV_U
    _CFG = cfg
    _ENV_U = build_env_unwrapped()


def _open_part(path: str):
    return gzip.open(path, "wt", encoding="utf-8", compresslevel=6)


def _finish_part(handle, tmp_path: str, final_path: str, n_rows: int) -> Optional[str]:
    if handle is not None:
        handle.close()
        if n_rows > 0:
            os.replace(tmp_path, final_path)
            return final_path
        os.unlink(tmp_path)
    return None


def worker_generated(task: Dict[str, Any]) -> Dict[str, Any]:
    """One contiguous seed range -> one part file (or in-memory rows)."""
    cfg = _CFG
    env_u = _ENV_U
    t_wall = time.perf_counter()
    idx, lo, hi = task["idx"], task["lo"], task["hi"]
    drops: Counter = Counter()
    rows_out: List[Dict[str, Any]] = []
    hashes: List[str] = []
    handle = None
    tmp_path = final_path = ""
    if not cfg["return_rows"]:
        final_path = os.path.join(cfg["parts_dir"], f"part_{idx:06d}.jsonl.gz")
        tmp_path = final_path + ".inprogress"
    n_scanned = 0
    n_rows = 0
    n_constrained = 0
    ms_sum = 0.0
    err_samples: List[str] = []
    for seed in range(lo, hi):
        t0 = time.perf_counter()
        try:
            env_u.reset(seed=seed)
            row, reason = evaluate_level(
                env_u,
                cfg,
                source="generated",
                seed=seed,
                stored_init_state=None,  # snapshot lazily inside, at init
                init_state_schema="repo_get_init_state",
                natural_provenance=None,
            )
        except Exception as exc:  # never kill the task on one level
            drops[f"exception:{type(exc).__name__}"] += 1
            if len(err_samples) < 3:
                err_samples.append(f"seed={seed}: {exc!r}")
            ms_sum += (time.perf_counter() - t0) * 1000.0
            n_scanned += 1
            continue
        if reason is not None:
            drops[reason] += 1
        else:
            n_rows += 1
            n_constrained += int(bool(row["checks"]["far_plan_constrained"]))
            hashes.append(row["init_state_hash"])
            if cfg["return_rows"]:
                rows_out.append(row)
            else:
                if handle is None:
                    handle = _open_part(tmp_path)
                handle.write(json.dumps(row, ensure_ascii=False,
                                        separators=(",", ":")) + "\n")
        ms_sum += (time.perf_counter() - t0) * 1000.0
        n_scanned += 1
    part = _finish_part(handle, tmp_path, final_path, n_rows)
    return {
        "idx": idx,
        "part": part,
        "rows": rows_out,
        "hashes": hashes,
        "n_scanned": n_scanned,
        "n_rows": n_rows,
        "n_constrained": n_constrained,
        "drops": dict(drops),
        "ms_sum": ms_sum,
        "wall_s": time.perf_counter() - t_wall,
        "err_samples": err_samples,
    }


def worker_natural(task: Dict[str, Any]) -> Dict[str, Any]:
    """One contiguous row range of one source shard -> one part file."""
    cfg = _CFG
    env_u = _ENV_U
    t_wall = time.perf_counter()
    idx, shard, lo, hi = task["idx"], task["shard"], task["lo"], task["hi"]
    drops: Counter = Counter()
    rows_out: List[Dict[str, Any]] = []
    hashes: List[str] = []
    handle = None
    tmp_path = final_path = ""
    if not cfg["return_rows"]:
        final_path = os.path.join(cfg["parts_dir"], f"part_{idx:06d}.jsonl.gz")
        tmp_path = final_path + ".inprogress"
    n_scanned = 0
    n_rows = 0
    n_constrained = 0
    ms_sum = 0.0
    err_samples: List[str] = []
    with gzip.open(shard, "rt", encoding="utf-8") as fin:
        row_i = -1
        for line in fin:
            row_i += 1
            if row_i < lo:
                continue
            if row_i >= hi:
                break
            t0 = time.perf_counter()
            try:
                src = json.loads(line)
                init_state = src["init_state"]
                if dict(src.get("env_kwargs") or {}) != ENV_KWARGS:
                    drops["env_kwargs_mismatch"] += 1
                    ms_sum += (time.perf_counter() - t0) * 1000.0
                    n_scanned += 1
                    continue
                env_u.reset(init_state=init_state)
                prov = {
                    "shard": os.path.basename(shard),
                    "row_index_in_shard": row_i,
                    "episode": src.get("episode"),
                    "order_key": src.get("__bagel_order_key"),
                    "alignment_index": src.get("alignment_index"),
                    "source_init_state_hash": src.get("init_state_hash"),
                }
                row, reason = evaluate_level(
                    env_u,
                    cfg,
                    source="natural",
                    seed=src.get("episode_seed", src.get("seed")),
                    stored_init_state=init_state,
                    init_state_schema="source_row",
                    natural_provenance=prov,
                )
            except Exception as exc:
                drops[f"exception:{type(exc).__name__}"] += 1
                if len(err_samples) < 3:
                    err_samples.append(f"{os.path.basename(shard)}:{row_i}: {exc!r}")
                ms_sum += (time.perf_counter() - t0) * 1000.0
                n_scanned += 1
                continue
            if reason is not None:
                drops[reason] += 1
            else:
                n_rows += 1
                n_constrained += int(bool(row["checks"]["far_plan_constrained"]))
                hashes.append(row["init_state_hash"])
                if cfg["return_rows"]:
                    rows_out.append(row)
                else:
                    if handle is None:
                        handle = _open_part(tmp_path)
                    handle.write(json.dumps(row, ensure_ascii=False,
                                            separators=(",", ":")) + "\n")
            ms_sum += (time.perf_counter() - t0) * 1000.0
            n_scanned += 1
    part = _finish_part(handle, tmp_path, final_path, n_rows)
    return {
        "idx": idx,
        "part": part,
        "rows": rows_out,
        "hashes": hashes,
        "n_scanned": n_scanned,
        "n_rows": n_rows,
        "n_constrained": n_constrained,
        "drops": dict(drops),
        "ms_sum": ms_sum,
        "wall_s": time.perf_counter() - t_wall,
        "err_samples": err_samples,
    }


def worker_filter_part(job: Tuple[str, List[int]]) -> Tuple[str, int]:
    """Rewrite a part file dropping the given 0-based row indices (dedup)."""
    part, drop_idx = job
    drop = set(drop_idx)
    tmp = part + ".dedup"
    kept = 0
    with gzip.open(part, "rt", encoding="utf-8") as fin, \
            _open_part(tmp) as fout:
        for i, line in enumerate(fin):
            if i in drop:
                continue
            fout.write(line)
            kept += 1
    if kept > 0:
        os.replace(tmp, part)
    else:
        os.unlink(tmp)
        os.unlink(part)
    return part, kept


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def build_tasks(args: argparse.Namespace) -> List[Dict[str, Any]]:
    n_tasks = max(1, args.workers * args.tasks_per_worker)
    tasks: List[Dict[str, Any]] = []
    if args.source == "generated":
        n = args.self_test if args.self_test else args.n_seeds
        n_tasks = min(n_tasks, n)
        base = args.base_seed
        per = n // n_tasks
        rem = n % n_tasks
        lo = base
        for i in range(n_tasks):
            hi = lo + per + (1 if i < rem else 0)
            tasks.append({"kind": "gen", "idx": i, "lo": lo, "hi": hi})
            lo = hi
    else:
        shards = sorted(glob.glob(os.path.join(args.natural_src, "*.jsonl.gz")))
        if not shards:
            raise FileNotFoundError(f"no *.jsonl.gz shards under {args.natural_src}")
        if args.self_test:
            n = args.self_test
            shards = shards[:1]
            rows_per_shard = [n]
        else:
            rows_per_shard = [args.natural_rows_per_shard] * len(shards)
        total_rows = sum(rows_per_shard)
        n_tasks = min(n_tasks, total_rows)
        per_shard_tasks = max(1, round(n_tasks / len(shards)))
        i = 0
        for shard, n_rows in zip(shards, rows_per_shard):
            k = min(per_shard_tasks, n_rows)
            per = n_rows // k
            rem = n_rows % k
            lo = 0
            for j in range(k):
                hi = lo + per + (1 if j < rem else 0)
                tasks.append({"kind": "nat", "idx": i, "shard": shard, "lo": lo, "hi": hi})
                i += 1
                lo = hi
    return tasks


def run_pool(args: argparse.Namespace, tasks: List[Dict[str, Any]],
             cfg: Dict[str, Any]) -> Dict[str, Any]:
    worker = worker_generated if args.source == "generated" else worker_natural
    results: List[Dict[str, Any]] = []
    total_rows = 0
    total_scanned = 0
    total_constrained = 0
    total_ms = 0.0
    drops: Counter = Counter()
    err_samples: List[str] = []
    stopped_early = False
    t0 = time.perf_counter()
    n_workers = min(args.workers, len(tasks))
    pool = mp.Pool(n_workers, initializer=_pool_init, initargs=(cfg,))
    try:
        for res in pool.imap_unordered(worker, tasks, chunksize=1):
            results.append(res)
            total_rows += res["n_rows"]
            total_scanned += res["n_scanned"]
            total_constrained += res["n_constrained"]
            total_ms += res["ms_sum"]
            drops.update(res["drops"])
            err_samples.extend(res["err_samples"][: max(0, 5 - len(err_samples))])
            done = len(results)
            if done % max(1, len(tasks) // 40) == 0 or done == len(tasks):
                rate = 100.0 * total_rows / max(1, total_scanned)
                print(
                    f"[progress] tasks {done}/{len(tasks)} scanned={total_scanned} "
                    f"qualifiers={total_rows} ({rate:.2f}%) "
                    f"mean={total_ms / max(1, total_scanned):.2f} ms/level "
                    f"wall={time.perf_counter() - t0:.0f}s",
                    flush=True,
                )
            if args.target_qualifiers and total_rows >= args.target_qualifiers:
                stopped_early = True
                print(
                    f"[early-stop] target_qualifiers={args.target_qualifiers} reached "
                    f"({total_rows}); halting dispatch (in-flight tasks discarded)",
                    flush=True,
                )
                break
    finally:
        pool.terminate()
        pool.join()
    return {
        "results": results,
        "total_rows": total_rows,
        "total_scanned": total_scanned,
        "total_constrained": total_constrained,
        "total_ms": total_ms,
        "drops": drops,
        "err_samples": err_samples,
        "stopped_early": stopped_early,
        "wall_s": time.perf_counter() - t0,
    }


def dedup_and_shard(args: argparse.Namespace, agg: Dict[str, Any],
                    out_root: Path) -> Dict[str, Any]:
    """Keep-first dedup by init_state_hash across completed parts (range order),
    then concatenate contiguous part groups into >=n_shards gzip shard files."""
    results = sorted(agg["results"], key=lambda r: r["idx"])
    seen: set = set()
    part_rows: List[Tuple[str, int]] = []  # (part path, kept rows) in range order
    filter_jobs: List[Tuple[str, List[int]]] = []
    dup_dropped = 0
    for res in results:
        if not res["part"]:
            continue
        drop_idx = []
        for i, h in enumerate(res["hashes"]):
            if h in seen:
                drop_idx.append(i)
            else:
                seen.add(h)
        kept = res["n_rows"] - len(drop_idx)
        dup_dropped += len(drop_idx)
        if drop_idx:
            filter_jobs.append((res["part"], drop_idx))
        if kept > 0:
            part_rows.append((res["part"], kept))
    if filter_jobs:
        print(f"[dedup] rewriting {len(filter_jobs)} part(s), dropping {dup_dropped} duplicate row(s)", flush=True)
        with mp.Pool(min(args.workers, len(filter_jobs))) as pool:
            pool.map(worker_filter_part, filter_jobs)

    total = sum(k for _p, k in part_rows)
    levels_dir = out_root / "levels"
    levels_dir.mkdir(parents=True, exist_ok=True)
    shard_files: List[Dict[str, Any]] = []
    if total > 0:
        n_shards = min(args.n_shards, len(part_rows))
        groups: List[List[Tuple[str, int]]] = [[] for _ in range(n_shards)]
        cum = 0
        for p, k in part_rows:
            gi = min(n_shards - 1, cum * n_shards // total)
            groups[gi].append((p, k))
            cum += k
        shard_i = 0
        for group in groups:
            if not group:
                continue
            shard_path = levels_dir / f"shard_{shard_i:04d}.jsonl.gz"
            shard_i += 1
            n_rows = 0
            with open(shard_path, "wb") as fout:
                for p, k in group:
                    with open(p, "rb") as fin:
                        shutil.copyfileobj(fin, fout, length=16 * 1024 * 1024)
                    n_rows += k
            shard_files.append({"path": str(shard_path), "rows": n_rows})
    return {"dup_dropped": dup_dropped, "total_rows_final": total, "shards": shard_files}


def _decode_b64_image(b64: str) -> Tuple[Tuple[int, int], str]:
    from PIL import Image

    img = Image.open(io.BytesIO(base64.b64decode(b64)))
    img.load()
    return img.size, img.mode


def print_selftest_report(args: argparse.Namespace, agg: Dict[str, Any]) -> bool:
    rows: List[Dict[str, Any]] = []
    for res in sorted(agg["results"], key=lambda r: r["idx"]):
        rows.extend(res["rows"])
    n = agg["total_scanned"]
    k = agg["total_rows"]
    mean_ms = agg["total_ms"] / max(1, n)
    # levels whose CERTIFICATE passed but a plan-construction gate dropped them
    plan_gate = sum(v for r, v in agg["drops"].items() if r in PLAN_GATE_REASONS)
    print("=" * 78)
    print(f"[self-test] source={args.source} scanned={n} qualifiers written={k} "
          f"({100.0 * k / max(1, n):.2f}%) | property-certified={k + plan_gate} "
          f"({100.0 * (k + plan_gate) / max(1, n):.2f}%) | "
          f"constrained far-first plans={agg['total_constrained']}/{k}")
    print(f"[self-test] mean per-level pipeline time: {mean_ms:.2f} ms/level "
          f"(pool wall {agg['wall_s']:.1f}s, workers={args.workers})")
    if agg["drops"]:
        print(f"[self-test] drop reasons: "
              f"{json.dumps(dict(sorted(agg['drops'].items())), sort_keys=True)}")
    if agg["err_samples"]:
        print(f"[self-test] error samples: {agg['err_samples']}")
    ok = True
    for qi, row in enumerate(rows[:3]):
        room = row["init_state"].get("room_fixed")
        end = row["near_plan"]["end_state"]
        recert = certify(
            {
                "room_fixed": room,
                "player_position": end["player_position"],
                "box_positions": end["box_positions"],
            },
            state_limit=args.certify_state_limit,
        )
        frames_ok = True
        for tag, b64 in (
            ("init", row["init_frame_b64"]),
            ("far_last", row["far_plan"]["frames_b64"][-1]),
            ("near_last", row["near_plan"]["frames_b64"][-1]),
        ):
            size, mode = _decode_b64_image(b64)
            if size != (256, 256) or mode != "RGB":
                frames_ok = False
                print(f"    [FRAME FAIL] {tag}: {size} {mode}")
        checks = row["checks"]
        print(f"  [qualifier {qi}] seed={row['seed']} "
              f"near_box={row['near_box']} (walk {row['near_walk_dist']}) "
              f"far_box={row['far_box']} (walk {row['far_walk_dist']})")
        print(f"      near_plan len={len(row['near_plan']['actions'])} "
              f"re-certified end verdict={recert!r} | "
              f"far_plan len={len(row['far_plan']['actions'])} (incl stop) "
              f"replay-solved={checks['far_replay_solved']} "
              f"first-pushed-box==far={checks['first_pushed_box_is_far']} "
              f"constrained={checks['far_plan_constrained']}")
        print(f"      dead states certified={row['n_dead_states_certified']} "
              f"certificates={json.dumps(row['certificates'])} "
              f"frames decode 256x256 RGB={frames_ok}")
        ok = ok and frames_ok and (recert == "unsolvable")
    if not rows:
        print("  (no qualifiers found in this sample)")
        ok = False
    print(f"[self-test] RESULT: {'PASS' if ok else 'FAIL'}")
    print("=" * 78)
    return ok


def git_rev() -> Optional[str]:
    try:
        return subprocess.run(
            ["git", "-C", str(_REPO_ROOT), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=10, check=True,
        ).stdout.strip()
    except Exception:
        return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--source", choices=("generated", "natural"), default="generated")
    ap.add_argument("--n-seeds", type=int, default=4_000_000,
                    help="generated mode: number of fresh seeds to scan")
    ap.add_argument("--base-seed", type=int, default=30_000_000,
                    help="generated mode: first seed of the contiguous range")
    ap.add_argument("--target-qualifiers", type=int, default=80_000,
                    help="stop dispatching once this many qualifiers exist (0=off)")
    ap.add_argument("--out", type=str, default=None,
                    help="output root (required unless --self-test)")
    ap.add_argument("--workers", type=int, default=0,
                    help="pool size (default: all cores; self-test: min(16, cores))")
    ap.add_argument("--tasks-per-worker", type=int, default=4,
                    help="pool tasks = workers * this (contiguous ranges, NOT shards)")
    ap.add_argument("--n-shards", type=int, default=8)
    ap.add_argument("--natural-src", type=str, default=DEFAULT_NATURAL_SRC)
    ap.add_argument("--natural-rows-per-shard", type=int, default=12_500,
                    help="expected rows per source shard (100k/8; workers tolerate EOF)")
    ap.add_argument("--max-plan-len", type=int, default=20,
                    help="drop qualifiers whose far plan (incl stop) exceeds this "
                         "(training episodes are capped at max_steps=20)")
    ap.add_argument("--phase1-limit", type=int, default=60_000)
    ap.add_argument("--certify-state-limit", type=int, default=200_000)
    ap.add_argument("--solve-state-limit", type=int, default=200_000)
    ap.add_argument("--self-test", type=int, default=0, metavar="N",
                    help="run on N levels/rows, print a verification report, write nothing")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    if args.source == "natural" and not args.natural_src:
        ap.error("--natural-src is required when --source natural")

    try:  # respect cgroup/SLURM CPU affinity (os.cpu_count() ignores it)
        ncpu = len(os.sched_getaffinity(0))
    except AttributeError:  # pragma: no cover - non-linux
        ncpu = os.cpu_count() or 1
    if args.workers <= 0:
        args.workers = min(16, ncpu) if args.self_test else ncpu

    # Warm the byte-compatible renderer in the driver so forked workers inherit
    # the loaded surface assets (copy-on-write).
    dummy_room = [[0] * 7 for _ in range(7)]
    for r in range(1, 6):
        for c in range(1, 6):
            dummy_room[r][c] = 1
    dummy_room[1][1] = 2
    render_branch_frame((2, 2), [(1, 1), (3, 3)], dummy_room, action_mode="move_only")

    cfg = {
        "phase1_limit": args.phase1_limit,
        "certify_state_limit": args.certify_state_limit,
        "solve_state_limit": args.solve_state_limit,
        "max_plan_len": args.max_plan_len,
        "return_rows": bool(args.self_test),
        "parts_dir": None,
    }

    if args.self_test:
        args.target_qualifiers = 0  # scan the full sample
        tasks = build_tasks(args)
        print(f"[self-test] {len(tasks)} range tasks over {args.self_test} "
              f"{'seeds' if args.source == 'generated' else 'rows'} "
              f"(workers={args.workers})", flush=True)
        agg = run_pool(args, tasks, cfg)
        return 0 if print_selftest_report(args, agg) else 1

    if not args.out:
        ap.error("--out is required for production runs")
    out_root = Path(args.out).resolve()
    levels_dir = out_root / "levels"
    if levels_dir.exists() and any(levels_dir.iterdir()):
        if not args.overwrite:
            raise FileExistsError(f"{levels_dir} already has shards (use --overwrite)")
        shutil.rmtree(levels_dir)
    parts_dir = out_root / "parts_tmp"
    if parts_dir.exists():
        shutil.rmtree(parts_dir)
    parts_dir.mkdir(parents=True, exist_ok=True)
    cfg["parts_dir"] = str(parts_dir)

    tasks = build_tasks(args)
    print(f"[driver] source={args.source} tasks={len(tasks)} workers={args.workers} "
          f"target_qualifiers={args.target_qualifiers or 'off'} out={out_root}",
          flush=True)
    agg = run_pool(args, tasks, cfg)
    shard_info = dedup_and_shard(args, agg, out_root)
    shutil.rmtree(parts_dir, ignore_errors=True)

    mean_ms = agg["total_ms"] / max(1, agg["total_scanned"])
    summary = {
        "generator": "scripts/data/sokoban_forward_level_gen.py",
        "git_rev": git_rev(),
        "argv": sys.argv[1:],
        "source": args.source,
        "env_id": ENV_ID,
        "env_kwargs": ENV_KWARGS,
        "base_seed": args.base_seed if args.source == "generated" else None,
        "levels_scanned": agg["total_scanned"],
        "qualifiers_written": shard_info["total_rows_final"],
        "qualifiers_before_dedup": agg["total_rows"],
        "duplicates_dropped": shard_info["dup_dropped"],
        "qualify_rate_pct": 100.0 * agg["total_rows"] / max(1, agg["total_scanned"]),
        "property_certified": agg["total_rows"] + sum(
            v for r, v in agg["drops"].items() if r in PLAN_GATE_REASONS),
        "far_plan_constrained_used": agg["total_constrained"],
        "mean_ms_per_level": mean_ms,
        "stopped_early": agg["stopped_early"],
        "tasks_completed": len(agg["results"]),
        "tasks_total": len(tasks),
        "pool_wall_s": agg["wall_s"],
        "drop_reasons": dict(sorted(agg["drops"].items())),
        "error_samples": agg["err_samples"],
        "shards": shard_info["shards"],
        "certify_state_limit": args.certify_state_limit,
        "solve_state_limit": args.solve_state_limit,
        "phase1_limit": args.phase1_limit,
        "max_plan_len": args.max_plan_len,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n",
                                           encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items()
                      if k not in ("shards", "drop_reasons")}, indent=2), flush=True)
    print(f"[driver] wrote {shard_info['total_rows_final']} rows into "
          f"{len(shard_info['shards'])} shard(s) under {levels_dir}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
