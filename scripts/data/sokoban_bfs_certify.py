#!/usr/bin/env python3
# Modified for Agima: portable paths and explicit dataset arguments.
"""SK1 -- robust sokoban solvability certifier.

Distinguishes three outcomes that the in-repo env conflates. The env's own
``BaseSokobanEnv._solve_bfs`` (gymnasium/envs/sokoban/sokoban.py:681) returns
``None`` for BOTH a proven-unsolvable board (queue exhausted) AND a truncated
search (state_limit hit), and ``solve()`` (:476) raises for both -- so a caller
can never tell "no solution exists" from "I gave up early". EXP2's counterfactual
splicer needs exactly that distinction: a counterfactual push is "critical" iff
it turns a solvable board into a *proven-unsolvable* one (not merely one the
search timed out on).

This module re-implements the BFS on top of the env's forward-model PRIMITIVES
(``_enumerate_forward_action_ids`` / ``_simulate_action_id`` / ``_is_solved_state``,
sokoban.py:739/731/788) with an explicit visited-set + queue, and returns:

    'solvable'    -- a goal state was reached
    'unsolvable'  -- the reachable state graph was fully exhausted below the
                     limit without reaching a goal (PROOF of unsolvability)
    'truncated'   -- state_limit expansions hit before either happened

Why the legacy move/push primitives certify the move_only dataset too
--------------------------------------------------------------------
The easy/small dataset runs ``action_mode='move_only'`` where moving INTO a box
implicitly pushes it (sokoban.py:602-603) and moving into floor just moves
(:605). ``_enumerate_forward_action_ids`` yields a legacy "push" id exactly when
a box sits ahead with a free cell behind, and a legacy "move" id exactly when
floor sits ahead -- i.e. the union of the two move_only outcomes. The reachable
(player, boxes) state graph is therefore identical, so solvability is invariant
to action_mode and this certifier is correct for both modes.

The BFS is a *pure exhaustive* search over (player, boxes) states -- no deadlock
heuristics -- so 'unsolvable' is a genuine proof (no false negatives from
pruning), and a box frozen in a non-target wall-corner collapses the reachable
graph to the player wandering, which exhausts fast.

NEW file; does not modify any existing script/env/loader.
"""
from __future__ import annotations

from collections import deque, namedtuple
from typing import Dict, List, Optional, Tuple

import numpy as np

from gymnasium.envs.sokoban.sokoban import BaseSokobanEnv, SokobanState


CertifyResult = namedtuple(
    "CertifyResult",
    ["verdict", "states_expanded", "states_discovered", "state_limit"],
)


def _boxes_from_state(state: Dict) -> List[Tuple[int, int]]:
    """Accept either the repo get_init_state schema ('box_positions': [[r,c],...])
    or the visgym/jsonl schema ('box_mapping': {goal_str: [r,c]}), matching
    BaseSokobanEnv.reset()'s dual-schema handling (sokoban.py:307-317)."""
    if state.get("box_positions"):
        raw = state["box_positions"]
    elif state.get("box_mapping"):
        raw = list(state["box_mapping"].values())
    else:
        raise KeyError("state must contain non-empty 'box_positions' or 'box_mapping'")
    return [tuple(int(v) for v in pos) for pos in raw]


def load_env_from_state(state: Dict) -> BaseSokobanEnv:
    """Build a BaseSokobanEnv whose fields are set DIRECTLY from an init_state
    dict, deliberately bypassing env.reset() (which renders an RGB frame and can
    require texture assets). We only populate the four fields the BFS primitives
    read: room_fixed, dim_room, player_position, box_positions. render_mode is
    'ansi' purely as a safe default; render() is never called."""
    room_fixed = np.asarray(state["room_fixed"], dtype=np.int8)
    dim_room = tuple(int(v) for v in room_fixed.shape)
    env = BaseSokobanEnv(
        dim_room=dim_room,
        max_steps=1,
        num_boxes=0,
        num_gen_steps=0,
        render_mode="ansi",
    )
    env.room_fixed = room_fixed
    # reset() does NOT propagate dim_room from init_state, so we align it to the
    # actual room_fixed shape ourselves (the BFS primitives index room_fixed via
    # self.dim_room in _is_floor()).
    env.dim_room = dim_room
    env.player_position = tuple(int(v) for v in state["player_position"])
    env.box_positions = tuple(sorted(_boxes_from_state(state)))
    return env


def certify_detailed(state: Dict, state_limit: int = 200000) -> CertifyResult:
    """Full certification with search statistics.

    ``state`` is an init_state dict (repo get_init_state output or a visgym
    jsonl init_state). Returns a CertifyResult(verdict, states_expanded,
    states_discovered, state_limit).

    Truncation semantics mirror the env's own _solve_bfs (sokoban.py:695): a node
    is counted as it is popped, and the search bails the moment the expansion
    count would exceed the limit."""
    env = load_env_from_state(state)
    start = env._current_state()
    if env._is_solved_state(start):
        return CertifyResult("solvable", 0, 1, int(state_limit))

    limit = max(1, int(state_limit))
    queue: deque = deque([start])
    visited = {start}
    expanded = 0

    while queue:
        node = queue.popleft()
        expanded += 1
        if expanded > limit:
            # limit hit; report the count actually completed (expanded - 1)
            return CertifyResult("truncated", expanded - 1, len(visited), int(state_limit))

        # _enumerate_forward_action_ids yields ONLY legal action ids (a push id
        # iff a box is ahead with floor behind; a move id iff floor is ahead), so
        # every _simulate_action_id below is a valid transition -- no need to
        # re-check "changed" the way _solve_bfs does.
        for action_id in env._enumerate_forward_action_ids(node):
            nxt = env._simulate_action_id(node, action_id)
            if nxt in visited:
                continue
            if env._is_solved_state(nxt):
                visited.add(nxt)
                return CertifyResult("solvable", expanded, len(visited), int(state_limit))
            visited.add(nxt)
            queue.append(nxt)

    # queue drained below the limit: the reachable graph is fully explored and no
    # goal exists -> PROVEN unsolvable.
    return CertifyResult("unsolvable", expanded, len(visited), int(state_limit))


def certify(state: Dict, state_limit: int = 200000) -> str:
    """Return one of 'solvable' | 'unsolvable' | 'truncated' for an init_state."""
    return certify_detailed(state, state_limit=state_limit).verdict


# --------------------------------------------------------------------------- #
# Self-test                                                                    #
# --------------------------------------------------------------------------- #
def _first_source_init_state(src: str) -> Dict:
    """First row's init_state from the explicitly selected CoT-inf source dataset."""
    import glob
    import gzip
    import json

    subdir = "trajectories/sokoban_easy/train"
    shards = sorted(glob.glob(f"{src}/{subdir}/*.jsonl.gz"))
    if not shards:
        raise FileNotFoundError(f"no shards under {src}/{subdir}")
    with gzip.open(shards[0], "rt") as fin:
        return json.loads(fin.readline())["init_state"]


def _find_corner_push_counterfactual(state: Dict) -> Optional[Dict]:
    """From a real init_state, find ONE legal counterfactual box push that lands a
    box in a 'dead corner' (a non-target floor cell walled on one vertical AND one
    horizontal side, from which a box can never move) and return the resulting
    init_state dict. Returns None if no such push exists on this board.

    A push in direction d needs the player behind the box (that cell must be
    floor, so the pusher can stand there) and the landing cell ahead to be floor.
    We enumerate over the board's own boxes so this stays faithful to "splice one
    counterfactual push into the real trajectory"."""
    from gymnasium.envs.sokoban.sokoban import CHANGE_COORDINATES, TARGET, WALL

    room_fixed = np.asarray(state["room_fixed"], dtype=np.int8)
    rows, cols = room_fixed.shape
    boxes = [tuple(p) for p in sorted(_boxes_from_state(state))]
    box_set = set(boxes)

    def in_bounds(rc):
        return 0 <= rc[0] < rows and 0 <= rc[1] < cols

    def is_wall(rc):
        return (not in_bounds(rc)) or room_fixed[rc[0], rc[1]] == WALL

    def is_floor(rc):
        return in_bounds(rc) and room_fixed[rc[0], rc[1]] != WALL and rc not in box_set

    def is_dead_corner(rc):
        if not in_bounds(rc):
            return False
        if room_fixed[rc[0], rc[1]] == WALL or room_fixed[rc[0], rc[1]] == TARGET:
            return False
        up = (rc[0] - 1, rc[1]); down = (rc[0] + 1, rc[1])
        left = (rc[0], rc[1] - 1); right = (rc[0], rc[1] + 1)
        vert_walled = is_wall(up) or is_wall(down)
        horiz_walled = is_wall(left) or is_wall(right)
        return vert_walled and horiz_walled

    for box in boxes:
        for _d, (dr, dc) in CHANGE_COORDINATES.items():
            behind = (box[0] - dr, box[1] - dc)   # where the pusher stands
            landing = (box[0] + dr, box[1] + dc)  # where the box ends up
            if not is_floor(behind):
                continue
            if not is_floor(landing):
                continue
            if not is_dead_corner(landing):
                continue
            new_boxes = [landing if b == box else b for b in boxes]
            cf = dict(state)
            cf.pop("box_mapping", None)          # override with explicit positions
            cf["box_positions"] = [list(b) for b in new_boxes]
            cf["player_position"] = list(box)    # pusher advances into the vacated cell
            cf["_cf_push"] = {"box_from": list(box), "box_to": list(landing)}
            return cf
    return None


def _selftest(src_root: str) -> int:
    print(f"[SK1 self-test] loading first source init_state from {src_root} ...", flush=True)
    state = _first_source_init_state(src_root)
    print(f"  room_fixed shape = {np.asarray(state['room_fixed']).shape}", flush=True)
    print(f"  player = {state.get('player_position')}  boxes = {sorted(_boxes_from_state(state))}", flush=True)

    r1 = certify_detailed(state)
    print(
        f"  [case 1: real solved episode]      verdict={r1.verdict!r:14} "
        f"states_expanded={r1.states_expanded} states_discovered={r1.states_discovered}",
        flush=True,
    )

    cf = _find_corner_push_counterfactual(state)
    if cf is None:
        print("  [case 2] could not find a corner-push counterfactual on this board", flush=True)
        return 1
    print(
        f"  counterfactual push: box {cf['_cf_push']['box_from']} -> {cf['_cf_push']['box_to']} "
        f"(dead corner), player -> {cf['player_position']}",
        flush=True,
    )
    r2 = certify_detailed(cf)
    print(
        f"  [case 2: one cornering counterfactual] verdict={r2.verdict!r:14} "
        f"states_expanded={r2.states_expanded} states_discovered={r2.states_discovered}",
        flush=True,
    )

    ok = (r1.verdict == "solvable") and (r2.verdict == "unsolvable")
    print(f"[SK1 self-test] {'PASS' if ok else 'FAIL'}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Verify the Sokoban BFS solver on a source episode.")
    parser.add_argument("--src-root", required=True)
    args = parser.parse_args()
    sys.exit(_selftest(args.src_root))
