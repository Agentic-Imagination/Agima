#!/usr/bin/env python3
# Modified for Agima: portable paths and explicit dataset arguments.
"""Build one-or-two-candidate Sokoban CoT-inf rows from vanilla two-box mazes.

This is the single-stage successor to the Stage-1/Stage-2 forward pipeline.  It
reads the existing vanilla CoT-inf rows directly, keeps their initial states and
user turns, and constructs model-imagined candidate plans:

* the vanilla expert plan (known to solve the puzzle), and
* the best counterfactual plan that starts by pushing a different box when
  possible.

All legal first-push directions for the alternate box are searched.  A solved
branch always outranks a dead branch, and the box-level candidate is called a
deadlock only if every complete branch is certified dead.  If no reliable
distinct second first-push candidate exists, the vanilla row is
still retained.  Its CoT evaluates the expert plan, states which relatively
named box is the only box that can be pushed first, and commits that plan.

The counterfactual first push is reached without moving any other box.  From the
post-push state a complete BFS either finds the shortest continuation, exhausts
the state graph (a certified deadlock), or hits the state limit (unknown; never
called a deadlock).  If one candidate deadlocks the solved one is committed.  If
both solve the shorter plan is committed, with the vanilla expert winning exact
ties.  A candidate whose full solution exceeds the environment horizon is kept,
but its displayed CoT rollout stops after ``max_steps`` moves and is labeled
``horizon_reached`` rather than deadlock.

All model-visible box references are English relative descriptions based on the
initial board ("the left box", "the upper-right box", ...).  The format never
says "two boxes", "both boxes", or "the remaining box" and therefore remains
valid for N-box evaluation even though this training generator requires exactly
two boxes by default.

The loader contract matches the existing CoT-inf datasets: the user initial
frame is loss=False; every assistant rollout frame is loss=True; the loader
prepends the literal ``<think>`` to the first assistant text segment.
"""

from __future__ import annotations

import argparse
import ast
import glob
import gzip
import hashlib
import json
import math
import multiprocessing as mp
import os
import re
import shutil
import sys
import time
from collections import Counter, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple


_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parents[1]
for _p in (str(_REPO_ROOT), str(_THIS_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    from remap_sokoban_cot_chunk_k import THINK_CLOSE
    from sokoban_bfs_certify import load_env_from_state
    from sokoban_branch_render import render_branch_frame
except ImportError:  # pragma: no cover - import path shim
    from scripts.data.remap_sokoban_cot_chunk_k import THINK_CLOSE
    from scripts.data.sokoban_bfs_certify import load_env_from_state
    from scripts.data.sokoban_branch_render import render_branch_frame

from gymnasium.envs.sokoban.sokoban import (
    CHANGE_COORDINATES,
    DIRECTION_TO_PAYLOAD,
    LEGACY_ACTION_TO_KIND_DIRECTION,
    PAYLOAD_TO_DIRECTION,
    SokobanState,
)


SCHEMA_VERSION = "sokoban_compare2_relative_v5"
COT_ROW_TYPE = "one_or_two_candidate_relative_cot_singleturn_openloop"
STOP_ACTION = str(("stop", "stop"))
TARGET = 2
PUSH_IDS = frozenset({1, 2, 3, 4})
MOVE_IDS = frozenset({5, 6, 7, 8})
TOKEN_BUDGET = 50_000
IMG_TOKENS_EST = 840

DEFAULT_SRC = str(
    _REPO_ROOT
    / "hf_datasets"
    / "sokoban_easy_v8_cot_chunk_kinf_world_model_20260707_perseg"
)
DEFAULT_OUT = str(
    _REPO_ROOT
    / "hf_datasets"
    / "sokoban_easy_v8_cot_chunk_kinf_compare2_relative_v5_20260724"
)

OPENER = (
    "\nI will evaluate the available candidate plans before committing. "
    "Box descriptions refer to their positions on the initial board.\n"
)


@dataclass
class SearchResult:
    status: str  # solved | certified_deadlock | truncated
    actions: Optional[List[str]]
    states_expanded: int


@dataclass
class Candidate:
    first_box: Tuple[int, int]
    actions: List[str]
    frames: List[str]
    status: str  # solved | certified_deadlock | horizon_reached
    source: str  # vanilla_expert | alternate_first_box | alternate_same_box
    states_expanded: int = 0
    full_status: str = ""  # solved | certified_deadlock before horizon projection
    full_move_count: int = 0
    horizon_truncated: bool = False

    @property
    def move_count(self) -> int:
        return sum(a != STOP_ACTION for a in self.actions)


def _project_candidate_to_horizon(
    init_state: Dict[str, Any],
    first_box: Tuple[int, int],
    full_actions: Sequence[str],
    full_status: str,
    source: str,
    states_expanded: int,
    max_moves: int,
) -> Candidate:
    """Keep a candidate but stop its visible rollout at the task horizon."""
    actions = [str(a) for a in full_actions]
    full_move_count = sum(a != STOP_ACTION for a in actions)
    horizon_truncated = full_move_count > max_moves
    if horizon_truncated:
        # STOP is terminal and never appears inside a valid solution, so the
        # first max_moves entries are exactly the visible move prefix.
        actions = [a for a in actions if a != STOP_ACTION][:max_moves]
        status = "horizon_reached"
    else:
        status = full_status
    frames = _render_actions(init_state, actions)
    return Candidate(
        first_box=first_box,
        actions=actions,
        frames=frames,
        status=status,
        source=source,
        states_expanded=states_expanded,
        full_status=full_status,
        full_move_count=full_move_count,
        horizon_truncated=horizon_truncated,
    )


def _boxes_from_init_state(init_state: Dict[str, Any]) -> List[Tuple[int, int]]:
    raw = init_state.get("box_positions")
    if raw is None:
        mapping = init_state.get("box_mapping")
        if isinstance(mapping, dict):
            raw = list(mapping.values())
    if raw is None:
        raise ValueError("init_state has neither box_positions nor box_mapping")
    return sorted((int(p[0]), int(p[1])) for p in raw)


def _parse_action(text: str) -> Tuple[str, Any]:
    value = ast.literal_eval(str(text).strip())
    if not isinstance(value, tuple) or len(value) != 2:
        raise ValueError(f"bad action tuple: {text!r}")
    return str(value[0]), value[1]


def _extract_committed_actions(row: Dict[str, Any]) -> List[str]:
    messages = row.get("messages") or []
    if len(messages) != 2 or messages[1].get("role") != "assistant":
        raise ValueError("expected exactly one user and one assistant message")
    tail: Optional[str] = None
    for item in messages[1].get("content") or []:
        if item.get("type") == "text" and "</think>" in str(item.get("text", "")):
            tail = str(item["text"]).rsplit("</think>", 1)[1]
    if tail is None:
        raise ValueError("assistant content has no </think> committed-plan tail")
    actions: List[str] = []
    for line in tail.splitlines():
        line = line.strip()
        if not line:
            continue
        kind, payload = _parse_action(line)
        if kind not in {"move", "push", "stop"}:
            raise ValueError(f"unsupported committed action: {line!r}")
        actions.append(str((kind, payload)))
    if not actions or actions[-1] != STOP_ACTION:
        raise ValueError("committed vanilla plan is empty or lacks terminal stop")
    return actions


def _legacy_action_id(state: SokobanState, action: str) -> Optional[int]:
    kind, payload = _parse_action(action)
    if kind == "stop":
        if payload != "stop":
            raise ValueError(f"bad stop payload: {payload!r}")
        return None
    if kind != "move" or not isinstance(payload, int) or payload not in PAYLOAD_TO_DIRECTION:
        raise ValueError(f"compare2 generator requires move_only actions, got {action!r}")
    direction = PAYLOAD_TO_DIRECTION[payload]
    dr, dc = CHANGE_COORDINATES[direction]
    ahead = (state.player[0] + dr, state.player[1] + dc)
    return direction + 1 if ahead in set(state.boxes) else direction + 5


def _serialize_move_only(aid: int) -> str:
    _, direction = LEGACY_ACTION_TO_KIND_DIRECTION[int(aid)]
    return str(("move", int(DIRECTION_TO_PAYLOAD[direction])))


def _apply_action(env: Any, state: SokobanState, action: str) -> SokobanState:
    aid = _legacy_action_id(state, action)
    if aid is None:
        return state
    legal = set(env._enumerate_forward_action_ids(state))
    if aid not in legal:
        raise ValueError(
            f"illegal action {action!r} at player={state.player}, boxes={state.boxes}; "
            f"legal={sorted(legal)}"
        )
    return env._simulate_action_id(state, aid)


def _replay(
    init_state: Dict[str, Any], actions: Sequence[str]
) -> Tuple[List[SokobanState], Optional[Tuple[int, int]], bool]:
    env = load_env_from_state(init_state)
    state = env._current_state()
    snapshots: List[SokobanState] = []
    first_box: Optional[Tuple[int, int]] = None
    for action in actions:
        aid = _legacy_action_id(state, action)
        if aid is None:
            snapshots.append(state)
            continue
        if aid in PUSH_IDS and first_box is None:
            _, direction = LEGACY_ACTION_TO_KIND_DIRECTION[aid]
            dr, dc = CHANGE_COORDINATES[direction]
            first_box = (state.player[0] + dr, state.player[1] + dc)
        state = _apply_action(env, state, action)
        snapshots.append(state)
    return snapshots, first_box, bool(env._is_solved_state(state))


def _render_actions(init_state: Dict[str, Any], actions: Sequence[str]) -> List[str]:
    snapshots, _, _ = _replay(init_state, actions)
    room = [[int(v) for v in row] for row in init_state["room_fixed"]]
    return [
        render_branch_frame(s.player, s.boxes, room, action_mode="move_only")
        for s in snapshots
    ]


def _shortest_solution(env: Any, start: SokobanState, state_limit: int) -> SearchResult:
    if env._is_solved_state(start):
        return SearchResult("solved", [STOP_ACTION], 0)
    q: deque[SokobanState] = deque([start])
    parents: Dict[SokobanState, Tuple[Optional[SokobanState], Optional[int]]] = {
        start: (None, None)
    }
    expanded = 0
    while q:
        state = q.popleft()
        expanded += 1
        if expanded > max(1, int(state_limit)):
            return SearchResult("truncated", None, expanded - 1)
        for aid in env._enumerate_forward_action_ids(state):
            nxt = env._simulate_action_id(state, aid)
            if nxt in parents:
                continue
            parents[nxt] = (state, int(aid))
            if env._is_solved_state(nxt):
                ids: List[int] = []
                cursor = nxt
                while parents[cursor][0] is not None:
                    prev, used = parents[cursor]
                    assert prev is not None and used is not None
                    ids.append(used)
                    cursor = prev
                ids.reverse()
                actions = [_serialize_move_only(aid0) for aid0 in ids]
                actions.append(STOP_ACTION)
                return SearchResult("solved", actions, expanded)
            q.append(nxt)
    return SearchResult("certified_deadlock", None, expanded)


def _walk_paths_without_pushes(
    env: Any, start: SokobanState
) -> Tuple[Dict[Tuple[int, int], Tuple[Optional[Tuple[int, int]], Optional[int]]], set]:
    parents: Dict[Tuple[int, int], Tuple[Optional[Tuple[int, int]], Optional[int]]] = {
        start.player: (None, None)
    }
    q: deque[Tuple[int, int]] = deque([start.player])
    boxes = set(start.boxes)
    while q:
        player = q.popleft()
        for direction, (dr, dc) in CHANGE_COORDINATES.items():
            nxt = (player[0] + dr, player[1] + dc)
            if nxt in parents or nxt in boxes:
                continue
            if not env._is_floor(nxt, boxes=start.boxes):
                continue
            parents[nxt] = (player, int(DIRECTION_TO_PAYLOAD[direction]))
            q.append(nxt)
    return parents, boxes


def _reconstruct_walk(
    parents: Dict[Tuple[int, int], Tuple[Optional[Tuple[int, int]], Optional[int]]],
    endpoint: Tuple[int, int],
) -> List[str]:
    payloads: List[int] = []
    cur = endpoint
    while parents[cur][0] is not None:
        prev, payload = parents[cur]
        assert prev is not None and payload is not None
        payloads.append(payload)
        cur = prev
    payloads.reverse()
    return [str(("move", int(p))) for p in payloads]


def _first_push_prefixes(
    init_state: Dict[str, Any], box: Tuple[int, int]
) -> List[Tuple[List[str], SokobanState, Tuple[int, int, Tuple[str, ...]]]]:
    env = load_env_from_state(init_state)
    start = env._current_state()
    if box not in set(start.boxes):
        return []
    parents, boxes = _walk_paths_without_pushes(env, start)
    targets = [
        (r, c)
        for r, row in enumerate(init_state["room_fixed"])
        for c, value in enumerate(row)
        if int(value) == TARGET
    ]
    out: List[Tuple[List[str], SokobanState, Tuple[int, int, Tuple[str, ...]]]] = []
    for direction, (dr, dc) in CHANGE_COORDINATES.items():
        behind = (box[0] - dr, box[1] - dc)
        landing = (box[0] + dr, box[1] + dc)
        if behind not in parents or landing in boxes:
            continue
        if not env._is_floor(landing, boxes=start.boxes):
            continue
        walk = _reconstruct_walk(parents, behind)
        push = str(("move", int(DIRECTION_TO_PAYLOAD[direction])))
        prefix = walk + [push]
        state = start
        for action in prefix:
            state = _apply_action(env, state, action)
        target_dist = min(
            (abs(landing[0] - t[0]) + abs(landing[1] - t[1]) for t in targets),
            default=0,
        )
        # A plausible push (toward a target) first, then shortest player walk,
        # then lexical action tie-break for exact determinism.
        score = (int(target_dist), len(prefix), tuple(prefix))
        out.append((prefix, state, score))
    out.sort(key=lambda item: item[2])
    return out


def _alternate_candidates_for_box(
    init_state: Dict[str, Any],
    box: Tuple[int, int],
    state_limit: int,
    max_moves: int,
    source: str,
) -> Tuple[List[Candidate], int]:
    env = load_env_from_state(init_state)
    candidates: List[Candidate] = []
    truncated = 0
    for prefix, successor, _ in _first_push_prefixes(init_state, box):
        solved = _shortest_solution(env, successor, state_limit)
        if solved.status == "truncated":
            truncated += 1
            continue
        if solved.status == "solved":
            assert solved.actions is not None
            actions = list(prefix) + list(solved.actions)
        else:
            actions = list(prefix)  # ends exactly at the certified-dead state
        candidates.append(
            _project_candidate_to_horizon(
                init_state=init_state,
                first_box=box,
                full_actions=actions,
                full_status=solved.status,
                source=source,
                states_expanded=solved.states_expanded,
                max_moves=max_moves,
            )
        )
    return candidates, truncated


def _pick_alternate(
    candidates: Sequence[Candidate],
    expert_actions: Sequence[str],
    search_truncated: int = 0,
) -> Optional[Candidate]:
    """Choose one representative for a box, preferring any known solution.

    A dead branch may represent the box only when every searched branch is dead
    and no branch search was truncated.  This prevents one bad push direction
    from masking another direction that solves with the same first box.
    """
    distinct = [c for c in candidates if list(c.actions) != list(expert_actions)]
    solved = [c for c in distinct if c.status == "solved"]
    if solved:
        return min(solved, key=lambda c: (c.move_count, tuple(c.actions)))
    horizon = [c for c in distinct if c.status == "horizon_reached"]
    if horizon:
        return min(horizon, key=lambda c: (c.full_move_count, tuple(c.actions)))
    dead = [c for c in distinct if c.status == "certified_deadlock"]
    if dead and int(search_truncated) == 0:
        return min(dead, key=lambda c: (c.move_count, tuple(c.actions)))
    return None


_ORDINALS = {
    0: "leftmost",
    1: "second from the left",
    2: "third from the left",
    3: "fourth from the left",
}


def relative_box_names(boxes: Sequence[Tuple[int, int]]) -> Dict[Tuple[int, int], str]:
    """Unique English names based only on the initial arrangement, never coordinates."""
    bs = sorted((int(b[0]), int(b[1])) for b in boxes)
    if len(bs) < 2:
        raise ValueError("relative box naming requires at least two boxes")
    if len(bs) == 2:
        a, b = bs
        if a[0] == b[0]:
            return {a: "the left box" if a[1] < b[1] else "the right box",
                    b: "the right box" if a[1] < b[1] else "the left box"}
        if a[1] == b[1]:
            return {a: "the upper box", b: "the lower box"}
        a_h = "left" if a[1] < b[1] else "right"
        b_h = "right" if a_h == "left" else "left"
        return {a: f"the upper-{a_h} box", b: f"the lower-{b_h} box"}

    # N-box fallback: name columns relative to one another, then disambiguate
    # boxes sharing a column vertically. This supports evaluation formatting for
    # larger boards while two-box training keeps the concise vocabulary above.
    columns = sorted({b[1] for b in bs})
    by_col: Dict[int, List[Tuple[int, int]]] = {c: [] for c in columns}
    for b in bs:
        by_col[b[1]].append(b)
    names: Dict[Tuple[int, int], str] = {}
    for ci, col in enumerate(columns):
        group = sorted(by_col[col])
        if len(columns) == 1:
            col_phrase = "central column"
        elif ci == 0:
            col_phrase = "leftmost column"
        elif ci == len(columns) - 1:
            col_phrase = "rightmost column"
        elif len(columns) == 3:
            col_phrase = "middle column"
        else:
            col_phrase = f"{_ORDINALS.get(ci, str(ci + 1) + 'th from the left')} column"
        if len(group) == 1:
            box = group[0]
            if ci == 0 and box[0] == min(b[0] for b in bs):
                names[box] = "the upper-left box"
            elif ci == 0 and box[0] == max(b[0] for b in bs):
                names[box] = "the lower-left box"
            elif ci == len(columns) - 1 and box[0] == min(b[0] for b in bs):
                names[box] = "the upper-right box"
            elif ci == len(columns) - 1 and box[0] == max(b[0] for b in bs):
                names[box] = "the lower-right box"
            else:
                names[box] = f"the box in the {col_phrase}"
            continue
        for ri, box in enumerate(group):
            if ri == 0:
                vertical = "upper"
            elif ri == len(group) - 1:
                vertical = "lower"
            else:
                vertical = f"{ri + 1}th from the top"
            names[box] = f"the {vertical} box in the {col_phrase}"
    if len(set(names.values())) != len(bs):
        raise ValueError(f"relative names are not unique: {names}")
    return names


def _estimate_tokens(messages: Sequence[Dict[str, Any]]) -> int:
    n_img = n_chars = 0
    for message in messages:
        for item in message.get("content") or []:
            if item.get("type") == "image":
                n_img += 1
            else:
                n_chars += len(str(item.get("text", "")))
    return n_img * IMG_TOKENS_EST + (n_chars + 2) // 3 + 128


def _candidate_result(candidate: Candidate) -> str:
    if candidate.status == "certified_deadlock":
        return (
            "\nResult: this plan reaches a certified deadlock; the puzzle cannot be "
            "solved from this state."
        )
    if candidate.status == "horizon_reached":
        return (
            f"\nResult: this plan has not solved the puzzle after "
            f"{candidate.move_count} moves. I stop evaluating this candidate at "
            "the task horizon."
        )
    return (
        f"\nResult: this plan succeeds; all boxes reach their targets in "
        f"{candidate.move_count} moves."
    )


def _selection(candidates: Sequence[Candidate]) -> Tuple[int, str]:
    solved = [i for i, c in enumerate(candidates) if c.status == "solved"]
    if not solved:
        raise ValueError("candidate pair has no solved plan")
    if len(solved) == 1:
        other = candidates[1 - solved[0]]
        if other.status == "certified_deadlock":
            return solved[0], "avoid_deadlock"
        if other.status == "horizon_reached":
            return solved[0], "solve_within_horizon"
        raise ValueError(f"unsupported unsolved candidate status: {other.status}")
    a, b = solved[0], solved[1]
    if candidates[a].move_count < candidates[b].move_count:
        return a, "shortest"
    if candidates[b].move_count < candidates[a].move_count:
        return b, "shortest"
    expert = next((i for i, c in enumerate(candidates) if c.source == "vanilla_expert"), a)
    return expert, "expert_tie_break"


def _comparison_text(
    displayed: Sequence[Candidate], displayed_chosen: int, reason: str
) -> str:
    winner = displayed_chosen + 1
    loser = 2 if winner == 1 else 1
    if reason == "avoid_deadlock":
        return (
            f"\nComparison: Candidate {loser} deadlocks, while Candidate {winner} solves "
            f"the puzzle. I will commit Candidate {winner}."
        )
    if reason == "solve_within_horizon":
        return (
            f"\nComparison: Candidate {loser} reaches the task horizon without solving, "
            f"while Candidate {winner} solves within the allowed moves. I will commit "
            f"Candidate {winner}."
        )
    if reason == "shortest":
        return (
            f"\nComparison: both candidates solve the puzzle, but Candidate {winner} is "
            f"shorter, requiring {displayed[displayed_chosen].move_count} moves instead of "
            f"{displayed[1 - displayed_chosen].move_count}. I will commit Candidate {winner}."
        )
    return (
        f"\nComparison: both candidates solve the puzzle in the same number of moves. "
        f"I will commit Candidate {winner} using the deterministic expert tie-break."
    )


def _messages_for_pair(
    compact: Dict[str, Any], candidates: Sequence[Candidate], chosen: int, reason: str
) -> Tuple[List[Dict[str, Any]], List[Candidate], int]:
    key = f"{compact.get('init_state_hash', '')}|{SCHEMA_VERSION}"
    swap = hashlib.sha256(key.encode("utf-8")).digest()[0] % 2 == 1
    order = [1, 0] if swap else [0, 1]
    displayed = [candidates[i] for i in order]
    displayed_chosen = order.index(chosen)
    names = relative_box_names(_boxes_from_init_state(compact["init_state"]))

    content: List[Dict[str, Any]] = []
    pending = OPENER
    for i, candidate in enumerate(displayed):
        pending += f"Candidate {i + 1}: start by pushing {names[candidate.first_box]}."
        for action, frame in zip(candidate.actions, candidate.frames):
            content.append({"type": "text", "text": pending + "\n" + action})
            content.append({"type": "image", "image": frame, "loss": True})
            pending = ""
        pending = _candidate_result(candidate) + "\n"
    pending += _comparison_text(displayed, displayed_chosen, reason)
    pending += "\n" + THINK_CLOSE + "\n</think>\n"
    pending += "\n".join(candidates[chosen].actions)
    content.append({"type": "text", "text": pending})
    messages = [compact["_user_message"], {"role": "assistant", "content": content}]
    return messages, displayed, displayed_chosen


def _messages_for_single(
    compact: Dict[str, Any], candidate: Candidate, availability_text: str
) -> Tuple[List[Dict[str, Any]], List[Candidate], int]:
    names = relative_box_names(_boxes_from_init_state(compact["init_state"]))
    content: List[Dict[str, Any]] = []
    pending = (
        OPENER
        + f"Candidate 1: start by pushing {names[candidate.first_box]}."
    )
    for action, frame in zip(candidate.actions, candidate.frames):
        content.append({"type": "text", "text": pending + "\n" + action})
        content.append({"type": "image", "image": frame, "loss": True})
        pending = ""
    pending = _candidate_result(candidate) + "\n"
    pending += "Availability check: " + availability_text
    pending += "\n" + THINK_CLOSE + "\n</think>\n"
    pending += "\n".join(candidate.actions)
    content.append({"type": "text", "text": pending})
    messages = [compact["_user_message"], {"role": "assistant", "content": content}]
    return messages, [candidate], 0


def _compact_input(row: Dict[str, Any]) -> Dict[str, Any]:
    messages = row.get("messages") or []
    if len(messages) != 2 or messages[0].get("role") != "user":
        raise ValueError("bad vanilla messages shape")
    compact = {k: v for k, v in row.items() if k != "messages"}
    compact["_user_message"] = messages[0]
    compact["_expert_actions"] = _extract_committed_actions(row)
    return compact


def transform_row(compact: Dict[str, Any], state_limit: int = 200_000) -> Dict[str, Any]:
    init_state = compact.get("init_state") or {}
    boxes = _boxes_from_init_state(init_state)
    if len(boxes) != 2:
        raise ValueError(f"training generator requires exactly 2 boxes, got {len(boxes)}")
    env_kwargs = compact.get("env_kwargs") or {}
    if str(env_kwargs.get("action_mode", "move_only")) != "move_only":
        raise ValueError("training generator requires action_mode=move_only")
    max_moves = int(env_kwargs.get("max_steps", compact.get("max_steps", 0)))
    if max_moves <= 0:
        raise ValueError(f"row has invalid max_steps horizon: {max_moves}")

    expert_actions = [str(a) for a in compact["_expert_actions"]]
    expert_frames = _render_actions(init_state, expert_actions)
    _, expert_first, expert_solved = _replay(init_state, expert_actions)
    if not expert_solved or expert_first not in set(boxes):
        raise ValueError("vanilla committed plan did not replay to a solved board")
    expert_move_count = sum(action != STOP_ACTION for action in expert_actions)
    if expert_move_count > max_moves:
        raise ValueError(
            f"vanilla expert exceeds max_steps: moves={expert_move_count} horizon={max_moves}"
        )
    expert = Candidate(
        first_box=expert_first,
        actions=expert_actions,
        frames=expert_frames,
        status="solved",
        source="vanilla_expert",
        full_status="solved",
        full_move_count=expert_move_count,
        horizon_truncated=False,
    )

    other_boxes = [b for b in boxes if b != expert_first]
    alternatives: List[Candidate] = []
    n_truncated = 0
    for box in other_boxes:
        found, truncated = _alternate_candidates_for_box(
            init_state, box, state_limit, max_moves, "alternate_first_box"
        )
        alternatives.extend(found)
        n_truncated += truncated
    other_box_first_pushable = bool(alternatives) or n_truncated > 0
    branch_counts = {
        "solved_within_horizon": sum(c.status == "solved" for c in alternatives),
        "solved_beyond_horizon": sum(
            c.status == "horizon_reached" for c in alternatives
        ),
        "certified_deadlock": sum(
            c.status == "certified_deadlock" for c in alternatives
        ),
        "search_truncated": int(n_truncated),
    }
    alternate = _pick_alternate(
        alternatives, expert_actions, search_truncated=n_truncated
    )
    if alternate is None:
        candidates = [expert]
        chosen = 0
        if other_box_first_pushable:
            reason = "no_reliable_second_candidate"
            availability_text = (
                "I could not establish a reliable distinct second candidate within "
                "the search limit. I will commit Candidate 1."
            )
        else:
            reason = "only_pushable_box_first"
            names = relative_box_names(boxes)
            availability_text = (
                f"It seems I can only push {names[expert.first_box]} first from this "
                "initial state. I will commit Candidate 1."
            )
        messages, displayed, displayed_chosen = _messages_for_single(
            compact, expert, availability_text
        )
    else:
        candidates = [expert, alternate]
        chosen, reason = _selection(candidates)
        messages, displayed, displayed_chosen = _messages_for_pair(
            compact, candidates, chosen, reason
        )
    token_estimate = _estimate_tokens(messages)
    if token_estimate >= TOKEN_BUDGET:
        raise RuntimeError(f"over_token_budget:{token_estimate}")

    out = {k: v for k, v in compact.items() if not k.startswith("_")}
    out["cot_schema_version"] = out.get("schema_version")
    out["messages"] = messages
    out["chunk_size"] = "inf"
    out["n_steps"] = len(candidates[chosen].actions)
    out["n_chunks"] = 1
    out["n_imagined_frames"] = sum(len(c.frames) for c in candidates)
    out["n_feedback_turns"] = 0
    out["n_feedback_frames"] = 0
    out["has_plan_comparison"] = len(candidates) == 2
    out["n_boxes"] = len(boxes)
    out["n_candidates"] = len(candidates)
    out["candidate_horizon_moves"] = int(max_moves)
    out["selection_reason"] = reason
    out["chosen_candidate"] = displayed_chosen + 1
    out["candidate_order_shuffled"] = (
        len(candidates) == 2 and displayed[0] is not candidates[0]
    )
    out["single_candidate_fallback"] = len(candidates) == 1
    out["alternate_box_branch_counts"] = branch_counts
    names = relative_box_names(boxes)
    out["candidates"] = [
        {
            "candidate": i + 1,
            "relative_box_name": names[c.first_box],
            "status": c.status,
            "full_status": c.full_status,
            "source": c.source,
            "plan_len": len(c.actions),
            "move_count": c.move_count,
            "full_move_count": c.full_move_count,
            "horizon_truncated": c.horizon_truncated,
            "states_expanded": c.states_expanded,
        }
        for i, c in enumerate(displayed)
    ]
    out["compare2_search_truncated"] = int(n_truncated)
    out["compare2_horizon_truncated_candidates"] = sum(
        int(c.horizon_truncated) for c in candidates
    )
    out["estimated_tokens"] = int(token_estimate)
    out["schema_version"] = SCHEMA_VERSION
    out["cot_row_type"] = COT_ROW_TYPE
    return out


def _validate_output(row: Dict[str, Any]) -> None:
    n_candidates = int(row.get("n_candidates", 0))
    if row.get("n_boxes") != 2 or n_candidates not in {1, 2}:
        raise AssertionError("wrong two-box/one-or-two-candidate metadata")
    if len(row.get("candidates") or []) != n_candidates:
        raise AssertionError("candidate metadata length mismatch")
    pair_reasons = {
        "avoid_deadlock", "solve_within_horizon", "shortest", "expert_tie_break"
    }
    single_reasons = {"only_pushable_box_first", "no_reliable_second_candidate"}
    reason = row.get("selection_reason")
    if reason not in pair_reasons | single_reasons:
        raise AssertionError("bad selection_reason")
    if n_candidates == 1:
        if reason not in single_reasons or row.get("chosen_candidate") != 1:
            raise AssertionError("bad single-candidate selection")
        if row["candidates"][0].get("status") != "solved":
            raise AssertionError("single candidate must be the solved vanilla expert")
    elif reason not in pair_reasons:
        raise AssertionError("two-candidate row has a single-candidate reason")
    branch_counts = row.get("alternate_box_branch_counts") or {}
    expected_branch_keys = {
        "solved_within_horizon",
        "solved_beyond_horizon",
        "certified_deadlock",
        "search_truncated",
    }
    if set(branch_counts) != expected_branch_keys or any(
        int(branch_counts[key]) < 0 for key in expected_branch_keys
    ):
        raise AssertionError("bad alternate_box_branch_counts")
    if n_candidates == 2:
        alternate = next(
            (c for c in row["candidates"] if c.get("source") != "vanilla_expert"),
            None,
        )
        if alternate is None or alternate.get("source") != "alternate_first_box":
            raise AssertionError("two-candidate row lacks alternate-first-box candidate")
        alt_status = alternate.get("status")
        if alt_status == "solved" and int(branch_counts["solved_within_horizon"]) <= 0:
            raise AssertionError("solved representative lacks a solved branch")
        if alt_status == "horizon_reached":
            if int(branch_counts["solved_within_horizon"]) != 0:
                raise AssertionError("horizon chosen despite an in-horizon solution")
            if int(branch_counts["solved_beyond_horizon"]) <= 0:
                raise AssertionError("horizon representative lacks a long solution")
        if alt_status == "certified_deadlock":
            if (
                int(branch_counts["solved_within_horizon"]) != 0
                or int(branch_counts["solved_beyond_horizon"]) != 0
                or int(branch_counts["search_truncated"]) != 0
                or int(branch_counts["certified_deadlock"]) <= 0
            ):
                raise AssertionError(
                    "dead representative is not an all-branches-certified box deadlock"
                )
    elif reason == "only_pushable_box_first":
        if any(int(value) != 0 for value in branch_counts.values()):
            raise AssertionError("only-pushable-box row has an alternate branch")
    elif reason == "no_reliable_second_candidate":
        if (
            int(branch_counts["search_truncated"]) <= 0
            or int(branch_counts["solved_within_horizon"]) != 0
            or int(branch_counts["solved_beyond_horizon"]) != 0
        ):
            raise AssertionError("bad search-limited single-candidate fallback")
    horizon = int(row.get("candidate_horizon_moves", 0))
    if horizon <= 0 or any(int(c.get("move_count", horizon + 1)) > horizon for c in row["candidates"]):
        raise AssertionError("candidate exceeds row horizon")
    for candidate in row["candidates"]:
        was_truncated = bool(candidate.get("horizon_truncated", False))
        if was_truncated:
            if candidate.get("status") != "horizon_reached":
                raise AssertionError("horizon-truncated candidate has wrong status")
            if int(candidate.get("move_count", -1)) != horizon:
                raise AssertionError("horizon-truncated candidate does not stop at horizon")
            if int(candidate.get("full_move_count", 0)) <= horizon:
                raise AssertionError("horizon-truncated candidate lacks longer full path")
        elif candidate.get("status") == "horizon_reached":
            raise AssertionError("unmarked horizon-reached candidate")
    if int(row.get("estimated_tokens", TOKEN_BUDGET)) >= TOKEN_BUDGET:
        raise AssertionError("row exceeds token budget")
    if len(row.get("messages") or []) != 2:
        raise AssertionError("bad message count")
    assistant = row["messages"][1]["content"]
    texts = "".join(str(x.get("text", "")) for x in assistant if x.get("type") == "text")
    if texts.count("Candidate 1:") != 1:
        raise AssertionError("Candidate 1 header is not exactly once")
    if texts.count("Candidate 2:") != (1 if n_candidates == 2 else 0):
        raise AssertionError("Candidate 2 header count does not match metadata")
    if n_candidates == 1:
        if "Availability check:" not in texts or "I will commit Candidate 1." not in texts:
            raise AssertionError("single-candidate availability statement missing")
        if reason == "only_pushable_box_first" and not re.search(
            r"It seems I can only push the .+ box first from this initial state\.",
            texts,
        ):
            raise AssertionError("relative only-pushable-box statement missing")
    if re.search(r"\b(?:near|far) box\b|Two boxes|Both boxes|remaining box", texts, re.I):
        raise AssertionError("two-box-specific or near/far language leaked")
    if re.search(r"box at \(\d+\s*,\s*\d+\)", texts):
        raise AssertionError("absolute box coordinates leaked")
    if re.search(r"[\u4e00-\u9fff]", texts):
        raise AssertionError("non-English CJK text leaked")
    image_losses = [x.get("loss") for x in assistant if x.get("type") == "image"]
    if not image_losses or any(v is not True for v in image_losses):
        raise AssertionError("assistant frames must all have loss=True")
    user_images = [
        x for x in row["messages"][0].get("content") or [] if x.get("type") == "image"
    ]
    if len(user_images) != 1 or user_images[0].get("loss") is not False:
        raise AssertionError("user must contain exactly one loss=False initial frame")


def _worker_task(task: Tuple[int, List[Dict[str, Any]], str, int]) -> Dict[str, Any]:
    idx, rows, parts_dir, state_limit = task
    final_path = os.path.join(parts_dir, f"part_{idx:06d}.jsonl.gz")
    tmp_path = final_path + ".inprogress"
    stats: Counter = Counter()
    errors: List[str] = []
    token_max = 0
    with gzip.open(tmp_path, "wt", encoding="utf-8", compresslevel=6) as fout:
        for local_i, compact in enumerate(rows):
            try:
                out = transform_row(compact, state_limit=state_limit)
                _validate_output(out)
            except Exception as exc:
                msg = str(exc)
                if msg.startswith("no_second_candidate"):
                    key = "drop:no_second_candidate"
                elif msg.startswith("over_token_budget"):
                    key = "drop:over_token_budget"
                else:
                    key = f"drop:error:{type(exc).__name__}"
                stats[key] += 1
                if len(errors) < 3:
                    errors.append(f"task={idx} row={local_i}: {type(exc).__name__}: {exc}")
                continue
            fout.write(json.dumps(out, ensure_ascii=True) + "\n")
            stats["written"] += 1
            stats[f"selection:{out['selection_reason']}"] += 1
            stats["candidate:horizon_truncated"] += int(
                out["compare2_horizon_truncated_candidates"]
            )
            branch_counts = out["alternate_box_branch_counts"]
            for key, value in branch_counts.items():
                stats[f"alternate_branch:{key}"] += int(value)
            n_solved_branches = int(branch_counts["solved_within_horizon"]) + int(
                branch_counts["solved_beyond_horizon"]
            )
            n_dead_branches = int(branch_counts["certified_deadlock"])
            n_truncated_branches = int(branch_counts["search_truncated"])
            if n_solved_branches:
                stats["row:alternate_box_has_solved_branch"] += 1
            if n_solved_branches and n_dead_branches:
                stats["row:alternate_box_mixed_solved_and_dead"] += 1
            if n_dead_branches and not n_solved_branches and not n_truncated_branches:
                stats["row:alternate_box_all_branches_dead"] += 1
            if not n_solved_branches and not n_dead_branches and not n_truncated_branches:
                stats["row:alternate_box_no_legal_first_push"] += 1
            if n_truncated_branches:
                stats["row:alternate_box_search_truncated"] += 1
            statuses = tuple(sorted(c["status"] for c in out["candidates"]))
            stats["pair:" + "+".join(statuses)] += 1
            token_max = max(token_max, int(out["estimated_tokens"]))
    os.replace(tmp_path, final_path)
    return {
        "idx": idx,
        "part": final_path,
        "input": len(rows),
        "stats": dict(stats),
        "errors": errors,
        "token_max": token_max,
    }


def _source_shards(src_root: str, subdir: str) -> List[str]:
    return sorted(glob.glob(os.path.join(src_root, subdir, "*.jsonl.gz")))


def _iter_compact_rows(shards: Sequence[str], max_rows: int) -> Iterator[Dict[str, Any]]:
    seen = 0
    for shard in shards:
        with gzip.open(shard, "rt", encoding="utf-8") as fin:
            for line in fin:
                if not line.strip():
                    continue
                yield _compact_input(json.loads(line))
                seen += 1
                if max_rows > 0 and seen >= max_rows:
                    return


def _task_stream(
    shards: Sequence[str], max_rows: int, rows_per_task: int, parts_dir: str, state_limit: int
) -> Iterator[Tuple[int, List[Dict[str, Any]], str, int]]:
    batch: List[Dict[str, Any]] = []
    idx = 0
    for compact in _iter_compact_rows(shards, max_rows):
        batch.append(compact)
        if len(batch) >= rows_per_task:
            yield idx, batch, parts_dir, state_limit
            idx += 1
            batch = []
    if batch:
        yield idx, batch, parts_dir, state_limit


def _concat_parts(results: Sequence[Dict[str, Any]], training_dir: Path, n_shards: int) -> List[Dict[str, Any]]:
    ordered = sorted(results, key=lambda x: int(x["idx"]))
    total = sum(int(x["stats"].get("written", 0)) for x in ordered)
    if total <= 0:
        raise RuntimeError("no output rows were written")
    target = max(1, math.ceil(total / max(1, int(n_shards))))
    manifests: List[Dict[str, Any]] = []
    shard_idx = 0
    shard_rows = 0
    out_path = training_dir / f"batch_{shard_idx:06d}.jsonl.gz"
    tmp_path = Path(str(out_path) + ".tmp")
    fout = tmp_path.open("wb")
    try:
        for result in ordered:
            part_rows = int(result["stats"].get("written", 0))
            if shard_rows > 0 and shard_rows + part_rows > target and shard_idx + 1 < n_shards:
                fout.close()
                os.replace(tmp_path, out_path)
                manifests.append({"path": out_path.name, "rows": shard_rows})
                shard_idx += 1
                shard_rows = 0
                out_path = training_dir / f"batch_{shard_idx:06d}.jsonl.gz"
                tmp_path = Path(str(out_path) + ".tmp")
                fout = tmp_path.open("wb")
            with open(result["part"], "rb") as fin:
                shutil.copyfileobj(fin, fout, length=16 * 1024 * 1024)
            shard_rows += part_rows
        fout.close()
        os.replace(tmp_path, out_path)
        manifests.append({"path": out_path.name, "rows": shard_rows})
    finally:
        if not fout.closed:
            fout.close()
    return manifests


def _self_test(args: argparse.Namespace) -> int:
    shards = _source_shards(args.src_root, args.src_subdir)
    if not shards:
        raise FileNotFoundError(f"no input shards under {args.src_root}/{args.src_subdir}")
    # Formatter-only checks exercise N-box descriptions without producing N-box
    # training examples.
    name_cases = [
        [(2, 1), (2, 4)],
        [(1, 2), (4, 2)],
        [(1, 1), (4, 4)],
        [(1, 1), (3, 1), (4, 4)],
        [(1, 1), (3, 3), (4, 1)],
    ]
    for case in name_cases:
        names = relative_box_names(case)
        if len(names) != len(case) or len(set(names.values())) != len(case):
            raise AssertionError(f"relative-name formatter failed: {case} -> {names}")
        if any(re.search(r"\(\d+\s*,\s*\d+\)", name) for name in names.values()):
            raise AssertionError(f"coordinate leaked from formatter: {names}")
    print("[compare2 self-test] N-box relative-name formatter: PASS", flush=True)

    checked = Counter()
    failures: List[str] = []
    t0 = time.perf_counter()
    for i, compact in enumerate(_iter_compact_rows(shards, args.self_test)):
        try:
            out = transform_row(compact, state_limit=args.state_limit)
            _validate_output(out)
        except Exception as exc:
            checked[f"drop:{type(exc).__name__}"] += 1
            failures.append(f"row {i}: {type(exc).__name__}: {exc}")
            continue
        checked["written"] += 1
        checked[f"selection:{out['selection_reason']}"] += 1
        pair = "+".join(sorted(c["status"] for c in out["candidates"]))
        checked[f"pair:{pair}"] += 1
        if checked["written"] <= 3:
            print(
                f"  row {i}: boxes={[c['relative_box_name'] for c in out['candidates']]} "
                f"statuses={[c['status'] for c in out['candidates']]} "
                f"moves={[c['move_count'] for c in out['candidates']]} "
                f"chosen={out['chosen_candidate']} reason={out['selection_reason']} "
                f"tokens~{out['estimated_tokens']}",
                flush=True,
            )
    elapsed = time.perf_counter() - t0
    print(
        f"[compare2 self-test] checked={args.self_test} elapsed={elapsed:.2f}s "
        f"stats={dict(checked)}",
        flush=True,
    )
    for failure in failures[:10]:
        print("  " + failure, flush=True)
    if checked["written"] <= 0:
        return 1
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-root", required=True)
    ap.add_argument("--src-subdir", default="training")
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--max-rows", type=int, default=0, help="0 means all source rows")
    ap.add_argument("--workers", type=int, default=min(96, os.cpu_count() or 8))
    ap.add_argument("--rows-per-task", type=int, default=32)
    ap.add_argument("--n-shards", type=int, default=8)
    ap.add_argument("--state-limit", type=int, default=200_000)
    ap.add_argument("--self-test", type=int, default=0, metavar="N", help="check N rows; write nothing")
    ap.add_argument("--keep-parts", action="store_true")
    return ap


def main() -> int:
    args = build_parser().parse_args()
    if args.self_test > 0:
        return _self_test(args)
    shards = _source_shards(args.src_root, args.src_subdir)
    if not shards:
        raise FileNotFoundError(f"no input shards under {args.src_root}/{args.src_subdir}")
    out_root = Path(args.out_root).resolve()
    training_dir = out_root / "training"
    metadata_dir = out_root / "metadata"
    parts_dir = out_root / ".parts"
    if training_dir.exists() and any(training_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {training_dir}")
    training_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)
    parts_dir.mkdir(parents=True, exist_ok=False)

    print(
        f"[compare2] src_shards={len(shards)} workers={args.workers} "
        f"rows_per_task={args.rows_per_task} max_rows={args.max_rows or 'ALL'} "
        f"state_limit={args.state_limit} out={out_root}",
        flush=True,
    )
    t0 = time.perf_counter()
    results: List[Dict[str, Any]] = []
    aggregate: Counter = Counter()
    token_max = 0
    errors: List[str] = []
    with mp.Pool(max(1, int(args.workers))) as pool:
        iterator = pool.imap_unordered(
            _worker_task,
            _task_stream(
                shards,
                max(0, int(args.max_rows)),
                max(1, int(args.rows_per_task)),
                str(parts_dir),
                max(1, int(args.state_limit)),
            ),
            chunksize=1,
        )
        for result in iterator:
            results.append(result)
            aggregate.update(result["stats"])
            token_max = max(token_max, int(result["token_max"]))
            errors.extend(result["errors"])
            if len(results) % 10 == 0:
                elapsed = time.perf_counter() - t0
                processed = sum(int(x["input"]) for x in results)
                rate = processed / max(elapsed, 1e-9)
                print(
                    f"[compare2] tasks={len(results)} input={processed} "
                    f"written={aggregate['written']} rate={rate:.2f} rows/s",
                    flush=True,
                )

    elapsed = time.perf_counter() - t0
    input_rows = sum(int(x["input"]) for x in results)
    written_rows = int(aggregate["written"])
    if written_rows != input_rows:
        raise RuntimeError(
            f"full coverage violated: input_rows={input_rows} written_rows={written_rows} "
            f"stats={dict(aggregate)}"
        )
    manifests = _concat_parts(results, training_dir, max(1, int(args.n_shards)))
    summary = {
        "schema_version": SCHEMA_VERSION,
        "cot_row_type": COT_ROW_TYPE,
        "source_root": str(Path(args.src_root).resolve()),
        "source_subdir": args.src_subdir,
        "source_shards": shards,
        "input_rows": input_rows,
        "written_rows": written_rows,
        "stats": dict(aggregate),
        "elapsed_seconds": elapsed,
        "throughput_input_rows_per_second": input_rows / max(elapsed, 1e-9),
        "state_limit": args.state_limit,
        "token_budget": TOKEN_BUDGET,
        "candidate_horizon_policy": (
            "keep over-horizon candidates but stop their displayed CoT at env max_steps"
        ),
        "row_retention_policy": (
            "retain every vanilla row; when no distinct second first-push candidate exists, "
            "evaluate and commit the vanilla candidate with an availability statement"
        ),
        "alternate_box_policy": (
            "choose the shortest solved branch for the alternate box; use horizon_reached "
            "only when every known solution exceeds max_steps; use certified_deadlock only "
            "when every complete first-push branch is dead and none were truncated"
        ),
        "max_estimated_tokens": token_max,
        "output_shards": manifests,
        "model_visible_language": "English",
        "training_num_boxes": 2,
        "format": (
            "N-box-neutral one-or-two-candidate single-turn CoT-inf; relative initial-board "
            "box descriptions; alternate-box branches are aggregated before status "
            "selection; over-horizon paths end with horizon_reached; rows with no legal "
            "alternate-box first push state which box must be pushed first"
        ),
        "errors_sample": errors[:50],
    }
    with (metadata_dir / "compare2_summary.json").open("w", encoding="utf-8") as fout:
        json.dump(summary, fout, indent=2, ensure_ascii=True)

    if not args.keep_parts:
        for result in results:
            Path(result["part"]).unlink(missing_ok=True)
        parts_dir.rmdir()
    print(
        f"[compare2] DONE input={input_rows} written={aggregate['written']} "
        f"elapsed={elapsed:.1f}s rate={input_rows / max(elapsed, 1e-9):.2f} rows/s",
        flush=True,
    )
    if errors:
        print(f"[compare2] first errors: {errors[:5]}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
