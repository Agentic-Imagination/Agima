#!/usr/bin/env python3
# Modified for Agima: portable paths and explicit dataset arguments.
"""Build Pacman K=inf forward-thinking rows with a shared prefix and two continuations.

The source is the raw successful-oracle dataset produced by
``scripts/pacman_gen/generate_pacman_success_dataset.py``.  A qualifying row:

1. replays the exact shortest expert trajectory;
2. searches expert states 2..10 for a non-wall, non-reversing alternate action;
3. uses complete status-aware BFS to find the shortest safe completion after that action;
4. chooses the largest-regret alternative that is at least two moves longer;
5. imagines the common expert prefix once, compares both continuations from the same
   checkpoint, and finally commits the complete expert plan in one open-loop turn.

The complete winning trajectory is always visible.  A repeated checkpoint frame resets the
second branch.  The alternative receives only the remaining image budget, so assistant
imagined frames are strictly capped (40 by default, matching Sokoban compare2's two 20-move
candidate rollouts).  Search-limit outcomes are recorded as ``unknown`` and are never used as
negative supervision or called deadlocks.
"""

from __future__ import annotations

import argparse
import ast
import base64
import gzip
import hashlib
import io
import json
import multiprocessing as mp
import os
import sys
import time
from collections import Counter, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

import numpy as np
from PIL import Image


_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from gymnasium.envs.pacman_2d.pacman_2d import (  # noqa: E402
    ACTION_NAMES,
    ACTION_ORDER,
    OPPOSITE,
    Pacman2DEnv,
)
from scripts.data.remap_pacman_cot_chunk_k import (  # noqa: E402
    FRAME0_LABEL,
    THINK_CLOSE,
    _rules_prefix,
    budget_sentence,
)


STOP_ACTION = "('stop', 'stop')"
SCHEMA_VERSION = "pacman_forward_shared_compare2_kinf_v1"
COT_ROW_TYPE = "shared_prefix_two_candidate_cot_singleturn_openloop"
TOKEN_BUDGET = 50_000
IMG_TOKENS_EST = 840
DEFAULT_SOURCE = _REPO / "hf_datasets" / "pacman_2d_easy_braidthin_oracle_success_20260815"
DEFAULT_OUTPUT = _REPO / "hf_datasets" / "pacman_2d_easy_braidthin_cot_chunk_kinf_forward_v1_20260815"


@dataclass
class SearchResult:
    status: str  # solved | certified_unsolvable | task_horizon_reached | unknown
    actions: Optional[list[int]]
    states_expanded: int
    depth_cut: bool = False


@dataclass
class BranchChoice:
    decision_step: int
    expert_action: int
    alternate_action: int
    alternate_suffix: list[int]
    alternate_total_moves: int
    regret_moves: int
    states_expanded: int


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_json_safe(v) for v in value]
    return value


def _stable_hash(value: Any) -> str:
    payload = json.dumps(_json_safe(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def _parse_action(text: str) -> tuple[str, Any]:
    value = ast.literal_eval(str(text).strip())
    if not isinstance(value, tuple) or len(value) != 2:
        raise ValueError(f"bad action tuple: {text!r}")
    return str(value[0]), value[1]


def _move_action(action_id: int) -> str:
    return str(("move", int(action_id)))


def _direction(action_id: int) -> str:
    return str(ACTION_NAMES.get(int(action_id), action_id)).lower()


def _image_to_base64(image: np.ndarray | Image.Image, quality: int) -> str:
    pil = Image.fromarray(np.asarray(image)) if not isinstance(image, Image.Image) else image
    if pil.mode != "RGB":
        pil = pil.convert("RGB")
    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=int(quality))
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _source_episode(row: dict[str, Any]) -> tuple[str, str, list[int], list[str], list[str]]:
    history = row.get("history") or []
    if not history:
        raise ValueError("source row has no history")
    initial_frame = history[0].get("image_prev") or history[0].get("image")
    rules = str(history[0].get("prompt", ""))
    if not initial_frame or not rules:
        raise ValueError("source row lacks initial frame or prompt")

    moves: list[int] = []
    move_actions: list[str] = []
    move_frames: list[str] = []
    all_actions: list[str] = []
    for index, step in enumerate(history):
        action = str(step.get("vlm_output") or step.get("action"))
        kind, payload = _parse_action(action)
        if kind == "move":
            if not isinstance(payload, int) or int(payload) not in ACTION_ORDER:
                raise ValueError(f"bad move action at history index {index}: {action!r}")
            canonical = _move_action(int(payload))
            moves.append(int(payload))
            move_actions.append(canonical)
            frame = step.get("image_next")
            if not frame:
                raise ValueError(f"move at history index {index} lacks image_next")
            move_frames.append(str(frame))
            all_actions.append(canonical)
        elif kind == "stop" and payload == "stop":
            if index != len(history) - 1:
                raise ValueError("STOP appears before the final history item")
            all_actions.append(STOP_ACTION)
        else:
            raise ValueError(f"unsupported source action: {action!r}")
    if not moves or not all_actions or all_actions[-1] != STOP_ACTION:
        raise ValueError("source plan is empty or lacks terminal STOP")
    if len(move_frames) != len(moves):
        raise ValueError("source move/frame count mismatch")
    stats = row.get("stats") or {}
    if not bool(stats.get("success")) or not bool(stats.get("won")):
        raise ValueError("source row is not a successful won trajectory")
    return rules, str(initial_frame), moves, move_actions, move_frames


def _env_from_row(row: dict[str, Any]) -> Pacman2DEnv:
    kw = row.get("env_kwargs") or {}
    init = row.get("init_state") or {}
    env = Pacman2DEnv(
        difficulty=str(kw.get("difficulty", init.get("difficulty", "easy"))),
        grid_size=int(kw.get("grid_size", init.get("grid_size", 9))),
        food_count=int(kw.get("food_count", init.get("food_count", len(init.get("food", []))))),
        num_ghosts=int(kw.get("num_ghosts", init.get("num_ghosts", 1))),
        wall_density=float(kw.get("wall_density", init.get("wall_density", 0.22))),
        max_steps=int(kw.get("max_steps", init.get("max_steps", 80))),
        horizon=int(kw.get("horizon", init.get("horizon", 8))),
        render_mode="rgb_array",
        render_size=kw.get("render_size", init.get("render_size", 256)),
        level=kw.get("level", init.get("level")),
        num_power_pellets=int(kw.get("num_power_pellets", init.get("num_power_pellets", 0))),
        frightened_steps=int(kw.get("frightened_steps", init.get("frightened_steps", 12))),
        ghost_spawn_dist=kw.get("ghost_spawn_dist"),
        diverse_ghosts=bool(kw.get("diverse_ghosts", init.get("diverse_ghosts", False))),
        braid=bool(kw.get("braid", init.get("braid", False))),
        thin_corridors=bool(kw.get("thin_corridors", init.get("thin_corridors", False))),
    )
    env.reset(seed=int(row.get("episode_seed", 0)), init_state=init)
    return env


def _expert_states(env: Pacman2DEnv, moves: Sequence[int]) -> list[dict[str, Any]]:
    state = env._state_dict()
    states = [state]
    for index, action_id in enumerate(moves):
        child, _, terminated, truncated, _ = env._transition_state(state, int(action_id))
        if terminated or truncated or not child["alive"]:
            raise ValueError(f"expert move {index} does not replay safely")
        if tuple(child["pacman_pos"]) == tuple(state["pacman_pos"]):
            raise ValueError(f"expert move {index} bumps a wall")
        states.append(child)
        state = child
    if state["food"] or not state["alive"]:
        raise ValueError("expert move sequence does not finish with all food collected")
    return states


def _shortest_completion(
    env: Pacman2DEnv,
    start: dict[str, Any],
    max_depth: int,
    state_limit: int,
) -> SearchResult:
    """Shortest safe completion with explicit exhaustion/depth/expansion-limit outcomes."""
    start = env._coerce_state(start)
    if not start["alive"]:
        return SearchResult("certified_unsolvable", None, 0)
    if not start["food"]:
        return SearchResult("solved", [], 0)
    if max_depth <= 0:
        return SearchResult("task_horizon_reached", None, 0, depth_cut=True)

    start_key = env._state_key(start)
    queue: deque[tuple[dict[str, Any], int]] = deque([(start, 0)])
    parents: dict[tuple, tuple[Optional[tuple], Optional[int]]] = {start_key: (None, None)}
    expanded = 0
    depth_cut = False
    while queue:
        if expanded >= max(1, int(state_limit)):
            return SearchResult("unknown", None, expanded, depth_cut=depth_cut)
        state, depth = queue.popleft()
        expanded += 1
        if depth >= max_depth:
            depth_cut = True
            continue
        for action_id in ACTION_ORDER:
            child, _, terminated, truncated, _ = env._transition_state(state, int(action_id))
            if terminated or truncated or not child["alive"]:
                continue
            if tuple(child["pacman_pos"]) == tuple(state["pacman_pos"]):
                continue
            child_key = env._state_key(child)
            if child_key in parents:
                continue
            parents[child_key] = (env._state_key(state), int(action_id))
            if not child["food"]:
                actions: list[int] = []
                cursor = child_key
                while parents[cursor][0] is not None:
                    previous, used = parents[cursor]
                    assert previous is not None and used is not None
                    actions.append(int(used))
                    cursor = previous
                actions.reverse()
                return SearchResult("solved", actions, expanded, depth_cut=depth_cut)
            queue.append((child, depth + 1))
    if depth_cut:
        return SearchResult("task_horizon_reached", None, expanded, depth_cut=True)
    return SearchResult("certified_unsolvable", None, expanded)


def _choose_branch(
    env: Pacman2DEnv,
    moves: Sequence[int],
    states: Sequence[dict[str, Any]],
    min_prefix_steps: int,
    max_prefix_steps: int,
    min_regret_moves: int,
    state_limit: int,
) -> tuple[Optional[BranchChoice], Counter]:
    expert_len = len(moves)
    choices: list[BranchChoice] = []
    counts: Counter = Counter()
    last_t = min(int(max_prefix_steps), expert_len - 2)
    for decision_step in range(int(min_prefix_steps), last_t + 1):
        state = states[decision_step]
        expert_action = int(moves[decision_step])
        previous_action = int(moves[decision_step - 1])
        for alternate_action in ACTION_ORDER:
            alternate_action = int(alternate_action)
            if alternate_action == expert_action:
                continue
            if alternate_action == OPPOSITE.get(previous_action):
                counts["filtered_immediate_reverse"] += 1
                continue
            child, _, terminated, truncated, _ = env._transition_state(state, alternate_action)
            if tuple(child["pacman_pos"]) == tuple(state["pacman_pos"]):
                counts["filtered_wall"] += 1
                continue
            if terminated or truncated or not child["alive"]:
                counts["filtered_immediate_fatal"] += 1
                continue
            counts["searched"] += 1
            # A successful STOP must still be possible after all move actions, so the last legal
            # move state has steps < max_steps.
            remaining = max(0, int(env.max_steps) - int(child["steps"]) - 1)
            result = _shortest_completion(env, child, remaining, state_limit)
            counts[f"search_status:{result.status}"] += 1
            if result.status != "solved" or result.actions is None:
                continue
            alternate_suffix = [alternate_action, *result.actions]
            alternate_total = decision_step + len(alternate_suffix)
            regret = alternate_total - expert_len
            if regret < int(min_regret_moves):
                counts["filtered_low_regret"] += 1
                continue
            choices.append(
                BranchChoice(
                    decision_step=decision_step,
                    expert_action=expert_action,
                    alternate_action=alternate_action,
                    alternate_suffix=alternate_suffix,
                    alternate_total_moves=alternate_total,
                    regret_moves=regret,
                    states_expanded=int(result.states_expanded),
                )
            )
            counts["qualified"] += 1
    if not choices:
        return None, counts
    # Largest downstream consequence first; exact ties use the earliest fork and action order.
    choices.sort(key=lambda c: (-c.regret_moves, c.decision_step, c.alternate_action))
    return choices[0], counts


def _render_suffix(
    env: Pacman2DEnv,
    start: dict[str, Any],
    actions: Sequence[int],
    jpeg_quality: int,
) -> list[str]:
    frames: list[str] = []
    state = start
    for index, action_id in enumerate(actions):
        child, _, terminated, truncated, _ = env._transition_state(state, int(action_id))
        if terminated or truncated or not child["alive"]:
            raise ValueError(f"alternate displayed action {index} is not safe")
        if tuple(child["pacman_pos"]) == tuple(state["pacman_pos"]):
            raise ValueError(f"alternate displayed action {index} bumps a wall")
        frames.append(_image_to_base64(env.render_state(child, mode="rgb_array"), jpeg_quality))
        state = child
    return frames


def forward_instruction(rules: str, max_steps: int, max_imagined_frames: int) -> str:
    cadence = (
        "You will see ONLY this initial frame and receive no further environment feedback. "
        "Before committing, imagine the route shared by the candidate plans until a meaningful "
        "fork, compare continuations from the same decision checkpoint, and use at most "
        f"{int(max_imagined_frames)} imagined frames in total. Then commit the complete plan in "
        "one shot."
    )
    return (
        _rules_prefix(rules).rstrip()
        + "\n\n"
        + budget_sentence(max_steps)
        + " "
        + cadence
        + "\n\n"
        + FRAME0_LABEL
    )


def _append_rollout(
    content: list[dict[str, Any]],
    pending: str,
    actions: Sequence[str],
    frames: Sequence[str],
) -> str:
    if len(actions) != len(frames):
        raise ValueError("rollout action/frame mismatch")
    for action, frame in zip(actions, frames):
        content.append({"type": "text", "text": pending + "\n" + str(action)})
        content.append({"type": "image", "image": str(frame), "loss": True})
        pending = ""
    return pending


def _candidate_result(candidate: dict[str, Any]) -> str:
    if candidate["source"] == "expert_shortest":
        return (
            "\nResult: this continuation safely collects all remaining food; including the "
            f"shared prefix, it finishes in {candidate['full_total_moves']} moves."
        )
    if candidate["status"] == "solved":
        return (
            "\nResult: this continuation also safely collects all remaining food, but including "
            f"the shared prefix it needs {candidate['full_total_moves']} moves."
        )
    return (
        f"\nResult: after {candidate['displayed_continuation_moves']} displayed continuation "
        "moves, this route has not finished within the shared comparison-frame budget. "
        f"Complete search finds a safe solution in {candidate['full_total_moves']} total moves."
    )


def _estimate_tokens(messages: Sequence[dict[str, Any]]) -> int:
    chars = 0
    images = 0
    for message in messages:
        for item in message.get("content") or []:
            if item.get("type") == "text":
                chars += len(str(item.get("text", "")))
            elif item.get("type") == "image":
                images += 1
    return (chars + 3) // 4 + images * IMG_TOKENS_EST


def _output_base(row: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "__bagel_order_key",
        "task",
        "env_id",
        "env_kwargs",
        "seed",
        "episode_seed",
        "episode",
        "extra_state",
        "init_state",
        "init_state_hash",
        "stats",
        "version",
        "alignment_index",
    )
    return {key: row[key] for key in keys if key in row}


def _single_row(
    row: dict[str, Any],
    rules: str,
    initial_frame: str,
    move_actions: Sequence[str],
    move_frames: Sequence[str],
    all_actions: Sequence[str],
    max_imagined_frames: int,
    search_counts: Counter,
) -> dict[str, Any]:
    max_steps = int((row.get("env_kwargs") or {}).get("max_steps", 80))
    content: list[dict[str, Any]] = []
    pending = (
        "\nI will inspect the route for a reliable meaningful fork before committing. "
        "No sufficiently distinct verified alternative is available, so I will imagine the "
        "single shortest safe plan.\nVerified plan:"
    )
    pending = _append_rollout(content, pending, move_actions, move_frames)
    pending += (
        f"\nResult: this plan safely collects all food in {len(move_actions)} moves."
        "\nAvailability check: no non-trivial alternate continuation with a verified regret of "
        "at least two moves was found in the decision window."
        f"\n{THINK_CLOSE}\n</think>\n"
        + "\n".join(all_actions)
    )
    content.append({"type": "text", "text": pending})
    messages = [
        {"role": "user", "content": [
            {"type": "text", "text": forward_instruction(rules, max_steps, max_imagined_frames)},
            {"type": "image", "image": initial_frame, "loss": False},
        ]},
        {"role": "assistant", "content": content},
    ]
    out = _output_base(row)
    out.update(
        {
            "messages": messages,
            "chunk_size": "inf",
            "n_steps": len(all_actions),
            "n_chunks": 1,
            "n_imagined_frames": len(move_frames),
            "n_feedback_turns": 0,
            "n_feedback_frames": 0,
            "max_steps": max_steps,
            "max_imagined_frames": int(max_imagined_frames),
            "has_plan_comparison": False,
            "n_candidates": 1,
            "shared_prefix_steps": 0,
            "decision_step": None,
            "decision_state_hash": None,
            "checkpoint_reset_frames": 0,
            "selection_reason": "no_reliable_critical_fork",
            "chosen_candidate": 1,
            "candidate_order_shuffled": False,
            "single_candidate_fallback": True,
            "expert_move_count": len(move_actions),
            "branch_search_counts": dict(search_counts),
            "candidates": [
                {
                    "candidate": 1,
                    "source": "expert_shortest",
                    "status": "solved",
                    "full_status": "solved",
                    "first_action": move_actions[0],
                    "first_direction": _direction(_parse_action(move_actions[0])[1]),
                    "displayed_continuation_moves": len(move_actions),
                    "full_continuation_moves": len(move_actions),
                    "displayed_total_moves": len(move_actions),
                    "full_total_moves": len(move_actions),
                    "regret_moves": 0,
                    "states_expanded": 0,
                }
            ],
            "schema_version": SCHEMA_VERSION,
            "cot_row_type": COT_ROW_TYPE,
        }
    )
    out["estimated_tokens"] = _estimate_tokens(messages)
    return out


def transform_row(row: dict[str, Any], cfg: dict[str, Any]) -> dict[str, Any]:
    rules, initial_frame, moves, move_actions, move_frames = _source_episode(row)
    expert_len = len(moves)
    source_max = int(cfg["source_max_expert_moves"])
    if expert_len > source_max:
        raise ValueError(f"expert has {expert_len} moves, exceeding source guarantee {source_max}")
    all_actions = [*move_actions, STOP_ACTION]
    env = _env_from_row(row)
    try:
        states = _expert_states(env, moves)
        choice, search_counts = _choose_branch(
            env,
            moves,
            states,
            int(cfg["min_prefix_steps"]),
            int(cfg["max_prefix_steps"]),
            int(cfg["min_regret_moves"]),
            int(cfg["state_limit"]),
        )
        if choice is None:
            out = _single_row(
                row,
                rules,
                initial_frame,
                move_actions,
                move_frames,
                all_actions,
                int(cfg["max_imagined_frames"]),
                search_counts,
            )
            _validate_output(out, all_actions, cfg)
            return out

        decision_step = int(choice.decision_step)
        max_frames = int(cfg["max_imagined_frames"])
        max_alt = int(cfg["max_alternate_display_moves"])
        alternate_budget = min(max_alt, max_frames - 1 - expert_len)
        if alternate_budget < 1:
            raise ValueError(
                f"no alternate frame budget: max={max_frames}, reset=1, expert={expert_len}"
            )
        alternate_visible_ids = choice.alternate_suffix[:alternate_budget]
        alternate_visible_actions = [_move_action(a) for a in alternate_visible_ids]
        alternate_frames = _render_suffix(
            env,
            states[decision_step],
            alternate_visible_ids,
            int(cfg["jpeg_quality"]),
        )
        alternate_status = (
            "solved"
            if len(alternate_visible_ids) == len(choice.alternate_suffix)
            else "comparison_budget_reached"
        )

        expert_candidate = {
            "source": "expert_shortest",
            "status": "solved",
            "full_status": "solved",
            "first_action": move_actions[decision_step],
            "first_direction": _direction(moves[decision_step]),
            "actions": list(move_actions[decision_step:]),
            "frames": list(move_frames[decision_step:]),
            "displayed_continuation_moves": expert_len - decision_step,
            "full_continuation_moves": expert_len - decision_step,
            "displayed_total_moves": expert_len,
            "full_total_moves": expert_len,
            "regret_moves": 0,
            "states_expanded": 0,
        }
        alternate_candidate = {
            "source": "alternate_shortest_after_divergence",
            "status": alternate_status,
            "full_status": "solved",
            "first_action": _move_action(choice.alternate_action),
            "first_direction": _direction(choice.alternate_action),
            "actions": alternate_visible_actions,
            "frames": alternate_frames,
            "displayed_continuation_moves": len(alternate_visible_actions),
            "full_continuation_moves": len(choice.alternate_suffix),
            "displayed_total_moves": decision_step + len(alternate_visible_actions),
            "full_total_moves": choice.alternate_total_moves,
            "regret_moves": choice.regret_moves,
            "states_expanded": choice.states_expanded,
        }
        raw_candidates = [expert_candidate, alternate_candidate]
        shuffle_key = f"{row.get('init_state_hash', '')}|{decision_step}|{SCHEMA_VERSION}"
        swap = hashlib.sha256(shuffle_key.encode("utf-8")).digest()[0] % 2 == 1
        order = [1, 0] if swap else [0, 1]
        displayed = [raw_candidates[i] for i in order]
        chosen_display_index = order.index(0)

        content: list[dict[str, Any]] = []
        pending = (
            "\nI will first imagine the route shared by both candidate plans until a meaningful "
            "fork.\nShared prefix:"
        )
        pending = _append_rollout(
            content,
            pending,
            move_actions[:decision_step],
            move_frames[:decision_step],
        )
        pending += (
            "\nA meaningful fork appears at this decision checkpoint. I will compare two "
            "continuations from exactly this same state.\n"
        )
        checkpoint_frame = move_frames[decision_step - 1]
        for displayed_index, candidate in enumerate(displayed):
            candidate_number = displayed_index + 1
            pending += (
                f"Candidate {candidate_number}: from the decision checkpoint, first move "
                f"{candidate['first_direction']}."
            )
            pending = _append_rollout(
                content,
                pending,
                candidate["actions"],
                candidate["frames"],
            )
            pending += _candidate_result(candidate)
            if displayed_index == 0:
                pending += (
                    "\nReturn to the exact decision checkpoint before evaluating the other "
                    "continuation:"
                )
                content.append({"type": "text", "text": pending})
                content.append({"type": "image", "image": checkpoint_frame, "loss": True})
                pending = "\n"

        winner_number = chosen_display_index + 1
        loser_number = 2 if winner_number == 1 else 1
        pending += (
            f"\nComparison: both complete searches are safe, but Candidate {winner_number} "
            f"finishes in {expert_len} moves instead of Candidate {loser_number}'s "
            f"{choice.alternate_total_moves}. I will commit Candidate {winner_number}."
            f"\n{THINK_CLOSE}\n</think>\n"
            + "\n".join(all_actions)
        )
        content.append({"type": "text", "text": pending})
        max_steps = int((row.get("env_kwargs") or {}).get("max_steps", 80))
        messages = [
            {"role": "user", "content": [
                {"type": "text", "text": forward_instruction(rules, max_steps, max_frames)},
                {"type": "image", "image": initial_frame, "loss": False},
            ]},
            {"role": "assistant", "content": content},
        ]
        decision_state = env._json_state(states[decision_step])
        candidate_metadata = []
        for index, candidate in enumerate(displayed, start=1):
            candidate_metadata.append(
                {
                    "candidate": index,
                    **{key: value for key, value in candidate.items() if key not in {"actions", "frames"}},
                }
            )
        n_frames = expert_len + 1 + len(alternate_frames)
        out = _output_base(row)
        out.update(
            {
                "messages": messages,
                "chunk_size": "inf",
                "n_steps": len(all_actions),
                "n_chunks": 1,
                "n_imagined_frames": n_frames,
                "n_feedback_turns": 0,
                "n_feedback_frames": 0,
                "max_steps": max_steps,
                "max_imagined_frames": max_frames,
                "has_plan_comparison": True,
                "n_candidates": 2,
                "shared_prefix_steps": decision_step,
                "decision_step": decision_step,
                "decision_state_hash": _stable_hash(decision_state),
                "checkpoint_reset_frames": 1,
                "selection_reason": "shorter_safe_completion",
                "chosen_candidate": chosen_display_index + 1,
                "candidate_order_shuffled": bool(swap),
                "single_candidate_fallback": False,
                "expert_move_count": expert_len,
                "alternate_full_total_moves": choice.alternate_total_moves,
                "regret_moves": choice.regret_moves,
                "alternate_display_budget": alternate_budget,
                "branch_search_counts": dict(search_counts),
                "candidates": candidate_metadata,
                "frame_layout": {
                    "shared_prefix": decision_step,
                    "candidate_1": len(displayed[0]["frames"]),
                    "checkpoint_reset": 1,
                    "candidate_2": len(displayed[1]["frames"]),
                },
                "schema_version": SCHEMA_VERSION,
                "cot_row_type": COT_ROW_TYPE,
            }
        )
        out["estimated_tokens"] = _estimate_tokens(messages)
        _validate_output(out, all_actions, cfg)
        return out
    finally:
        env.close()


def _assistant_images(row: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        item
        for message in row.get("messages") or []
        if message.get("role") == "assistant"
        for item in message.get("content") or []
        if item.get("type") == "image"
    ]


def _committed_actions(row: dict[str, Any]) -> list[str]:
    text = "".join(
        str(item.get("text", ""))
        for message in row.get("messages") or []
        if message.get("role") == "assistant"
        for item in message.get("content") or []
        if item.get("type") == "text"
    )
    if "</think>" not in text:
        raise AssertionError("assistant text lacks </think>")
    tail = text.rsplit("</think>", 1)[1]
    return [line.strip() for line in tail.splitlines() if line.strip()]


def _validate_output(row: dict[str, Any], expected_actions: Sequence[str], cfg: dict[str, Any]) -> None:
    messages = row.get("messages") or []
    if len(messages) != 2 or [m.get("role") for m in messages] != ["user", "assistant"]:
        raise AssertionError("row must contain exactly one user and one assistant turn")
    user_images = [item for item in messages[0].get("content") or [] if item.get("type") == "image"]
    if len(user_images) != 1 or bool(user_images[0].get("loss", True)):
        raise AssertionError("initial frame must be the single loss=False user image")
    images = _assistant_images(row)
    if any(not bool(item.get("loss")) or not item.get("image") for item in images):
        raise AssertionError("every assistant image must be a populated loss=True MSE target")
    if len(images) != int(row.get("n_imagined_frames", -1)):
        raise AssertionError("assistant image count does not match n_imagined_frames")
    if len(images) > int(cfg["max_imagined_frames"]):
        raise AssertionError("imagined-frame budget exceeded")
    if _committed_actions(row) != list(expected_actions):
        raise AssertionError("committed plan differs from the source expert plan")
    if int(row.get("estimated_tokens", TOKEN_BUDGET)) >= TOKEN_BUDGET:
        raise AssertionError("estimated token budget exceeded")
    if row.get("chunk_size") != "inf" or int(row.get("n_feedback_turns", -1)) != 0:
        raise AssertionError("row is not K=inf/no-feedback")
    if row.get("schema_version") != SCHEMA_VERSION:
        raise AssertionError("wrong schema version")

    candidates = row.get("candidates") or []
    complete_visible_experts = [
        candidate
        for candidate in candidates
        if candidate.get("source") == "expert_shortest"
        and candidate.get("status") == "solved"
        and candidate.get("full_status") == "solved"
        and int(candidate.get("displayed_continuation_moves", -1))
        == int(candidate.get("full_continuation_moves", -2))
        and int(candidate.get("displayed_total_moves", -1))
        == int(candidate.get("full_total_moves", -2))
    ]
    if len(complete_visible_experts) != 1:
        raise AssertionError("row must contain exactly one complete visible solved expert plan")
    expert = complete_visible_experts[0]
    if int(expert.get("full_total_moves", -1)) != int(row.get("expert_move_count", -2)):
        raise AssertionError("complete visible expert length differs from source expert")
    chosen_index = int(row.get("chosen_candidate", 0)) - 1
    if chosen_index not in range(len(candidates)) or candidates[chosen_index] is not expert:
        raise AssertionError("committed candidate is not the complete visible expert plan")

    if row.get("has_plan_comparison"):
        decision = int(row.get("decision_step", -1))
        if not int(cfg["min_prefix_steps"]) <= decision <= int(cfg["max_prefix_steps"]):
            raise AssertionError("decision step outside configured window")
        if int(row.get("n_candidates", 0)) != 2 or int(row.get("regret_moves", 0)) < int(cfg["min_regret_moves"]):
            raise AssertionError("bad two-candidate metadata")
        layout = row.get("frame_layout") or {}
        if sum(int(v) for v in layout.values()) != len(images):
            raise AssertionError("frame layout does not sum to assistant image count")
        if int(layout.get("checkpoint_reset", 0)) != 1:
            raise AssertionError("two-candidate row lacks exactly one checkpoint reset")
        if row.get("selection_reason") != "shorter_safe_completion":
            raise AssertionError("unexpected pair selection reason")
    else:
        if int(row.get("n_candidates", 0)) != 1 or not row.get("single_candidate_fallback"):
            raise AssertionError("bad single-candidate fallback metadata")


def _merge_counters(target: Counter, source: dict[str, Any] | Counter, prefix: str = "") -> None:
    for key, value in source.items():
        target[prefix + str(key)] += int(value)


def _merge_stats(target: Counter, source: dict[str, Any] | Counter) -> None:
    """Merge per-shard stats while preserving corpus-wide maxima."""
    for key, value in source.items():
        if str(key).startswith("max_"):
            target[str(key)] = max(target[str(key)], int(value))
        else:
            target[str(key)] += int(value)


def _process_shard(task: tuple[str, str, dict[str, Any]]) -> dict[str, Any]:
    input_path_s, output_path_s, cfg = task
    input_path = Path(input_path_s)
    output_path = Path(output_path_s)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = output_path.with_suffix(output_path.suffix + f".{os.getpid()}.tmp")
    stats: Counter = Counter()
    prefix_hist: Counter = Counter()
    regret_hist: Counter = Counter()
    frame_hist: Counter = Counter()
    started = time.time()
    with gzip.open(input_path, "rt", encoding="utf-8") as source, gzip.open(
        tmp, "wt", encoding="utf-8", compresslevel=int(cfg["gzip_level"])
    ) as destination:
        for line_index, line in enumerate(source):
            if not line.strip():
                continue
            raw = json.loads(line)
            out = transform_row(raw, cfg)
            destination.write(json.dumps(out, ensure_ascii=False, separators=(",", ":")) + "\n")
            stats["rows"] += 1
            if out["has_plan_comparison"]:
                stats["pair_rows"] += 1
                prefix_hist[str(out["decision_step"])] += 1
                regret_hist[str(out["regret_moves"])] += 1
                stats[f"chosen_candidate:{out['chosen_candidate']}"] += 1
                if any(c["status"] == "comparison_budget_reached" for c in out["candidates"]):
                    stats["alternate_comparison_budget_reached"] += 1
            else:
                stats["single_rows"] += 1
            frame_hist[str(out["n_imagined_frames"])] += 1
            stats["max_imagined_frames_seen"] = max(
                stats["max_imagined_frames_seen"], int(out["n_imagined_frames"])
            )
            stats["max_estimated_tokens"] = max(
                stats["max_estimated_tokens"], int(out["estimated_tokens"])
            )
            _merge_counters(stats, out.get("branch_search_counts") or {}, "search:")
            if int(cfg["progress_every"]) > 0 and stats["rows"] % int(cfg["progress_every"]) == 0:
                print(
                    json.dumps(
                        {
                            "shard": input_path.name,
                            "rows": stats["rows"],
                            "pairs": stats["pair_rows"],
                            "rows_per_s": round(stats["rows"] / max(1e-6, time.time() - started), 2),
                        }
                    ),
                    flush=True,
                )
    tmp.replace(output_path)
    return {
        "input": str(input_path.resolve()),
        "output": str(output_path.resolve()),
        "output_bytes": output_path.stat().st_size,
        "elapsed_sec": time.time() - started,
        "stats": dict(stats),
        "prefix_histogram": dict(prefix_hist),
        "regret_histogram": dict(regret_hist),
        "imagined_frame_histogram": dict(frame_hist),
    }


def _summarize_existing_shard(task: tuple[str, str]) -> dict[str, Any]:
    """Rebuild generation statistics from a completed output shard."""
    input_path = Path(task[0])
    output_path = Path(task[1])
    stats: Counter = Counter()
    prefix_hist: Counter = Counter()
    regret_hist: Counter = Counter()
    frame_hist: Counter = Counter()
    started = time.time()
    with gzip.open(output_path, "rt", encoding="utf-8") as source:
        for line in source:
            if not line.strip():
                continue
            out = json.loads(line)
            stats["rows"] += 1
            if out["has_plan_comparison"]:
                stats["pair_rows"] += 1
                prefix_hist[str(out["decision_step"])] += 1
                regret_hist[str(out["regret_moves"])] += 1
                stats[f"chosen_candidate:{out['chosen_candidate']}"] += 1
                if any(c["status"] == "comparison_budget_reached" for c in out["candidates"]):
                    stats["alternate_comparison_budget_reached"] += 1
            else:
                stats["single_rows"] += 1
            frame_hist[str(out["n_imagined_frames"])] += 1
            stats["max_imagined_frames_seen"] = max(
                stats["max_imagined_frames_seen"], int(out["n_imagined_frames"])
            )
            stats["max_estimated_tokens"] = max(
                stats["max_estimated_tokens"], int(out["estimated_tokens"])
            )
            _merge_counters(stats, out.get("branch_search_counts") or {}, "search:")
    return {
        "input": str(input_path.resolve()),
        "output": str(output_path.resolve()),
        "output_bytes": output_path.stat().st_size,
        "elapsed_sec": time.time() - started,
        "stats": dict(stats),
        "prefix_histogram": dict(prefix_hist),
        "regret_histogram": dict(regret_hist),
        "imagined_frame_histogram": dict(frame_hist),
        "summary_source": "completed_output_shard",
    }


def _cfg(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "min_prefix_steps": int(args.min_prefix_steps),
        "max_prefix_steps": int(args.max_prefix_steps),
        "min_regret_moves": int(args.min_regret_moves),
        "state_limit": int(args.state_limit),
        "max_imagined_frames": int(args.max_imagined_frames),
        "max_alternate_display_moves": int(args.max_alternate_display_moves),
        "source_max_expert_moves": int(args.source_max_expert_moves),
        "jpeg_quality": int(args.jpeg_quality),
        "gzip_level": int(args.gzip_level),
        "progress_every": int(args.progress_every),
    }


def _self_test(args: argparse.Namespace) -> int:
    cfg = _cfg(args)
    shards = sorted((args.src_root / args.src_subdir).glob("*.jsonl.gz"))
    if not shards:
        raise FileNotFoundError(f"no source shards under {args.src_root / args.src_subdir}")
    stats: Counter = Counter()
    examples: list[dict[str, Any]] = []
    with gzip.open(shards[0], "rt", encoding="utf-8") as source:
        for line in source:
            if stats["rows"] >= args.self_test:
                break
            raw = json.loads(line)
            out = transform_row(raw, cfg)
            stats["rows"] += 1
            stats["pairs"] += int(out["has_plan_comparison"])
            stats["singles"] += int(not out["has_plan_comparison"])
            stats["max_frames"] = max(stats["max_frames"], out["n_imagined_frames"])
            stats["max_tokens"] = max(stats["max_tokens"], out["estimated_tokens"])
            if len(examples) < 3:
                examples.append(
                    {
                        "episode": out.get("episode"),
                        "hash": out.get("init_state_hash"),
                        "pair": out["has_plan_comparison"],
                        "decision_step": out.get("decision_step"),
                        "regret_moves": out.get("regret_moves"),
                        "n_imagined_frames": out["n_imagined_frames"],
                        "chosen_candidate": out["chosen_candidate"],
                        "candidate_statuses": [c["status"] for c in out["candidates"]],
                    }
                )
    if stats["rows"] != args.self_test:
        raise RuntimeError(f"self-test read only {stats['rows']}/{args.self_test} rows")
    print(json.dumps({"self_test": "passed", "stats": dict(stats), "examples": examples}, indent=2))
    return 0


def _readme(summary: dict[str, Any]) -> str:
    cfg = summary["config"]
    totals = summary["totals"]
    return f"""# Pacman-2D shared-prefix forward-thinking K=inf v1

This dataset remaps the 100,000 successful braided+thinned Pacman oracle episodes into
single-turn open-loop (`K=inf`) forward-thinking rows.

Each two-candidate row imagines a shared expert prefix, compares the shortest expert
continuation with a verified safe but longer continuation from the same dynamic decision
state, explicitly restores the decision checkpoint before the second branch, and commits
the complete expert plan. Candidate order is deterministically shuffled.

- Rows: {totals.get('rows', 0):,}
- Two-candidate rows: {totals.get('pair_rows', 0):,}
- Single-plan fallbacks: {totals.get('single_rows', 0):,}
- Prefix decision window: {cfg['min_prefix_steps']}..{cfg['max_prefix_steps']} moves
- Minimum alternate regret: {cfg['min_regret_moves']} moves
- Maximum assistant imagined frames: {cfg['max_imagined_frames']}
- Maximum alternate displayed continuation: {cfg['max_alternate_display_moves']} moves
- Maximum source expert trajectory: {cfg['source_max_expert_moves']} moves plus STOP
- Search expansion limit: {cfg['state_limit']:,}

Only alternatives with a complete BFS-verified safe solution are used as comparison
targets. Search-limit results remain `unknown` in metadata and are never labeled as a
deadlock. Initial user images have `loss=False`; all assistant imagined/checkpoint images
have `loss=True`. No environment feedback is provided.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src-root", type=Path, required=True)
    parser.add_argument("--src-subdir", default="data/train")
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--min-prefix-steps", type=int, default=2)
    parser.add_argument("--max-prefix-steps", type=int, default=10)
    parser.add_argument("--min-regret-moves", type=int, default=2)
    parser.add_argument("--state-limit", type=int, default=200_000)
    parser.add_argument("--max-imagined-frames", type=int, default=40)
    parser.add_argument("--max-alternate-display-moves", type=int, default=20)
    parser.add_argument("--source-max-expert-moves", type=int, default=27)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--gzip-level", type=int, default=6)
    parser.add_argument("--progress-every", type=int, default=1000)
    parser.add_argument("--expected-rows", type=int, default=100_000)
    parser.add_argument("--self-test", type=int, default=0)
    parser.add_argument(
        "--summarize-existing",
        action="store_true",
        help="scan completed output shards and rebuild metadata without regenerating rows",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.min_prefix_steps < 1 or args.max_prefix_steps < args.min_prefix_steps:
        raise ValueError("invalid prefix decision window")
    if args.max_imagined_frames <= args.source_max_expert_moves + 1:
        raise ValueError("frame budget must fit the complete expert, reset, and an alternate frame")
    if args.self_test:
        return _self_test(args)

    source_dir = args.src_root / args.src_subdir
    shards = sorted(source_dir.glob("*.jsonl.gz"))
    if not shards:
        raise FileNotFoundError(f"no source shards under {source_dir}")
    training_dir = args.out_root / "training"
    metadata_dir = args.out_root / "metadata"
    training_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)
    outputs = [training_dir / shard.name for shard in shards]
    existing = [path for path in outputs if path.exists()]
    if args.summarize_existing and len(existing) != len(outputs):
        raise FileNotFoundError(
            f"summarize-existing requires all {len(outputs)} output shards; found {len(existing)}"
        )
    if existing and not args.overwrite and not args.summarize_existing:
        raise FileExistsError(f"{len(existing)} output shards already exist; pass --overwrite")
    cfg = _cfg(args)
    started = time.time()
    context = mp.get_context("fork")
    if args.summarize_existing:
        tasks = [(str(src), str(dst)) for src, dst in zip(shards, outputs)]
        print(
            f"[Pacman forward K=inf] summarizing {len(tasks)} completed shards "
            f"with {min(args.workers, len(tasks))} workers",
            flush=True,
        )
        with context.Pool(min(max(1, args.workers), len(tasks))) as pool:
            results = pool.map(_summarize_existing_shard, tasks)
    else:
        tasks = [(str(src), str(dst), cfg) for src, dst in zip(shards, outputs)]
        print(
            f"[Pacman forward K=inf] shards={len(tasks)} workers={min(args.workers, len(tasks))} "
            f"source={source_dir} out={args.out_root}",
            flush=True,
        )
        with context.Pool(min(max(1, args.workers), len(tasks))) as pool:
            results = pool.map(_process_shard, tasks)

    totals: Counter = Counter()
    prefix_hist: Counter = Counter()
    regret_hist: Counter = Counter()
    frame_hist: Counter = Counter()
    for result in results:
        _merge_stats(totals, result["stats"])
        _merge_counters(prefix_hist, result["prefix_histogram"])
        _merge_counters(regret_hist, result["regret_histogram"])
        _merge_counters(frame_hist, result["imagined_frame_histogram"])
    if totals["rows"] != int(args.expected_rows):
        raise RuntimeError(f"output coverage {totals['rows']} != expected {args.expected_rows}")
    if totals["max_imagined_frames_seen"] > int(args.max_imagined_frames):
        raise RuntimeError("generated output exceeds imagined-frame budget")
    summary = {
        "schema_version": SCHEMA_VERSION,
        "source_root": str(args.src_root.resolve()),
        "source_subdir": args.src_subdir,
        "output_root": str(args.out_root.resolve()),
        "config": cfg,
        "totals": dict(totals),
        "prefix_histogram": dict(sorted(prefix_hist.items(), key=lambda item: int(item[0]))),
        "regret_histogram": dict(sorted(regret_hist.items(), key=lambda item: int(item[0]))),
        "imagined_frame_histogram": dict(sorted(frame_hist.items(), key=lambda item: int(item[0]))),
        "shards": results,
        "elapsed_sec": time.time() - started,
        "loss_contract": "user initial frame loss=False; every assistant imagined/reset frame loss=True; CE on assistant text",
        "search_contract": "only complete BFS-solved longer alternatives supervise comparisons; unknown is never called deadlock",
    }
    _write_json(metadata_dir / "forward_summary.json", summary)
    (args.out_root / "README.md").write_text(_readme(summary), encoding="utf-8")
    print(json.dumps({"done": True, "totals": dict(totals), "elapsed_sec": summary["elapsed_sec"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
