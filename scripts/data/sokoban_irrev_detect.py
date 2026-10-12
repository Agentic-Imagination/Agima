#!/usr/bin/env python3
# Modified for Agima: portable paths and explicit dataset arguments.
"""SK2 -- Sokoban irreversibility critical-step detector.

Exp2 "branching" wants to teach the model, INSIDE its private <think>, to imagine a
tempting-but-fatal alternative move, see it produce a *proven*-unsolvable board, and
backtrack to the expert line. This module finds, for one solved expert episode, the
EARLIEST interior step at which a single legal counterfactual BOX-PUSH turns a
still-solvable board into a proven deadlock -- exactly the decision point a branch
should be spliced before.

Public API
----------
    detect_critical_step(episode) ->
        {step, alt_action, alt_deadlock_frame_b64, expert_action}  |  None

    `episode` is one source row from remap_sokoban_cot_chunk_inf.DEFAULT_SRC (has
    `history` and `init_state`). The four contract keys are:
      step                  -- 0-based step index j the branch belongs BEFORE (the
                               expert action a_j is what gets committed after the
                               abandoned detour). Feed straight to
                               branch_format.step_action_index / branch_metadata.
      alt_action            -- the counterfactual push serialized EXACTLY like the
                               CoT-inf remap serializes actions, i.e. the tuple-string
                               str(vlm_output) form: "('move', P)" for move_only data
                               ("('push', P)" for move_push data). In move_only mode a
                               box is pushed BY moving into it, so a push in direction d
                               is the move-into-box "('move', DIRECTION_TO_PAYLOAD[d])".
      alt_deadlock_frame_b64-- the board AFTER that counterfactual push, rendered
                               byte-compatibly via SK3.render_branch_frame (256px q95).
      expert_action         -- the expert's action at step j, verbatim as the remap
                               would emit it (str(vlm_output or action)).
    Plus diagnostics (safe to ignore; the four keys above are the contract):
      certify_states_expanded, n_deadlock_pushes, alt_push_legacy_id, action_mode.

Algorithm
---------
From init_state (SK1 schema) replay the expert actions to materialize every board
state S_0..S_n (only init_state stores coordinates; per-step frames don't). For each
INTERIOR step j (1..n-1) whose expert action is a real board MOVE and whose S_j is not
already solved:
  * enumerate the LEGAL forward action ids at S_j (env primitive), keep only the
    BOX-PUSH ids {1,2,3,4} that DIFFER from the expert's own action id;
  * simulate each push -> S' and SK1.certify(S').
  * A push is "critical" iff certify(S') == 'unsolvable' (queue-EXHAUSTED proof of a
    deadlock). 'truncated' is NOT a proof and is never counted (SK1 contract).
The earliest j with >=1 such proven-deadlocking push is returned; among that step's
deadlocking pushes the lowest-id one is taken (decision S3 "single clean deadlocking
push"). The terminal ('stop','stop') step and any already-solved S_j are skipped
(branching a box push there is degenerate).

Reuses the env's forward-model primitives (via SK1.load_env_from_state:
_enumerate_forward_action_ids / _simulate_action_id / _is_solved_state) so the state
graph, and hence 'critical', is identical to what the certifier sees.

NEW file; does not modify any existing script/env/loader.
"""
from __future__ import annotations

import ast
from typing import Dict, List, Optional, Tuple

# Robust to both run-as-script (scripts/data on sys.path) and import-as-module
# (scripts.data.<mod>) contexts, matching the other remap/data modules.
try:
    from sokoban_bfs_certify import certify_detailed, load_env_from_state
    from sokoban_branch_render import render_branch_frame
except ImportError:  # pragma: no cover - import-path shim
    from scripts.data.sokoban_bfs_certify import certify_detailed, load_env_from_state
    from scripts.data.sokoban_branch_render import render_branch_frame

from gymnasium.envs.sokoban.sokoban import (
    CHANGE_COORDINATES,
    DIRECTION_TO_PAYLOAD,
    LEGACY_ACTION_TO_KIND_DIRECTION,
    PAYLOAD_TO_DIRECTION,
    SokobanState,
)

# Legacy action ids that _enumerate_forward_action_ids emits for a BOX PUSH are
# direction+1 in {1,2,3,4}; a plain MOVE is direction+5 in {5,6,7,8}.
_PUSH_IDS = frozenset({1, 2, 3, 4})


# --------------------------------------------------------------------------- #
# Action (de)serialization                                                    #
# --------------------------------------------------------------------------- #
def _action_text(hist_entry: Dict) -> str:
    """The action string EXACTLY as the CoT-inf remap emits it: str(vlm_output or
    action)."""
    return str(hist_entry.get("vlm_output") or hist_entry.get("action"))


def _parse_action(text: str) -> Tuple[str, object]:
    """Parse a tuple-string action into (kind, payload). kind in {move,push,stop};
    payload is an int 1..4 for move/push, or 'stop' for the terminal action."""
    parsed = ast.literal_eval(text)
    return str(parsed[0]), parsed[1]


def _detect_action_mode(episode: Dict, parsed_kinds: List[str]) -> str:
    """'move_only' or 'move_push'. Prefer the row's declared env_kwargs/init_state
    action_mode; else infer from the expert action kinds (any explicit 'push' ->
    move_push, otherwise move_only)."""
    for src in (episode.get("env_kwargs") or {}, episode.get("init_state") or {}):
        am = src.get("action_mode")
        if am in ("move_only", "move_push"):
            return am
        if am in ("move_and_push", None) and am is not None:
            return "move_push"
    return "move_push" if any(k == "push" for k in parsed_kinds) else "move_only"


def _serialize_push(action_mode: str, direction: int) -> str:
    """Serialize a box-push in `direction` the way the source dataset would. In
    move_only the push is expressed as a move INTO the box: "('move', P)". In
    move_push it is an explicit "('push', P)". P = DIRECTION_TO_PAYLOAD[direction]."""
    payload = DIRECTION_TO_PAYLOAD[direction]
    kind = "move" if action_mode == "move_only" else "push"
    return f"('{kind}', {payload})"


# --------------------------------------------------------------------------- #
# Replay                                                                       #
# --------------------------------------------------------------------------- #
def _legacy_id_for(env, state: SokobanState, kind: str, payload: int) -> int:
    """Legacy forward-action id (1..4 push / 5..8 move) that (kind, payload) means
    at `state`. In move_only a 'move' INTO a box is really a push; an explicit
    'push' is always a push."""
    direction = PAYLOAD_TO_DIRECTION[int(payload)]
    change = CHANGE_COORDINATES[direction]
    ahead = (state.player[0] + change[0], state.player[1] + change[1])
    if kind == "push" or (kind == "move" and ahead in set(state.boxes)):
        return direction + 1  # push id
    return direction + 5      # move id


def _apply(env, state: SokobanState, kind: str, payload: int) -> SokobanState:
    """Apply one expert action, returning the next board state. Asserts the action
    is legal at `state` (it always is for a real solved episode); a violation means
    a replay bug, so we surface it loudly rather than fabricate a bad board."""
    legacy_id = _legacy_id_for(env, state, kind, payload)
    legal = set(env._enumerate_forward_action_ids(state))
    if legacy_id not in legal:
        raise ValueError(
            f"expert action ({kind},{payload}) -> legacy id {legacy_id} illegal at "
            f"player={state.player} boxes={state.boxes}; legal={sorted(legal)}")
    return env._simulate_action_id(state, legacy_id)


def _state_dict(room_fixed_list: List[List[int]], s: SokobanState) -> Dict:
    """Build an init_state-shaped dict SK1.certify consumes from a SokobanState."""
    return {
        "room_fixed": room_fixed_list,
        "player_position": [int(s.player[0]), int(s.player[1])],
        "box_positions": [[int(b[0]), int(b[1])] for b in s.boxes],
    }


# --------------------------------------------------------------------------- #
# Detector                                                                     #
# --------------------------------------------------------------------------- #
def detect_critical_step(episode: Dict) -> Optional[Dict]:
    """Find the earliest interior step whose counterfactual box-push proves a
    deadlock. See module docstring. Returns the branch spec dict or None."""
    history = episode.get("history") or []
    if not history:
        return None
    init_state = episode.get("init_state")
    if not init_state:
        return None

    env = load_env_from_state(init_state)
    room_fixed_list = env.room_fixed.tolist()

    texts = [_action_text(h) for h in history]
    parsed = [_parse_action(t) for t in texts]
    action_mode = _detect_action_mode(episode, [k for k, _ in parsed])
    n = len(history)

    # Replay: states[j] = board state S_j BEFORE expert action j (states[0] = init).
    states: List[SokobanState] = [env._current_state()]
    for kind, payload in parsed:
        if kind == "stop":
            states.append(states[-1])  # 'stop' does not change the board
        else:
            states.append(_apply(env, states[-1], kind, payload))

    for j in range(1, n):
        kind_j, payload_j = parsed[j]
        if kind_j == "stop":
            continue                       # never branch before the terminal stop
        s_j = states[j]
        if env._is_solved_state(s_j):
            continue                       # degenerate: already solved, expert stops
        expert_legacy = _legacy_id_for(env, s_j, kind_j, payload_j)

        deadlocks: List[Tuple[int, SokobanState, int]] = []
        for aid in sorted(env._enumerate_forward_action_ids(s_j)):
            if aid not in _PUSH_IDS or aid == expert_legacy:
                continue                   # only box-pushes that DIFFER from expert
            s_prime = env._simulate_action_id(s_j, aid)
            res = certify_detailed(_state_dict(room_fixed_list, s_prime))
            if res.verdict == "unsolvable":   # PROVEN deadlock only (never 'truncated')
                deadlocks.append((aid, s_prime, res.states_expanded))

        if deadlocks:
            aid, s_prime, states_expanded = deadlocks[0]   # lowest id: single clean push
            direction = LEGACY_ACTION_TO_KIND_DIRECTION[aid][1]
            return {
                "step": j,
                "alt_action": _serialize_push(action_mode, direction),
                "alt_deadlock_frame_b64": render_branch_frame(
                    s_prime.player, s_prime.boxes, room_fixed_list,
                    action_mode=action_mode,
                ),
                "expert_action": texts[j],
                # diagnostics (not part of the contract)
                "certify_states_expanded": int(states_expanded),
                "n_deadlock_pushes": len(deadlocks),
                "alt_push_legacy_id": int(aid),
                "action_mode": action_mode,
            }
    return None


# --------------------------------------------------------------------------- #
# Self-test                                                                    #
# --------------------------------------------------------------------------- #
def _load_solved_episodes(n_want: int = 30, src_root: str | None = None) -> List[Dict]:
    """First `n_want` solved easy episodes from the selected CoT-inf source."""
    import glob
    import gzip
    import json

    try:
        from remap_sokoban_cot_chunk_inf import DEFAULT_SRC
    except ImportError:
        from scripts.data.remap_sokoban_cot_chunk_inf import DEFAULT_SRC

    src_root = src_root or DEFAULT_SRC
    subdir = "trajectories/sokoban_easy/train"
    shards = sorted(glob.glob(f"{src_root}/{subdir}/*.jsonl.gz"))
    if not shards:
        raise FileNotFoundError(f"no shards under {src_root}/{subdir}")
    out: List[Dict] = []
    for shard in shards:
        with gzip.open(shard, "rt") as fin:
            for line in fin:
                if not line.strip():
                    continue
                row = json.loads(line)
                stats = row.get("stats") or {}
                if stats.get("reward") == 1.0 and stats.get("terminated"):
                    out.append(row)
                    if len(out) >= n_want:
                        return out
    return out


def _selftest(n_want: int = 30, src_root: str | None = None) -> int:
    print(f"[SK2 self-test] loading first ~{n_want} solved easy episodes ...", flush=True)
    episodes = _load_solved_episodes(n_want, src_root=src_root)
    print(f"[SK2 self-test] loaded {len(episodes)} solved episodes", flush=True)

    detected: List[Tuple[int, Dict]] = []
    n_fail = 0
    for i, ep in enumerate(episodes):
        try:
            res = detect_critical_step(ep)
        except Exception as exc:  # replay/parse bug -> report, keep going
            n_fail += 1
            print(f"  ep {i}: ERROR {type(exc).__name__}: {exc}", flush=True)
            continue
        if res is not None:
            detected.append((i, res))

    n = len(episodes)
    frac = len(detected) / n if n else 0.0
    print(
        f"[SK2 self-test] detected critical step in {len(detected)}/{n} "
        f"({frac:.0%}) episodes; replay failures={n_fail}",
        flush=True,
    )

    print("[SK2 self-test] 2 concrete examples:", flush=True)
    for i, res in detected[:2]:
        print(
            f"  ep {i}: step j={res['step']}  expert_action={res['expert_action']!r}  "
            f"alt_deadlocking_push={res['alt_action']!r}  "
            f"certify_states_expanded={res['certify_states_expanded']}  "
            f"(n_deadlock_pushes={res['n_deadlock_pushes']}, "
            f"frame_len={len(res['alt_deadlock_frame_b64'])})",
            flush=True,
        )

    # PASS = every episode processed without a replay error AND the detection rate
    # is in a sane band around the ~68% scoping figure.
    ok = (n > 0) and (n_fail == 0) and (0.40 <= frac <= 0.95) and (len(detected) >= 2)
    print(f"[SK2 self-test] RESULT: {'PASS' if ok else 'FAIL'}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Verify irreversible Sokoban branch detection.")
    parser.add_argument("--src-root", required=True)
    parser.add_argument("--episodes", type=int, default=30)
    args = parser.parse_args()
    sys.exit(_selftest(args.episodes, src_root=args.src_root))
