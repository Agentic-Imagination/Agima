from __future__ import annotations

import ast
import random
from collections import deque
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box, Discrete, FuncConditional, Text

from gymnasium.envs.sokoban.cheap_generator import generate_level
from gymnasium.envs.sokoban.render_utils import room_to_rgb


WALL = 0
FLOOR = 1
TARGET = 2
BOX_ON_TARGET = 3
BOX = 4
PLAYER = 5
PLAYER_ON_TARGET = 6

DIRECTION_NAMES = {
    1: "up",
    2: "down",
    3: "left",
    4: "right",
}

MOVE_ONLY_ACTION_LOOKUP = {
    1: "move up",
    2: "move down",
    3: "move left",
    4: "move right",
}

MOVE_ACTION_LOOKUP = {
    0: "move no operation",
    1: "move up",
    2: "move down",
    3: "move left",
    4: "move right",
}

PUSH_ACTION_LOOKUP = {
    1: "push up",
    2: "push down",
    3: "push left",
    4: "push right",
}

LEGACY_ACTION_LOOKUP = {
    0: "no operation",
    1: "push up",
    2: "push down",
    3: "push left",
    4: "push right",
    5: "move up",
    6: "move down",
    7: "move left",
    8: "move right",
}

CHANGE_COORDINATES = {
    0: (-1, 0),
    1: (1, 0),
    2: (0, -1),
    3: (0, 1),
}

PAYLOAD_TO_DIRECTION = {
    1: 0,
    2: 1,
    3: 2,
    4: 3,
}

DIRECTION_TO_PAYLOAD = {
    0: 1,
    1: 2,
    2: 3,
    3: 4,
}

LEGACY_ACTION_TO_KIND_DIRECTION = {
    1: ("push", 0),
    2: ("push", 1),
    3: ("push", 2),
    4: ("push", 3),
    5: ("move", 0),
    6: ("move", 1),
    7: ("move", 2),
    8: ("move", 3),
}


def _normalize_action_mode(action_mode: str | None) -> str:
    if action_mode in {None, "move_push", "move_and_push"}:
        return "move_push"
    if action_mode == "move_only":
        return "move_only"
    raise ValueError(f"Unsupported Sokoban action_mode: {action_mode!r}")


def _normalize_success_mode(success_mode: str | None) -> str:
    if success_mode in {None, "stop_then_solved"}:
        return "stop_then_solved"
    if success_mode == "auto_solved":
        return "auto_solved"
    raise ValueError(f"Unsupported Sokoban success_mode: {success_mode!r}")


@dataclass(frozen=True)
class SokobanState:
    player: Tuple[int, int]
    boxes: Tuple[Tuple[int, int], ...]


class BaseSokobanEnv(gym.Env, gym.VLMEnvMixin):
    metadata = {"render_modes": ["rgb_array", "ansi"]}

    def __init__(
        self,
        dim_room: Tuple[int, int],
        max_steps: int,
        num_boxes: int,
        num_gen_steps: int,
        seed: Optional[int] = None,
        render_mode: str = "rgb_array",
        tile_size: int = 16,
        legacy_single_move_branch: bool = False,
        action_mode: str = "move_push",
        action_mode_override: str | None = None,
        success_mode: str = "stop_then_solved",
    ):
        super().__init__()
        self.dim_room = tuple(int(v) for v in dim_room)
        self.max_steps = int(max_steps)
        self.num_boxes = int(num_boxes)
        self.num_gen_steps = int(num_gen_steps)
        self.render_mode = render_mode
        self.tile_size = int(tile_size)
        self.legacy_single_move_branch = bool(legacy_single_move_branch)
        self._forced_action_mode = (
            None if action_mode_override is None else _normalize_action_mode(action_mode_override)
        )
        self.action_mode = (
            self._forced_action_mode if self._forced_action_mode is not None else _normalize_action_mode(action_mode)
        )
        self.success_mode = _normalize_success_mode(success_mode)

        height = self.dim_room[0] * self.tile_size
        width = self.dim_room[1] * self.tile_size
        self.action_space = self._build_action_space()
        self.observation_space = Box(
            low=0,
            high=255,
            shape=(height, width, 3),
            dtype=np.uint8,
        )

        self.seed(seed)
        self.room_fixed: np.ndarray | None = None
        self.player_position: Tuple[int, int] | None = None
        self.box_positions: Tuple[Tuple[int, int], ...] = ()
        self._cached_solution_actions: Optional[Tuple[str, ...]] = None
        self._cached_solution_schema: str = "move_push_internal"
        self.step_count = 0
        self.env_feedback = ""
        self._last_is_success = False

    def seed(self, seed: Optional[int] = None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def _build_action_space(self):
        if self.action_mode == "move_only":
            return FuncConditional(
                {
                    "move": Discrete(4, start=1),
                    "stop": Text(4),
                }
            )
        if self.legacy_single_move_branch:
            return FuncConditional(
                {
                    "move": Discrete(9),
                    "stop": Text(4),
                }
            )
        return FuncConditional(
            {
                "move": Discrete(5),
                "push": Discrete(5),
                "stop": Text(4),
            }
        )

    def get_prompt(self, **kwargs) -> str:
        if self.action_mode == "move_only":
            return (
                "You are playing Sokoban, a box-pushing puzzle. "
                "Your job is to control the green animal character and push every crate onto a goal tile.\n"
                "The layout is a small top-down room. Walls are stone tiles, normal floor is pale, crates are movable boxes, "
                "goal tiles are highlighted floor markers, and you are the green animal.\n\n"
                "Available actions:\n"
                "1. `move`: Move in one of four directions. Format: `('move', action_id)`\n"
                "- 1 = move up\n"
                "- 2 = move down\n"
                "- 3 = move left\n"
                "- 4 = move right\n"
                "If a crate is directly in the chosen direction and the next cell is free, the move will push that crate.\n"
                "2. `stop`: End the episode. Format: `('stop', 'stop')`\n\n"
                "Rules: you may move into empty floor or goal tiles, and moving into a crate pushes it only if the cell behind the crate is empty.\n"
                "Success is only scored when you submit with `('stop', 'stop')` after every crate is on a goal tile. "
                "If you stop before the puzzle is solved, the episode ends with zero reward.\n\n"
                "Examples:\n"
                "- Move up: `('move', 1)`\n"
                "- Move left: `('move', 3)`\n"
                "- Stop: `('stop', 'stop')`"
            )
        if self.legacy_single_move_branch:
            return (
                "You are playing Sokoban, a box-pushing puzzle. "
                "Your job is to control the green animal character and push every crate onto a goal tile.\n"
                "The layout is a small top-down room. Walls are stone tiles, normal floor is pale, crates are movable boxes, "
                "goal tiles are highlighted floor markers, and you are the green animal.\n\n"
                "Available actions:\n"
                "1. `move`: Execute one Sokoban move. Format: `('move', action_id)`\n"
                "- 0 = no operation\n"
                "- 1 = push up, 2 = push down, 3 = push left, 4 = push right\n"
                "- 5 = move up, 6 = move down, 7 = move left, 8 = move right\n"
                "2. `stop`: End the episode. Format: `('stop', 'stop')`\n\n"
                "Rules: action ids 1-4 require a crate directly in that direction and an empty destination behind it; "
                "action ids 5-8 move without pushing and therefore require the destination cell itself to be empty.\n"
                "Success is only scored when you submit with `('stop', 'stop')` after every crate is on a goal tile. "
                "If you stop before the puzzle is solved, the episode ends with zero reward.\n\n"
                "Examples:\n"
                "- Push right: `('move', 4)`\n"
                "- Move left: `('move', 7)`\n"
                "- Stop: `('stop', 'stop')`"
            )
        return (
            "You are playing Sokoban, a box-pushing puzzle. "
            "Your job is to control the green animal character and push every crate onto a goal tile.\n"
            "The layout is a small top-down room. Walls are stone tiles, normal floor is pale, crates are movable boxes, "
            "goal tiles are highlighted floor markers, and you are the green animal.\n\n"
            "Available actions:\n"
            "1. `move`: Move without pushing a crate. Format: `('move', action_id)`\n"
            "- 0 = no operation\n"
            "- 1 = move up, 2 = move down, 3 = move left, 4 = move right\n"
            "2. `push`: Attempt to push a crate. Format: `('push', action_id)`\n"
            "- 1 = push up, 2 = push down, 3 = push left, 4 = push right\n"
            "3. `stop`: End the episode. Format: `('stop', 'stop')`\n\n"
            "Rules: you may move into empty floor or goal tiles, and you may push a crate only if the cell behind it is empty.\n"
            "Success is only scored when you submit with `('stop', 'stop')` after every crate is on a goal tile. "
            "If you stop before the puzzle is solved, the episode ends with zero reward.\n\n"
            "Examples:\n"
            "- Move left: `('move', 3)`\n"
            "- Push right: `('push', 4)`\n"
            "- Stop: `('stop', 'stop')`"
        )

    def get_action_lookup(self) -> Dict[int, str]:
        if self.action_mode == "move_only":
            return dict(MOVE_ONLY_ACTION_LOOKUP)
        return dict(LEGACY_ACTION_LOOKUP)

    def get_init_state(self) -> Dict:
        return {
            "room_fixed": self.room_fixed.tolist(),
            "player_position": list(self.player_position),
            "box_positions": [list(pos) for pos in self.box_positions],
            "step_count": int(self.step_count),
            "max_steps": int(self.max_steps),
            "dim_room": list(self.dim_room),
            "num_boxes": int(self.num_boxes),
            "num_gen_steps": int(self.num_gen_steps),
            "legacy_single_move_branch": bool(self.legacy_single_move_branch),
            "action_mode": str(self.action_mode),
            "success_mode": str(self.success_mode),
            "constructive_solution": list(self._cached_solution_actions) if self._cached_solution_actions else None,
            "constructive_solution_schema": str(self._cached_solution_schema),
        }

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
        init_state: Optional[Dict] = None,
    ):
        super().reset(seed=seed)
        if seed is not None:
            self.np_random, _ = gym.utils.seeding.np_random(seed)
            random.seed(int(seed))
            np.random.seed(int(seed) % (2**32))

        if init_state is not None:
            self.room_fixed = np.asarray(init_state["room_fixed"], dtype=np.int8)
            self.player_position = tuple(int(v) for v in init_state["player_position"])
            # Box positions can come in two upstream schemas:
            #   - "box_positions": [[r,c], ...]            (this repo's get_init_state output)
            #   - "box_mapping":   {goal_pos_str: [r,c]}   (visgym_3d / sokoban_v6 jsonl)
            if "box_positions" in init_state:
                _box_positions_raw = init_state["box_positions"]
            elif "box_mapping" in init_state:
                _box_positions_raw = list(init_state["box_mapping"].values())
            else:
                raise KeyError(
                    "init_state must contain 'box_positions' or 'box_mapping'"
                )
            self.box_positions = tuple(
                sorted(tuple(int(v) for v in pos) for pos in _box_positions_raw)
            )
            cached_solution = init_state.get("constructive_solution")
            self._cached_solution_actions = tuple(cached_solution) if isinstance(cached_solution, list) else None
            self._cached_solution_schema = str(init_state.get("constructive_solution_schema", "move_push_internal"))
            self.step_count = int(init_state.get("step_count", 0))
            if "legacy_single_move_branch" in init_state:
                self.legacy_single_move_branch = bool(init_state["legacy_single_move_branch"])
            if "action_mode" in init_state:
                self.action_mode = _normalize_action_mode(init_state["action_mode"])
            if "success_mode" in init_state:
                self.success_mode = _normalize_success_mode(init_state["success_mode"])
            if self._forced_action_mode is not None:
                if self.action_mode != self._forced_action_mode and self._cached_solution_schema == "move_only":
                    self._cached_solution_actions = None
                    self._cached_solution_schema = "move_push_internal"
                self.action_mode = self._forced_action_mode
            self.action_space = self._build_action_space()
        else:
            self.room_fixed, self.player_position, self.box_positions, self._cached_solution_actions = self._generate_level()
            self._cached_solution_schema = "move_push_internal"
            self.step_count = 0
            if self._forced_action_mode is not None:
                self.action_mode = self._forced_action_mode
            self.action_space = self._build_action_space()

        self.env_feedback = ""
        self._last_is_success = False
        return self.render(), self._get_info()

    def step(self, action: str):
        terminated = False
        truncated = False
        reward = 0.0
        self._last_is_success = False

        try:
            parsed = ast.literal_eval(action)
        except Exception as exc:
            self.env_feedback = self._format_env_feedback(
                executed=False,
                extra=f"Cannot parse action string {action!r}: {exc}.",
            )
            return self.render(), reward, terminated, truncated, self._get_info()

        if not isinstance(parsed, tuple) or len(parsed) != 2:
            self.env_feedback = self._format_env_feedback(
                executed=False,
                extra="Action must be a tuple of the form (branch, payload).",
            )
            return self.render(), reward, terminated, truncated, self._get_info()

        branch, payload = parsed
        if branch in {"move", "push"}:
            _changed, consumed_step, feedback = self._apply_branch_payload(
                self._current_state(),
                branch,
                payload,
                mutate=True,
            )
            self._cached_solution_actions = None
            self.env_feedback = feedback
            if consumed_step:
                self.step_count += 1
        elif branch == "stop":
            if payload != "stop":
                self.env_feedback = self._format_env_feedback(
                    executed=False,
                    extra="For the stop action the payload must be 'stop'.",
                )
            else:
                terminated = True
                self._last_is_success = self._is_solved()
                reward = 1.0 if self._last_is_success else 0.0
                self._cached_solution_actions = None
                self.env_feedback = self._format_env_feedback(
                    executed=True,
                    is_success=self._last_is_success,
                )
        else:
            self.env_feedback = self._format_env_feedback(
                executed=False,
                extra=f"Unknown action branch {branch!r}.",
            )

        if self.success_mode == "auto_solved" and not terminated and self._is_solved():
            reward = 1.0
            terminated = True
            self._last_is_success = True
            self.env_feedback = self._format_env_feedback(
                executed=True,
                is_success=True,
            )

        if self.step_count >= self.max_steps and not terminated:
            truncated = True
            self.env_feedback = self._format_env_feedback(
                executed=True,
                extra=f"Reached max_steps={self.max_steps}.",
                is_success=False,
            )

        return self.render(), reward, terminated, truncated, self._get_info()

    def _convert_solution_actions_for_mode(
        self,
        actions: Sequence[str],
        *,
        source_schema: str,
    ) -> List[str]:
        if self.action_mode == "move_push":
            if source_schema == "move_only":
                raise RuntimeError("Cannot reuse cached move_only constructive_solution in move_push mode.")
            return list(actions)

        out: List[str] = []
        for action in actions:
            parsed = ast.literal_eval(action)
            branch, payload = parsed
            if branch == "stop":
                out.append(str(("stop", "stop")))
                continue
            if source_schema == "move_only":
                out.append(str(("move", int(payload))))
                continue
            if branch == "push" and isinstance(payload, int) and payload in PAYLOAD_TO_DIRECTION:
                out.append(str(("move", int(payload))))
                continue
            if branch == "move" and isinstance(payload, int):
                if payload in PAYLOAD_TO_DIRECTION:
                    out.append(str(("move", int(payload))))
                    continue
                if payload in LEGACY_ACTION_TO_KIND_DIRECTION:
                    _, direction = LEGACY_ACTION_TO_KIND_DIRECTION[int(payload)]
                    out.append(str(("move", int(DIRECTION_TO_PAYLOAD[direction]))))
                    continue
            raise RuntimeError(f"Unsupported constructive action {action!r} for move_only mode.")
        return out

    def render(self, mode: str = "rgb_array"):
        if mode == "ansi":
            return self._render_ansi()
        if mode != "rgb_array":
            raise NotImplementedError(f"Invalid render mode: {mode}")
        return self._render_rgb()

    def close(self):
        return None

    @property
    def room_state(self) -> np.ndarray:
        grid = self.room_fixed.copy()
        for box in self.box_positions:
            grid[box] = BOX_ON_TARGET if self.room_fixed[box] == TARGET else BOX
        if self.player_position is not None:
            grid[self.player_position] = (
                PLAYER_ON_TARGET if self.room_fixed[self.player_position] == TARGET else PLAYER
            )
        return grid

    def solve(
        self,
        strategy: str = "constructive",
        state_limit: int = 50000,
        fallback_strategy: Optional[str] = None,
    ) -> List[str]:
        strategy_name = str(strategy).lower()
        if strategy_name in {"constructive", "cached"}:
            if self._cached_solution_actions is not None:
                return self._convert_solution_actions_for_mode(
                    self._cached_solution_actions,
                    source_schema=self._cached_solution_schema,
                )
            if fallback_strategy:
                return self.solve(strategy=fallback_strategy, state_limit=state_limit)
            raise RuntimeError("Constructive Sokoban solution is unavailable for the current state.")
        if strategy_name == "auto":
            if self._cached_solution_actions is not None:
                return self._convert_solution_actions_for_mode(
                    self._cached_solution_actions,
                    source_schema=self._cached_solution_schema,
                )
            strategy_name = "bfs"
        if strategy_name != "bfs":
            raise ValueError(f"Unsupported Sokoban solve strategy: {strategy}")

        bfs_actions = self._solve_bfs(state_limit=state_limit)
        if bfs_actions is not None:
            return self._convert_solution_actions_for_mode(
                bfs_actions,
                source_schema="move_push_internal",
            )

        raise RuntimeError(
            f"BFS failed to find a solution within the configured state_limit={state_limit}."
        )

    def _generate_level(
        self,
        max_attempts: int = 64,
    ) -> Tuple[np.ndarray, Tuple[int, int], Tuple[Tuple[int, int], ...], Tuple[str, ...]]:
        for _ in range(max_attempts):
            try:
                room_fixed, player, box_positions, constructive_solution = generate_level(
                    dim_room=self.dim_room,
                    num_boxes=self.num_boxes,
                    num_gen_steps=self.num_gen_steps,
                    rng=self.np_random,
                )
            except RuntimeError:
                continue

            return np.asarray(room_fixed, dtype=np.int8), player, box_positions, constructive_solution

        raise RuntimeError("Unable to generate a Sokoban level from the cheap native generator.")

    def _format_env_feedback(
        self,
        *,
        executed: bool,
        extra: Optional[str] = None,
        is_success: Optional[bool] = None,
    ) -> str:
        prefix = "Action executed successfully." if executed else "Action executed unsuccessfully."
        solved_text = "true" if self._is_solved() else "false"
        success_text = "true" if (self._last_is_success if is_success is None else is_success) else "false"
        if extra:
            return f"{prefix} Solved: {solved_text}. Success: {success_text}. {extra}"
        return f"{prefix} Solved: {solved_text}. Success: {success_text}."

    def _apply_forward_action(
        self,
        state: SokobanState,
        branch: str,
        direction: int,
        *,
        mutate: bool,
    ) -> Tuple[bool, str]:
        change = CHANGE_COORDINATES[direction]
        next_pos = (state.player[0] + change[0], state.player[1] + change[1])
        boxes = set(state.boxes)

        if branch == "move":
            if not self._is_floor(next_pos, boxes=state.boxes):
                return False, f"Cannot move {DIRECTION_NAMES[DIRECTION_TO_PAYLOAD[direction]]} into an occupied cell."
            next_state = SokobanState(player=next_pos, boxes=tuple(sorted(boxes)))
        else:
            if next_pos not in boxes:
                return False, f"No box to push {DIRECTION_NAMES[DIRECTION_TO_PAYLOAD[direction]]}."
            box_target = (next_pos[0] + change[0], next_pos[1] + change[1])
            if not self._is_floor(box_target, boxes=state.boxes):
                return False, f"Cannot push {DIRECTION_NAMES[DIRECTION_TO_PAYLOAD[direction]]} because the destination is blocked."
            boxes.remove(next_pos)
            boxes.add(box_target)
            next_state = SokobanState(player=next_pos, boxes=tuple(sorted(boxes)))

        if mutate:
            self.player_position = next_state.player
            self.box_positions = next_state.boxes
        return True, self._format_env_feedback(executed=True)

    def _apply_branch_payload(
        self,
        state: SokobanState,
        branch: str,
        payload: int,
        *,
        mutate: bool,
    ) -> Tuple[bool, bool, str]:
        if self.action_mode == "move_only":
            if branch != "move":
                return False, False, self._format_env_feedback(
                    executed=False,
                    extra=f"Unknown action branch {branch!r}.",
                )
            if not isinstance(payload, int) or payload not in PAYLOAD_TO_DIRECTION:
                return False, False, self._format_env_feedback(
                    executed=False,
                    extra=f"Invalid action id {payload!r} for move_only mode.",
                )
            direction = PAYLOAD_TO_DIRECTION[payload]
            action_label = MOVE_ONLY_ACTION_LOOKUP[payload]
            next_pos = (
                state.player[0] + CHANGE_COORDINATES[direction][0],
                state.player[1] + CHANGE_COORDINATES[direction][1],
            )
            if next_pos in set(state.boxes):
                changed, feedback = self._apply_forward_action(state, "push", direction, mutate=mutate)
            else:
                changed, feedback = self._apply_forward_action(state, "move", direction, mutate=mutate)
            if changed:
                return changed, True, feedback
            return changed, True, self._format_env_feedback(
                executed=False,
                extra=f"{action_label} failed.",
            )

        if not isinstance(payload, int):
            return False, False, self._format_env_feedback(
                executed=False,
                extra=f"Invalid action id {payload!r} for branch {branch!r}.",
            )

        action_label: Optional[str] = None
        effective_branch = branch
        direction: Optional[int] = None

        if branch == "move" and self.legacy_single_move_branch and payload in LEGACY_ACTION_LOOKUP:
            return self._apply_legacy_action_id(state, payload, mutate=mutate)

        if branch == "move":
            if payload == 0:
                return True, True, self._format_env_feedback(executed=True)
            if payload in PAYLOAD_TO_DIRECTION:
                direction = PAYLOAD_TO_DIRECTION[payload]
                action_label = MOVE_ACTION_LOOKUP[payload]
            elif payload in {5, 6, 7, 8}:
                effective_branch, direction = LEGACY_ACTION_TO_KIND_DIRECTION[payload]
                action_label = LEGACY_ACTION_LOOKUP[payload]
            else:
                return False, False, self._format_env_feedback(
                    executed=False,
                    extra=f"Invalid action id {payload!r} for branch 'move'.",
                )
        elif branch == "push":
            if payload in PAYLOAD_TO_DIRECTION:
                direction = PAYLOAD_TO_DIRECTION[payload]
                action_label = PUSH_ACTION_LOOKUP[payload]
            else:
                return False, False, self._format_env_feedback(
                    executed=False,
                    extra=f"Invalid action id {payload!r} for branch 'push'.",
                )
        else:
            return False, False, self._format_env_feedback(
                executed=False,
                extra=f"Unknown action branch {branch!r}.",
            )

        changed, feedback = self._apply_forward_action(state, effective_branch, direction, mutate=mutate)
        if changed:
            return changed, True, feedback
        return changed, True, self._format_env_feedback(
            executed=False,
            extra=f"{action_label} failed.",
        )

    def _apply_legacy_action_id(
        self,
        state: SokobanState,
        action_id: int,
        *,
        mutate: bool,
    ) -> Tuple[bool, bool, str]:
        if int(action_id) == 0:
            return True, True, self._format_env_feedback(executed=True)
        branch, direction = LEGACY_ACTION_TO_KIND_DIRECTION[int(action_id)]
        changed, feedback = self._apply_forward_action(state, branch, direction, mutate=mutate)
        if changed:
            return changed, True, feedback
        return changed, True, self._format_env_feedback(
            executed=False,
            extra=f"{LEGACY_ACTION_LOOKUP[int(action_id)]} failed.",
        )

    def _solve_bfs(self, state_limit: int) -> Optional[List[str]]:
        start = self._current_state()
        if self._is_solved_state(start):
            return ["('stop', 'stop')"]

        queue = deque([start])
        parents: Dict[SokobanState, Tuple[Optional[SokobanState], Optional[str]]] = {
            start: (None, None)
        }
        explored = 0

        while queue:
            state = queue.popleft()
            explored += 1
            if explored > max(1, int(state_limit)):
                return None

            for action_id in self._enumerate_forward_action_ids(state):
                changed, _, _ = self._apply_legacy_action_id(state, action_id, mutate=False)
                if not changed:
                    continue
                next_state = self._simulate_action_id(state, action_id)
                if next_state in parents:
                    continue
                legacy_branch, direction = LEGACY_ACTION_TO_KIND_DIRECTION[int(action_id)]
                action_str = str((legacy_branch, int(DIRECTION_TO_PAYLOAD[direction])))
                parents[next_state] = (state, action_str)
                if self._is_solved_state(next_state):
                    path = self._reconstruct_actions(next_state, parents)
                    path.append("('stop', 'stop')")
                    return path
                queue.append(next_state)

        return None

    def _simulate_forward_action(
        self,
        state: SokobanState,
        branch: str,
        direction: int,
    ) -> SokobanState:
        change = CHANGE_COORDINATES[direction]
        next_pos = (state.player[0] + change[0], state.player[1] + change[1])
        boxes = set(state.boxes)
        if branch == "move":
            return SokobanState(player=next_pos, boxes=tuple(sorted(boxes)))
        boxes.remove(next_pos)
        boxes.add((next_pos[0] + change[0], next_pos[1] + change[1]))
        return SokobanState(player=next_pos, boxes=tuple(sorted(boxes)))

    def _simulate_action_id(
        self,
        state: SokobanState,
        action_id: int,
    ) -> SokobanState:
        branch, direction = LEGACY_ACTION_TO_KIND_DIRECTION[int(action_id)]
        return self._simulate_forward_action(state, branch, direction)

    def _enumerate_forward_action_ids(
        self,
        state: SokobanState,
    ) -> Iterable[int]:
        boxes = set(state.boxes)
        for direction, change in CHANGE_COORDINATES.items():
            next_pos = (state.player[0] + change[0], state.player[1] + change[1])
            if next_pos in boxes:
                beyond = (next_pos[0] + change[0], next_pos[1] + change[1])
                if self._is_floor(beyond, boxes=state.boxes):
                    yield direction + 1
            elif self._is_floor(next_pos, boxes=state.boxes):
                yield direction + 5

    def _reconstruct_actions(
        self,
        goal_state: SokobanState,
        parents: Dict[SokobanState, Tuple[Optional[SokobanState], Optional[str]]],
    ) -> List[str]:
        out: List[str] = []
        cursor = goal_state
        while True:
            parent, action = parents[cursor]
            if parent is None or action is None:
                break
            out.append(action)
            cursor = parent
        out.reverse()
        return out

    def _current_state(self) -> SokobanState:
        return SokobanState(player=tuple(self.player_position), boxes=tuple(self.box_positions))

    def _is_floor(
        self,
        pos: Tuple[int, int],
        *,
        boxes: Tuple[Tuple[int, int], ...],
    ) -> bool:
        row, col = pos
        if row < 0 or row >= self.dim_room[0] or col < 0 or col >= self.dim_room[1]:
            return False
        if self.room_fixed[row, col] == WALL:
            return False
        return pos not in boxes

    def _is_solved(self) -> bool:
        return self._is_solved_state(self._current_state())

    def _is_solved_state(self, state: SokobanState) -> bool:
        return all(self.room_fixed[box] == TARGET for box in state.boxes)

    def _get_info(self) -> Dict:
        boxes_on_target = int(sum(self.room_fixed[pos] == TARGET for pos in self.box_positions))
        return {
            "env_feedback": self.env_feedback,
            "is_solved": self._is_solved(),
            "is_success": bool(self._last_is_success),
            "success_mode": str(self.success_mode),
            "boxes_on_target": boxes_on_target,
            "num_boxes": int(self.num_boxes),
            "steps": int(self.step_count),
            "max_steps": int(self.max_steps),
            "num_gen_steps": int(self.num_gen_steps),
            "action_mode": str(self.action_mode),
        }

    def _render_ansi(self) -> str:
        chars = np.full(self.dim_room, " ", dtype="<U1")
        chars[self.room_fixed == WALL] = "#"
        chars[self.room_fixed == TARGET] = "."
        for box in self.box_positions:
            chars[box] = "*" if self.room_fixed[box] == TARGET else "$"
        chars[self.player_position] = "+" if self.room_fixed[self.player_position] == TARGET else "@"
        return "\n".join("".join(row) for row in chars)

    def _render_rgb(self) -> np.ndarray:
        return room_to_rgb(self.room_state, self.room_fixed)


class SokobanEnv(BaseSokobanEnv):
    def __init__(self, **kwargs):
        kwargs.setdefault("dim_room", (10, 10))
        kwargs.setdefault("max_steps", 120)
        kwargs.setdefault("num_boxes", 4)
        kwargs.setdefault("num_gen_steps", 34)
        super().__init__(**kwargs)


class SokobanSmallEnv(BaseSokobanEnv):
    def __init__(self, **kwargs):
        kwargs.setdefault("dim_room", (7, 7))
        kwargs.setdefault("max_steps", 200)
        kwargs.setdefault("num_boxes", 2)
        kwargs.setdefault("num_gen_steps", 23)
        super().__init__(**kwargs)
